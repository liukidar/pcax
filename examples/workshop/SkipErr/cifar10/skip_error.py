from typing import Callable
import math
from pathlib import Path
import logging
import sys
import argparse

# Core dependencies
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

# pcax
import pcax as px
import pcax.predictive_coding as pxc
import pcax.nn as pxnn
import pcax.utils as pxu
import pcax.functional as pxf
from pcax import RKG

sys.path.insert(0, "../../../")
from data_utils import get_vision_dataloaders, seed_everything, get_config_value  # noqa: E402

sys.path.pop(0)


def seed_pcax_and_everything(seed: int | None = None):
    if seed is None:
        seed = 0
    RKG.seed(seed)
    seed_everything(seed)


logging.basicConfig(level=logging.INFO)


STATUS_FORWARD = "forward"
ACTIVATION_FUNCS = {
    None: lambda x: x,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "hard_tanh": jax.nn.hard_tanh,
    "sigmoid": jax.nn.sigmoid,
}


class SkipError(pxc.EnergyModule):
    def __init__(
        self,
        num_layers: int,
        input_dim: tuple[int, int, int],
        hidden_dim: int,
        num_classes: int,
        skip_error_layer_indices: list,
        freeze_skip_layers: bool,
        act_fn: Callable[[jax.Array], jax.Array],
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.act_fn = px.static(act_fn)
        self.skip_error_layer_indices = px.static(skip_error_layer_indices)

        self.layer_dims = [math.prod(input_dim)] + [hidden_dim for _ in range(num_layers - 1)] + [num_classes]

        self.layers = []
        for layer_input, layer_output in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            self.layers.append(pxnn.Linear(layer_input, layer_output))

        self.skip_error_layers = [
            pxnn.Linear(num_classes, hidden_dim, bias=False) for i in range(len(skip_error_layer_indices))
        ]

        self.vodes = []
        for layer_output in self.layer_dims[1:-1]:
            self.vodes.append(pxc.Vode())
        self.vodes.append(pxc.Vode(pxc.se_energy))
        self.vodes[-1].h.frozen = True

        if freeze_skip_layers:
            for index in self.skip_error_layer_indices:
                self.vodes[index].h.frozen = True

    def __call__(self, x, y=None, beta=0.0):
        x = x.flatten()
        for i, layer in enumerate(self.layers):
            act_fn = self.act_fn if i < len(self.layers) - 1 else lambda x: x
            x = layer(x)
            x = act_fn(self.vodes[i](x))

        if y is not None:
            self.vodes[-1].set("h", y)

            error = y - self.vodes[-1].get("u")

            for i, index in enumerate(self.skip_error_layer_indices):
                skip_error = self.skip_error_layers[i](error)
                current_state = self.vodes[index].get("h")
                # jax.debug.print("current state variance: {x}", x=jnp.var(current_state))
                # jax.debug.print("skip error variance: {x}", x=jnp.var(skip_error))
                self.vodes[index].set("h", current_state - beta * (current_state - skip_error))

        return self.vodes[-1].get("u")


@pxf.vmap(pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)), in_axes=(0, 0, None), out_axes=0)
def forward(x, y=None, beta=0.0, *, model: SkipError):
    return model(x, y, beta)


@pxf.vmap(
    pxu.M(pxc.VodeParam | pxc.VodeParam.Cache).to((None, 0)),
    in_axes=(0,),
    out_axes=(None, 0),
    axis_name="batch",
)
def energy(x, *, model: SkipError):
    y_ = model(x)
    return jax.lax.psum(model.energy(), "batch"), y_


@pxf.jit(static_argnums=0)
def train_on_batch(
    T: int, x: jax.Array, y: jax.Array, beta: float, *, model: SkipError, optim_w: pxu.Optim, optim_h: pxu.Optim
):
    model.train()

    # Init step
    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        forward(x, y, beta, model=model)
    optim_h.init(pxu.M_hasnot(pxc.VodeParam, frozen=True)(model))

    # Inference steps
    for _ in range(T):
        with pxu.step(model, clear_params=pxc.VodeParam.Cache):
            _, g = pxf.value_and_grad(pxu.M_hasnot(pxc.VodeParam, frozen=True).to(([False, True])), has_aux=True)(
                energy
            )(x, model=model)

        optim_h.step(model, g["model"])
    optim_h.clear()

    # Learning step
    with pxu.step(model, clear_params=pxc.VodeParam.Cache):
        _, g = pxf.value_and_grad(pxu.M(pxnn.LayerParam).to([False, True]), has_aux=True)(energy)(x, model=model)

        for i, index in enumerate(model.skip_error_layer_indices):
            error = jnp.expand_dims(model.vodes[-1].get("h") - model.vodes[-1].get("u"), -1)
            state = jnp.expand_dims(model.vodes[index].get("h"), -1)
            grad = jnp.mean(
                -state @ jnp.transpose(error, (0, 2, 1)),
                axis=0,
            )
            # jax.debug.print("grad variance: {x}", x=jnp.var(grad))
            g["model"].skip_error_layers[i].nn.weight.set(grad)
    optim_w.step(model, g["model"], scale_by=1.0 / x.shape[0])


@pxf.jit()
def eval_on_batch(x: jax.Array, y: jax.Array, *, model: SkipError):
    model.eval()

    with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
        y_ = forward(x, None, 0.0, model=model).argmax(axis=-1)

    return (y_ == y).mean(), y_


def train(dl, T, *, model: SkipError, optim_w: pxu.Optim, optim_h: pxu.Optim, beta: float):
    for i, (x, y) in enumerate(dl):
        train_on_batch(T, x, jax.nn.one_hot(y, 10), beta, model=model, optim_w=optim_w, optim_h=optim_h)

    for i in range(len(model.skip_error_layers)):
        print(f"E{i} variance: ", jnp.var(model.skip_error_layers[i].nn.weight.get()))
        print(f"E{i} mean: ", jnp.mean(model.skip_error_layers[i].nn.weight.get()))


def eval(dl, *, model: SkipError):
    acc = []
    ys_ = []

    for x, y in dl:
        a, y_ = eval_on_batch(x, y, model=model)
        acc.append(a)
        ys_.append(y_)

    return np.mean(acc), np.concatenate(ys_)


def run_experiment(
    *,
    dataset_name: str,
    num_classes: int,
    num_layers: int,
    hidden_dim: int,
    skip_error_layer_indices: list[int],
    freeze_skip_layers: bool,
    beta: float,
    beta_annealing: bool,
    act_fn: str | None,
    batch_size: int,
    epochs: int,
    T: int,
    optim_x_lr: float,
    optim_x_momentum: float,
    optim_w_name: str,
    optim_w_lr: float,
    optim_w_wd: float,
    optim_w_momentum: float,
    checkpoint_dir: Path | None = None,
    seed: int | None = None,
) -> float:
    seed_pcax_and_everything(seed)

    # Channel first: (batch, channel, height, width)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_vision_dataloaders(dataset_name=dataset_name, batch_size=batch_size, should_normalize=False)

    input_dim = dataset.train_dataset[0][0].shape

    model = SkipError(
        num_layers=num_layers,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        act_fn=ACTIVATION_FUNCS[act_fn],
        skip_error_layer_indices=skip_error_layer_indices,
        freeze_skip_layers=freeze_skip_layers,
    )

    # with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
    #     forward(jnp.zeros((batch_size, math.prod(input_dim))), None, model=model)

    optim_h = pxu.Optim(optax.sgd(learning_rate=optim_x_lr, momentum=optim_x_momentum))
    mask = pxu.M(pxnn.LayerParam)(model)
    # mask.skip_error_layers = jax.tree_util.tree_map(
    #     lambda x: None, mask.skip_error_layers, is_leaf=lambda x: isinstance(x, pxnn.LayerParam)
    # )

    if optim_w_name == "adamw":
        optim_w = pxu.Optim(
            optax.adamw(learning_rate=optim_w_lr, weight_decay=optim_w_wd),
            mask,
        )
    elif optim_w_name == "sgd":
        optim_w = pxu.Optim(optax.sgd(learning_rate=optim_w_lr, momentum=optim_w_momentum), mask)
    else:
        raise ValueError(f"Unknown optimizer name: {optim_w_name}")

    model_save_dir: Path | None = checkpoint_dir / dataset_name / "best_model" if checkpoint_dir is not None else None
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)

    print("Training...")

    best_acc: float | None = None
    test_acc: list[float] = []
    for epoch in range(epochs):
        effective_beta = beta
        if beta_annealing:
            effective_beta = max(0.1, min(1.0, (beta + 1.0) / (epoch + 1.0)))
        train(
            dataset.train_dataloader,
            T=T,
            model=model,
            optim_w=optim_w,
            optim_h=optim_h,
            beta=effective_beta,
        )
        mean_acc, _ = eval(dataset.test_dataloader, model=model)
        if np.isnan(mean_acc):
            logging.warning("Model diverged. Stopping training.")
            break
        test_acc.append(mean_acc)
        if epochs > 1 and model_save_dir is not None and (best_acc is None or mean_acc >= best_acc):
            best_acc = mean_acc
        print(f"Epoch {epoch + 1}/{epochs} - Test Accuracy: {mean_acc:.4f}")

    print(f"\nBest accuracy: {best_acc}")

    return min(test_acc) if test_acc else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/skiperror_cifar10_adamw_hypertune.yaml", help="Path to the config file."
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    run_experiment(
        dataset_name=get_config_value(config, "dataset_name"),
        num_classes=get_config_value(config, "num_classes"),
        seed=get_config_value(config, "seed", required=False),
        num_layers=get_config_value(config, "hp/num_layers"),
        hidden_dim=get_config_value(config, "hp/hidden_dim"),
        skip_error_layer_indices=get_config_value(config, "hp/skip_error_layer_indices"),
        freeze_skip_layers=get_config_value(config, "hp/freeze_skip_layers"),
        beta=get_config_value(config, "hp/beta"),
        beta_annealing=get_config_value(config, "hp/beta_annealing"),
        act_fn=get_config_value(config, "hp/act_fn"),
        batch_size=get_config_value(config, "hp/batch_size"),
        epochs=get_config_value(config, "hp/epochs"),
        T=get_config_value(config, "hp/T"),
        optim_x_lr=get_config_value(config, "hp/optim/x/lr"),
        optim_x_momentum=get_config_value(config, "hp/optim/x/momentum"),
        optim_w_name=get_config_value(config, "hp/optim/w/name"),
        optim_w_lr=get_config_value(config, "hp/optim/w/lr"),
        optim_w_wd=get_config_value(config, "hp/optim/w/wd"),
        optim_w_momentum=get_config_value(config, "hp/optim/w/momentum"),
        checkpoint_dir=Path("results/skip_error"),
    )
