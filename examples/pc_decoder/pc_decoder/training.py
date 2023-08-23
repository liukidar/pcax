import gc
import json
import logging
import os
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from matplotlib import pyplot as plt
from pc_decoder.data_loading import get_data_loaders
from pc_decoder.logging import (
    init_wandb,
    log_test_t_step_metrics,
    log_train_batch_metrics,
    log_train_t_step_metrics,
)
from pc_decoder.model import PCDecoder, feed_forward_predict, model_energy_loss
from pc_decoder.params import ModelParams, Params
from pc_decoder.visualization import create_all_visualizations, plot_training_exmaple
from ray import tune
from ray.air import session
from tqdm import tqdm

import pcax as px  # type: ignore
import pcax.core as pxc
import pcax.utils as pxu  # type: ignore
from pcax.pc import node  # type: ignore

DEBUG = os.environ.get("DEBUG", "0") == "1"
DEBUG_BATCH_NUMBER = 1

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

if DEBUG:
    from itertools import islice

    class ReentryIsliceIterator:
        def __init__(self, iterable, limit):
            self.iterable = iterable
            self.limit = limit

        def __iter__(self):
            return iter(islice(self.iterable, self.limit))

        def __len__(self):
            return min(self.limit, len(self.iterable))

        def __getattr__(self, attr):
            return getattr(self.iterable, attr)


def internal_state_init(
    params: ModelParams,
    prng_key: jax.random.KeyArray,
) -> tuple[jax.Array, jax.random.KeyArray]:
    # TODO: Play with different initialization strategies
    value = jnp.zeros((params.internal_dim,))
    return value, prng_key


class TrainOnBatchResult(NamedTuple):
    mse: jax.Array
    energies: jax.Array
    iterations_done: jax.Array
    num_x_updates: jax.Array
    num_w_updates: jax.Array


class _LoopState(NamedTuple):
    iter_index: jax.Array
    num_x_updates: jax.Array
    num_w_updates: jax.Array

    prev_energy: jax.Array
    curr_energy: jax.Array

    all_energies: jax.Array


def _build_loop_body(
    loss: Callable,
    update_x: bool = False,
    update_w: bool = False,
):
    def loop_body(
        state: _LoopState,
        examples: jax.Array,
        *,
        model: PCDecoder,
        optim_x: pxu.Optim,
        optim_w: pxu.Optim,
    ) -> _LoopState:
        grad_and_values = pxu.grad_and_values(
            px.f(px.NodeParam)(frozen=False) | px.f(px.LayerParam),  # type: ignore
        )(loss)

        with pxu.step(model):
            gradients, (prev_energy,) = grad_and_values(examples, model=model)

            if update_x:
                optim_x(gradients)
            if update_w:
                optim_w(gradients)

        with pxu.step(model):
            # Re-compute energies after parameter updates
            (curr_energy,) = loss(examples, model=model)

            nodes_energies = jnp.array([jnp.sum(x.energy()) for x in model.pc_nodes])
            all_energies = state.all_energies.at[state.iter_index].set(nodes_energies)

        # grads = {}
        # for param_name, param_value in model_parameters.items():
        #     if id(param_value) in g:
        #         grads[param_name] = g[id(param_value)]
        # all_gradients = state.all_gradients.at[state.iter_index].set(grads)

        return (
            _LoopState(
                iter_index=state.iter_index + 1,
                num_x_updates=state.num_x_updates + int(update_x),
                num_w_updates=state.num_w_updates + int(update_w),
                prev_energy=jnp.sum(prev_energy),
                curr_energy=jnp.sum(curr_energy),
                all_energies=all_energies,
            ),
            examples,
        )

    return loop_body


@pxu.jit()
def train_on_batch(
    examples: jax.Array,
    *,
    model: PCDecoder,
    optim_x: pxu.Optim,
    optim_w: pxu.Optim,
    loss: Callable,
    params: Params,
) -> TrainOnBatchResult:
    t_iterations = params.T
    if params.pc_mode == "efficient_ppc":
        t_iterations -= params.T_min_w_updates
    total_iterations = params.T
    if params.pc_mode == "pc":
        total_iterations += 1

    def t_loop_should_continue(state: _LoopState, *_) -> jax.Array:
        cond = state.iter_index < t_iterations
        if params.pc_mode == "efficient_ppc":
            eppc_cond = jnp.logical_or(
                jnp.abs(state.curr_energy - state.prev_energy)
                > params.energy_convergence_threshold,
                state.iter_index < params.T_min_x_updates,
            )
            cond = jnp.logical_and(
                cond,
                eppc_cond,
            )
        return cond

    def w_loop_should_continue(state: _LoopState, *_) -> jax.Array:
        return state.iter_index < total_iterations

    with pxu.pc_train_on_batch(model), pxu.train(model, examples):
        if params.reset_optimizer_x_state:
            optim_x.init_state()
        if params.preserve_all_pc_states_between_batches and model.saved_pc_states:
            for pc_node, state in zip(model.pc_nodes[1:], model.saved_pc_states):
                pc_node["x"] = state

        initial_state = _LoopState(
            iter_index=jnp.array(0),
            num_x_updates=jnp.array(0),
            num_w_updates=jnp.array(0),
            prev_energy=jnp.array(0.0),
            curr_energy=jnp.array(0.0),
            all_energies=jnp.zeros(
                (total_iterations, len(model.pc_nodes)), dtype=jnp.float32
            ),
        )

        # Make sure to remove "u" parameters from the model before passing it to the while_loop,
        # because in the loop body we use the pxu.step() decorator that calls model.clear_cache() that drops "u" parameters.
        # As the result, the list of parameters passed to the loop body and returned by the loop body differs, which is strictly forbidden by jax.lax.while_loop.
        model.clear_cache()

        t_loop_outputs = pxu.flow.while_loop(
            _build_loop_body(loss, update_x=True, update_w=params.pc_mode == "ppc"),
            t_loop_should_continue,
        )(
            initial_state,
            examples,
            model=model,
            optim_x=optim_x,
            optim_w=optim_w,
        )
        final_state = t_loop_outputs[0]
        if params.pc_mode in ["pc", "efficient_ppc"]:
            model.clear_cache()
            w_loop_outputs = pxu.flow.while_loop(
                _build_loop_body(loss, update_x=False, update_w=True),
                w_loop_should_continue,
            )(
                final_state,
                examples,
                model=model,
                optim_x=optim_x,
                optim_w=optim_w,
            )
            final_state = w_loop_outputs[0]

    predictions = feed_forward_predict(model.internal_state, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return TrainOnBatchResult(
        mse=mse,
        energies=final_state.all_energies,
        iterations_done=final_state.iter_index,
        num_x_updates=final_state.num_x_updates,
        num_w_updates=final_state.num_w_updates,
    )


# @pxu.jit()
def test_on_batch(
    examples, *, model: PCDecoder, optim_x, loss, T
) -> tuple[jax.Array, list[list[jax.Array]]]:
    energies = model.converge_on_batch(examples, optim_x=optim_x, loss=loss, T=T)
    predictions = feed_forward_predict(model.internal_state, model=model)[0]
    mse = jnp.mean((predictions - examples) ** 2)
    return mse, energies


def run_training_experiment(params: Params) -> None:
    results_dir = Path(params.results_dir) / params.experiment_name  # type: ignore
    if results_dir.exists() and any(results_dir.iterdir()):
        if params.do_hypertunning and params.hypertunning_resume_run:
            shutil.move(
                results_dir, results_dir.with_suffix(f".backup-{tune.get_trial_id()}")
            )
        elif params.overwrite_results_dir:
            shutil.rmtree(results_dir)
        else:
            raise RuntimeError(
                f"Results dir {results_dir} already exists and is not empty!"
            )
    results_dir.mkdir(parents=True, exist_ok=True)

    model = PCDecoder(
        params=params,
        internal_state_init_fn=internal_state_init,
    )

    if params.load_weights_from is not None:
        model.load_weights(params.load_weights_from)

    if params.wandb_logging:
        with init_wandb(params=params, results_dir=results_dir) as run:
            train_model(model=model, params=params, results_dir=results_dir, run=run)
    else:
        train_model(model=model, params=params, results_dir=results_dir)


def train_model(
    model: PCDecoder,
    params: Params,
    results_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None = None,
) -> None:
    best_epoch_dir = results_dir / "best"
    with pxu.train(model, jnp.zeros((params.batch_size, params.output_dim))):
        if params.optimizer_x == "sgd":
            optim_x = pxu.Optim(
                optax.chain(
                    optax.add_decayed_weights(weight_decay=params.optim_x_l2),
                    optax.sgd(params.optim_x_lr / params.batch_size),
                ),
                model.x_parameters(),
                allow_none_grads=True,
            )
        elif params.optimizer_x == "adamw":
            optim_x = pxu.Optim(
                optax.adamw(
                    params.optim_x_lr / params.batch_size,
                    weight_decay=params.optim_x_l2,
                ),
                model.x_parameters(),
                allow_none_grads=True,
            )
        else:
            raise ValueError(f"Unknown optimizer_x: {params.optimizer_x}")
        if params.optimizer_w == "sgd":
            optim_w = pxu.Optim(
                optax.chain(
                    optax.add_decayed_weights(weight_decay=params.optim_w_l2),
                    optax.sgd(params.optim_w_lr / params.batch_size),
                ),
                model.w_parameters(),
                allow_none_grads=True,
            )
        elif params.optimizer_w == "adamw":
            optim_w = pxu.Optim(
                optax.chain(
                    optax.adamw(
                        params.optim_w_lr / params.batch_size,
                        weight_decay=params.optim_w_l2,
                    ),
                ),
                model.w_parameters(),
                allow_none_grads=True,
            )
        else:
            raise ValueError(f"Unknown optimizer_w: {params.optimizer_w}")

    train_batch_fn = partial(
        train_on_batch,
        model=model,
        optim_x=optim_x,
        optim_w=optim_w,
        loss=model_energy_loss,  # type: ignore
        params=params,
    )

    test_batch_fn = partial(
        test_on_batch,
        model=model,
        optim_x=optim_x,
        loss=model_energy_loss,  # type: ignore
        T=params.T,  # type: ignore
    )

    train_loader, test_loader, train_data_mean, train_data_std = get_data_loaders(
        params
    )
    if DEBUG:
        train_loader = ReentryIsliceIterator(train_loader, DEBUG_BATCH_NUMBER)  # type: ignore
        test_loader = ReentryIsliceIterator(test_loader, DEBUG_BATCH_NUMBER)  # type: ignore

    train_mses = []
    test_mses = []
    best_train_mse = float("inf")
    best_test_mse = float("inf")
    train_t_step: int = 0
    test_t_step: int = 0

    with tqdm(range(params.epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Train Epoch {epoch + 1}")
            logging.info(f"Starting epoch {epoch + 1}")

            epoch_train_mses = []
            with tqdm(train_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Train Batch {tbatch.n + 1}")
                    batch_res: TrainOnBatchResult = train_batch_fn(examples)
                    mse = batch_res.mse.item()
                    log_train_t_step_metrics(
                        run=run,
                        t_step=train_t_step,
                        # FIXME: report energies and gradients
                        energies=batch_res.energies,
                        # gradients=[],
                        params=params,
                    )
                    log_train_batch_metrics(
                        run=run,
                        epochs=epoch,
                        batches_per_epoch=len(train_loader),
                        batch=tbatch.n,
                        num_x_updates=batch_res.num_x_updates.item(),
                        num_w_updates=batch_res.num_w_updates.item(),
                    )
                    train_t_step += batch_res.iterations_done.item()
                    epoch_train_mses.append(mse)
                    tbatch.set_postfix(mse=mse)

                    # Force GC to free some RAM and GPU memory
                    del batch_res
                    gc.collect()

            epoch_train_mse: float = float(
                np.mean(
                    epoch_train_mses[-params.use_last_n_batches_to_compute_metrics :]
                )
            )
            train_mses.append(epoch_train_mse)
            logging.info(f"Finished training in epoch {epoch + 1}")

            epoch_test_mses = []
            with tqdm(test_loader, unit="batch") as tbatch:
                for examples, _ in tbatch:
                    tbatch.set_description(f"Test Batch {tbatch.n + 1}")
                    mse, energies = test_batch_fn(examples)
                    mse = mse.item()
                    log_test_t_step_metrics(
                        run=run,
                        t_step=test_t_step,
                        energies=energies,
                    )
                    test_t_step += len(energies)
                    epoch_test_mses.append(mse)
                    tbatch.set_postfix(mse=mse)
            epoch_test_mse: float = float(np.mean(epoch_test_mses))
            test_mses.append(epoch_test_mse)
            logging.info(f"Finished testing in epoch {epoch + 1}")

            epoch_report = {
                "epochs": epoch + 1,
                "train_mse": epoch_train_mse,
                "test_mse": epoch_test_mse,
            }

            should_save_intermediate_results = (
                params.save_intermediate_results
                and (epoch + 1) % params.save_results_every_n_epochs == 0
            )
            should_save_best_results = (
                params.save_best_results and epoch_test_mse < best_test_mse
            )
            best_train_mse = min(best_train_mse, epoch_train_mse)
            best_test_mse = min(best_test_mse, epoch_test_mse)
            if should_save_intermediate_results or should_save_best_results:
                logging.info(
                    f"Saving results for epoch {epoch + 1}. Best epoch: {should_save_best_results}. MSE: {epoch_test_mse}"
                )
                epoch_results = results_dir / f"epochs_{epoch + 1}"
                epoch_results.mkdir()

                if should_save_best_results:
                    best_epoch_dir.unlink(missing_ok=True)
                    best_epoch_dir.symlink_to(
                        epoch_results.relative_to(results_dir),
                        target_is_directory=True,
                    )

                model.save_weights(str(epoch_results))  # type: ignore
                with open(os.path.join(epoch_results, "report.json"), "w") as outfile:
                    json.dump(epoch_report, outfile, indent=4)

            if run is not None:
                run.log(epoch_report)
            if params.do_hypertunning:
                session.report(epoch_report)

            logging.info(f"Finished epoch {epoch + 1}")
            tepoch.set_postfix(train_mse=epoch_train_mse, test_mse=epoch_test_mse)

    # Generate images for the best epoch only
    if best_epoch_dir.exists():
        internal_states_mean, internal_states_std = visualize_epoch(
            epoch_dir=best_epoch_dir,
            run=run,
            model=model,
            optim_x=optim_x,
            test_loader=test_loader,
            params=params,
            train_data_mean=train_data_mean,
            train_data_std=train_data_std,
        )
        plot_training_exmaple(
            example=next(iter(train_loader))[0],
            prediction=feed_forward_predict(model.internal_state, model=model)[0],
            out_dir=best_epoch_dir,
            run=run,
            train_data_mean=train_data_mean,
            train_data_std=train_data_std,
        )
        if run is not None:
            run.summary["internal_states_mean"] = internal_states_mean
            run.summary["internal_states_std"] = internal_states_std
    else:
        logging.error(f"No best epoch exists for this run")

    if run is not None:
        run.summary["train_mse"] = best_train_mse
        run.summary["test_mse"] = best_test_mse

    logging.info(
        f"Finished training for {params.epochs} epochs, test_mse={best_test_mse}"
    )


def visualize_epoch(
    *,
    epoch_dir: Path,
    run: wandb.wandb_sdk.wandb_run.Run | None,
    model: PCDecoder,
    optim_x: pxu.Optim,
    test_loader,
    params: Params,
    train_data_mean: float,
    train_data_std: float,
) -> tuple[float, float]:
    if not epoch_dir.exists():
        raise ValueError(f"Epoch dir {epoch_dir} does not exist!")
    logging.info(f"Visualizing epoch from {epoch_dir.resolve()}...")
    with open(epoch_dir / "report.json") as infile:
        epoch_report = json.load(infile)
        epochs = epoch_report["epochs"]
    model.load_weights(str(epoch_dir))  # type: ignore
    internal_states = create_all_visualizations(
        out_dir=epoch_dir,
        run=run,
        epochs=epochs,
        model=model,
        optim_x=optim_x,
        test_loader=test_loader,
        params=params,
        train_data_mean=train_data_mean,
        train_data_std=train_data_std,
    )
    internal_states_mean = jnp.mean(internal_states).item()
    internal_states_std = jnp.std(internal_states).item()
    with open(epoch_dir / "report.json", "w") as outfile:
        json.dump(
            {
                **epoch_report,
                "internal_states_mean": internal_states_mean,
                "internal_states_std": internal_states_std,
            },
            outfile,
            indent=4,
        )
    logging.info(f"Finished visualizing epoch from {epoch_dir}...")
    return internal_states_mean, internal_states_std


class Trainable:
    def __init__(self, params: Params) -> None:
        self.params = params

    def __call__(self, config: dict):
        gc.collect()
        params = self.params.update(config, inplace=False, validate=True)
        # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.utils.wait_for_gpu.html#ray.tune.utils.wait_for_gpu
        if params.hypertunning_gpu_memory_fraction_per_trial > 0:
            tune.utils.wait_for_gpu(  # type: ignore
                target_util=1.0 - params.hypertunning_gpu_memory_fraction_per_trial,
                retry=50,
                delay_s=12,
            )
        run_training_experiment(params)
        gc.collect()
