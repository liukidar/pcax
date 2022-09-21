from pcax.core.energy import EnergyCriterion
from pcax.core.node import NODE_TYPE
import pcax.core as pcax
import pcax.nn as nn
import jax
import jax.numpy as jnp
import optax
import pcax.interface as pxi
import numpy as np
from torchvision.datasets import MNIST
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class Model(pcax.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    linear3: nn.Linear
    pc1: pcax.Layer
    pc2: pcax.Layer
    pc3: pcax.Layer

    def __init__(self, key, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        key, subkey = jax.random.split(key)
        self.linear1 = nn.Linear(input_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, _key=subkey)
        key, subkey = jax.random.split(key)
        self.linear3 = nn.Linear(hidden_dim, output_dim, _key=subkey)

        self.pc1 = pcax.Layer()
        self.pc2 = pcax.Layer()
        self.pc3 = pcax.Layer()
        self.pc3._node_info.status = pcax.node.NODE_STATUS.FROZEN

    def init(self, state, x, t=None):
        act_fn = jax.nn.tanh

        self.pc1.at().set(act_fn(self.linear1(x)))
        self.pc2.at().set(act_fn(self.linear2(self.pc1.x.get())))

        if t is None:
            self.pc3.at().set(jax.nn.softmax(self.linear3(self.pc2.x.get())))
        else:
            self.pc3.at().set(t)

        return state

    def __call__(self, x):
        act_fn = jax.nn.tanh

        x = self.pc1(act_fn(self.linear1(x)))
        x = self.pc2(act_fn(self.linear2(*x.get())))
        x = self.pc3(jax.nn.softmax(self.linear3(*x.get())))

        y = self.pc3.at(type="output").get()[0]

        return y


def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


class FlattenAndCast:
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


batch_size = 128
input_dim = 28 * 28
hidden_dim = 256
output_dim = 10

mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())
training_generator = pxi.data.Dataloader(
    mnist_dataset, batch_size=batch_size, num_workers=8
)

rseed = 0
rkey = jax.random.PRNGKey(rseed)
rkey, rsubkey = jax.random.split(rkey)


state = pxi.DefaultState()
trainer = pxi.Trainer()

state, model, optim = state.init(
    Model(rsubkey, input_dim, hidden_dim, output_dim),
    "*",
    batch_size=batch_size,
    input_shape=(input_dim,),
    optim_fn=lambda state: pxi.optim.combine(
        {
            NODE_TYPE.X: optax.sgd(1e-4),
            NODE_TYPE.W: optax.chain(pxi.optim.reduce(), optax.adam(1e-3)),
        },
        state.get_masks("type"),
    ),
    trainer=trainer,
    init_fn_args=(None,),
)()


def loss_fn(state, model, y, t):
    return model.energy(EnergyCriterion())


@pxi.jit(loss_fn=loss_fn, show_jit_count=True)
def run_on_batch(state, model, x, t, loss_fn):
    model = trainer.init_fn(state, model, x, t)

    r, y = pxi.flow.scan(
        pxi.flow.switch(
            lambda j: j % 2,
            trainer.update_fn[NODE_TYPE.X, NODE_TYPE.W],
            loss_fn=loss_fn,
            optim=optim,
        ),
        js=np.arange(8),
    )(state=state, model=model, x_args=[x], loss_fn_args=[t])

    target_class = jnp.argmax(t, axis=1)
    predicted_class = jnp.argmax(y[0][0, ...], axis=1)
    accuracy = jnp.mean(predicted_class == target_class)

    return r["state"], r["model"], jnp.mean(accuracy)


epoch_times = []
for e in range(16):
    accuracies = []
    start_time = time.time()
    for x, y in training_generator:
        state, model, accuracy = run_on_batch(state, model, x, one_hot(y, output_dim))
        accuracies.append(accuracy)

    epoch_time = time.time() - start_time
    if e > 1:
        epoch_times.append(epoch_time)

    print("Epoch {} in {:0.2f} sec".format(e, epoch_time))
    print("Accuracy:", np.mean(accuracies))
print(f"Avg epoch time: {np.mean(epoch_times)}")
