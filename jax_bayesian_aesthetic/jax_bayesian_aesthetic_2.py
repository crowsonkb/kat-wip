#!/usr/bin/env python3

"""Trains a Bayesian aesthetic model on a dataset of CLIP image embeddings and
aesthetic ratings."""

import argparse
import pickle
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich import print
from rich.traceback import install
from tqdm.auto import trange, tqdm

import adamld

print = tqdm.external_write_mode()(print)


def train_val_split(key, x, y, val_frac=0.1):
    n = x.shape[0]
    n_val = int(n * val_frac)
    n_train = n - n_val
    perm = jax.random.permutation(key, n)
    x_train = x[perm[:n_train]]
    y_train = y[perm[:n_train]]
    x_val = x[perm[n_train:]]
    y_val = y[perm[n_train:]]
    return x_train, y_train, x_val, y_val


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x * jnp.sqrt(x.shape[-1]) / jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = nn.Dense(16)(x)
        x = nn.swish(x)
        x = nn.Dense(1)(x)
        return x[..., 0]


class TrainState(flax.struct.PyTreeNode):
    count: jax.Array
    params: Any
    opt_state: optax.OptState
    params_m1: Any
    params_m2: Any


def main():
    install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i", type=str, default="./stable_horde_ratings_embeds.pkl"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="./bayesian_aesthetic_model.pkl"
    )
    parser.add_argument(
        "--model-output", type=str, default="./bayesian_aesthetic_model_output.pkl"
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    n_components = 20
    n_models_per_component = 10
    n_steps = 30000
    batch_size = 10000
    ema_decay = 0.999
    lr = 0.01

    # load training set
    pkl = pickle.load(open(args.input, "rb"))
    data_x, data_y = jax.tree_map(jnp.array, (pkl["embeds"], pkl["ratings"]))
    filenames = pkl["filenames"]
    del pkl
    print("Dataset size:", data_x.shape[0])

    # train/val split
    key, subkey = jax.random.split(key)
    train_x, train_y, val_x, val_y = train_val_split(subkey, data_x, data_y)
    n, ndim = train_x.shape
    print("Training set size:", train_x.shape[0])
    print("Validation set size:", val_x.shape[0])

    # init model
    model = Model()
    key, subkey = jax.random.split(key)
    params_prototype = model.init(subkey, jnp.ones([ndim]))
    priors = adamld.make_priors_flax(params_prototype)
    print("Number of parameters:", adamld.tree_size(params_prototype))

    # sample initial ensemble
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_components)
    params = jax.vmap(adamld.prior_sample, in_axes=(0, None, None))(
        keys, params_prototype, priors
    )

    # set up training state
    key, subkey = jax.random.split(key)
    opt = adamld.adamld(lr, subkey, priors, tau=1.0)
    opt_state = jax.vmap(opt.init)(params)
    state = TrainState(
        count=jnp.array(0, jnp.int32),
        params=params,
        opt_state=opt_state,
        params_m1=jax.tree_map(jnp.zeros_like, params),
        params_m2=jax.tree_map(jnp.zeros_like, params),
    )

    @jax.jit
    def update(state, x, y):
        # update params
        def loss_fun(params, x, y):
            y_pred = model.apply(params, x)
            return jnp.mean((y_pred - y) ** 2) / 2 * n

        grad_fun = jax.value_and_grad(loss_fun)
        prior_potentials = jax.vmap(adamld.prior_potential, in_axes=(0, None))(
            state.params, priors
        )
        losses, grads = jax.vmap(grad_fun, in_axes=(0, None, None))(state.params, x, y)
        updates, opt_state = jax.vmap(opt.update)(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # update param moments
        params_m1 = jax.tree_map(
            lambda a, b: ema_decay * a + (1 - ema_decay) * b, state.params_m1, params
        )
        params_m2 = jax.tree_map(
            lambda a, b: ema_decay * a + (1 - ema_decay) * b**2,
            state.params_m2,
            params,
        )

        # update state
        state = state.replace(
            count=state.count + 1,
            params=params,
            opt_state=opt_state,
            params_m1=params_m1,
            params_m2=params_m2,
        )
        return state, prior_potentials, losses

    # train the ensemble
    for i in trange(n_steps):
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(subkey, [batch_size], 0, train_x.shape[0])
        x, y = train_x[indices], train_y[indices]
        state, prior_potentials, losses = update(state, x, y)
        if i % 100 == 0:
            print("i:", i)
            print("prior:", prior_potentials)
            print("likelihood:", losses)
            print()

    # get ensemble from train state
    params_mean = jax.tree_map(
        lambda x: x / (1 - ema_decay**state.count), state.params_m1
    )
    params_m2 = jax.tree_map(
        lambda x: x / (1 - ema_decay**state.count), state.params_m2
    )
    params_var = jax.tree_map(
        lambda m1, m2: jax.nn.relu(m2 - m1**2), params_mean, params_m2
    )
    ensemble = params_mean, params_var

    # sample from ensemble
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_models_per_component)
    ensemble_samples = jax.vmap(adamld.prior_sample, in_axes=(0, None, None))(
        keys, ensemble[0], ensemble
    )
    ensemble_samples = jax.tree_map(
        lambda x: x.reshape([x.shape[0] * x.shape[1], *x.shape[2:]]), ensemble_samples
    )

    # evaluate ensemble on training and validation set
    @jax.jit
    def apply_ensemble(params, x):
        return jax.vmap(model.apply, in_axes=(0, None))(params, x)

    y_pred_train = apply_ensemble(ensemble_samples, train_x)
    y_pred_train_mean = jnp.mean(y_pred_train, axis=0)
    y_pred_train_std = jnp.std(y_pred_train, axis=0)
    train_loss = jnp.mean((y_pred_train_mean - train_y) ** 2)
    train_std = jnp.sqrt(jnp.mean(y_pred_train_std**2))
    print("Training loss (MSE):", train_loss)
    print("Training std:", train_std)

    y_pred_val = apply_ensemble(ensemble_samples, val_x)
    y_pred_val_mean = jnp.mean(y_pred_val, axis=0)
    y_pred_val_std = jnp.std(y_pred_val, axis=0)
    val_loss = jnp.mean((y_pred_val_mean - val_y) ** 2)
    val_std = jnp.sqrt(jnp.mean(y_pred_val_std**2))
    print("Validation loss (MSE):", val_loss)
    print("Validation std:", val_std)

    # save outputs
    y_pred_data = apply_ensemble(ensemble_samples, data_x)
    y_pred_data_mean = jnp.mean(y_pred_data, axis=0)
    y_pred_data_std = jnp.std(y_pred_data, axis=0)
    indices = jnp.argsort(y_pred_data_std)
    obj = {
        "rating_mean": y_pred_data_mean[indices],
        "rating_std": y_pred_data_std[indices],
        "filenames": filenames[indices],
    }
    obj = jax.tree_map(np.array, obj)
    with open(args.model_output, "wb") as f:
        pickle.dump(obj, f)
        print(f"Saved outputs to: [bright_magenta]{args.model_output}[/]")

    # save ensemble
    obj = {"ensemble": ensemble, "ensemble_samples": ensemble_samples}
    obj = jax.tree_map(np.array, obj)
    with open(args.output, "wb") as f:
        pickle.dump(obj, f)
        print(f"Saved ensemble to: [bright_magenta]{args.output}[/]")


if __name__ == "__main__":
    main()
