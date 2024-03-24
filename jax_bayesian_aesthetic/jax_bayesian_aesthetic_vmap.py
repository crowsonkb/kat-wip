#!/usr/bin/env python3

from typing import NamedTuple

import flax
import flax.linen as nn
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
from rich import print
from rich.traceback import install
import torch
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


class EMAAccumulator(NamedTuple):
    value: jax.Array
    accum: jax.Array
    decay: jax.Array

    @classmethod
    def create(cls, value, decay):
        return cls(value, jnp.array(1.0, jnp.float32), decay)

    @classmethod
    def update(cls, state, new_value):
        value, accum, decay = state
        accum = accum * decay
        value = jax.tree_map(lambda a, b: decay * a + (1 - decay) * b, value, new_value)
        return cls(value, accum, decay)

    @classmethod
    def get_value(cls, state):
        value, accum, decay = state
        return jax.tree_map(lambda a: a / (1 - accum), value)


def gaussian_potential(x, mean, var):
    return jnp.sum(jnp.square(x - mean) / var) / 2


def prior(params, variances):
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_map(lambda a, b: gaussian_potential(a, 0.0, b), params, variances))


class LinearModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        bias_init = nn.initializers.normal(1.0)
        x = x * jnp.sqrt(x.shape[-1]) / jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = nn.Dense(1, kernel_init=kernel_init, bias_init=bias_init)(x)
        return x

    def init_variances(self, params):
        variances = {}
        for name, layer in params.items():
            variances[name] = {}
            if 'bias' in layer:
                variances[name]['bias'] = jnp.array(1.0)
            if 'kernel' in layer:
                fan_in = layer['kernel'].shape[-2]
                variances[name]['kernel'] = 1.0 / jnp.array(fan_in, jnp.float32)
        return flax.core.FrozenDict(variances)


class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        bias_init = nn.initializers.normal(1.0)
        x = x * jnp.sqrt(x.shape[-1]) / jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = nn.Dense(32, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(16, kernel_init=kernel_init, bias_init=bias_init)(x)
        x = nn.gelu(x)
        x = nn.Dense(1, kernel_init=kernel_init, bias_init=bias_init)(x)
        return x

    def init_variances(self, params):
        variances = {}
        for name, layer in params.items():
            variances[name] = {}
            if 'bias' in layer:
                variances[name]['bias'] = jnp.array(1.0)
            if 'kernel' in layer:
                fan_in = layer['kernel'].shape[-2]
                variances[name]['kernel'] = 1.0 / jnp.array(fan_in, jnp.float32)
        return flax.core.FrozenDict(variances)


class TrainState(flax.struct.PyTreeNode):
    step: jax.Array
    params: jax.Array
    opt_state: optax.OptState
    params_moms: EMAAccumulator

    @classmethod
    def create(cls, params, opt_state, beta):
        n = params.shape[0]
        return cls(
            step=jnp.array(0, jnp.int32), params=params, opt_state=opt_state,
            params_moms=EMAAccumulator.create((jnp.zeros_like(params), jnp.zeros([n])), beta),
        )


def main():
    install()

    key = jax.random.PRNGKey(284031)

    # load training set
    pth = torch.load('sac_embeds/vit_l_with_filenames.pth', map_location='cpu')
    data_x, data_y = jnp.array(pth['embeds'].float()), jnp.array(pth['ratings'].float())
    filenames = np.array(pth['filenames'])
    print('Number of items in the dataset:', data_x.shape[0])

    key, subkey = jax.random.split(key)
    train_x, train_y, val_x, val_y = train_val_split(subkey, data_x, data_y)
    n = train_x.shape[0]

    n, ndim = train_x.shape
    linear_model = LinearModel()
    key, subkey = jax.random.split(key)
    _, params = linear_model.init(subkey, jnp.ones([ndim])).pop('params')
    _, unravel_linear = jax.flatten_util.ravel_pytree(params)

    n_models = 20

    model = Model()
    params_lst = []
    init_jit = jax.jit(model.init)
    for i in range(n_models):
        key, subkey = jax.random.split(key)
        variables, params = init_jit(subkey, jnp.ones([ndim])).pop('params')
        params, unravel = jax.flatten_util.ravel_pytree(params)
        params_lst.append(params)
    params = jnp.stack(params_lst)
    print('Number of parameters:', params[0].size)

    # optimization parameters
    dt_base = 0.005
    # sched = adamld.inverse_schedule(dt_base, 0.1, 0.5)
    n_steps = 15000
    # sched = lambda count: dt_base * jnp.cos(count * jnp.pi / n_steps) ** 2
    # sched = optax.cosine_decay_schedule(dt_base, n_steps)
    sched = optax.constant_schedule(dt_base)
    batch_size = 10000

    priors = adamld.make_priors_flax(unravel(params[0]))
    means, _ = jax.flatten_util.ravel_pytree(jax.tree_map(jnp.full_like, unravel(params[0]), priors[0]))
    variances, _ = jax.flatten_util.ravel_pytree(jax.tree_map(jnp.full_like, unravel(params[0]), priors[1]))
    priors = means, variances

    key, subkey = jax.random.split(key)
    opt = adamld.adamld(sched, subkey, priors, tau=1.0)
    opt_state = jax.vmap(opt.init)(params)
    state = jax.vmap(lambda p, o: TrainState.create(params=p, opt_state=o, beta=0.999))(params, opt_state)

    def update_one(state, key, x, y):
        def prior_fun(params):
            params_unflat = unravel(params)
            variances = model.init_variances(params_unflat)
            return prior(params_unflat, variances)

        def likelihood_fun(params):
            model_out = model.apply({'params': unravel(params)}, x)
            likelihood_batch = gaussian_potential(model_out[:, 0], y, 1.0)
            return likelihood_batch / x.shape[0] * n

        # make flat prior variance
        # params_unflat = unravel(state.params)
        # variances = model.init_variances(params_unflat)
        # prior_grad_mag = jax.tree_map(lambda a, b: jnp.full_like(b, 1 / a), variances, params_unflat)
        # prior_grad_mag, _ = jax.flatten_util.ravel_pytree(prior_grad_mag)

        # compute potentials and gradients
        p_p = prior_fun(state.params)
        l_p, l_grad = jax.value_and_grad(likelihood_fun)(state.params)

        # compute step size
        # dt = dt_base * (state.step * 0.1 + 1) ** -0.5
        step = state.step + 1

        # do the update
        updates, opt_state = opt.update(l_grad, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)

        # update params moments
        # params_moms = EMAAccumulator.update(state.params_moms, (params, jnp.outer(params, params)))
        params_moms = EMAAccumulator.update(state.params_moms, (params, params ** 2))
        # params_moms = jax.lax.cond(
        #     step % 50 == 0,
        #     lambda: EMAAccumulator.update(state.params_moms, (params, params ** 2)),
        #     lambda: state.params_moms,
        # )

        state = state.replace(step=step, params=params, opt_state=opt_state, params_moms=params_moms)

        return state, p_p, l_p

    update = jax.jit(jax.vmap(update_one, in_axes=(0, 0, None, None)))

    for i in trange(n_steps):
        key, *keys = jax.random.split(key, 3)
        indices = jax.random.randint(keys[0], [batch_size], 0, train_x.shape[0])
        x, y = train_x[indices], train_y[indices]
        keys_ = jax.random.split(keys[1], n_models)
        state, prior_p, likelihood_p = update(state, keys_, x, y)
        if i % 20 == 0:
            print(i, prior_p, likelihood_p)

    params_mean, params_m2 = jax.vmap(EMAAccumulator.get_value)(state.params_moms)
    params_var = jax.nn.relu(params_m2 - params_mean ** 2)

    def sample_models(key, n, params_mean, params_var):
        return jax.random.normal(key, [n, *params_mean.shape]) * jnp.sqrt(params_var) + params_mean

    # print(params_mean, jnp.sqrt(jnp.diag(params_cov)), params_cov)

    # fit a linear regression the normal way
    ridge_coeff = 1000
    train_x_norm = train_x * jnp.sqrt(train_x.shape[-1]) / jnp.linalg.norm(train_x, axis=-1, keepdims=True)
    val_x_norm = val_x * jnp.sqrt(val_x.shape[-1]) / jnp.linalg.norm(val_x, axis=-1, keepdims=True)
    x = jnp.concatenate([jnp.ones([train_x_norm.shape[0], 1]), train_x_norm], axis=-1)
    with jax.experimental.enable_x64():
        params_ridge = jnp.linalg.inv(x.T @ x + ridge_coeff * jnp.eye(x.shape[-1])) @ x.T @ train_y

    # print(jnp.stack([params_mean, params_ridge], axis=-1)[:25])
    # print(jnp.mean(jnp.sign(params_ridge) == jnp.sign(params_mean)))
    # sim = jnp.vdot(params_ridge, params_mean) / (jnp.linalg.norm(params_ridge) * jnp.linalg.norm(params_mean))
    # print('cosine similarity:', sim)

    @jax.jit
    def evaluate_linear(params, x):
        return linear_model.apply({'params': unravel_linear(params)}, x)[..., 0]

    @jax.jit
    def evaluate(params, x):
        return model.apply({'params': unravel(params)}, x)[..., 0]

    def evaluate_ensemble(params, x):
        evaluate_vmap = jax.vmap(evaluate, in_axes=(0, None))
        out = evaluate_vmap(params, x)
        return jnp.mean(out, axis=0), jnp.std(out, axis=0)

    # x_to_eval = train_x[:25]
    # real_y = train_y[:25]
    key, subkey = jax.random.split(key)
    params_ensemble = sample_models(subkey, 10, params_mean, params_var)
    params_ensemble = jnp.reshape(params_ensemble, [10 * n_models, -1])

    mean_y, std_y = evaluate_ensemble(params_ensemble, train_x)
    ridge_y = evaluate_linear(params_ridge, train_x_norm)
    mean_loss = jnp.mean(jnp.square(mean_y - train_y))
    ridge_loss = jnp.mean(jnp.square(ridge_y - train_y))
    print('adamld train loss:', mean_loss, ', ridge train loss:', ridge_loss)
    print('adamld train std:', jnp.sqrt(jnp.mean(std_y ** 2)))

    mean_y, std_y = evaluate_ensemble(params_ensemble, val_x)
    ridge_y = evaluate_linear(params_ridge, val_x_norm)
    mean_loss = jnp.mean(jnp.square(mean_y - val_y))
    ridge_loss = jnp.mean(jnp.square(ridge_y - val_y))
    print('adamld val loss:', mean_loss, ', ridge val loss:', ridge_loss)
    print('adamld val std:', jnp.sqrt(jnp.mean(std_y ** 2)))

    mean_y, std_y = evaluate_ensemble(params_ensemble, data_x)
    indices = jnp.argsort(std_y)
    for a, b, c in zip(mean_y[indices[:10]], std_y[indices[:10]], filenames[indices[:10]]):
        print(a, b, c)
    for a, b, c in zip(mean_y[indices[-10:]], std_y[indices[-10:]], filenames[indices[-10:]]):
        print(a, b, c)


if __name__ == '__main__':
    main()
