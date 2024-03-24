"""Adam Langevin Dynamics for optax, by Katherine Crowson."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


def tree_size(tree):
    """Returns the total size of a tree."""
    return jax.tree_util.tree_reduce(lambda x, y: x + y.size, tree, 0)


def tree_sum(tree):
    """Returns the sum of a tree."""
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y), tree, 0)


def tree_mean(tree):
    """Returns the mean of a tree."""
    return tree_sum(tree) / tree_size(tree)


def unwrap_schedule(scalar_or_schedule, count):
    """Unwraps a scalar or schedule into a scalar."""
    if callable(scalar_or_schedule):
        return scalar_or_schedule(count)
    return scalar_or_schedule


def keys_like_tree(key, tree):
    """Generates a tree of random keys with the same structure as `tree`."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, jax.random.split(key, len(leaves)))


def noise_like_tree(key, tree):
    """Generates standard normal noise with the same structure as `tree`."""
    keys = keys_like_tree(key, tree)
    return jax.tree_map(
        lambda x, key: jax.random.normal(key, x.shape, x.dtype), tree, keys
    )


def inverse_schedule(init_value, gamma=1.0, power=1.0):
    """Constructs an inverse decay schedule (for the SGLD convergence
    guarantee).

    Args:
        init_value: the initial value of the schedule.
        gamma: the multiplicative factor.
        power: the power of the inverse decay.
    """

    def schedule(count):
        return init_value * (1.0 + count * gamma) ** -power

    return schedule


def make_priors_flax(params, prior_fun=None):
    """Constructs the prior mean and variance trees for a Flax model.

    Args:
        params: the Flax model parameters.
        prior_fun: a function that takes a path and value and returns a tuple of
        (mean, variance) for the prior distribution of the value. If None, uses a prior
        corresponding to the Flax default initialization.
    """
    import flax

    def default_prior_fun(path, value):
        if path[-1] == "bias":
            return jnp.array(0.0), jnp.array(1.0)
        if path[-1] == "embedding":
            fan_out = value.shape[-1]
            return jnp.array(0.0), jnp.array(1.0 / fan_out)
        if path[-1] == "kernel":
            fan_in = value.shape[-2]
            return jnp.array(0.0), jnp.array(1.0 / fan_in)
        if path[-1] == "scale":
            fan_in = value.shape[-1]
            return jnp.array(1.0), jnp.array(1.0 / fan_in)
        raise ValueError(f'Unknown param type: {"/".join(path)}')

    prior_fun = prior_fun or default_prior_fun

    priors = flax.traverse_util.path_aware_map(prior_fun, params)
    means = flax.core.FrozenDict(
        jax.tree_map(lambda x: x[0], priors, is_leaf=lambda x: isinstance(x, tuple))
    )
    variances = flax.core.FrozenDict(
        jax.tree_map(lambda x: x[1], priors, is_leaf=lambda x: isinstance(x, tuple))
    )
    return means, variances


def prior_potential(tree, priors):
    """Computes the potential of a prior distribution evaluated at a tree of
    parameters."""
    return jax.tree_util.tree_reduce(
        jnp.add,
        jax.tree_map(
            lambda x, m, v: 0.5 * jnp.sum((x - m) ** 2 / v), tree, priors[0], priors[1]
        ),
        0.0,
    )


def prior_sample(key, params, priors):
    """Samples parameters from the prior distribution."""
    noise = noise_like_tree(key, params)
    return jax.tree_map(
        lambda x, m, v: x * jnp.sqrt(v) + m, noise, priors[0], priors[1]
    )


class SGLDState(NamedTuple):
    count: jax.Array
    key: jax.Array
    tau_ema: jax.Array


def sgld(learning_rate, key, priors, tau=1.0, tau_decay=0.99):
    """Stochastic Gradient Langevin Dynamics.

    The gradient provided to the update function should be a stochastic estimate
    of the gradient of the true unnormalized negative log likelihood. (That is,
    it should already be divided by the batch size and multiplied by the number of
    samples in the training set.)

    Args:
        learning_rate: a scalar or a function that maps the step count to a scalar.
        key: a JAX PRNG key.
        priors: a tuple of two trees with the same structure of the parameters,
            which contain the means and variances of the diagonal normal prior.
        tau: the temperature of the Langevin dynamics.
        tau_decay: the exponential moving average decay for the thermostat.
    """

    prior_means, prior_variances = priors

    def init_fn(params):
        return SGLDState(count=jnp.zeros([], jnp.int32), key=key, tau_ema=jnp.zeros([]))

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("No params provided to update_fn.")

        count = state.count + 1

        posterior_grads = jax.tree_map(
            lambda x, m, v, u: (x - m) / v + u,
            params,
            prior_means,
            prior_variances,
            updates,
        )

        key, subkey = jax.random.split(state.key)
        noise = noise_like_tree(subkey, params)

        lr = unwrap_schedule(learning_rate, state.count)

        tau_empirical = tree_mean(
            jax.tree_map(
                lambda pg: pg**2 * jnp.abs(lr) / 2,
                posterior_grads,
            )
        )
        tau_ema = state.tau_ema * tau_decay + tau_empirical * (1 - tau_decay)
        tau_hat = tau_ema / (1 - tau_decay**count)
        tau_adj = jax.nn.relu(tau - tau_hat)

        updates = jax.tree_map(
            lambda g, n: -lr * g + jnp.sqrt(2 * lr * tau_adj) * n,
            posterior_grads,
            noise,
        )

        state = SGLDState(count=count, key=key, tau_ema=tau_ema)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


class AdamLDState(NamedTuple):
    count: jax.Array
    key: jax.Array
    tau_ema: jax.Array
    m1: optax.Updates
    v2: optax.Updates


def adamld(
    learning_rate, key, priors, tau=1.0, tau_decay=0.99, b1=0.9, b2=0.99, eps=1e-8
):
    """Adam Langevin Dynamics, by Katherine Crowson.

    The gradient provided to the update function should be a stochastic estimate
    of the gradient of the true unnormalized negative log likelihood. (That is,
    it should already be divided by the batch size and multiplied by the number of
    samples in the training set.)

    Args:
        learning_rate: a scalar or a function that maps the step count to a scalar.
        key: a JAX PRNG key.
        priors: a tuple of two trees with the same structure of the parameters,
            which contain the means and variances of the diagonal normal prior.
        tau: the temperature of the Langevin dynamics.
        tau_decay: the exponential moving average decay for the thermostat.
        b1: the exponential decay rate for the first moment estimate.
        b2: the exponential decay rate for the second moment estimate.
        eps: a small constant for numerical stability.
    """

    prior_means, prior_variances = priors

    def init_fn(params):
        m1 = jax.tree_map(jnp.zeros_like, params)
        v2 = jax.tree_map(jnp.zeros_like, params)
        return AdamLDState(
            count=jnp.zeros([], jnp.int32),
            key=key,
            tau_ema=jnp.zeros([]),
            m1=m1,
            v2=v2,
        )

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("No params provided to update_fn.")

        count = state.count + 1

        m1 = jax.tree_map(lambda x, u: b1 * x + (1 - b1) * u, state.m1, updates)
        v2 = jax.tree_map(lambda x, u: b2 * x + (1 - b2) * u**2, state.v2, updates)
        m1_hat = jax.tree_map(lambda x: x / (1 - b1**count), m1)
        v2_hat = jax.tree_map(lambda x: x / (1 - b2**count), v2)

        posterior_grads = jax.tree_map(
            lambda x, m, v, u: (x - m) / v + u,
            params,
            prior_means,
            prior_variances,
            m1_hat,
        )
        precond = jax.tree_map(
            lambda v, pv: 1 / (jnp.sqrt(v + pv**-2.0) + eps), v2_hat, prior_variances
        )

        key, subkey = jax.random.split(state.key)
        noise = noise_like_tree(subkey, params)

        lr = unwrap_schedule(learning_rate, state.count)

        tau_empirical = tree_mean(
            jax.tree_map(
                lambda pg, p: pg**2 * p * jnp.abs(lr) / 2, posterior_grads, precond
            )
        )
        tau_ema = state.tau_ema * tau_decay + tau_empirical * (1 - tau_decay)
        tau_hat = tau_ema / (1 - tau_decay**count)
        tau_adj = jax.nn.relu(tau - tau_hat)

        updates = jax.tree_map(
            lambda p, g, n: -lr * p * g + jnp.sqrt(jnp.abs(2 * lr * p * tau_adj)) * n,
            precond,
            posterior_grads,
            noise,
        )

        state = AdamLDState(count=count, key=key, tau_ema=tau_ema, m1=m1, v2=v2)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
