#!/usr/bin/env python3

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from rich import print
from rich.traceback import install

import jax_bayesian_aesthetic_2 as training


def main():
    install()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=str, help="the input embedding file")
    p.add_argument("--model", type=str, required=True, help="the model file")
    p.add_argument("--output", "-o", type=str, required=True, help="the output file")
    args = p.parse_args()

    input_pkl = pickle.load(open(args.input, "rb"))
    embeds = jnp.array(input_pkl["embeds"])
    filenames = input_pkl["filenames"]

    model_pkl = pickle.load(open(args.model, "rb"))
    ensemble = jax.tree_map(jnp.array, model_pkl["ensemble_samples"])

    model = training.Model()

    @jax.jit
    def apply_ensemble(params, x):
        return jax.vmap(model.apply, in_axes=(0, None))(params, x)

    y_pred = apply_ensemble(ensemble, embeds)
    y_pred_mean = jnp.mean(y_pred, axis=0)
    y_pred_std = jnp.std(y_pred, axis=0)
    print("y_pred_mean:", jnp.mean(y_pred_mean))
    print("y_pred_std:", jnp.sqrt(jnp.mean(y_pred_std**2)))

    indices = jnp.flip(jnp.argsort(y_pred_std))
    filenames = filenames[indices]
    y_pred_mean = y_pred_mean[indices]
    y_pred_std = y_pred_std[indices]

    cols = {"filename": filenames, "mean": y_pred_mean, "std": y_pred_std}
    cols = jax.tree_map(np.array, cols)
    df = pd.DataFrame(cols)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
