import jax.numpy as jnp

def postprocessing(input, Cl):
    return Cl * jnp.exp(3-input[0])