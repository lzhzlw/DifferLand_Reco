# Function to convert the parameters from output of FCNN to the priori range


import jax.numpy as jnp


# reverse of _par2nor
def _nor2par(p, mn, mx):

    return mn * (mx/mn)**p


# reverse of par2nor
def nor2par(x, mn, mx):

    return _nor2par(jnp.arctan(x)/jnp.pi+0.5, mn, mx)


# log normalization, convert p (between mn, mx) to [0, 1]
def _par2nor(p, mn, mx):

    return jnp.log(p/mn)/jnp.log(mx/mn)


# tan normalization, convert x (between 0, 1) to [-finite, +finite]
def par2nor(x, mn, mx):

    return jnp.tan((_par2nor(x, mn, mx)-0.5)*jnp.pi)


# same as nor2par, convert a param from [-finite, +finite] to the physical constrained range
def unnormalize(norm_param, parmin, parmax):

    return nor2par(norm_param, parmin, parmax)

