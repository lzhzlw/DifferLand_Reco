# Function to update carbon pools and fluxes
# according to prior carbon pool range and mass balance


import jax.numpy as jnp


def update_cpool_flux(current_pool, pool_min, pool_max, total_influx, eflux, delta_t, other_eflux=0.0):
    
    '''Function to update carbon pool (g C m-2) and out_flux (g C m-2 day-1)

    :param current_pool: current carbon pool, g C m-2
    :param pool_min: prior minimum carbon pool, g C m-2
    :param pool_max: prior maximum carbon pool, g C m-2
    :param total_influx: total carbon influx to the pool, g C m-2 day-1
    :param eflux: carbon outflux from the pool, which will be updated when carbon pool beyond the prior range,  C m-2 day-1
    :param delta_t: time step, day(s)
    :param other_eflux: other outflux from the pool (default=0), g C m-2 day-1

    :return: next_pool (g C m-2), updated out_flux (g C m-2 day-1)

    '''

    # calculate next pool
    next_pool = current_pool + (total_influx - eflux - other_eflux) * delta_t
    # constraint carbon pool by prior minimum or maximum value
    next_pool = jnp.clip(next_pool, pool_min, pool_max)
    # update eflux acording to the constrained pool
    eflux = total_influx - other_eflux - (next_pool - current_pool) / delta_t
    
    return next_pool, eflux

