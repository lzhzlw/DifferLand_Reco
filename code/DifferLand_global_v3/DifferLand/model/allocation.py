# Function for allocation of carbon to the different pools


import jax
import jax.numpy as jnp
from DifferLand.model.update_carbon import update_cpool_flux


def plant_allocation(pa_pool, pa_forcing, pa_param, pa_pool_min, pa_pool_max):
    '''
    Function for plant carbon allocation
    
    :param pa_pool: current plant carbon pools, g C m-2 (5, )
    :param pa_forcing: (5, ), 4 + 1 = 5 (time variant + time invariant)
    :param pa_param: (8, )
    :param pa_pool_min: prior minimum pool, g C m-2 (5, )
    :param pa_pool_max: prior maximum pool, g C m-2 (5, )
    
    :return next_pa_pool: next plant carbon pools, g C m-2 (5, )
    :return ra: autotrophic respiration, g C m-2 day-1
    :return npp: net primary productivity, g C m-2 day-1
    :return rh_litter: litter heterotrophic respiration, g C m-2 day-1

    '''
    
    # parameters need to be constrained
    # allocation coefficients
    f_auto = pa_param[0]  # fraction of GPP to autotrophic respiration, unitless
    f_fol = pa_param[1]  # fraction of (NPP - labile production) to foliar pool, unitless
    f_lab = pa_param[2]  # fraction of NPP to labile pool, unitless
    f_root = pa_param[3]  # fraction of (NPP - foliar production - labile production) to root pool, unitless
    # turnover rate
    k_wood = pa_param[4]  # turnover rate of wood to litter, day-1
    k_root = pa_param[5]  # turnover rate of root to litter, day-1
    k_litter = pa_param[6]  # turnover rate of litter to respiration, day-1
    k_decomp = pa_param[7]  # turnover rate of litter to soil, day-1
    
    # forcings
    # time variant
    gpp = pa_forcing[0]  # g C m-2 day-1
    ffr = pa_forcing[1]  # foliar falling rate, day-1
    lrr = pa_forcing[2]  # labile releasing rate, day-1
    tpf = pa_forcing[3]  # temperature and precipitation limitation factor, unitless
    # time invariant
    delta_t = pa_forcing[4]  # time step, day(s)

    # pools and prior range, g C m-2
    foliar_pool = pa_pool[0]
    labile_pool = pa_pool[1]
    wood_pool = pa_pool[2]
    root_pool = pa_pool[3]
    litter_pool = pa_pool[4]
    foliar_min = pa_pool_min[0]
    foliar_max = pa_pool_max[0]
    labile_min = pa_pool_min[1]
    labile_max = pa_pool_max[1]
    wood_min = pa_pool_min[2]
    wood_max = pa_pool_max[2]    
    root_min = pa_pool_min[3]
    root_max = pa_pool_max[3]
    litter_min = pa_pool_min[4]
    litter_max = pa_pool_max[4]

    # calculation
    # ========================================================================================================
    # plant carbon fluxes, pools and steady state
    # fluxes, g C m-2 day-1
    # Ra and npp
    ra = f_auto * gpp
    npp = gpp - ra
    # foliar
    leaf_production = npp * f_fol
    leaf2litter = foliar_pool * (1 - (1 - ffr) ** delta_t) / delta_t
    # labile
    labile_production = (npp - leaf_production) * f_lab
    labile2leaf = labile_pool * (1 - (1 - lrr) ** delta_t) / delta_t
    # root
    root_production = (npp - leaf_production - labile_production) * f_root
    root2litter = root_pool * (1 - (1 - k_root) ** delta_t) / delta_t
    # wood
    wood_production = npp - leaf_production - labile_production - root_production
    wood2litter = wood_pool * (1 - (1 - k_wood) ** delta_t) / delta_t
    # litter
    rh_litter = litter_pool * (1 - (1 - tpf * k_litter) ** delta_t) / delta_t
    litter2soc = litter_pool * (1 - (1 - tpf * k_decomp) ** delta_t) / delta_t

    # update pools, g C m-2
    next_labile_pool, labile2leaf = update_cpool_flux(
        labile_pool, labile_min, labile_max, labile_production, labile2leaf, delta_t
    )
    next_foliar_pool, leaf2litter = update_cpool_flux(
        foliar_pool, foliar_min, foliar_max, leaf_production + labile2leaf, leaf2litter, delta_t
    )
    next_root_pool, root2litter = update_cpool_flux(
        root_pool, root_min, root_max, root_production, root2litter, delta_t
    )
    next_wood_pool, wood2litter = update_cpool_flux(
        wood_pool, wood_min, wood_max, wood_production, wood2litter, delta_t
    )
    next_litter_pool, litter2soc = update_cpool_flux(
        litter_pool, litter_min, litter_max, leaf2litter + root2litter + wood2litter, litter2soc, 
        delta_t, rh_litter
    )
    litter2soc = jnp.maximum(litter2soc, 1e-6)
    rh_litter = leaf2litter + root2litter + wood2litter - litter2soc - (next_litter_pool - litter_pool) / delta_t
    
    # next pool, g C m-2
    next_pa_pool = jnp.array([next_foliar_pool, next_labile_pool, next_wood_pool, next_root_pool, next_litter_pool])
    
    # jax.debug.breakpoint()

    return next_pa_pool, ra, npp, rh_litter, litter2soc

