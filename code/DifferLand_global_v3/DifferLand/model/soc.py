# Function for soil organic carbon module


import jax.numpy as jnp
import jax
from DifferLand.model.update_carbon import update_cpool_flux


def one_pool(soc_pool, soc_forcing, soc_param, soc_pool_min, soc_pool_max):
    '''
    Function for one-pool soil organic carbon model

    :param soc_pool: current SOC pool, g C m-2 (1, )
    :param soc_forcing: (3, ), 2 + 1 = 3 (time variant + time invariant)
    :param soc_param: (1, )
    :param soc_pool_min: prior minimum pool, g C m-2 (1, )
    :param soc_pool_max: prior maximum pool, g C m-2 (1, )

    :return next_soc_pool: next SOC pool, g C m-2 (1, )
    :return rh_soil: soil heterotrophic respiration, g C m-2 day-1

    '''

    # parameter need to be constrained
    k_soc = soc_param[0]  # turnover rate of SOC to respiration, day-1
    
    # forcings
    # time variant
    c_in = soc_forcing[0]  # total carbon from litter to soil, g C m-2 day-1
    tpf = soc_forcing[1]  # temperature and precipitation limitation factor, unitless
    # time invariant
    delta_t = soc_forcing[2]  # time step in days, day(s)

    # pool
    soc_pool = soc_pool[0]  # total soc, g C m-2
    soc_pool_min = soc_pool_min[0]  # minimum soc, g C m-2
    soc_pool_max = soc_pool_max[0]  # maximum soc, g C m-2

    # soil heterotrophic respiration, g C m-2 day-1
    rh_soil = soc_pool * (1 - (1 - tpf * k_soc) ** delta_t) / delta_t
    # next soc pool, g C m-2
    next_soc_pool, rh_soil = update_cpool_flux(
        soc_pool, soc_pool_min, soc_pool_max, c_in, rh_soil, delta_t
    )

    return next_soc_pool, rh_soil


def Century_simplified(soc_pool, soc_forcing, soc_param, soc_pool_min, soc_pool_max):
    '''
    Function for simplified Century soil organic carbon model without depth consideration

    :param soc_pool: current SOC pools, g C m-2 (3, )
    :param soc_forcing: (3, ), 2 + 1 = 3 (time variant + time invariant)
    :param soc_param: (5, )
    :param soc_pool_min: prior minimum pools, g C m-2 (3, )
    :param soc_pool_max: prior maximum pools, g C m-2 (3, )

    :return next_soc_pool: next SOC pools, g C m-2 (3, )
    :return soc_pool: total SOC pool, g C m-2
    :return soc1_pool: active SOC pool, g C m-2
    :return soc2_pool: slow SOC pool, g C m-2
    :return soc3_pool: passive SOC pool, g C m-2
    :return rh_soil: soil heterotrophic respiration, g C m-2 day-1
    
    '''

    # constant parameters
    f_litter2soc1 = 1.0  # fraction of litter to SOC1, unitless
    f_litter2soc2 = 0.0  # fraction of litter to SOC2, unitless
    f_litter2soc3 = 0.0  # fraction of litter to SOC3, unitless
    f_soc31 = 0.0  # fraction of SOC1 to SOC3, unitless
    f_soc12 = 0.0  # fraction of SOC2 to SOC1, unitless
    f_soc13 = 0.0  # fraction of SOC3 to SOC1, unitless
    f_soc23 = 0.0  # fraction of SOC3 to SOC2, unitless

    # parameters need to be constrained
    f_soc21 = soc_param[0]  # fraction of SOC1 to SOC2, unitless
    f_soc32 = soc_param[1]  # fraction of SOC2 to SOC3, unitless
    k_soc1 = soc_param[2]  # turnover rate for SOC1, day-1
    k_soc2 = soc_param[3]  # turnover rate for SOC2, day-1
    k_soc3 = soc_param[4]  # turnover rate for SOC3, day-1

    # forcings
    # time variant
    c_in = soc_forcing[0]  # total carbon from litter to soil, g C m-2 day-1
    tpf = soc_forcing[1]  # temperature and precipitation limitation factor, unitless
    # time invariant
    delta_t = soc_forcing[2]  # time step in days, day(s)

    # calculate in matric approach, dx/dt = b*u + A*k*x
    # where x is soil carbon pool vector, u is litter input, b is allocation matrix
    # A is transfer coefficient matrix, k is turnover rate matrix
    # ==========================================================================================
    # define matrices
    # allocation matrix, unitless
    b_matrix = jnp.array([f_litter2soc1, f_litter2soc2, f_litter2soc3])
    # transfer matrix, unitless
    A_matrix = jnp.array([[-1, f_soc12, f_soc13],
                          [f_soc21, -1, f_soc23],
                          [f_soc31, f_soc32, -1]])
    # turnover rate matrix (diagonal matrix) with environment factor, day-1
    k_elements = jnp.array([k_soc1, k_soc2, k_soc3])
    k_elements = (1 - (1 - tpf * k_elements) ** delta_t) / delta_t
    k_matrix = jnp.diag(k_elements)
    # current pool matrix, g C m-2
    x_matrix = jnp.array(soc_pool)
    # matrix calculation
    # part 1 total output, g C m-2 day-1
    out_matrix = k_matrix @ x_matrix
    # part2, output to other soc pools, g C m-2 day-1
    transfer_in_coe_matrix = jnp.array([[0, f_soc12, f_soc13],
                                        [f_soc21, 0, f_soc23],
                                        [f_soc31, f_soc32, 0]])
    transfer_out_coe_elements = jnp.nansum(transfer_in_coe_matrix, axis=0)
    # convert to diagonal matrix
    transfer_out_coe_matrix = jnp.diag(transfer_out_coe_elements)
    transfer_out_matrix = transfer_out_coe_matrix @ out_matrix
    # part3, output to rh
    rh_matrix = out_matrix - transfer_out_matrix
    # part4, input from litter to soc pools, g C m-2 day-1
    in_matrix = c_in * b_matrix
    # part5, input from other soc pools, replace the diagonal elements to 0
    transfer_in_matrix = transfer_in_coe_matrix @ out_matrix
    # part6, net fluxes, g C m-2 day-1
    flux_matrix = in_matrix + A_matrix @ out_matrix

    # update SOC pools, g C m-2
    # ==========================================================================================
    next_soc_pool = x_matrix + flux_matrix * delta_t
    # adjustment, assuming the transfer between soil carbon pools unchanged, adjust rh
    min_matrix = jnp.array(soc_pool_min)
    max_matrix = jnp.array(soc_pool_max)
    next_soc_pool = jnp.clip(next_soc_pool, min_matrix, max_matrix)
    rh_matrix = in_matrix + transfer_in_matrix - transfer_out_matrix - (next_soc_pool - x_matrix) / delta_t

    # cal total and each SOC pools and soil heterotrophic respiration
    soc_pool = jnp.nansum(x_matrix, axis=0)  # total soc, g C m-2
    soc1_pool = x_matrix[0]  # active soc, g C m-2
    soc2_pool = x_matrix[1]  # slow soc, g C m-2
    soc3_pool = x_matrix[2]  # passive soc, g C m-2
    rh_soil = jnp.sum(rh_matrix, axis=0)  # soil heterotrophic respiration, g C m-2 day-1

    return next_soc_pool, soc_pool, soc1_pool, soc2_pool, soc3_pool, rh_soil


def Vmax_func(Vs, Ts, Vint, av, Vmod):
    '''
    Function to calculate Vmax, maximum reaction velocity, g Cs (g MIC)-1 day-1

    :param Vs: Temperature sensitivity for Vmax, ln(g Cs (g MIC)-1 hour-1) degC-1
    :param Ts: Soil temperature, degC
    :param Vint: Intercept for Vmax, ln(g Cs (g MIC)-1 hour-1)
    :param av: Tuning coefficient for Vmax, unitless
    :param Vmod: Modifier for Vmax, unitless

    :return: Vmax, g Cs (g MIC)-1 day-1
    
    '''

    Vmax = jnp.exp(Vs * Ts + Vint) * av * Vmod * 24.0  # from hour-1 to day-1

    return Vmax


def Km_func(Ks, Ts, Kint, ak, Kmod):
    '''
    Function to calculate Km, half saturation constant, g MIC m-3 soil

    :param Ks: Temperature sensitivity for Km, ln(mg MIC cm-3 soil) degC-1
    :param Ts: Soil temperature, degC
    :param Kint: Intercept for Km, ln(mg MIC cm-3 soil)
    :param ak: Tuning coefficient for Km, unitless
    :param Kmod: Modifier for Km, unitless

    :return: Km, g MIC m-3 soil

    '''

    Km = jnp.exp(Ks * Ts + Kint) * ak * Kmod * 1000.0  # from mg MIC cm-3 soil to g MIC m-3 soil

    return Km


def MM_func(Vmax, Km, substrate, soil_thick):
    '''
    Function to calculate reaction rate based on Michaelis-Menten kinetics, g Cs (g MIC)-1 day-1

    :param Vmax: Maximum reaction velocity, g Cs (g MIC)-1 day-1
    :param Km: Half saturation constant, g MIC m-3 soil
    :param substrate: Substrate concentration, g MIC m-2 soil
    :param soil_thick: Soil thickness for the considered soil layer, m

    :return: rate, g Cs (g MIC)-1 day-1

    '''
    
    rate = Vmax * substrate / (Km * soil_thick + substrate)

    return rate


def MGE_func(coe, Ts):
    '''
    Function to calculate microbial growth efficiency (MGE), mg mg-1
    
    :param coe: Coefficient for MGE, mg mg-1
    :param Ts: Soil temperature, degC

    :return: MGE, mg mg-1

    '''

    # -0.015 degC-1 is the temperature sensitivity for MGE 
    Tscalar = jnp.exp(-0.015 * Ts)
    MGE = coe * Tscalar

    return MGE


def MES_C(soc_pool, soc_forcing, soc_param, soc_pool_min, soc_pool_max):
    '''
    Function for MES-C SOC model without depth consideration
    
    Here the variable names are same as the supplementary document of MES-C model for clarity
    The variable names in the original fortran code are also provided in the comments for reference

    :param soc_pool: current SOC pools, g C m-2 (9, )
    :param soc_forcing: (10, ), 5 + 5 = 10 (time variant + time invariant)
    :param soc_param: (12, )
    :param soc_pool_min: prior minimum pools, g C m-2 (9, )
    :param soc_pool_max: prior maximum pools, g C m-2 (9, )

    :return next_soc_pool: next SOC pools, g C m-2 (9, )
    :return soc_pool: total SOC pool, g C m-2
    :return soc1_pool: particulate organic carbon pool (POC), g C m-2
    :return soc2_pool: mineral associated organic carbon pool (MAOC), g C m-2
    :return soc3_pool: live microbial biomass carbon pool, g C m-2
    :return rh_soil: soil heterotrophic respiration, g C m-2 day-1

    '''

    # constant parameters
    # (Wieder, 2015)
    Vslope = 0.063  # temperature sensitivity for Vmax (sv), ln(mg Cs (mg MIC)-1 hour-1) degC-1
    Vint = 5.47  # intercept for Vmax (bv), ln(mg Cs (mg MIC)-1 hour-1)
    Vmod_r1 = 10.0  # modifier for Vmax, LIT_m to MIC_r (xv1), unitless
    Vmod_r2 = 2.0  # modifier for Vmax, LIT_s to MIC_r (xv2), unitless
    Vmod_r3 = 10.0  # modifier for Vmax, LMWC to MIC_r (xv3), unitless
    Vmod_k1 = 3.0  # modifier for Vmax, LIT_m to MIC_k (xw1), unitless
    Vmod_k2 = 3.0  # modifier for Vmax, LIT_s to MIC_k (xw2), unitless
    Vmod_k3 = 2.0  # modifier for Vmax, LMWC to MIC_k (xw3), unitless
    Kslope1 = 0.017  # temperature sensitivity for Km, LIT_m to MIC_r/k (sk), ln(mg MIC cm-3 soil) degC-1
    Kslope2 = 0.027  # temperature sensitivity for Km, LIT_s to MIC_r/k (skx), ln(mg MIC cm-3 soil) degC-1
    Kslope3 = 0.017  # temperature sensitivity for Km, LMWC to MIC_r/k (sk), ln(mg MIC cm-3 soil) degC-1
    Kint = 3.19  # intercept for Km (bk), ln(mg MIC cm-3 soil)
    Kmod_r1 = 1 / 8.0  # modifier for Km, LIT_m to MIC_r (1/xk1), unitless
    Kmod_r2 = 1 / 2.0  # modifier for Km, LIT_s to MIC_r (1/xk2), unitless
    Kmod_r3 = 1 / 4.0  # modifier for Km, LMWC to MIC_r (1/xk3), unitless
    Kmod_k1 = 1 / 2.0  # modifier for Km, LIT_m to MIC_k (1/xj1), unitless
    Kmod_k2 = 1 / 4.0  # modifier for Km, LIT_s to MIC_k (1/xj2), unitless
    Kmod_k3 = 1 / 6.0  # modifier for Km, LMWC to MIC_k (1/xj3), unitless
    coe_mge_r1 = 0.55 # coefficient for microbial growth efficiency, LIT_m to MIC_r (epsilon1), mg mg-1
    coe_mge_r2 = 0.25  # coefficient for microbial growth efficiency, LIT_s to MIC_r (epsilon2)
    coe_mge_r3 = 0.55  # coefficient for microbial growth efficiency, LMWC to MIC_r (epsilon1), mg mg-1
    coe_mge_k1 = 0.75  # coefficient for microbial growth efficiency, LIT_m to MIC_k (epsilon3), mg mg-1
    coe_mge_k2 = 0.35  # coefficient for microbial growth efficiency, LIT_s to MIC_k (epsilon4)
    coe_mge_k3 = 0.75  # coefficient for microbial growth efficiency, LMWC to MIC_k (epsilon3), mg mg-1
    # (Rose, 2022)
    lamda = 2.1e-4  # dependence of rate on matric potential, kpa-1
    kamin = 0.2  # minimum relative rate in saturated soil, unitless
    phcoeff1 = 0.186  # coefficient for estimating th ebinding affinity for LMWC sorption, unitless
    phcoeff2 = 0.216  # coefficient for estimating th ebinding affinity for LMWC sorption, unitless
    kadsorp = 1.0  # adsorption rate for LMWC to MAOC, mg C L-1 day-1

    # parameters need to be constrained
    fmet = soc_param[0]  # fraction of carbon input to metabolic litter
    kab = soc_param[1]  # breakdown rate for AGG,day-1
    fm = soc_param[2]  # fraction of LIT_m to AGG_m, unitless
    av = soc_param[3]  # tuning coefficient for Vmax, unitless
    ak = soc_param[4]  # tuning coefficient for Km, unitless
    fs = soc_param[5]  # fraction of LIT_s to AGG_s, unitless
    tau_r = soc_param[6]  # turnover rate for MIC_r, hour-1
    tau_k = soc_param[7]  # turnover rate for MIC_k, hour-1
    ksorp = soc_param[8]  # equilibrium constant, unitless
    qmaxcoeff = soc_param[9]  # coefficient for qmax, unitless
    kma = soc_param[10]  # formation rate for MAOC to AGG_maoc, day-1
    floss = soc_param[11]  # leaching loss rate for LMWC, day-1

    # forcings
    # time variant
    c_in = soc_forcing[0]  # total carbon from litter to soil, g C m-2 day-1
    npp = soc_forcing[1] * 365.25  # NPP from g C m-2 day-1 to g C m-2 year-1
    soil_moist = soc_forcing[2]  # soil moisture, m3 m-3
    Ts = soc_forcing[3]  # soil temperature, degC
    matpot = soc_forcing[4]  # matric potential, kPa
    
    # time invariant
    bulkd = soc_forcing[5]  # soil bulk density, kg m-3
    pore = 1.0 - bulkd / 2650.0  # soil porosity, mm3 mm-3
    ph = soc_forcing[6]  # soil pH, unitless
    clay = soc_forcing[7]  # soil clay fraction, unitless, 0-1
    silt = soc_forcing[8]  # soil silt fraction, unitless, 0-1
    delta_t = soc_forcing[9]  # time step in days, day(s)
    soil_thick = soc_forcing[10]  # soil thickness for the considered soil layer, m

    # pools, g C m-2
    litm_pool = soc_pool[0]  # Metabolic litter pool, LIT_m (m)
    lits_pool = soc_pool[1]  # Structural litter pool, LIT_s (s)
    micr_pool = soc_pool[2]  # Microbial biomass pool, r-stratagy, MIC_r (r)
    mick_pool = soc_pool[3]  # Microbial biomass pool, k-stratagy, MIC_k (k)
    lmwc_pool = soc_pool[4]  # Light molecule weight carbon pool, LMWC (a)
    maoc_pool = soc_pool[5]  # Mineral associated SOC pool, MAOC (c)
    # 3 aggregated organic matter pools, AGG_m, AGG_s, AGG_maoc (p)
    aggm_pool = soc_pool[6]
    aggs_pool = soc_pool[7]
    aggmaoc_pool = soc_pool[8]

    litm_min = soc_pool_min[0]
    lits_min = soc_pool_min[1]
    micr_min = soc_pool_min[2]
    mick_min = soc_pool_min[3]
    lmwc_min = soc_pool_min[4]
    maoc_min = soc_pool_min[5]
    aggm_min = soc_pool_min[6]
    aggs_min = soc_pool_min[7]
    aggmaoc_min = soc_pool_min[8]

    litm_max = soc_pool_max[0]
    lits_max = soc_pool_max[1]
    micr_max = soc_pool_max[2]
    mick_max = soc_pool_max[3]
    lmwc_max = soc_pool_max[4]
    maoc_max = soc_pool_max[5]
    aggm_max = soc_pool_max[6]
    aggs_max = soc_pool_max[7]
    aggmaoc_max = soc_pool_max[8]

    # LIT_m
    # ===================================================================================
    # Litter to LIT_m, g C m-2 day-1
    # fmet is a linear function of litter nitrogen to lignin ratio in the original code
    # Here it is taken as a parameter need to be constrained
    c_in_met = c_in * fmet
    # AGG_m to LIT_m, g C m-2 day-1
    # Water limitation on AGG breakdown, unitless, 0-1 (Ghezzehei et al., 2019)
    xwater1 = jnp.sqrt(jnp.clip(soil_moist / pore, 1e-6, 1.0))
    # breakdown rate for AGG, day-1, calculated the averaged daily flux during a delta_t
    # when the turnover is linear to pool size, solution for x(t) = x0 * exp(-kt)
    # when timestep is small, use the linear approximation, exp(-kt) ~ 1-kt
    # when delta_t is large, use the linear approximation, exp(-k)t ~ (1-k)exp(t)
    fluxp2m = aggm_pool* (1 - (1 - xwater1 * kab) ** delta_t) / delta_t  # g C m-2 day-1
    # LIT_m to AGG_m, g C m-2 day-1
    fluxm2p = c_in_met * fm
    # LIT_m to MIC_r, reverse Michaelis-Menten kinetics, including respiration, g C m-2 day-1
    vmax_r1 = Vmax_func(Vslope, Ts, Vint, av, Vmod_r1)  # maximum reaction velocity, g Cs (g MIC)-1 day-1
    km_r1 = Km_func(Kslope1, Ts, Kint, ak, Kmod_r1)  # half saturation constant, g MIC m-3 soil
    fm2r = MM_func(vmax_r1, km_r1, micr_pool, soil_thick)  # reaction rate based on Michaelis-Menten kinetics, g Cs (g MIC)-1 day-1
    # Water limitation on microbial uptake, unitless, 0-1 (Abramoff et al., 2022, eqn. 15)
    tempt = jnp.sqrt(jnp.clip(1.0 - soil_moist / pore, 1e-6, 1.0))
    xwater2 = jnp.exp(lamda * matpot) * (kamin + (1.0 - kamin) * tempt) * xwater1
    xwater2 = jnp.clip(xwater2, 1e-6, 1.0)
    xph2 = 1.0 / (1.0 + jnp.exp(-(ph - 4.798) / 0.4246))  # pH limitation on microbial uptake for r strategist, unitless
    fluxm2r = litm_pool * (1 - (1 - xwater2 * xph2 * fm2r) ** delta_t) / delta_t  # g C m-2 day-1 
    # LIT_m to MIC_k, reverse Michaelis-Menten kinetics, including respiration, g C m-2 day-1
    vmax_k1 = Vmax_func(Vslope, Ts, Vint, av, Vmod_k1)  # g Cs (g MIC)-1 day-1
    km_k1 = Km_func(Kslope1, Ts, Kint, ak, Kmod_k1)  # g MIC m-3 soil
    fm2k = MM_func(vmax_k1, km_k1, mick_pool, soil_thick)  # g Cs (g MIC)-1 day-1
    xph3 = 1.0 / (1.0 + jnp.exp(-(ph - 3.022) / 0.428))  # pH limitation on microbial uptake for k strategist, unitless
    fluxm2k = litm_pool * (1 - (1 - xwater2 * xph3 * fm2k) ** delta_t) / delta_t  # g C m-2 day-1

    # LIT_s
    # ===================================================================================
    # Litter to LIT_s， g C m-2 day-1
    c_in_str = c_in * (1.0 - fmet)
    # AGG_s to LIT_s, g C m-2 day-1
    fluxp2s = aggs_pool * (1 - (1 - xwater1 * kab) ** delta_t) / delta_t
    # LIT_s to AGG_s, g C m-2 day-1
    fluxs2p = c_in_str * fs
    # LIT_s to MIC_r, reverse Michaelis-Menten kinetics, including respiration, g C m-2 day-1
    vmax_r2 = Vmax_func(Vslope, Ts, Vint, av, Vmod_r2)  # g Cs (g MIC)-1 day-1
    km_r2 = Km_func(Kslope2, Ts, Kint, ak, Kmod_r2)  # g MIC m-3 soil
    fs2r = MM_func(vmax_r2, km_r2, micr_pool, soil_thick)  # g Cs (g MIC)-1 day-1
    fluxs2r = lits_pool * (1 - (1 - xwater2 * xph2 * fs2r) ** delta_t) / delta_t  # g C m-2 day-1
    # LIT_s to MIC_k, reverse Michaelis-Menten kinetics, including respiration, g C m-2 day-1
    vmax_k2 = Vmax_func(Vslope, Ts, Vint, av, Vmod_k2)  # g Cs (g MIC)-1 day-1
    km_k2 = Km_func(Kslope2, Ts, Kint, ak, Kmod_k2)  # g MIC m-3 soil
    fs2k = MM_func(vmax_k2, km_k2, mick_pool, soil_thick)  # g Cs (g MIC)-1 day-1
    fluxs2k = lits_pool * (1 - (1 - xwater2 * xph3 * fs2k) ** delta_t) / delta_t  # g C m-2 day-1

    # MIC_r
    # ===================================================================================
    # LIT_m to MIC_r, g C m-2 day-1
    mge_r1 = MGE_func(coe_mge_r1, Ts)  # microbial growth efficiency, mg mg-1
    fluxm2r_net = fluxm2r * mge_r1  # exclude respiration, g C m-2 day-1
    fluxm2r_res = fluxm2r * (1.0 - mge_r1)  # respiration, g C m-2 day-1
    # LIT_s to MIC_r, g C m-2 day-1
    mge_r2 = MGE_func(coe_mge_r2, Ts)  # microbial growth efficiency, mg mg-1
    fluxs2r_net = fluxs2r * mge_r2  # exclude respiration, g C m-2 day-1
    fluxs2r_res = fluxs2r * (1.0 - mge_r2)  # respiration, g C m-2 day-1
    # LMWC to MIC_r, forward Michaelis-Menten kinetics, g C m-2 day-1
    vmax_r3 = Vmax_func(Vslope, Ts, Vint, av, Vmod_r3)  # g Cs (g MIC)-1 day-1
    km_r3 = Km_func(Kslope3, Ts, Kint, ak, Kmod_r3)  # g MIC m-3 soil
    # adjust Km based on clay content, only for LMWC to MIC_r/k, unitless
    Pscalar = 1.0 / (2.0 * jnp.exp(-2 * jnp.sqrt(clay)))
    km_r3 = km_r3 * Pscalar  # g MIC m-3 soil
    fa2r = MM_func(vmax_r3, km_r3, lmwc_pool, soil_thick)  # g Cs (g MIC)-1 day-1
    fluxa2r = lmwc_pool * (1 - (1 - xwater2 * xph2 * fa2r) ** delta_t) / delta_t  # g C m-2 day-1
    mge_r3 = MGE_func(coe_mge_r3, Ts)  # microbial growth efficiency, mg mg-1
    fluxa2r_net = fluxa2r * mge_r3  # exclude respiration, g C m-2 day-1
    fluxa2r_res = fluxa2r * (1.0 - mge_r3)  # respiration, g C m-2 day-1
    # MIC_r to SOC, including LMWC and MAOC, g C m-2 day-1
    # modifier for microbial turnover (Wielder et al., 2015, 0.8-1.2) (tvref)
    # 100 g C m-2 yr-1, referecne value for npp
    tau_mod = jnp.clip(jnp.sqrt(npp/100), 0.6, 1.3)  # unitless
    tau_r = tau_r * tau_mod * jnp.exp(0.3 * fmet) * 24.0  #  from per hour to per day
    # calculate the averaged daily flux during a delta_t
    # as the turnovet is non-linear to pool size, solution for x(t) = x0 / (1 + x0 * k * t)
    fluxr2ac = micr_pool ** 2 * tau_r / (micr_pool * tau_r * delta_t + 1.0)  # g C m-2 day-1

    # MIC_k
    # ===================================================================================
    # LIT_m to MIC_k, g C m-2 day-1
    mge_k1 = MGE_func(coe_mge_k1, Ts)  # microbial growth efficiency, mg mg-1
    fluxm2k_net = fluxm2k * mge_k1  # exclude respiration, g C m-2 day-1
    fluxm2k_res = fluxm2k * (1.0 - mge_k1)  # respiration, g C m-2 day-1
    # LIT_s to MIC_k, g C m-2 day-1
    mge_k2 = MGE_func(coe_mge_k2, Ts)  # microbial growth efficiency, mg mg-1
    fluxs2k_net = fluxs2k * mge_k2  # exclude respiration, g C m-2 day-1
    fluxs2k_res = fluxs2k * (1.0 - mge_k2)  # respiration, g C m-2 day-1
    # LMWC to MIC_k, forward Michaelis-Menten kinetics, g C m-2 day-1
    vmax_k3 = Vmax_func(Vslope, Ts, Vint, av, Vmod_k3)  # g Cs (g MIC)-1 day-1
    km_k3 = Km_func(Kslope3, Ts, Kint, ak, Kmod_k3)  # g MIC m-3 soil
    km_k3 = km_k3 * Pscalar
    fa2k = MM_func(vmax_k3, km_k3, lmwc_pool, soil_thick)  # g Cs (g MIC)-1 day-1
    fluxa2k = lmwc_pool * (1 - (1 - xwater2 * xph3 * fa2k) ** delta_t) / delta_t  # g C m-2 day-1
    mge_k3 = MGE_func(coe_mge_k3, Ts)  # microbial growth efficiency, mg mg-1
    fluxa2k_net = fluxa2k * mge_k3  # exclude respiration, g C m-2 day-1
    fluxa2k_res = fluxa2k * (1.0 - mge_k3)  # respiration, g C m-2 day-1
    # MIC_k to SOC, including LMWC and MAOC, g C m-2 day-1
    tau_k = tau_k * tau_mod * jnp.exp(0.1 * fmet) * 24.0  # per day
    fluxk2ac = mick_pool ** 2 * tau_k / (mick_pool * tau_k * delta_t + 1.0)  # g C m-2 day-1

    # LMWC
    # ===================================================================================
    # MIC_r to LMWC, g C m-2 day-1
    fr2p = 0.3 * jnp.exp(1.3 * clay)  # fraction of MIC_r to AGG, temporary value, unitless
    fr2c = jnp.clip(fr2p + 0.1 * jnp.exp(-3.0 * fmet), 0.0, 1.0)  # fraction of MIC_r to MAOC, unitless
    fr2p = 0
    fr2a = jnp.clip(1.0 - fr2c, 1e-6, 1.0)  # fraction of MIC_r to LMWC, unitless
    fluxr2a = fluxr2ac * fr2a
    # MIC_k to LMWC, g C m-2 day-1
    fk2p = 0.2 * jnp.exp(0.8 * clay)  # fraction of MIC_k to AGG, temporary value, unitless
    fk2c = jnp.clip(fk2p + 0.3 * jnp.exp(-3.0 * fmet), 0.0, 1.0)  # fraction of MIC_k to MAOC, unitless
    fk2p = 0
    fk2a = jnp.clip(1.0 - fk2c, 1e-6, 1.0)  # fraction of MIC_k to LMWC, unitless
    fluxk2a = fluxk2ac * fk2a
    # MAOC to LMWC, g C m-2 day-1
    kdesorp = kadsorp / ksorp  # desorption rate for MAOC, from mg C L-1 day -1 to g C m-2 day-1
    qmax =  qmaxcoeff * (clay + silt) * 100.0 * bulkd * soil_thick  # from g C kg-1 soil to g C m-2 soil, bulkd in kg m-3, soil_thick in m
    fluxc2a = maoc_pool * (1 - ( 1 - kdesorp / qmax) ** delta_t) / delta_t  # g C m-2 day-1
    # LMWC to MIC_r, already included in MIC_r section
    # LMWC to MIC_k, already included in MIC_k section
    # LMWC to MAOC, g C m-2 day-1
    xph1 = jnp.exp(-phcoeff1 * ph - phcoeff2)  # pH limitation on adsorption to MAOC, L mg-1, (Abramoff et al., 2022, eqn. 10)
    # need to considder lmwc and maoc pool size simutaneously
    # solution for x(t) = (C - qmax) / (1 - A(1+B)^t)
    # C = maoc0 + lmwc0, A = (qmax - maoc0) / lmwc0, B = (qmax - C) * k / qmax0
    tempt_k = xwater1 * xph1 * kadsorp  # effective adsorption rate, from mg C L-1 day-1
    tempt_C = maoc_pool + lmwc_pool  # total carbon for maoc and lmwc, g C m-2
    tempt_A = (qmax - maoc_pool) / lmwc_pool  # unitless
    tempt_B = (qmax - tempt_C) * tempt_k / qmax  # day-1
    lmwc_pool_delta_t = (tempt_C - qmax) / (1 - tempt_A * (1 + tempt_B) ** delta_t)  # g C m-2
    fluxa2c = (lmwc_pool - lmwc_pool_delta_t) / delta_t  # g C m-2 day-1
    # Leaching loss of LMWC, g C m-2 day-1
    fluxaloss = lmwc_pool * (1 - (1 - xwater1 * floss) ** delta_t) / delta_t  # g C m-2 day-1

    # MAOC
    # ===================================================================================
    # MIC_r to MAOC, g C m-2 day-1
    fluxr2c = fluxr2ac * fr2c
    # MIC_k to MAOC, g C m-2 day-1
    fluxk2c = fluxk2ac * fk2c
    # LMWC to MAOC, already included in LMWC section
    # AGG_maoc to MAOC, g C m-2 day-1
    fluxp2c = aggmaoc_pool * (1 - (1 - xwater1 * kab) ** delta_t) / delta_t
    # MAOC to LMWC, already included in LMWC section
    # MAOC to AGG_maoc, g C m-2 day-1
    fluxc2p = maoc_pool * (1 - (1 - xwater1 * kma) ** delta_t) / delta_t

    # AGG
    # ===================================================================================
    # LIT_m to AGG_m, already included in LIT_m section
    # AGG_m to LIT_m, already included in LIT_m section

    # LIT_s to AGG_s, already included in LIT_s section
    # AGG_s to LIT_s, already included in LIT_s section

    # MAOC to AGG_maoc, already included in MAOC section
    # AGG_maoc to MAOC, already included in MAOC section

    # Update all pools
    # next_aggm_pool = aggm_pool + (fluxm2p - fluxp2m) * delta_t
    # next_aggs_pool = aggs_pool + (fluxs2p - fluxp2s) * delta_t
    # next_aggmaoc_pool = aggmaoc_pool + (fluxc2p - fluxp2c) * delta_t
    # next_maoc_pool = maoc_pool + (fluxp2c + fluxr2c + fluxk2c + fluxa2c - fluxc2a - fluxc2p) * delta_t
    # next_litm_pool = litm_pool + (c_in_met + fluxp2m - fluxm2p - fluxm2r - fluxm2k) * delta_t
    # next_lits_pool = lits_pool + (c_in_str + fluxp2s - fluxs2p - fluxs2r - fluxs2k) * delta_t
    # next_micr_pool = micr_pool + (fluxm2r_net + fluxs2r_net + fluxa2r_net - fluxr2a - fluxr2c) * delta_t
    # next_mick_pool = mick_pool + (fluxm2k_net + fluxs2k_net + fluxa2k_net - fluxk2a - fluxk2c) * delta_t
    # next_lmwc_pool = lmwc_pool + (fluxr2a + fluxk2a + fluxc2a - fluxa2c - fluxa2r - fluxa2k - fluxaloss) * delta_t

    # Agg_m
    next_aggm_pool, fluxp2m = update_cpool_flux(aggm_pool, aggm_min, aggm_max, fluxm2p, fluxp2m, delta_t)
    # Agg_s
    next_aggs_pool, fluxp2s = update_cpool_flux(aggs_pool, aggs_min, aggs_max, fluxs2p, fluxp2s, delta_t)
    # Agg_maoc
    next_aggmaoc_pool, fluxp2c = update_cpool_flux(aggmaoc_pool, aggmaoc_min, aggmaoc_max, fluxc2p, fluxp2c, delta_t)
    # MAOC
    next_maoc_pool, fluxc2a = update_cpool_flux(maoc_pool, maoc_min, maoc_max, fluxp2c + fluxr2c + fluxk2c + fluxa2c, 
                                                fluxc2a, delta_t, fluxc2p)
    # LIT_m
    fluxm_res = fluxm2r_res + fluxm2k_res
    fluxm2r_res_frac = fluxm2r_res / (fluxm_res + 1e-6)
    fluxm2r_res_frac = jnp.clip(fluxm2r_res_frac, 1e-6, 1.0)
    next_litm_pool, fluxm_res = update_cpool_flux(litm_pool, litm_min, litm_max, c_in_met + fluxp2m, 
                                                  fluxm_res, delta_t, fluxm2p + fluxm2r_net + fluxm2k_net)
    fluxm2r_res = fluxm_res * fluxm2r_res_frac
    fluxm2k_res = fluxm_res * (1.0 - fluxm2r_res_frac)
    # LIT_s
    fluxs_res = fluxs2r_res + fluxs2k_res
    fluxs2r_res_frac = fluxs2r_res / (fluxs_res + 1e-6)
    fluxs2r_res_frac = jnp.clip(fluxs2r_res_frac, 1e-6, 1.0)
    next_lits_pool, fluxs_res = update_cpool_flux(lits_pool, lits_min, lits_max, c_in_str + fluxp2s, 
                                                  fluxs_res, delta_t, fluxs2p + fluxs2r_net + fluxs2k_net)
    fluxs2r_res = fluxs_res * fluxs2r_res_frac
    fluxs2k_res = fluxs_res * (1.0 - fluxs2r_res_frac)
    # MIC_r
    next_micr_pool, fluxr2a = update_cpool_flux(micr_pool, micr_min, micr_max, fluxm2r_net + fluxs2r_net + fluxa2r_net, 
                                                fluxr2a, delta_t, fluxr2c)
    # MIC_k
    next_mick_pool, fluxk2a = update_cpool_flux(mick_pool, mick_min, mick_max, fluxm2k_net + fluxs2k_net + fluxa2k_net, 
                                                fluxk2a, delta_t, fluxk2c)
    # LMWC
    next_lmwc_pool, fluxaloss = update_cpool_flux(lmwc_pool, lmwc_min, lmwc_max, fluxr2a + fluxk2a + fluxc2a, 
                                                  fluxaloss, delta_t, fluxa2c + fluxa2r + fluxa2k)

    # total soc, maoc and poc
    next_soc_pool = jnp.array([next_litm_pool, next_lits_pool, next_micr_pool, next_mick_pool, next_lmwc_pool, 
                           next_maoc_pool, next_aggm_pool, next_aggs_pool, next_aggmaoc_pool])
    soc1_pool = litm_pool + lits_pool + aggm_pool + aggs_pool + lmwc_pool  # poc
    soc2_pool = maoc_pool + aggmaoc_pool  # maoc
    soc3_pool = micr_pool + mick_pool  # live microbial biomass
    soc_pool = soc1_pool + soc2_pool + soc3_pool  # total soc

    # Soil heterotrophic respiration, g C m-2 day-1
    rh_soil = fluxm2r_res + fluxs2r_res + fluxa2r_res + fluxm2k_res + fluxs2k_res + fluxa2k_res

    # jax.debug.breakpoint()

    return next_soc_pool, soc_pool, soc1_pool, soc2_pool, soc3_pool, rh_soil

