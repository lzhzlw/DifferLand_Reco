# Function for environment factor


import jax.numpy as jnp
import jax


def offset(L, w):
    '''
    Auxiliary function to calculate labile releasing and foliar falling rate, day-1
    
    :param L: residence time of carbon, year
    :param w: carbon releasing/falling period, day

    '''

    p1 = 0.000023599784710
    p2 = 0.000332730053021
    p3 = 0.000901865258885
    p4 = -0.005437736864888
    p5 = -0.020836027517787
    p6 = 0.126972018064287
    p7 = -0.188459767342504

    lf = jnp.log(L - 1)
    os = p1 * lf ** 6 + p2 * lf ** 5 + p3 * lf ** 4 + p4 * lf ** 3 + p5 * lf ** 2 + p6 * lf + p7
    os = os * w

    return os


def lab_release_rate(lrr_forcing, lrr_param):
    '''
    Function to calculate labile releasing rate, day-1
    
    :lrr_forcing: (1, ), 1 + 0 = 1 (time variant + time invariant)
    :lrr_param: (3, )

    :return lrr: labile releasing rate, day-1

    '''

    # parameters need to be constrained
    rt_lab = lrr_param[0]  # residence time of labile carbon releasing to foliar pool, year
    lab_release_period = lrr_param[1]  # labile releasing period, day
    doy_lab = lrr_param[2]  # day of year with maximum labile releasing rate, unitless

    # forcings
    # time variant
    doy = lrr_forcing[0]  # day of year, unitless

    # calculating labile releasing rate, day-1
    fl = (jnp.log(rt_lab) - jnp.log(rt_lab - 1)) * 0.5
    wl = lab_release_period * jnp.sqrt(2) / 2
    osl = offset(rt_lab, wl)
    sf = 365.25 / jnp.pi
    lrr = (2 / jnp.sqrt(jnp.pi)) * (fl / wl) * jnp.exp(-(jnp.sin((doy - doy_lab + osl) / sf) * sf / wl) ** 2)

    return lrr


def foliar_fall_rate(ffr_forcing, ffr_param):
    '''
    Function to calculate foliar falling rate, day-1
    
    :ffr_forcing: (1, ), 1 + 0 = 1 (time variant + time invariant)
    :ffr_param: (3, )

    :return ffr: foliar falling rate, day-1

    '''

    # parameters need to be constrained
    rt_fol = ffr_param[0]  # residence time of foliar carbon falling to litter pool, year
    foliar_fall_period = ffr_param[1]  # foliar falling period, day
    doy_fol = ffr_param[2]  # day of year with maximum foliar falling rate, unitless

    # forcing
    # time variant
    doy = ffr_forcing[0]  # day of year, unitless

    # calculating foliar falling rate, day-1
    ff = (jnp.log(rt_fol) - jnp.log(rt_fol - 1)) * 0.5
    wf = foliar_fall_period * jnp.sqrt(2) / 2
    osf = offset(rt_fol, wf)
    sf = 365.25 / jnp.pi
    ffr = (2 / jnp.sqrt(jnp.pi)) * (ff / wf) * jnp.exp(-(jnp.sin((doy - doy_fol + osf) / sf) * sf / wf) ** 2)

    return ffr


def tmp_prcp_factor(tpf_forcing, tpf_param):
    '''
    Function to calculate temperature and precipitation limitation factor on litter and soil turnover rate, unitless
     
    :param tpf_forcing: (5, ), 3 + 2 = 5 (time variant + time invariant)
    :param tpf_param: (2, )

    :return tpf: temperature and precipitation factor, unitless
    
    '''

    # parameters need to be constrained
    q10 = tpf_param[0]  # temperature sensitivity of litter and soil turnover rate, unitless
    prcp_sensitivity = tpf_param[1]  # precipitation sensitivity of litter and soil turnover rate, unitless

    # forcings
    # time variant
    ta_max = tpf_forcing[0]  # daily maximum air temperature, degC
    ta_min = tpf_forcing[1]  # daily minimum air temperature, degC
    prcp = tpf_forcing[2]  # daily precipitation, mm day-1
    # time invariant
    mean_ta = tpf_forcing[3]  # multi-year averaged daily air temperature, degC
    mean_prcp = tpf_forcing[4]  # multi-year averaged daily precipitation, mm day-1

    # calculating temperature (tf) and precipitation (pf) factor, unitless
    tf = jnp.exp(q10 * (0.5 * (ta_max + ta_min) - mean_ta))  # unitless
    pf = (prcp / mean_prcp - 1) * prcp_sensitivity + 1  # unitless
    tpf = tf * pf

    # jax.debug.breakpoint()

    return tpf
     

def soil_moisture_factor(smf_pool, smf_param):
	'''
	Function to calculate soil moisture limitation factor on gpp, unitless

	:param smf_pool: (1, )
	:param smf_param: (2, ) 
     
    :return smf: soil moisture limitation factor, unitless

	'''

	# parameters need to be constrained
	paw_fc = smf_param[0]  # field capacity for PAW pool, kg H2O m-2 or mm
	f_wp = smf_param[1]  # wilting point fraction as of field capacity, wilting point / filed capacity, unitless

	# pool
	paw_pool = smf_pool[0]  # plant available water pool, mm

	# calculating soil moisture limitation factor, unitless
	wp = paw_fc * f_wp  # wilting point, kg H2O m-2 or mm
	smf = (paw_pool - wp) / (paw_fc - wp)
	smf = jnp.clip(smf, 1e-6, 1.0)
	
	return smf

