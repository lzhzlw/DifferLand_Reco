# Function for GPP and ET module


import jax
import jax.numpy as jnp
from DifferLand.model.update_carbon import update_cpool_flux


def LAI(lai_pool, lai_param):
	'''
	Function to calculate leaf area index (LAI), m2 m-2
	
	:param lai_pool: (1, )
	:param lai_param: (1, )

	:return lai: leaf area index, m2 m-2

	'''

	# parameter need to be constrained
	lma = lai_param[0]  # leaf mass per area, g C m-2

	# pool
	foliar_pool = lai_pool[0]  # g C m-2

	# calculate LAI, m2 m-2
	lai= foliar_pool/lma

	return lai


def ACM(acm_forcing, acm_param):
	'''
	Function to calculate GPP based on ACM model
	With nitrogen, temperature, CO2 and light limitations but without soil moisture limitation
	
	:param acm_forcing: (8, ), 6 + 2 = 8 (time variant + time invariant)
    :param acm_param: (1, )

	:return gpp: g C m-2 day-1

	'''

	# constant parameters
	d1 = 0.0156935  # slope for day length correction factor, hour(s)-1
	d2 = 0.0453194  # intercept for day length correction factor, unitless
	c1 = 7.19298  # coefficient 1 used to calculate canopy-level quantum yield, g C MJ-1
	c2 = 2.1001  # coefficient 2 used to calculate canopy-level quantum yield, unitless
	phi_d = -2  # leaf to soil water potential difference, MPa
	eb1T = 0.789798  # exponent of phi_d, unitless
	b2 = 0.37836  # coefficient for canopy height, unitless
	a2 = 0.011136  # coefficient for maximum Ta, degC-1
	theta = 4.22273  # coeficient 1 used for internal CO2 concentration, unitless
	k = 208.868  # coeficient 2 used for internal CO2 concentration, unitless
	
	# parameters need to be constrained
	# PR_N = a1 * N
	# where N refers to the foliar nitogen concentration, g C m-2
	# a1 refers to a scalar converting nitrogen concentration to photosynthetic rate, day-1
	pr_n = acm_param[0]  # nitrogen-limited canopy photosynthetic rate, g C m-2 day-1

	# forcings
	# time variant
	doy = acm_forcing[0]  # day of year, 1-366
	ta_max = acm_forcing[1]  # daily maximum air temperature, degC
	ta_min = acm_forcing[2]  # daily minimum air temperature, degC
	lai = acm_forcing[3]  # leaf area index, m2 m-2
	Rs = acm_forcing[4] * 0.0864  # solar downward shortwave radiation, from W m-2 to MJ m-2 day-1
	Ca = acm_forcing[5]  # atmospheric CO2 concentration, umol mol-1 or ppm
	# time invariant
	ch = acm_forcing[6]  # canopy height, m
	lat = acm_forcing[7]  # latitude, degN
	
	# calculation
	# ==================================================================================================================
    # daily canopy conductance, g C m-2 day-1
	gc = jnp.abs(phi_d) ** eb1T / (b2 * ch + 0.5 * (ta_max - ta_min))
    # internal CO2 concentration, ppm
	p = lai * pr_n * jnp.exp(a2 * ta_max) / gc
	q = theta - k
	ci = 0.5 * (Ca + q - p + jnp.sqrt((Ca + q - p) ** 2 - 4 * (Ca * q - p * theta)))
	# CO2-limited of photosynthetic rate, g C m-2 day-1
	pd = gc * (Ca - ci)
	# canopy-level quantum yield, a function of LAI, g C MJ-1
	# measure how many carbon will be fixed (g C) by per unit energy photon (MJ)
	# unit energy generally has 1/E0 photon
	# E0 refers to the averaged energy of each photon
	e0 = c1 * (lai ** 2) / (c2 + lai ** 2)
    # light & CO2 limited photosynthetic rate, g C m-2 day-1
	# e0 * rs refers to the light limited photosynthetic rate, g C m-2 day-1
	pi = e0 * Rs * pd / (e0 * Rs + pd)
	# day length correction factor, hour(s)
	dec = -23.4 * jnp.cos((360 * (doy + 10) / 365) * jnp.pi / 180) * jnp.pi / 180
	mult = jnp.tan(lat * jnp.pi / 180) * jnp.tan(dec)
	mult_valid = (mult < 1) * (mult > -1)
	mult_temp = mult * mult_valid
	dayl = 24 * jnp.arccos(-mult_temp) / jnp.pi
	mult_geq_one_sel = (mult < 1)
	dayl = dayl * mult_geq_one_sel + (1 - mult_geq_one_sel) * 24
	mult_leq_minus_one_sel = (mult > -1)
	dayl = dayl * mult_leq_minus_one_sel + (1 - mult_geq_one_sel) * 0
    # gpp, g C m-2 day-1
	gpp = pi * (d1 * dayl + d2)
	
	return gpp


def WUE(wue_forcing, wue_param):
	'''
	Function to calculate ET based on WUE model

	:param wue_forcing: (3, ), 3 + 0 = 3 (time variant + time invariant)
	:param wue_param: (2, )

	:return et: mm day-1

	'''

	# parameters need to be constrained
	wue = wue_param[0]  # water use efficiency, g C H2O-1 hPa-0.5
	boese_r = wue_param[1]  # SSDR adjustment factor to ET, kg H2O MJ-1
	
	# wue forcings
	# time variant
	Rs = wue_forcing[0] * 0.0864  # solar downward shortwave radiation, from W m-2 to MJ m-2 day-1
	gpp = wue_forcing[1]  # with soil moisture limitation, g C m-2 day-1
	vpd = wue_forcing[2]  # hpa

	# calculation
	# ==================================================================================================================
	et = gpp * jnp.sqrt(vpd) / wue + Rs * boese_r   # mm day-1

	# jax.debug.breakpoint()

	return et


def vapor_pressure(tmp):
	'''
	Function to calculate saturate vapor pressure, kPa

	:param tmp: air temperature, degC

	:return svp: saturate vapor pressure, kPa
	
	'''

	# Note the original code is 0.6108 * jnp.exp(17.27 * tmp / (tmp + 237.3))
	# Here we change the function to align with that we used to calculate the VPD before
	svp = 0.61121 * jnp.exp(17.502 * tmp / (tmp + 240.97))

	return svp


def PML2(PML_pool, PML_forcing, PML_param):
	'''Function to calculate GPP (without soil moisture limitation) and ET based on PML2 model
	Zhang, Y. et al., 2019. Coupled estimation of 500 m and 8-day resolution global evapotranspiration and 
	gross primaryproduction in 2002-C2017. Remote Sens. Environ., 222: 165-182. https://doi.org/10.1016/j.rse.2018.12.031
	https://github.com/gee-hydro/gee_PML/blob/stable/src/PML_v2_v0.1.5.js

	:param PML_pool: (1, )
	:param PML_forcing: (12, ), 11 + 1 = 12 (time variant + time invariant)
	:param PML_param: (11, )

	:return gpp: g C m-2 day-1
	:return et: mm day-1

	'''
	# Constant parameters
	Stefan = 5.67e-8  # SteFan-Boltzmann constant 5.67e-8, W m-2 K-4
	Cp = 1.0164  # 4.2 * 0.242, specific heat at constant pressure, 1.013, J g-1 degC-1
	mr = 0.622	#  moisture ratio, unitless
	lai_ref = 5  # reference LAI, m2 m-2
	coe_a = 0.031  # temperature coefficient a, used to convet Vm_25 to any temperature, unitless
	coe_b = 0.115  # temperature coefficient b, used to convet Vm_25 to any temperature
	kmar = 0.41  # von Karman's constant, unitless
	zm = 45  # height, making sure higher than hc (canopy height), m

	# forcings
	# time variant
	lai = PML_forcing[0]  # leaf area index, m2 m-2
	prcp = PML_forcing[1]  # daily precipitation, mm day-1
	Rs = PML_forcing[2]  # solar shortwave radiation, W m-2
	Ca = PML_forcing[3]  # surface air CO2 concentration, ppm
	vpd = PML_forcing[4] / 10  # water vapour deficit, from hpa to kpa
	ta_avg = PML_forcing[5]  # daily averaged air temperature, degC
	albedo = PML_forcing[6]  # 0-1, unitless
	Rli = PML_forcing[7]  # downward longwave radiation, W m-2
	emiss = PML_forcing[8]  # surface emissivity, 0-1, unitless
	Pres = PML_forcing[9]  # air pressure, kPa
	u2 = PML_forcing[10]  # wind speed at 2 m, m s-1
	# time invariant
	ch = PML_forcing[11]  # canopy height, m

	# pool
	paw_pool = PML_pool[0]  # plant available water pool, mm

	# PML-v2 parameters need to be constrained
	kA = PML_param[0]  # extinction coefficient of available energy, unitless
	# specific canopy rainfall storage capacity per unit leaf area, mm day-1
	Sl = PML_param[1]
	# specific ratio of evaporation rate over rainfall intensity per unit vegetation cover, unitless
	f0 = PML_param[2]
	beta_qe = PML_param[3]  # quantum efficiency, umol CO2 (umol PAR)-1
	eta_ce = PML_param[4]  # carbonxylation efficiency, umol m-2 s-1 (umol m-2 s-1)-1
	Vm_25 = PML_param[5]  # maximum catalytic capacity of Rubisco per unit leaf area at 25 degC, umol m-2 s-1
	m_scc = PML_param[6]  # stomatal conductance (Ball-Berry) coefficient, unitless
	kQ = PML_param[7]  # extinction coefficient of PAR, unitless
	Dmax = PML_param[8]  # above which there is no assimilation, kPa
	Dmin = PML_param[9]  # below which there is no VPD constraint, kPa
	d0 = PML_param[10]  # sensitivity of stomatal conductance to leaf surface VPD, kPa

	# Es: Soil Evaporation [mm day-1]
	# ==================================================================================================================
	svp = vapor_pressure(ta_avg)	# Saturation water vapour pressure [kPa]
	# Slope of the curve relating saturation water vapor pressure to temperature [kPa degC-1]
	slope = 4098 * svp / jnp.power(ta_avg + 237.2, 2)
	# Latent heat of vaporization, 2500 [J g-1] at 25 degC
	lamada = 2500 - 2.2 * ta_avg
	gamma = Cp * Pres / (mr * lamada)  # Psychrometric constant [kPa degC-1]
	epsilon = slope / gamma

	# Refer to Wang et al., 2013, Atpspheric Research, https://doi.org/10.1016/j.atmosres.2022.106475
	Rlo = emiss * Stefan * jnp.power(ta_avg + 273.15, 4)  # Upward longwave radiation [W m-2]
	Rn = (1 - albedo) * Rs + Rli - Rlo	# Net radiation [W m-2]
	Rn = jnp.maximum(Rn, 1e-6)
	Tau = jnp.exp(-kA * lai)  # Ratio of available energy of surface
	As = Rn * Tau
	Es_eq = epsilon * As / (epsilon + 1)  # Soil evaporation at equilibrium [MJ]
	coef_MJ2mm = lamada / 86.4  # Unit conversion coefficient from MJ to mm day-1
	Es_eq = jnp.maximum(Es_eq / coef_MJ2mm, 1e-6)	# Soil evaporation at equilibrium [mm day-1]

	# Valid fraction of Es, we change the multi-period averaged precipitation to plant available pool here, unitless
	fval_Es = paw_pool / Es_eq
	fval_Es = jnp.clip(fval_Es, 1e-6, 1)

	Es = fval_Es * Es_eq

	# Ei: Interception Evaporation by Canopy Vegetation [mm day-1]
	# ==================================================================================================================
	Sveg = Sl * lai  # canopy vegetation storage capacity [mm day-1]
	fveg = jnp.maximum(1 - jnp.exp(-lai / lai_ref), 1e-4)  # Fraction of vegetation coverage
	# Ratio of the average evaporation rate over the average rainfall intensity during storms
	fER = jnp.maximum(fveg * f0, 1e-4)
	# Reference threshold precipitation rate when the canopy is wet [mm day-1]
	prcp_wet = - jnp.log(1 - f0) * Sveg / fER

	Ei_dry = fveg * prcp
	Ei_wet = fveg * prcp_wet + fER * (prcp - prcp_wet)
	Ei = jnp.where(prcp < prcp_wet, Ei_dry, Ei_wet)	# [mm day-1]

	# Et: Plant/Canopy transpiration [mm day-1] and GPP [gC m-2 day-1]
	# ==================================================================================================================
	# Gc: canopy conductance [m s-1]
	par = jnp.maximum(Rs * 0.45, 1e-6)  # Photosynthetically active radiation [W m-2]
	par_mol = par * 4.57  # From [W m-2] to [umol m-2 s-1]
	Am = Vm_25	# Maximum photosynthetic rate [umol m-2 s-1]
	fP4 = jnp.exp(coe_a * (ta_avg - 25)) / (1 + (jnp.exp(coe_b * (ta_avg - 41))))	# Vm_25/Vm
	fP4 = jnp.minimum(fP4, 1.0)
	p1 = Am * beta_qe * eta_ce * par_mol
	p2 = Am * beta_qe * par_mol
	p3 = Am * eta_ce * Ca
	p4 = beta_qe * eta_ce * par_mol * Ca * fP4
	inter_var1 = p1 / (kQ * (p2 + p4))
	inter_var2 = kQ * lai + jnp.log((p2 + p3 + p4) / (p2 + p3 * jnp.exp(kQ * lai) + p4))
	fvpd_gc = 1 / (1 + vpd / d0)
	# 1.6, conductance of water / conductance of CO2 [mol m-2 s-1]
	Gc = m_scc * inter_var1 * inter_var2 * fvpd_gc * 1.6
	# Convert from [mol m-2 s-1] to [m s-1]
	Gc = jnp.maximum(Gc * 1e-2 / (0.446 * (273 / (273 + ta_avg)) * (Pres / 101.3)), 1e-6)

	# Ga:aerodynamic conductance [m s-1]
	uz = jnp.log(67.8 * zm - 5.42) / 4.87 * u2	# Convert 2 m wind speed to height of zm
	d = ch * 0.67
	zom = ch * 0.13
	zov = zom * 0.1
	Ga = uz * jnp.power(kmar, 2) / (jnp.log((zm - d) / zom) * jnp.log((zm - d) / zov))

	# Transpiration from plant cause by radiation water transfer
	Ac = Rn * (1 - Tau)
	Et_rad = jnp.maximum(epsilon * Ac / (epsilon + 1 + Ga / Gc), 1e-6)
	rou_air = 3846 * Pres / (ta_avg + 273.15)  # Density of air [g m-3]
	# Transpiration from plant cause by aerodynamic water transfer
	Et_aero = jnp.maximum((rou_air * Cp * Ga * vpd / gamma) / (epsilon + 1 + Ga / Gc), 1e-6)
	Et = Et_rad + Et_aero
	Et = Et / coef_MJ2mm

	# ET: total evapotranspiration [mm day-1]
	et = Es + Ei + Et

	# GPP [gC m-2 day-1]
	# ==================================================================================================================
	# Potential GPP [umol cm-2 s-1]
	GPP_p = Ca * inter_var1 * inter_var2
	# GPP under VPD constraint
	fvpd = (Dmax - vpd) / (Dmax - Dmin)
	fvpd = jnp.clip(1e-6, 1.0)
	gpp = GPP_p * fvpd  # [umol cm-2 s-1]
	gpp = gpp * 1.0368  # 86400 / 1e6 * 12 [gC m-2 day-1]

	return gpp, et


def water_budget(wb_pool, wb_forcing, wb_param, wb_pool_min, wb_pool_max):
	'''
	Function to update plant available water pools and fluxes
	
	:param wb_pool: currrent water pools, mm (2, )
	:param wue_forcing: (4, ), 2 + 3 = 5 (time variant + time invariant)
	:param wue_param: (3, )
	:param wb_pool_min: prior minimum pool, mm (2, )
	:param wb_pool_max: prior maximum pool, mm (2, )

	:return next_wb_pool: next water pool, mm (2, )
	:return et: mm day-1
	:return runoff: mm day-1

	'''

	# parameters need to be constrained
	PAW_Qmax = wb_param[0]  # threshold for paw's eflux, kg H2O m-2 or mm
	PUW_Qmax = wb_param[1]  # threshold for puw's eflux, kg H2O m-2 or mm
	fpaw2puw = wb_param[2]  # fraction of paw's eflux to puw, unitless

	# forcings
	# time variant
	prcp = wb_forcing[0]  # mm day-1
	et = wb_forcing[1]  # mm day-1
	# time invariant
	bulkd = wb_forcing[2]  # bulk density, kg m-3
	delta_t = wb_forcing[3]  # time step, day(s)
	soil_thick = wb_forcing[4]  # soil thickness, m

	# pools and prior range, mm
	paw_pool = wb_pool[0]  # plant available water pool
	puw_pool = wb_pool[1]  # plant usable water pool
	paw_min = wb_pool_min[0]
	puw_min = wb_pool_min[1]
	paw_max = wb_pool_max[0]
	puw_max = wb_pool_max[1]

	# calculation
	# ==================================================================================================================
	# water budget
	# current soil moisture
	soil_moist = paw_pool + puw_pool # mm
	soil_moist = soil_moist / (bulkd * soil_thick) # m3 m-3

	# paw eflux, mm day-1
	paw_eflux = paw_pool ** 2 / PAW_Qmax / delta_t
	paw_eflux_focal_sel = paw_eflux <= PAW_Qmax / 2
	paw_eflux = jnp.where(paw_eflux_focal_sel, paw_eflux, (paw_pool - PAW_Qmax / 4) / delta_t)
	paw_eflux = jnp.maximum(paw_eflux, 1e-6)
	# paw to puw, mm day-1
	paw2puw = paw_eflux * fpaw2puw
	# paw to runoff, mm day-1
	paw2r = paw_eflux * (1 - fpaw2puw)

	# puw eflux, mm day-1
	puw_eflux = puw_pool ** 2 / PUW_Qmax / delta_t
	puw_eflux_focal_sel = puw_eflux <= PUW_Qmax / 2
	puw_eflux = jnp.where(puw_eflux_focal_sel, puw_eflux, (puw_pool - PUW_Qmax / 4) / delta_t)
	puw_eflux = jnp.maximum(puw_eflux, 1e-6)
	
	next_paw_pool, paw_eflux = update_cpool_flux(paw_pool, paw_min, paw_max, prcp, paw_eflux, delta_t, et)
	paw_eflux = jnp.maximum(paw_eflux, 1e-6)
	et = prcp - paw_eflux - (next_paw_pool - paw_pool) / delta_t

	paw2puw = paw_eflux * fpaw2puw
	paw2r = paw_eflux * (1 - fpaw2puw)
	next_puw_pool, puw_eflux = update_cpool_flux(puw_pool, puw_min, puw_max, paw2puw, puw_eflux, delta_t)

	next_wb_pool = jnp.array([next_paw_pool, next_puw_pool])  # mm
	runoff = paw2r + puw_eflux  # total runoff, mm day-1

	# jax.debug.breakpoint()

	return next_wb_pool, et, runoff, soil_moist

