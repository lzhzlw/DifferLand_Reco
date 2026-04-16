# Basic information about the DifferLand framework


from collections import namedtuple
import jax.numpy as jnp


# extrenal forcing and parameter numbers of each module
# internal forcings and parameters are excluded
# and those have been used before will not be double counted
# number of [time-variant forcing, time-invariant frocing, pools, parameters]
lrr_info = jnp.array([1, 0, 0, 3])  # labile releasing module
ffr_info = jnp.array([0, 0, 0, 3])  # foliar falling module
tpf_info = jnp.array([3, 2, 0, 2])  # temperatuer and precipitation factor module
smf_info = jnp.array([0, 0, 1, 2])  # soil moisture factor module
lai_info = jnp.array([0, 0, 1, 1])  # leaf area index module
wb_info = jnp.array([0, 1, 1, 3])  # water budget module
pa_info = jnp.array([0, 0, 4, 8])  # plant carbon allocation module
common_info = lrr_info + ffr_info + tpf_info + smf_info + lai_info + wb_info + pa_info  # [4, 3, 7, 22]

acm_info = jnp.array([2, 2, 0, 1])  # ACM module
wue_info = jnp.array([1, 0, 0, 2])  # WUE module
acm_wue_info = acm_info + wue_info  # [3, 2, 0, 3]

pml2_info = jnp.array([9, 1, 0, 11])  # PML2 module

op_info = jnp.array([0, 0, 1, 1])  # one-pool SOC module
cs_info = jnp.array([0, 0, 3, 5])  # Century simplified SOC module
mesc_info = jnp.array([3, 3, 9, 12])  # MES-C SOC module


# parameters
param_name = namedtuple('param_name', ['RT_lab', 'Lab_release_period', 'DOY_lab',  # 3, labile releasing module
                                       'RT_fol', 'Foliar_fall_period', 'DOY_fol',  # 3, foliar falling module
                                       'Q10', 'Prcp_sensitivity',  # 2, temperature and precipitation factor module
                                       'PAW_fc', 'f_wp',  # 2, soil moisture factor module
                                       'LMA',  # 1, LAI module
                                       'PAW_Qmax', 'PUW_Qmax', 'f_paw2puw',  # 3, water budget module
                                       'f_auto', 'f_fol', 'flab', 'f_root',  # 8, plant carbon allocation module
                                       'k_wood', 'k_root', 'k_litter', 'k_decomp',
                                       'PR_N',  # 1, ACM module
                                       'WUE', 'Boese_r',  # 2, WUE module
                                       'kA', 'Sl', 'F0', 'beta_qe', 'eta_ce', # 11, PML2 module
                                       'Vm_25', 'm_scc', 'kQ', 'Dmax', 'Dmin', 'D0',
                                       'k_soc',  # 1, one-pool SOC module
                                       'f_soc21', 'f_soc32', 'k_soc1', 'k_soc2', 'k_soc3',  # 5, Century simplified SOC module
                                       'fmet', 'kab', 'fm', 'av', 'ak', 'fs', 'tau_r',  # 12, MES-C SOC module
                                       'tau_k', 'ksorp', 'qmaxcoeff', 'kma', 'floss'])
param_min = jnp.array([1.001, 30.4375, 1.0,
                       1.001, 30.4375, 1.0,
                       0.018, 0.01,
                       1.0, 0.01,
                       10.0,
                       1.0, 1.0, 0.01, 
                       0.2, 0.01, 0.01, 0.01,
                       2.5e-7, 1.0e-4, 1.0e-4, 1.0e-5,
                       5.0,
                       0.5, 0.01,
                       0.61, 1.0e-4, 1.0e-4, 0.02, 0.0022,
                       0.96, 0.83, 1.0e-4, 2.3, 0.33, 0.06,
                       1.0e-7,
                       1.0e-7, 1.0e-7, 0.0027, 5.5e-5, 2.7e-6,
                       1.0e-6, 0.0019, 0.005, 8.0e-7, 1.0, 0.005, 5.2e-5,
                       2.4e-5, 0.6, 0.43, 0.002, 1.5e-4])
param_max = jnp.array([8.0, 100.0, 366.0, 
                       8.0, 150.0, 366.0,
                       0.08, 1.0,
                       1.0e4, 0.5,
                       400.0,
                       1.0e5, 1.0e5, 1.0,
                       0.8, 0.5, 0.5, 1.0, 
                       0.001, 0.01, 0.01, 0.01,
                       50.0,
                       30.0, 0.3,
                       0.97, 0.23, 0.21, 0.06, 0.09,  
                       56.3, 31.3, 1.0, 7.7, 1.82, 2.44,
                       0.001,
                       0.4, 0.1, 1.0, 0.0027, 1.4e-5,
                       0.85, 0.19, 0.25, 2.4e-4, 300.0, 0.25, 0.0052, 
                       0.0024, 60.0, 4.3, 0.2, 0.015])
param_unit = ['year', 'day', '',
              'year', 'day', '',
              '', '',
              'mm', '',
              'g C m$^{-2}$',
              'mm', 'mm', '',
              '', '', '', '',
              'day$^{-1}$', 'day$^{-1}$', 'day$^{-1}$', 'day$^{-1}$',
              'g C m$^{-2}$ day$^{-1}$',
              'g C kg H$_2$O$^{-1}$ hPa$^{-0.5}$', 'kg H$_2$O MJ$^{-1}$',
              '', 'mm day$^{-1}$', '', 'umol CO$_2$ (umol PAR)$^{-1}$', 'umol m$^{-2}$ s$^{-1}$ (umol m$^{-2}$ s$^{-1}$)$^{-1}$',
              'umol m$^{-2}$ s$^{-1}$', '', '', 'kPa', 'kPa', 'kPa',
              'day$^{-1}$', 
              '', '', 'day$^{-1}$', 'day$^{-1}$', 'day$^{-1}$',
              '', 'day-1', '', '', '', '', 'hour-1',
              'hour-1', '', '', 'day-1', 'day-1']
num_fields = len(param_name._fields)
param_info = param_name(*jnp.arange(num_fields))

# divide into different module
# common module params
st = 0
end = common_info[3]
common_param_min = param_min[st:end]
common_param_max = param_max[st:end]
common_param_unit = param_unit[st:end]
# ACM and WUE module params
st = end
end = st + acm_wue_info[3]
acm_wue_param_min = param_min[st:end]
acm_wue_param_max = param_max[st:end]
acm_wue_param_unit = param_unit[st:end]
# PML2 module params
st = end
end = st + pml2_info[3]
pml2_param_min = param_min[st:end]
pml2_param_max = param_max[st:end]
pml2_param_unit = param_unit[st:end]
# one-pool SOC module params
st = end
end = st + op_info[3]
op_param_min = param_min[st:end]
op_param_max = param_max[st:end]
op_param_unit = param_unit[st:end]
# Century simplified SOC module params
st = end
end = st + cs_info[3]
cs_param_min = param_min[st:end]
cs_param_max = param_max[st:end]
cs_param_unit = param_unit[st:end]
# MES-C SOC module params
st = end
end = st + mesc_info[3]
mesc_param_min = param_min[st:end]
mesc_param_max = param_max[st:end]
mesc_param_unit = param_unit[st:end]

# pools
pool_name = namedtuple('pool_name', ['PAW',  # 1, soil moisture factor module
                                     'Foliar',  # 1, LAI module
                                     'PUW',  # 1, water budget module
                                     'Labile', 'Wood', 'Root', 'Litter',  # 4, plant carbon allocation module
                                     'SOC',  # 1, one-pool SOC module
                                     'SOC1', 'SOC2', 'SOC3',  # 3, Century simplified SOC module
                                     'LITm', 'LITS', 'MICr', 'MICk', 'LMWC', 'MAOC',  # 9, MES-C SOC module
                                     'AGGm', 'AGGs', 'AGGmaoc'])
pool_min = jnp.array([1.0,
                      1.0,
                      1.0,
                      1.0, 1.0, 1.0, 1.0,
                      1.0,
                      0.009, 0.058, 0.503,
                      2.5e-3, 2.5e-3, 2.0e-3, 2.0e-3, 0.074, 0.35,
                      0.076, 0.076, 0.076])
pool_max = jnp.array([1.0e4,
                      2000.0,
                      1.0e4,
                      2000, 2000, 1.0e5, 2000,
                      2.0e5,
                      1.14e4, 9.52e4, 1.86e5,
                      6000.0, 6000.0, 2200.0, 2200.0, 5.86e4, 1.38e5,
                      2.26e4, 2.26e4, 2.26e4])
pool_unit = ['mm',
             'g C m$^{-2}$',
             'mm',
             'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$',
             'g C m$^{-2}$',
             'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$',
             'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$',
             'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$']

num_fields = len(pool_name._fields)
pool_info = pool_name(*jnp.arange(num_fields))

# divide into different module
# common module pools
st = 0
end = common_info[2]
common_pool_min = pool_min[st:end]
common_pool_max = pool_max[st:end]
common_pool_unit = pool_unit[st:end]
# acm and wue module pools
st = end
end = st + acm_wue_info[2]
acm_wue_pool_min = pool_min[st:end]
acm_wue_pool_max = pool_max[st:end]
acm_wue_pool_unit = pool_unit[st:end]
# pml2 module pools
st = end
end = st + pml2_info[2]
pml2_pool_min = pool_min[st:end]
pml2_pool_max = pool_max[st:end]
pml2_pool_unit = pool_unit[st:end]
# one-pool SOC module pools
st = end
end = st + op_info[2]
op_pool_min = pool_min[st:end]
op_pool_max = pool_max[st:end]
op_pool_unit = pool_unit[st:end]
# Century simplified SOC module pools
st = end
end = st + cs_info[2]
cs_pool_min = pool_min[st:end]
cs_pool_max = pool_max[st:end]
cs_pool_unit = pool_unit[st:end]
# MES-C SOC module pools
st = end
end = st + mesc_info[2]
mesc_pool_min = pool_min[st:end]
mesc_pool_max = pool_max[st:end]
mesc_pool_unit = pool_unit[st:end]

# time variant forcings
varf_name = namedtuple('varf_name', ['DOY',  # 1, labile releasing module
                                     'Ta_max', 'Ta_min', 'Prcp',  # 3, temperature and precipitation factor module
                                     'SSDR', 'CO2',  # 2, ACM module
                                     'VPD', # 1, WUE module
                                     'Ta_mean', 'Albedo', 'ALDR', 'Emiss', 'Pres', 'Wind',  # 9, PML2 module, including SSDR, CO2, VPD
                                     'SM', 'Ts', 'SMP']) # 3, MES-C module
varf_unit = ['',
             'degC', 'degC', 'mm day$^{-1}$',
             'W m$^{-2}$', 'ppm',
             'kPa',
             'degC', '', 'W m$^{-2}$', '', 'kPa', 'm s$^{-1}$',
             'm3 m-3', 'degC', 'kPa']

# time invariant forcings
invarf_name = namedtuple('invarf_name', ['ave_Ta_mean', 'ave_Prcp',  # 2, temperature and precipitation factor module
                                         'BD', # 1, water budget module
                                         'Canopy_height', 'Lat',  # 2, ACM module, CH also used in PML2
                                         'pH', 'Clay', 'Silt'])  # 3, MES-C SOC module
invarf_unit = ['degC', 'mm day$^{-1}$',
               'kg m$^{-3}$',
               'm', 'degN',
               '', '', '']


# targets
target_name = namedtuple('target_name', ['SM', 'Biomass', 'Above_biomass', 'Below_biomass', 
                                         'Litter', 'SOC', 'SOC1', 'SOC2', 
                                         'ET', 'Runoff', 'GPP', 'NPP', 
                                         'Reco', 'Ra', 'Rh', 'LAI'])
target_unit = ['m3 m$^{-3}$', 'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$',
               'g C m$^{-2}$', 'g C m$^{-2}$', 'g C m$^{-2}$',
               'mm day$^{-1}$', 'mm day$^{-1}$', 'g C m$^{-2}$ day$^{-1}$', 'g C m$^{-2}$ day$^{-1}$', 
               'g C m$^{-2}$ day$^{-1}$', 'g C m$^{-2}$ day$^{-1}$', 'g C m$^{-2}$ day$^{-1}$', 'm$^2$ m$^{-2}$']
num_fields = len(target_name._fields)
target_info = target_name(*jnp.arange(num_fields))

