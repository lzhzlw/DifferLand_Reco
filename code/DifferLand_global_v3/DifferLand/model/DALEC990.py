# Function for the DALEC model


import jax
import jax.numpy as jnp
from DifferLand.model.DALEC import DALEC
from functools import partial
import sys
from DifferLand.model.environment_factor import lab_release_rate, foliar_fall_rate, tmp_prcp_factor, soil_moisture_factor
from DifferLand.model.DALEC990_info import (common_info, acm_info, wue_info, acm_wue_info, pml2_info, 
                                            op_info, cs_info, mesc_info, 
                                            common_param_min, common_param_max, common_pool_min, common_pool_max,
                                            acm_wue_param_min, acm_wue_param_max, acm_wue_pool_min, acm_wue_pool_max,
                                            pml2_param_min, pml2_param_max, pml2_pool_min, pml2_pool_max,
                                            op_param_min, op_param_max, op_pool_min, op_pool_max,
                                            cs_param_min, cs_param_max, cs_pool_min, cs_pool_max,
                                            mesc_param_min, mesc_param_max, mesc_pool_min, mesc_pool_max)
from DifferLand.model.gpp_et import LAI, ACM, WUE, PML2, water_budget
from DifferLand.model.allocation import plant_allocation
from DifferLand.model.soc import one_pool, Century_simplified, MES_C
from DifferLand.util.normalization import unnormalize


class DALEC990(DALEC):
    def __init__(self, ge_type='pml2', soc_type='mesc', time_step=30, soil_depth=200):
        super().__init__()
        self.ge_type = ge_type  # type of GPP and ET coupling module
        self.soc_type = soc_type  # type of SOC module
        self.time_step = time_step # time step in days
        self.soil_depth = soil_depth / 100  # soil depth in m
        # extrenal forcing and parameter numbers of each module, (4, )
        # number of [time-variant forcing, time-invariant frocing, pools, parameters]
        self.common_info = common_info.tolist()
        self.acm_info = acm_info.tolist()
        self.wue_info = wue_info.tolist()
        self.acm_wue_info = acm_wue_info.tolist()
        self.pml2_info = pml2_info.tolist()
        self.op_info = op_info.tolist()
        self.cs_info = cs_info.tolist()
        self.mesc_info = mesc_info.tolist()
        # prior range of pools and parameters, further combine these 
        if ge_type == 'acm_wue':
            self.ge_param_min = acm_wue_param_min
            self.ge_param_max = acm_wue_param_max
            self.ge_pool_min = acm_wue_pool_min
            self.ge_pool_max = acm_wue_pool_max
        elif ge_type == 'pml2':
            self.ge_param_min = pml2_param_min
            self.ge_param_max = pml2_param_max
            self.ge_pool_min = pml2_pool_min
            self.ge_pool_max = pml2_pool_max
        else:
            sys.exit('Error: wrong type of GPP and ET coupling module')
        if soc_type == 'onepool':
            self.soc_param_min = op_param_min
            self.soc_param_max = op_param_max
            self.soc_pool_min = op_pool_min
            self.soc_pool_max = op_pool_max
        elif soc_type == 'century':
            self.soc_param_min = cs_param_min
            self.soc_param_max = cs_param_max
            self.soc_pool_min = cs_pool_min
            self.soc_pool_max = cs_pool_max
        elif soc_type == 'mesc':
            self.soc_param_min = mesc_param_min
            self.soc_param_max = mesc_param_max
            self.soc_pool_min = mesc_pool_min
            self.soc_pool_max = mesc_pool_max
        else:
            sys.exit('Error: wrong type of SOC module')
        self.param_min = jnp.concatenate((common_param_min, self.ge_param_min, self.soc_param_min))
        self.param_max = jnp.concatenate((common_param_max, self.ge_param_max, self.soc_param_max))
        self.pool_min = jnp.concatenate((common_pool_min, self.ge_pool_min, self.soc_pool_min))
        # Adjust the water pool maximum (ref, 10 m depth) according to the soil depth
        wp_idx = jnp.array([0, 2])
        update_common_pool_max = common_pool_max.at[wp_idx].set(common_pool_max[wp_idx] * self.soil_depth / 10) # PAW and PUW pool
        # Adjust the SOC pool maximum (ref, 30 cm depth) according to the soil depth
        beta = 0.9786
        update_soc_pool_max = self.soc_pool_max * (1 - jnp.power(beta, self.soil_depth * 100))/(1 - jnp.power(beta, 30))
        self.pool_max = jnp.concatenate((update_common_pool_max, self.ge_pool_max, update_soc_pool_max))


    @partial(jax.jit, static_argnums=(0,))
    def step(self, pools, var_forcings, invar_forcings, params):
        '''
        Function for the DALEC model at specific grrid and time

        :param pools: (n_pool, )
        :param var_forcings: (n_var, )
        :param invar_forcings: (n_invar, )
        :param params: (n_param, )

        :return new_pools: (n_pool, )
        :return targets: (n_tar, )

        '''
        
        # environment factor module
        # ========================================================================================================
        # labile releasing rate, day-1 (for plant C allocation module)
        lrr_forcing = var_forcings[0:1]  # 1, doy
        lrr_param = params[:3]  # 3， rt_lab, lab_release_period, doy_lab
        lrr = lab_release_rate(lrr_forcing, lrr_param)

        # foliar falling rate, day-1 (for plant C allocation module)
        ffr_forcing = lrr_forcing  # same as lrr_forcing, doy
        ffr_param = params[3:6]  # 3, rt_fol, leaf_fall_period, doy_fol
        ffr = foliar_fall_rate(ffr_forcing, ffr_param)

        # temperature and preparation lmitation factor on litter and soil turnover rate, unitless
        tpf_vf = var_forcings[1:4]  # 3, ta_max, ta_min, prcp
        tpv_ivf = invar_forcings[:2]  #  2, mean_ta, mean_prcp
        tpf_forcing = jnp.concatenate((tpf_vf, tpv_ivf))
        tpf_param = params[6:8]  # 2, q10, prcp_sensitivity
        tpf = tmp_prcp_factor(tpf_forcing, tpf_param)

        # soil moisure limitation factor on GPP, unitless
        smf_pool = pools[0:1]  # 1, paw_pool
        smf_param = params[8:10]  # 2, paw_fc, f_wp
        smf = soil_moisture_factor(smf_pool, smf_param)

        # GPP and ET coupling module
        # ========================================================================================================
        # LAI module, used for calculating GPP
        lai_pool = pools[1:2]  # 1, foliar_pool
        lai_param = params[10:11]  # 1, LMA
        lai = LAI(lai_pool, lai_param)
        
        if self.ge_type == 'acm_wue':
            # ACM module
            acm_vf = var_forcings[self.common_info[0]:self.common_info[0]+self.acm_info[0]]  # 2, Rs, CO2
            acm_ivf = invar_forcings[self.common_info[1]:self.common_info[1]+self.acm_info[1]]  # 2, canopy height, latitude
            acm_forcing = jnp.concatenate((lrr_forcing, tpf_forcing[0:2], jnp.array([lai]), acm_vf, acm_ivf))  # add doy, ta__max, ta_min, lai
            acm_param = params[self.common_info[3]:self.common_info[3]+self.acm_info[3]]  # 1, pr_n
            gpp = ACM(acm_forcing, acm_param)
            gpp = gpp * smf
            gpp = jnp.maximum(gpp, 1e-12)

            # WUE model
            st_vf = self.common_info[0]+self.acm_info[0]
            wue_vf = var_forcings[st_vf:st_vf+self.wue_info[0]]  # 1, VPD
            wue_forcing = jnp.concatenate((acm_vf[0:1], jnp.array([gpp]), wue_vf)) # add Ra, gpp
            st_param = self.common_info[3]+self.acm_info[3]
            wue_param = params[st_param:st_param+self.wue_info[3]]  # 2, wue, boese_r
            et = WUE(wue_forcing, wue_param)

            st_vf = st_vf + self.wue_info[0]
            st_ivf = self.common_info[1] + self.acm_info[1]
            st_pool = self.common_info[2]
            st_param = st_param + self.wue_info[3]
        
        elif self.ge_type == 'pml2':
            # PML2 moedule
            # 9, Rs, CO2, VPD, ta_avg, albedo, Rl_downward, emiss, pres, wind
            pml_vf = var_forcings[self.common_info[0]:self.common_info[0]+self.pml2_info[0]]
            pml_ivf = invar_forcings[self.common_info[1]:self.common_info[1]+self.pml2_info[1]]  # 1, canopy height
            pml_forcing = jnp.concatenate((jnp.array([lai]), tpf_forcing[2:3], pml_vf, pml_ivf))  # add lai, prcp
            pml_pool = smf_pool  # 1, paw_pool
            # 11, ka, sl, f0, beta_qe, eta_ce, vm_25, m_scc, kq, dmax, dmin, d0
            pml_param = params[self.common_info[3]:self.common_info[3]+self.pml2_info[3]]
            gpp, et = PML2(pml_pool, pml_forcing, pml_param)
            gpp = gpp * smf
            gpp = jnp.maximum(gpp, 1e-12)

            st_vf = self.common_info[0]+self.pml2_info[0]
            st_ivf = self.common_info[1]+self.pml2_info[1]
            st_pool = self.common_info[2]
            st_param = self.common_info[3]+self.pml2_info[3]

        else:
            sys.exit('Error: wrong type of GPP and ET coupling module')
        
        # water budget module
        # ========================================================================================================
        wb_ivf = invar_forcings[2:3]  # bulk density
        wb_forcing = jnp.concatenate((tpf_forcing[2:3], jnp.array([et]), wb_ivf, jnp.array([self.time_step, self.soil_depth])))  # add prcp, et, time step, soil depth
        wb_pool_new = pools[2:3]  # 1, puw_pool
        wb_pool = jnp.concatenate((smf_pool, wb_pool_new))  # add paw_pool
        wb_param = params[11:14]  # 3, paw_qmax, puw_qmax, fpaw2puw
        wb_pool_min = jnp.concatenate((self.pool_min[0:1], self.pool_min[2:3]))
        wb_pool_max = jnp.concatenate((self.pool_max[0:1], self.pool_max[2:3]))
        # next_wb_pool = [next_paw_pool, next_puw_pool]
        next_wb_pool, et, runoff, soil_moist = water_budget(wb_pool, wb_forcing, wb_param, wb_pool_min, wb_pool_max)

        # plant allocation module
        # =======================================================================================================
        pa_forcing = jnp.concatenate((jnp.array([gpp, ffr, lrr, tpf]), jnp.array([self.time_step]))) # 5, gpp, ffr, lrr, tpf, time step
        pa_pool_new = pools[3:7]  # 4, labile_pool, wood_pool, root_pool, litter_pool
        pa_pool = jnp.concatenate((lai_pool, pa_pool_new))  # add foliar_pool
        # 8, f_auto, f_fol, f_lab, f_root, k_wood, k_root, k_litter, k_decomp
        pa_param = params[14:22]
        pa_pool_min = jnp.concatenate((self.pool_min[1:2],self.pool_min[3:7]))
        pa_pool_max = jnp.concatenate((self.pool_max[1:2],self.pool_max[3:7]))
        # next_aa_pool = [next_foliar_pool, next_labile_pool, next_wood_pool, next_root_pool, next_litter_pool]
        next_pa_pool, ra, npp, rh_litter, litter2soc = plant_allocation(pa_pool, pa_forcing, pa_param, pa_pool_min, pa_pool_max)
        above_biomass = jnp.sum(pa_pool[:3], axis=0)  # total aboveground biomass, g C m-2
        root_biomass = pa_pool[3]  # root biomass, g C m-2
        litter_pool = pa_pool[4]  # litter pool, g C m-2

        # soc module
        # ========================================================================================================
        # one-pool SOC module
        if self.soc_type == 'onepool':
            soc_forcing = jnp.concatenate((jnp.array([litter2soc, tpf]), jnp.array([self.time_step])))  # 3, litter2soc, tpf, time step
            soc_pool = pools[st_pool:st_pool+self.op_info[2]]  # 1, soc_pool
            soc_param = params[st_param:st_param+self.op_info[3]]  # 1, k_soc
            soc_pool_min = self.pool_min[st_pool:st_pool+self.op_info[2]]
            soc_pool_max = self.pool_max[st_pool:st_pool+self.op_info[2]]
            # soc1-3 are zero here for placeholder
            next_soc_pool, rh_soil = one_pool(
                soc_pool, soc_forcing, soc_param, soc_pool_min, soc_pool_max
            )
            soc_pool = soc_pool[0]
            next_soc_pool = jnp.array([next_soc_pool])
            soc1_pool, soc2_pool, soc3_pool, mic_biomass = 0.0, 0.0, 0.0, 0.0  # placeholder

        # simplified Century SOC module
        elif self.soc_type == 'century':
            soc_forcing = jnp.concatenate((jnp.array([litter2soc, tpf]), jnp.array([self.time_step])))  # 3, litter2soc, tpf, time step
            soc_pool = pools[st_pool:st_pool+self.cs_info[2]]  # 3, soc1_pool, soc2_pool, soc3_pool
            # 5, f_soc21, f_soc32, k_soc1, k_soc2, k_soc3
            soc_param = params[st_param:st_param+self.cs_info[3]]
            soc_pool_min = self.pool_min[st_pool:st_pool+self.cs_info[2]]
            soc_pool_max = self.pool_max[st_pool:st_pool+self.cs_info[2]]
            next_soc_pool, soc_pool, soc1_pool, soc2_pool, soc3_pool, rh_soil = Century_simplified(
                soc_pool, soc_forcing, soc_param, soc_pool_min, soc_pool_max
            )
            mic_biomass = 0.0  # placeholder

        # MES-C soc module
        elif self.soc_type == 'mesc':
            # 3, Soil mositure, soil temperature, soil matric potential
            soc_vf = var_forcings[st_vf:st_vf+self.mesc_info[0]]
            # 3, soil pH, soil clay, soil silt
            soc_ivf = invar_forcings[st_ivf:st_ivf+self.mesc_info[1]]
            # add litter2soc, npp, bulk density, time step, soil depth
            soc_forcing = jnp.concatenate((jnp.array([litter2soc, npp]), soc_vf, wb_ivf, soc_ivf, jnp.array([self.time_step, self.soil_depth])))
            # 9, litm, lits, micr, mick, lmwc, maoc, aggm, aggs, aggmaoc
            soc_pool = pools[st_pool:st_pool+self.mesc_info[2]]
            # 11, fmet, xkab, xfm, xav, xak, xfs, xtau, xksorp, xqmaxcoeff, xkma, xfloss
            soc_param = params[st_param:st_param+self.mesc_info[3]]
            soc_pool_min = self.pool_min[st_pool:st_pool+self.mesc_info[2]]
            soc_pool_max = self.pool_max[st_pool:st_pool+self.mesc_info[2]]
            next_soc_pool, soc_pool, soc1_pool, soc2_pool, soc3_pool, rh_soil = MES_C(
                soc_pool, soc_forcing, soc_param, soc_pool_min, soc_pool_max
            )
            mic_biomass = soc3_pool
            
        else:
            sys.exit('Error: wrong type of SOC module')

        rh = rh_litter + rh_soil
        reco = ra + rh
        
        below_biomass = root_biomass + mic_biomass
        biomass = above_biomass + below_biomass

        # next pools
        next_pools = jnp.concatenate((
            next_wb_pool[0:1], next_pa_pool[0:1], next_wb_pool[1:2], next_pa_pool[1:], next_soc_pool
        ))
        
        # target pool/flux
        targets = jnp.array([soil_moist, biomass, above_biomass, below_biomass, 
                             litter_pool, soc_pool, soc1_pool, soc2_pool, 
                             et, runoff, gpp, npp, 
                             reco, ra, rh, lai, rh_litter, rh_soil])
        
        # jax.debug.breakpoint()

        return next_pools, targets
    

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_params(self, norm_param):
        '''Convert model parameters from the real space to the physcial range'''

        return unnormalize(norm_param, parmin = self.param_min, parmax = self.param_max)


    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_pools(self, norm_pool):
        '''Convert pool values from the real space to the physical value range'''

        return unnormalize(norm_pool, parmin = self.pool_min, parmax = self.pool_max)

 
    @partial(jax.jit, static_argnums=(0,))
    def step_tem_batch(self, pools, var_forcings, invar_forcings, params):
        '''
        Function for the DALEC model batch for all time
        
        :param pools: (n_pool, )
        :param var_forcings: (n_time, n_var)
        :param invar_forcings: (n_invar, )
        :param params: (n_param, )

        :return targets: (n_time, n_tar)

        '''

        params = self.unnormalize_params(params)
        pools = self.unnormalize_pools(pools)
        next_pools, targets = jax.lax.scan(
            partial(self.step, invar_forcings=invar_forcings, params=params), pools, var_forcings
        )

        return targets.squeeze()


    @partial(jax.jit, static_argnums=(0,))
    def step_spa_batch(self, pools, var_forcings, invar_forcings, params):
        '''
        Function for the DALEC model batch for all grids
        
        :param pools: (n_grid,n_pool)
        :param var_forcings: (n_grid, n_time, n_var)
        :param invar_forcings: (n_grid, n_invar)
        :param params: (n_grid, n_param)

        :return targets: (n_grid, n_time, n_tar)

        '''
        
        batched_step = jax.vmap(self.step_tem_batch, in_axes=([0, 0, 0, 0]))
        targets = batched_step(pools, var_forcings, invar_forcings, params)

        return targets
    

    @partial(jax.jit, static_argnums=(0,))
    def compute_tempo_loss(self, pools, var_forcings, invar_forcings, params, var_obs, invar_obs, grid_weights,
                           obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind):
        '''
        Function to calculate the loss at specific grid

        :param pools: (n_pool, )
        :param var_forcings: (n_time, n_var)
        :param invar_forcings: (n_invar, )
        :param params: (n_param, )
        :param var_obs: (n_time, n_var)
        :param invar_obs: (n_invar, )
        :param grid_weights: (n_invar, )
        :param obs_mean: (n_var+n_invar, )
        :param obs_std: (n_var+n_invar, )
        :param var_obs_ind: (n_var, )
        :param invar_obs_ind: (n_invar, )
        :param params_ind: (n_param, )
        :param pools_ind: (n_pool, )

        :return var_loss: (n_var, )
        :return invar_loss: (n_invar, )

        '''

        # get shape
        if var_obs.ndim == 1:
            n_var = 1
        else:
            n_var = var_obs.shape[1]
        if len(invar_obs.shape) == 0:
            n_invar = 1
        else:
            n_invar = invar_obs.shape[0]
        n_total = n_var + n_invar
        # run model
        models = self.step_tem_batch(pools, var_forcings, invar_forcings, params)
        # standardization
        var_obs = (var_obs - obs_mean[:n_var]) / obs_std[:n_var]
        invar_obs = (invar_obs - obs_mean[n_var:n_total]) / obs_std[n_var:n_total]

        var_mod = models[:, var_obs_ind]
        var_mod = (var_mod - obs_mean[:n_var]) / obs_std[:n_var]

        invar_mod_flux = jnp.mean(models[:, invar_obs_ind], axis=0)
        unnorm_params = self.unnormalize_params(params)
        unnorm_pools = self.unnormalize_pools(pools)
        invar_mod_params = None if len(params_ind) == 0 else unnorm_params[params_ind]
        invar_mod_pools = None if len(pools_ind) == 0 else unnorm_pools[pools_ind]
        stacked_result = []
        stacked_result.append(invar_mod_flux)
        if invar_mod_params is not None:
            stacked_result.append(invar_mod_params)
        if invar_mod_pools is not None:
            stacked_result.append(invar_mod_pools)
        # Perform the hstack only if there are values to stack
        invar_mod = jnp.hstack(stacked_result)
        invar_mod = (invar_mod - obs_mean[n_var:n_total]) / obs_std[n_var:n_total]

        # cal mse
        var_loss = jnp.mean((var_mod - var_obs) ** 2, axis=0) * grid_weights[:n_var]
        invar_loss = (invar_mod - invar_obs) ** 2 * grid_weights[n_var:]

        # jax.debug.breakpoint()

        return var_loss, invar_loss


    @partial(jax.jit, static_argnums=(0,))
    def compute_loss_batch(self, pools, var_forcings, invar_forcings, params, var_obs, invar_obs, grid_weights,
                           obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind):
        '''
        Function to calculate the loss, batch for all grids

        :param pools: (n_grid, n_pool)
        :param var_forcings: (n_grid, n_time, n_var)
        :param invar_forcings: (n_grid, n_invar)
        :param params: (n_grid, n_param)
        :param var_obs: (n_grid, n_time, n_var)
        :param invar_obs: (n_grid, n_invar)
        :param grid_weights: (n_grid, n_invar)
        :param obs_mean: (n_var+n_invar, )
        :param obs_std: (n_var+n_invar, )
        :param var_obs_ind: (n_var, )
        :param invar_obs_ind: (n_invar, )
        :param params_ind: (n_param, )
        :param pools_ind: (n_pool, )

        :return loss_ls: (n_var+n_invar, )

        '''
        
        batched_loss = jax.vmap(self.compute_tempo_loss,
                                in_axes=([0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None]))
        # var_loss, invar_loss = batched_loss(pools[65:67, :], var_forcings[65:67, 4:6, :], invar_forcings[65:67, :], params[65:67, :], var_obs[65:67, 4:6, :], invar_obs[65:67, :],
        #                                    grid_weights[65:67, :], obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind)
        var_loss, invar_loss = batched_loss(pools, var_forcings, invar_forcings, params, var_obs, invar_obs, grid_weights,
                                           obs_mean, obs_std, var_obs_ind, invar_obs_ind, params_ind, pools_ind)
        if var_obs.ndim == 2:
            n_var = 1
        else:
            n_var = var_obs.shape[2]
        if invar_obs.ndim == 1:
            n_invar = 1
        else:
            n_invar = invar_obs.shape[1]

        loss_ls = []
        for vari in range(n_var):
            loss_ls.append(var_loss[:, vari])
        for vari in range(n_invar):
            loss_ls.append(invar_loss[:, vari])
        
        # jax.debug.breakpoint()

        return loss_ls

        