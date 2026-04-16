# -*- coding: utf-8 -*-
import jax.numpy as jnp
import jax
from functools import partial
from DifferLand.model.auxi.phenology import leaf_fall_factor, lab_release_factor
from DifferLand.model.auxi.ACM import ACM
from DifferLand.model.DALEC_990_parinfo import dalec990_parmax, dalec990_parmin
from DifferLand.model.DALEC_990_parinfo import dalec990_param_parmin, dalec990_param_parmax, dalec990_pfn
from DifferLand.model.DALEC_990_parinfo import dalec990_pool_parmax, dalec990_pool_parmin
from DifferLand.optimization.forward import parameter_prediction_forward
from DifferLand.model.DALEC import DALEC
from DifferLand.util.normalization import unnormalize_parameters, nor2par
from DifferLand.optimization.loss_functions import negative_log_sigmoid, compute_nnse


class DALEC990(DALEC):
    def __init__(self, train_end_idx, water_stress_type="default", scm_type="onepool", with_som=1, reco=False):
        super().__init__()
        self.parmin = dalec990_parmin
        self.parmax = dalec990_parmax
        self.param_parmin = dalec990_param_parmin
        self.param_parmax = dalec990_param_parmax
        self.pool_parmin = dalec990_pool_parmin
        self.pool_parmax = dalec990_pool_parmax
        self.with_som = with_som
        self.water_stress_type = water_stress_type
        self.scm_type = scm_type
        self.pfn = dalec990_pfn
        self.id = 990
        self.train_end_idx = train_end_idx
        range1 = jnp.arange(self.pfn.next_paw_pool, self.pfn.next_paw_pool + 1)
        range2 = jnp.arange(self.pfn.next_foliar_pool, self.pfn.next_soc3_pool + 1)
        self.pool_range = jnp.concatenate([range1, range2])
        self.steady_range = jnp.arange(self.pfn.steady_fol_pool, self.pfn.steady_soc3_pool + 1)
        self.reco = reco

    def update_pool_and_flux(self, this_pool, pool_min, pool_max, in_flux, out_flux, delta_t, other_out_flux=0):
        # cal flux
        next_pool = this_pool + (in_flux - out_flux - other_out_flux) * delta_t
        # constraint by min or max value
        next_pool = jnp.maximum(next_pool, pool_min)
        next_pool = jnp.minimum(next_pool, pool_max)
        # update out_flux
        out_flux = (in_flux - other_out_flux - (next_pool - this_pool) / delta_t)

        return next_pool, out_flux

    @partial(jax.jit, static_argnums=(0,))
    def step(self, pools, met, dalec_parameters, gpp_params):
        """A forward step of the model"""
        # ========================================================================================================
        # basic  info
        # driver
        # for calculating GPP and ET
        lat = met[0]
        doy = met[1]
        t_max = met[2]
        t_min = met[3]
        rad = met[4]
        ca = met[5]
        vpd = met[6]
        norm_temp = met[7]
        norm_solar = met[8]
        norm_vpd = met[9]
        norm_ca = met[10]
        # for other internal calculating process
        time = met[11]
        delta_t = met[12]
        t_mean = met[13]
        precipitation = met[14]
        mean_precipitation = met[15]
        mean_gpp = met[16]

        # Params
        # Initial pool params
        paw_pool = pools[0]  # water available pool
        foliar_pool = pools[1]
        labile_pool = pools[2]
        wood_pool = pools[3]
        root_pool = pools[4]
        litter_pool = pools[5]
        soc_pool = pools[6]  # total soil carbon pool
        soc1_pool = pools[7]  # active soil carbon pool
        soc2_pool = pools[8]  # slow soil carbon pool
        soc3_pool = pools[9]  # passive soil carbon pool

        # Other params used in the model
        # ACM model related params
        LCMA = dalec_parameters[0]  # Leaf carbon mass per area
        lai = foliar_pool / LCMA
        ce = dalec_parameters[1]  # Canopy Efficiency (time invariant)

        # ET related params
        uWUE = dalec_parameters[2]  # water use efficiency
        boese_r = dalec_parameters[3]  # a site-specific parameter controlling the radiation-associated equilibrium transpiration
        field_capacity = dalec_parameters[4]
        wilting_point_frac = dalec_parameters[5]

        # water pool fluxes related params
        PAW_Qmax = dalec_parameters[6]  # max water fluxes of available water pool

        # aboveground carbon fluxes related params
        # allocation coefficients
        f_auto = dalec_parameters[7]  # fraction of GPP respired
        f_fol = dalec_parameters[8]  # fraction of (1-fpp) to foliage
        f_lab = dalec_parameters[9]  # Fraction to clab
        f_root = dalec_parameters[10]  # fraction of (1-fgpp) to roots
        # turnover rate
        leaf_lifespan = dalec_parameters[11]  # leaf lifespan
        leaf_fall_period = dalec_parameters[12]  # leaf fall period
        Fday = dalec_parameters[13]  # Peak day of year for foliar turnover
        lff = leaf_fall_factor(time, leaf_lifespan, leaf_fall_period, Fday)  # turnover rate of foliar
        lab_lifespan = dalec_parameters[14]  # labile pool lifespan
        clab_release_period = dalec_parameters[15]  # Clab release period
        Bday = dalec_parameters[16]  # Peak day of year for labile turnover
        lrf = lab_release_factor(time, lab_lifespan, clab_release_period, Bday)  # turnover rate of labile
        tor_wood = dalec_parameters[17]  # turn over rate wood - 1% loss per year value
        tor_root = dalec_parameters[18]  # turn over rate root
        tor_litter = dalec_parameters[19]  # part of Rh
        decomposition_rate = dalec_parameters[20]  # part of litter to soil
        # influence of temperature and precipitation on the litter output
        Q10 = dalec_parameters[21]  # Temp factor
        moisture_factor = dalec_parameters[22]  # moisture factor
        temperate = (jnp.exp(Q10 * (0.5 * (t_max + t_min) - t_mean))
                     * ((precipitation / mean_precipitation - 1) * moisture_factor + 1))

        # params related to the soil carbon fluxes
        # turnover rate of soil carbon pool
        tor_soc = dalec_parameters[23]
        tor_soc1 = dalec_parameters[24]
        tor_soc2 = dalec_parameters[25]
        tor_soc3 = dalec_parameters[26]
        # allocation
        f_litter2soc1 = dalec_parameters[27]
        f_litter2soc2 = dalec_parameters[28]
        f_litter2soc3 = dalec_parameters[29]  # if needed, here we only use f_litter2som1 and f_litter2som2, sum of them is 1
        # transfer coefficient between soil carbon pool
        f_soc21 = dalec_parameters[30]
        f_soc31 = dalec_parameters[31]
        f_soc12 = dalec_parameters[32]
        f_soc32 = dalec_parameters[33]
        f_soc13 = dalec_parameters[34]
        f_soc23 = dalec_parameters[35]  # if needed, here f_soc23 is 0

        # ========================================================================================================
        # calculate GPP and ET based on different water stress configurations
        if self.water_stress_type == "baseline":
            beta = 1
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "default":
            wilting_point = field_capacity * wilting_point_frac
            beta = (paw_pool - wilting_point) / (field_capacity - wilting_point)
            beta = jnp.where(beta <= 1, beta, 1)
            beta = jnp.where(beta >= 0, beta, 0.0001) # use a minimal value to replace 0
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "nn_paw":
            beta = jax.nn.sigmoid(parameter_prediction_forward(gpp_params, jnp.array([paw_pool / 1500, ]))[0])
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            ET = gpp * jnp.sqrt(vpd) / uWUE + rad * boese_r
        elif self.water_stress_type == "nn_whole":
            beta = -9999
            raw_gpp, raw_ET = parameter_prediction_forward(gpp_params, jnp.array(
                [norm_temp, lai / 8, norm_solar, paw_pool / 1500, norm_vpd, norm_ca]))
            gpp = jnp.maximum(0.01 * raw_gpp, raw_gpp)
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)
        elif self.water_stress_type == "nn_whole_no_lai":
            beta = -9999
            raw_gpp, raw_ET = parameter_prediction_forward(gpp_params, jnp.array(
                [norm_temp, norm_solar, paw_pool / 1500, norm_vpd, norm_ca]))
            gpp = jnp.maximum(0.01 * raw_gpp, raw_gpp)
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)
        elif self.water_stress_type == "gpp_acm_et_nn":
            beta = -9999
            beta_params, et_params = gpp_params
            beta = jax.nn.sigmoid(parameter_prediction_forward(beta_params, jnp.array([paw_pool / 1500, ]))[0])
            gpp = ACM(lat=lat, doy=doy, t_max=t_max, t_min=t_min, lai=lai, rad=rad, ca=ca, ce=ce) * beta
            raw_ET = parameter_prediction_forward(et_params, jnp.array(
                [norm_temp, lai / 8, norm_solar, paw_pool / 1500, norm_vpd, norm_ca, gpp / 8]))[0]
            ET = jnp.maximum(0.01 * raw_ET, raw_ET)

        # ========================================================================================================
        # water pool fluxes and pools
        # fluxes, q_paw
        q_paw = paw_pool ** 2 / (PAW_Qmax * delta_t)
        # q_paw <= paw_pool / (2*delta_t)
        q_paw = jnp.minimum(q_paw, paw_pool / (2 * delta_t))
        q_paw = jnp.maximum(q_paw, 0)

        # pools, adjustments such that each pool never exceeds its minimum or maximum
        # also need to greater than or equal to 0
        # if exceeds the threshold, adjust the fluxes of output
        next_paw_pool, ET = self.update_pool_and_flux(
            paw_pool,
            self.parmin.initial_PAW,
            self.parmax.initial_PAW,
            precipitation,
            ET,
            delta_t,
            q_paw
        )

        # ========================================================================================================
        # aboveground carbon fluxes, pools and steady state
        # fluxes
        # Ra
        respiration_auto = f_auto * gpp
        # foliar
        leaf_production = (gpp - respiration_auto) * f_fol
        leaf_litter = foliar_pool * (1 - (1 - lff) ** delta_t) / delta_t
        # labile
        labile_production = (gpp - respiration_auto - leaf_production) * f_lab
        labile_release = labile_pool * (1 - (1 - lrf) ** delta_t) / delta_t
        # root
        root_production = (gpp - respiration_auto - leaf_production - labile_production) * f_root
        root_litter = root_pool * (1 - (1 - tor_root) ** delta_t) / delta_t
        # wood
        wood_production = gpp - respiration_auto - leaf_production - labile_production - root_production
        wood_litter = wood_pool * (1 - (1 - tor_wood) ** delta_t) / delta_t
        # litter
        respiration_hetero_litter = litter_pool * (1 - (1 - temperate * tor_litter) ** delta_t) / delta_t
        litter_to_soc = litter_pool * (1 - (1 - temperate * decomposition_rate) ** delta_t) / delta_t

        # pools
        next_labile_pool, labile_release = self.update_pool_and_flux(
            labile_pool,
            self.parmin.Clab,
            self.parmax.Clab,
            labile_production,
            labile_release,
            delta_t
        )
        next_foliar_pool, leaf_litter = self.update_pool_and_flux(
            foliar_pool,
            self.parmin.Cfol,
            self.parmax.Cfol,
            leaf_production + labile_release,
            leaf_litter,
            delta_t
        )
        next_root_pool, root_litter = self.update_pool_and_flux(
            root_pool,
            self.parmin.Croot,
            self.parmax.Croot,
            root_production,
            root_litter,
            delta_t
        )
        next_wood_pool, wood_litter = self.update_pool_and_flux(
            wood_pool,
            self.parmin.Cwood,
            self.parmax.Cwood,
            wood_production,
            wood_litter,
            delta_t
        )
        next_litter_pool, litter_to_soc = self.update_pool_and_flux(
            litter_pool,
            self.parmin.Clitter,
            self.parmax.Clitter,
            leaf_litter + root_litter + wood_litter,
            litter_to_soc,
            delta_t,
            other_out_flux=respiration_hetero_litter
        )
        litter_to_soc = jnp.maximum(litter_to_soc, 0)
        respiration_hetero_litter = (leaf_litter + root_litter + wood_litter - litter_to_soc -
                                     (next_litter_pool - litter_pool) / delta_t)

        # # steady state, if necessary
        # # cal the true allocation coefficients, f_auto + f_fol + f_lab + f_root + f_wood = 1
        # f_fol1 = leaf_production / gpp
        # f_lab1 = labile_production / gpp
        # f_root1 = root_production / gpp
        # f_wood1 = wood_production / gpp
        #
        # steady_lab_pool = mean_gpp * f_lab1 / lrf
        # steady_fol_pool = mean_gpp * (f_lab1 + f_fol1) / lff
        # steady_root_pool = mean_gpp * f_root1 / tor_root
        # steady_wood_pool = mean_gpp * f_wood1 / tor_wood
        # steady_litter_pool = mean_gpp * (f_lab1 + f_fol1 + f_root1 + f_wood1) / (
        #             temperate * (tor_litter + decomposition_rate))

        # ========================================================================================================
        # soil carbon fluxes baesd on different soil carbon model type
        if self.scm_type == "onepool":
            # Rh
            respiration_hetero_soc = soc_pool * (1 - (1 - temperate * tor_soc) ** delta_t) / delta_t
            # next som pool
            next_soc_pool, respiration_hetero_soc = self.update_pool_and_flux(
                soc_pool,
                self.parmin.Csoc,
                self.parmax.Csoc,
                litter_to_soc,
                respiration_hetero_soc,
                delta_t
            )

            # create nan array
            litter_to_soc1 = 0
            litter_to_soc2 = 0
            litter_to_soc3 = 0

            respiration_hetero_soc1 = 0
            respiration_hetero_soc2 = 0
            respiration_hetero_soc3 = 0
            
            transfer_in_soc1 = 0
            transfer_in_soc2 = 0
            transfer_in_soc3 = 0

            transfer_out_soc1 = 0
            transfer_out_soc2 = 0
            transfer_out_soc3 = 0

            next_soc1_pool = 0
            next_soc2_pool = 0
            next_soc3_pool = 0

            # steady_soc1_pool = 0
            # steady_soc2_pool = 0
            # steady_soc3_pool = 0

        elif self.scm_type == "century":
            # som carbon pool input
            in_flux = litter_to_soc
            # allocation matrix
            b_matrix = jnp.array([f_litter2soc1, 1 - f_litter2soc1, 0])
            # transfer matrix
            transfer_coe_matrix = jnp.array([[-1, f_soc12, f_soc13],
                                             [f_soc21, -1, 0],
                                             [f_soc31, f_soc32, -1]])

            # turnover rate matrix (diagonal matrix) with environment factor
            tor_soc_jax = jnp.array([tor_soc1, tor_soc2, tor_soc3])
            k_elements = (1 - (1 - temperate * tor_soc_jax) ** delta_t) / delta_t
            k_matrix = jnp.diag(k_elements)
            # initial pool matrix
            soc_pool_matrix = jnp.array([soc1_pool, soc2_pool, soc3_pool])

            # matrix calculation
            # part 1 total output
            out_matrix = k_matrix @ soc_pool_matrix
            # part2, output to other som pools
            transfer_in_coe_matrix = jnp.array([[0, f_soc12, f_soc13],
                                                [f_soc21, 0, 0],
                                                [f_soc31, f_soc32, 0]])
            transfer_out_coe_elements = jnp.nansum(transfer_in_coe_matrix, axis=0)
            # convert to diagonal matrix
            transfer_out_coe_matrix = jnp.diag(transfer_out_coe_elements)
            transfer_out_matrix = transfer_out_coe_matrix @ out_matrix
            # part3, output to rh
            respiration_hetero_soc_matrix = out_matrix - transfer_out_matrix
            # part4, input from litter to som
            litter2soc_matrix = in_flux * b_matrix
            # part5, input from other som pools, replace the diagonal elements to 0
            transfer_in_matrix = transfer_in_coe_matrix @ out_matrix
            # part6, net fluxes
            soc_pool_flux_matrix = litter2soc_matrix + transfer_coe_matrix @ out_matrix

            # next som pool
            next_soc_pool_matrix = soc_pool_matrix + soc_pool_flux_matrix * delta_t
            # adjustment, assuming the transfer between soil carbon pools unchanged, adjust rh
            soil_min_matrix = jnp.array([self.parmin.Csoc1, self.parmin.Csoc2, self.parmin.Csoc3])
            soil_max_matrix = jnp.array([self.parmax.Csoc1, self.parmax.Csoc2, self.parmax.Csoc3])

            next_soc_pool_matrix = jnp.maximum(next_soc_pool_matrix, soil_min_matrix)
            next_soc_pool_matrix = jnp.minimum(next_soc_pool_matrix, soil_max_matrix)

            respiration_hetero_soc_matrix = (litter2soc_matrix + transfer_in_matrix - transfer_out_matrix
                                             - (next_soc_pool_matrix - soc_pool_matrix) / delta_t)
            soc_pool_flux_matrix = (next_soc_pool_matrix - soc_pool_matrix) / delta_t

            # # steady state
            # # cal the new k matrix
            # right_matrix = (transfer_out_matrix + respiration_hetero_soc_matrix) / temperate
            # left_matrix = soc_pool_matrix
            # new_k_elements = right_matrix.flatten() / left_matrix.flatten()
            # new_k_matrix = jnp.diag(new_k_elements)
            #
            # # cal steady state
            # right_matrix1 = -steady_litter_pool * decomposition_rate * b_matrix
            # left_matrix1 = temperate * transfer_coe_matrix @ new_k_matrix
            # Csoc_steady_matrix = jnp.linalg.solve(left_matrix1, right_matrix1)

            # cal total and each soil carbon pools and fluxes
            soc_pool = jnp.nansum(soc_pool_matrix, axis=0)
            litter_to_soc = jnp.nansum(litter2soc_matrix, axis=0)
            litter_to_soc1 = litter2soc_matrix[0]
            litter_to_soc2 = litter2soc_matrix[1]
            litter_to_soc3 = litter2soc_matrix[2]

            respiration_hetero_soc = jnp.sum(respiration_hetero_soc_matrix, axis=0)
            respiration_hetero_soc1 = respiration_hetero_soc_matrix[0]
            respiration_hetero_soc2 = respiration_hetero_soc_matrix[1]
            respiration_hetero_soc3 = respiration_hetero_soc_matrix[2]

            transfer_in_soc1 = transfer_in_matrix[0]
            transfer_in_soc2 = transfer_in_matrix[1]
            transfer_in_soc3 = transfer_in_matrix[2]

            transfer_out_soc1 = transfer_out_matrix[0]
            transfer_out_soc2 = transfer_out_matrix[1]
            transfer_out_soc3 = transfer_out_matrix[2]

            next_soc_pool = jnp.nansum(next_soc_pool_matrix, axis=0)
            next_soc1_pool = next_soc_pool_matrix[0]
            next_soc2_pool = next_soc_pool_matrix[1]
            next_soc3_pool = next_soc_pool_matrix[2]

            # steady_soc_pool = jnp.nansum(Csoc_steady_matrix, axis=0)
            # steady_soc1_pool = Csoc_steady_matrix[0]
            # steady_soc2_pool = Csoc_steady_matrix[1]
            # steady_soc3_pool = Csoc_steady_matrix[2]

        reco = respiration_auto + respiration_hetero_litter + respiration_hetero_soc
        nee = -gpp + reco

        # new pools for the next time steps
        new_pools = jnp.array([next_paw_pool, next_foliar_pool, next_labile_pool,
                               next_wood_pool, next_root_pool, next_litter_pool,
                               next_soc_pool, next_soc1_pool, next_soc2_pool, next_soc3_pool])
        # new fluxes
        new_fluxes = jnp.array(
            [ET, q_paw, paw_pool, next_paw_pool, gpp, 
             nee, lai, reco, respiration_auto, respiration_hetero_litter,
             respiration_hetero_soc, respiration_hetero_soc1, respiration_hetero_soc2, respiration_hetero_soc3,
             transfer_in_soc1, transfer_in_soc2, transfer_in_soc3,
             transfer_out_soc1, transfer_out_soc2, transfer_out_soc3, 
             leaf_production, labile_production, wood_production, root_production,
             leaf_litter, labile_release, wood_litter, root_litter,
             litter_to_soc, litter_to_soc1, litter_to_soc2, litter_to_soc3,
             foliar_pool, labile_pool, wood_pool, root_pool, litter_pool,
             soc_pool, soc1_pool, soc2_pool, soc3_pool,
             next_foliar_pool, next_labile_pool, next_wood_pool, next_root_pool, next_litter_pool,
             next_soc_pool, next_soc1_pool, next_soc2_pool, next_soc3_pool])
        # steady_fol_pool, steady_lab_pool, steady_wood_pool, steady_root_pool, steady_litter_pool,
        # steady_soc_pool, steady_soc1_pool, steady_soc2_pool, steady_soc3_pool])

        return new_pools, new_fluxes

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize(self, normalized_parameters):
        """Convert model parameters from the real space to the physcial range"""
        return unnormalize_parameters(normalized_parameters, param_parmin=self.param_parmin,
                                      param_parmax=self.param_parmax)

    @partial(jax.jit, static_argnums=(0,))
    def unnormalize_pools(self, normalized_pools):
        """Convert pool values from the real space to the physical value range"""
        return unnormalize_parameters(normalized_pools, param_parmin=self.pool_parmin, param_parmax=self.pool_parmax)

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, param_initial, pool_initial, gpp_params, met):
        """Forward hybrid model integration"""
        dp = self.unnormalize(param_initial)
        ip = self.unnormalize_pools(pool_initial)
        final_state, all_fluxes = jax.lax.scan(jax.jit(partial(self.step,
                                                               dalec_parameters=dp,
                                                               gpp_params=gpp_params)), ip, met)

        return all_fluxes.squeeze()


    @partial(jax.jit, static_argnums=(0,))
    def cal_edc(self, output_matrix, dalec_params, mean_temp, Pstart, total_precip, end_of_year_idx, k):
        """EDC constraints applied before model integration"""
        # part1, turnover rate constraints
        # EDC 1: Thita(som) < Thita(lit), DC 2: Thita(som) < Thita(min), if one pool
        # EDC 4: (1 - Thita(root))^N < (1 - Thita(som)*e^(Q10*Ti)) (i=1,..N, multiplied together), if one pool
        # Simplify, Thita(root) > Thita(som)*e^(Q10*Tmean), according to the text in ref
        # but if according to eq. 4, it is <, maybe type mistakes
        tor_lit = dalec_params[19]
        decom_lit = dalec_params[20]
        tor_root = dalec_params[28]
        Q10 = dalec_params[21]
        if self.scm_type == "onepool":
           tor_som = dalec_params[23]
           edc1 = negative_log_sigmoid(tor_lit, tor_som, 100000 * k)
           edc2 = negative_log_sigmoid(decom_lit, tor_som, 10000 * k)
           edc4 = negative_log_sigmoid(tor_root, tor_som * jnp.exp(Q10 * mean_temp), 10 * k)
        elif self.scm_type == "century":
           edc1 = 0
           edc2 = 0
           edc4 = 0

        # EDC 3: clf > 1 - (1 - Thita(woo))^365.25, i.e. (1 - Thita(woo))^365.25 > 1 - clf
        # simplify, Thita(leaf) > Thita(woo), Thita(leaf) = 1/(leaf_lifespan * 365.25)
        leaf_lifespan = dalec_params[11]
        tor_wood = dalec_params[17]
        edc3 = negative_log_sigmoid(1 / (leaf_lifespan * 365.25), tor_wood, 200000 * k)

        # part2, allocation coefficient constraints
        # EDC 5: 0.2 * froot < fol + flab < 5 * froot
        a_auto = dalec_params[7]
        a_fol = (1 - a_auto) * dalec_params[8]
        a_lab = (1 - a_auto - a_fol) * dalec_params[9]
        a_root = (1 - a_auto - a_fol - a_lab) * dalec_params[10]
        edc5 = (negative_log_sigmoid(5 * a_root, (a_fol + a_lab), 100 * k) +
               negative_log_sigmoid(5 * (a_fol + a_lab), a_root, 100 * k))

        # EDC 6: 0.2*Cfol_mean < Croo_mean < 5*Cfol_mean
        # calculating the multi-year averaged Carbon pool
        Cfol_mean = jnp.nanmean(output_matrix[:self.train_end_idx, self.pfn.foliar_pool])
        Croot_mean = jnp.nanmean(output_matrix[:self.train_end_idx, self.pfn.root_pool])
        edc6 = (negative_log_sigmoid(5 * Cfol_mean, Croot_mean, 100 * k) +
               negative_log_sigmoid(5 * Croot_mean, Cfol_mean, 100 * k))

        # part3, carbon pool growth constraints, carbon pool exponential decay trajectories and steady state constraint
        # EDC 7: C_mean/C_initial < 1+growth_rate * (n-1)/10, growth_rate = 0.1
        # here use expected growth * (actual growth - 1) < 0.1, expected growth, Fin/Fout, actual growth, Cmean/C0
        # EDC 8: c < -365.25 ¡Á 3 / log(2), here use log(expected growth) + log(actual growth) < 2
        etol = 0.1  # max growing rate
        MPOOLS_Jan = ((Pstart + jnp.sum(output_matrix[:, self.pool_range].T * end_of_year_idx, axis=1)) /
                      (jnp.sum(end_of_year_idx) + 1))
        FTOTAL = jnp.sum(output_matrix, axis=0)
        # fluxes of paw pool
        Fin_0 = total_precip
        Fout_0 = FTOTAL[self.pfn.ET]
        # foliar pool
        Fin_1 = FTOTAL[self.pfn.leaf_production] + FTOTAL[self.pfn.labile_release]
        Fout_1 = FTOTAL[self.pfn.leaf_litter]
        # labile pool
        Fin_2 = FTOTAL[self.pfn.labile_production]
        Fout_2 = FTOTAL[self.pfn.labile_release]
        # wood pool
        Fin_3 = FTOTAL[self.pfn.wood_production]
        Fout_3 = FTOTAL[self.pfn.wood_litter]
        # root pool
        Fin_4 = FTOTAL[self.pfn.root_production]
        Fout_4 = FTOTAL[self.pfn.root_litter]
        # litter pool
        Fin_5 = FTOTAL[self.pfn.leaf_litter] + FTOTAL[self.pfn.root_litter] + FTOTAL[self.pfn.wood_litter]
        Fout_5 = FTOTAL[self.pfn.respiration_hetero_litter] + FTOTAL[self.pfn.litter_to_soc]
        # soil pool
        Fin_6 =  FTOTAL[self.pfn.litter_to_soc]
        Fout_6 = FTOTAL[self.pfn.respiration_hetero_soc]
        # soil pool1
        Fin_7 = FTOTAL[self.pfn.litter_to_soc1] + FTOTAL[self.pfn.transfer_in_soc1]
        Fout_7 = FTOTAL[self.pfn.respiration_hetero_soc1] +  FTOTAL[self.pfn.transfer_out_soc1]
        # soil pool2
        Fin_8 = FTOTAL[self.pfn.litter_to_soc2] + FTOTAL[self.pfn.transfer_in_soc2]
        Fout_8 = FTOTAL[self.pfn.respiration_hetero_soc2] +  FTOTAL[self.pfn.transfer_out_soc2]
        # soil pool3
        Fin_9 = FTOTAL[self.pfn.litter_to_soc3] + FTOTAL[self.pfn.transfer_in_soc3]
        Fout_9 = FTOTAL[self.pfn.respiration_hetero_soc3] +  FTOTAL[self.pfn.transfer_out_soc3]


        Fin = jnp.array([Fin_0, Fin_1, Fin_2, Fin_3, Fin_4, Fin_5, Fin_6, Fin_7, Fin_8, Fin_9])
        Fout = jnp.array([Fout_0, Fout_1, Fout_2, Fout_3, Fout_4, Fout_5, Fout_6, Fout_7, Fout_8, Fout_9])

        Rm = Fin / Fout
        Rs = Rm * MPOOLS_Jan / Pstart
        EQF = 2

        if self.scm_type == "onepool":
           num_pool = 7
        elif self.scm_type == "century":
           num_pool = 10
        edc78 = 0
        #jax.debug.breakpoint()
        for i in range(1, num_pool):
           edc78 += negative_log_sigmoid(jnp.log(EQF), jnp.abs(jnp.log(Rs[i])), 3 * k)
           edc78 += negative_log_sigmoid(etol, jnp.abs(Rs[i] - Rm[i]), 3 * k)

        return edc1 + edc2 + edc3 + edc4 + edc5 + edc6 + edc78

    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, param_initial, pool_initial, gpp_params, met_matrix, target_matrix, k):
        "Compute value of the loss function"
        output_matrix = self.forward(param_initial, pool_initial, gpp_params, met_matrix)
        gpp_loss = compute_nnse(target_matrix[:self.train_end_idx, 0],
                                output_matrix[:self.train_end_idx, self.pfn.gpp],
                                target_matrix[:self.train_end_idx, 1])
        # use ecosystem respiration (RECO) for
        if self.reco:
            nee_reco_loss = compute_nnse(target_matrix[:self.train_end_idx, -2],
                                         output_matrix[:self.train_end_idx, self.pfn.nee] +
                                         output_matrix[:self.train_end_idx, self.pfn.gpp],
                                         target_matrix[:self.train_end_idx, -1])
        else:
            nee_reco_loss = compute_nnse(target_matrix[:self.train_end_idx, 2],
                                         output_matrix[:self.train_end_idx, self.pfn.nee],
                                         target_matrix[:self.train_end_idx, 3])
        et_loss = compute_nnse(target_matrix[:self.train_end_idx, 4],
                               output_matrix[:self.train_end_idx, self.pfn.ET],
                               target_matrix[:self.train_end_idx, 5])
        lai_loss = compute_nnse(target_matrix[:self.train_end_idx, 6],
                                output_matrix[:self.train_end_idx, self.pfn.lai],
                                target_matrix[:self.train_end_idx, 7])
        # som, only one observation at each location, cal MRE
        model_mean_soc = jnp.mean(output_matrix[:, self.pfn.soc_pool])
        som_loss = jnp.abs(target_matrix[0, 8] - model_mean_soc) / target_matrix[0, 8]

        dalec_params = self.unnormalize(param_initial)
        mean_temp = jnp.mean(met_matrix[:self.train_end_idx, 13])
        Pstart = self.unnormalize_pools(pool_initial)
        if self.scm_type == "century":
            Pstart = Pstart.at[-4].set(jnp.nansum(Pstart[-3:]))
        end_of_year_idx = met_matrix[:, 17]
        total_precip = jnp.sum(met_matrix[:, 14])
        edc_loss = self.cal_edc(output_matrix, dalec_params, mean_temp, Pstart, total_precip, end_of_year_idx, k)

        if self.with_som == 1:
            if self.scm_type == "onepool":
                total_loss = -gpp_loss - nee_reco_loss - et_loss - lai_loss + som_loss + edc_loss
            elif self.scm_type == "century":
                total_loss = -gpp_loss - nee_reco_loss - et_loss - lai_loss + 0.5 * som_loss + edc_loss
        else:
            total_loss = -gpp_loss - nee_reco_loss - et_loss - lai_loss + edc_loss

        return total_loss

