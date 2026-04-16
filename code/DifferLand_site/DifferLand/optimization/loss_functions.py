import jax.numpy as jnp
from DifferLand.model.DALEC_990_parinfo import dalec990_pfn


# this function is used to EDCs constraints
def negative_log_sigmoid(a, b, k=1000000):
    return -jnp.log(1 / (1 + jnp.exp(-k * (a - b))))


# cal nse and nnse
def compute_nse(observed, modeled, mask):
    numerator = jnp.sum((observed - modeled) ** 2 * mask)
    observed_mean = jnp.sum(observed * mask) / jnp.sum(mask)
    denominator = jnp.sum((observed - observed_mean) ** 2 * mask)
    return 1 - numerator / denominator


def compute_nnse(observed, modeled, mask):
    return 1 / (2 - compute_nse(observed, modeled, mask))


def sum_of_error(predicted, true, mask, weight=1):
    return jnp.sum((predicted - true) ** 2 * mask) * weight


def compute_nnse_eval(output_matrix, target_matrix, train_sel, test_sel, reco=False):
    gpp_train_nnse = compute_nnse(target_matrix[train_sel, 0],
                                  output_matrix[train_sel, dalec990_pfn.gpp],
                                  target_matrix[train_sel, 1])
    if reco:
        nee_reco_train_nnse = compute_nnse(target_matrix[train_sel, -2],
                                           output_matrix[train_sel, dalec990_pfn.gpp] + output_matrix[train_sel, dalec990_pfn.nee],
                                           target_matrix[train_sel, -1])
    else:
        nee_reco_train_nnse = compute_nnse(target_matrix[train_sel, 2],
                                           output_matrix[train_sel, dalec990_pfn.nee],
                                           target_matrix[train_sel, 3])
    et_train_nnse = compute_nnse(target_matrix[train_sel, 4],
                                 output_matrix[train_sel, dalec990_pfn.ET],
                                 target_matrix[train_sel, 5])
    lai_train_nnse = compute_nnse(target_matrix[train_sel, 6],
                                  output_matrix[train_sel, dalec990_pfn.lai],
                                  target_matrix[train_sel, 7])
    model_mean_soc = jnp.mean(output_matrix[:, dalec990_pfn.soc_pool])
    som_train_mre = jnp.abs(target_matrix[0, 8] - model_mean_soc) / target_matrix[0, 8]

    gpp_test_nnse = compute_nnse(target_matrix[test_sel, 0],
                                 output_matrix[test_sel, dalec990_pfn.gpp],
                                 target_matrix[test_sel, 1])
    if reco:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, -2],
                                          output_matrix[test_sel, dalec990_pfn.gpp] + output_matrix[test_sel, dalec990_pfn.nee],
                                          target_matrix[test_sel, -1])
    else:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 2],
                                          output_matrix[test_sel, dalec990_pfn.nee],
                                          target_matrix[test_sel, 3])
    et_test_nnse = compute_nnse(target_matrix[test_sel, 4],
                                output_matrix[test_sel, dalec990_pfn.ET],
                                target_matrix[test_sel, 5])
    lai_test_nnse = compute_nnse(target_matrix[test_sel, 6],
                                 output_matrix[test_sel, dalec990_pfn.lai],
                                 target_matrix[test_sel, 7])

    return (gpp_train_nnse, nee_reco_train_nnse, et_train_nnse, lai_train_nnse, som_train_mre,
            gpp_test_nnse, nee_reco_test_nnse, et_test_nnse, lai_test_nnse)


def compute_test_nnse(output_matrix, target_matrix, test_sel, reco=False):
    gpp_test_nnse = compute_nnse(target_matrix[test_sel, 0],
                                 output_matrix[test_sel, dalec990_pfn.gpp],
                                 target_matrix[test_sel, 1])
    if reco:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, -2],
                                          output_matrix[test_sel, dalec990_pfn.gpp] + output_matrix[test_sel, dalec990_pfn.nee],
                                          target_matrix[test_sel, -1])
    else:
        nee_reco_test_nnse = compute_nnse(target_matrix[test_sel, 2],
                                          output_matrix[test_sel, dalec990_pfn.nee],
                                          target_matrix[test_sel, 3])
    et_test_nnse = compute_nnse(target_matrix[test_sel, 4],
                                output_matrix[test_sel, dalec990_pfn.ET],
                                target_matrix[test_sel, 5])
    lai_test_nnse = compute_nnse(target_matrix[test_sel, 6],
                                 output_matrix[test_sel, dalec990_pfn.lai],
                                 target_matrix[test_sel, 7])

    return gpp_test_nnse + nee_reco_test_nnse + et_test_nnse + lai_test_nnse
