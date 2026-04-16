# Code for visualizing the results of a specific simulation

import os
import pickle
import sys
sys.path.insert(1, '../')
from DifferLand.util.visualization import visualization_global


def main():

    result_path = r'../../../output/soc3_ge2/mesc_pml2_all_time_ts_30_bs_400_lr_0.0005_epo_1_it_1_r_42_unc_0_p2g_0_i_1.pkl'
    with open(result_path, 'rb') as fp:
        result_ls = pickle.load(fp)
    
    fig_dir_base = r'../../../fig/soc3_ge2'
    fig_dir = os.path.join(fig_dir_base, os.path.basename(result_path)[:-4])
    MASK = 'test'
    METRIC = 'R2'

    visualization_global(result_ls, fig_dir, mask=MASK, metric=METRIC)

    return


if __name__ == "__main__":

    main()

