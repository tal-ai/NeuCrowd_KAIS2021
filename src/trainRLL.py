from model import RLLWORKER
import numpy as np
import pandas as pd
import glob
import sys
from data_cfg import cfg
which_data = sys.argv[1]

dimension = cfg[which_data]['dimension']
DATA_NAME = cfg[which_data]['DATA_NAME']
train_df_name = cfg[which_data]['train_df_name']
valid_df_name = cfg[which_data]['valid_df_name']


# first column is ground truth label, second column is crowdsourced votes (set to 1 if you have no crowdsourced labels)
train = pd.read_csv(DATA_NAME + train_df_name)
validation = pd.read_csv(DATA_NAME + valid_df_name)

#####
#####
# grid search for parameters
max_iter = 50
batchSize = 256

sorted(json_file_list)
lr_rate_lst = [0.01, 0.05]
# lr_rate_lst = [0.01]
l1_n_lst = [256, 128]
# l1_n_lst = [256]
# l2_n_lst = [128, 64, 32]
l2_n_lst = [32]

reg_scale_lst = [2.0, 5.0]
reg_scale_lst = [2.0]
dropout_rate_lst = [0.2]
sim_type_lst = ['euc']
anchor_type_lst = ['default']
seq_length_list = [8, 9, 10]
# seq_length_list = [7]
gamma_list = [2.0, 5.0, 10.0]
gamma_list = [2.0]
# RLL_l1_256_l2_64_lr_0.05_penalty_2.0_bs_256_dropout_0.2_sim_euc_anchor_default_ON_class_rehearsal_seq_5_history
for lr_rate in lr_rate_lst:
    for dropout_rate in dropout_rate_lst:
        for l1_n in l1_n_lst:
            for l2_n in l2_n_lst:
                for reg_scale in reg_scale_lst:
                        if l1_n < l2_n:
                            continue
                        for sim_type in sim_type_lst:
                            for anchor_type in anchor_type_lst:
                                for gamma in gamma_list:
                                    for seq_length in seq_length_list:
                                        model = RLLWORKER(dimension, l1_n, l2_n, gamma, sim_type, anchor_type, grade=which_data)
                                        model.build(train, validation)
                                        model.train(batchSize, lr_rate, reg_scale, dropout_rate, max_iter)

