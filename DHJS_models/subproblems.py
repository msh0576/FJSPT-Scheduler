import random
import time
import torch
import numpy as np
from copy import deepcopy

def train_one_batch_subprob(
    num_subprob, batch_size, num_opes, num_mas, num_vehs,
    nums_ope_batch, proc_times_batch, ope_ma_adj_batch, trans_times_batch
):
    '''
    :param nums_ope_batch [B, n_jobs]
    :param proc_times_batch [B, n_opes, n_mas]
    :param ope_ma_adj_batch [B, n_opes, n_mas]
    :param trans_times_batch [B, n_mas, n_mas]
    
    :return sub_ope_ma_adj [B, n_subprob, n_opes, n_mas]
    '''
    
    # === get subproblems ===
    for batch in range(batch_size):
        num_vehs_list = [random.randint(2, num_vehs) for _ in range(num_subprob)]
        num_mas_list = [random.randint(2, num_mas) for _ in range(num_subprob)]
        mas_idxes_list = [sample_n_indexes(n_ma, num_mas) for n_ma in num_mas_list]
        vehs_idxes_list = [sample_n_indexes(n_veh, num_vehs) for n_veh in num_vehs_list]
        
        
    
    pass


def sample_n_indexes(N, max_idx):
    return random.sample(range(max_idx + 1), N)

