import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def random_act(state):
    '''
    Output:
        action: [ope_idx, mas_idx, job_idx, veh_idx]: [4, B]
    '''
    batch_idxes = state.batch_idxes
    batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
    num_jobs = state.mask_job_procing_batch.size(1)
    num_vehs = state.mask_veh_procing_batch.size(1)
    
    rand_act = torch.ones(size=(batch_size, num_mas * num_jobs * num_vehs), dtype=torch.float)
    ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)   # [B, n_jobs]
    
    # === eligible action check: Detect eligible O-M pairs (eligible action) ===
    # = find some eligible operation indexes, which are non-finished ones =
    dummy_shape = torch.zeros(size=(batch_size, num_jobs, num_mas, num_vehs))

    # = whether processing is possible. for each operation, there are processible machines =
    eligible_proc = state.ope_ma_adj_batch.gather(1, ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1)))   # [B, n_jobs, n_mas]
    eligible_proc = eligible_proc[..., None].expand_as(dummy_shape) # [B, n_jobs, n_mas, n_vehs]
    
    # = whether machine/job/vehicle is possible =
    # : operation-machine eligible
    ma_eligible = ~state.mask_ma_procing_batch[:, None, :].expand(-1, num_jobs, -1)       # [B, n_jobs, n_mas]
    job_eligible = ~(state.mask_job_procing_batch + state.mask_job_finish_batch)[..., None].expand(-1, -1, num_mas)
    ope_ma_adj = state.ope_ma_adj_batch.gather(1, ope_step_batch[..., None].expand(-1, -1, num_mas)).bool()   # [B, n_jobs, n_mas]
    ope_ma_eligible = job_eligible & ma_eligible & ope_ma_adj   # [B, n_jobs, n_mas]
    # : operation-machine-vehicle eligible
    veh_eligible = ~state.mask_veh_procing_batch[:, None, None, :].expand_as(dummy_shape)
    ope_ma_veh_eligible = ope_ma_eligible[..., None].expand_as(dummy_shape) & veh_eligible  # [B, n_jobs, n_mas, n_vehs]
    
    # ma_eligible = ~state.mask_ma_procing_batch[:, None, :, None].expand_as(dummy_shape)    # [B, n_jobs, n_mas, n_vehs]
    # job_eligible = ~(state.mask_job_procing_batch + state.mask_job_finish_batch)[..., None, None].expand_as(dummy_shape)    # [B, n_jobs, n_mas, n_vehs]
    # veh_eligible = ~state.mask_veh_procing_batch[:, None, None, :].expand_as(dummy_shape)  # [B, n_jobs, n_mas, n_vehs]
    # eligible = ma_eligible & job_eligible & veh_eligible & (eligible_proc == 1)  # [B, n_jobs, n_mas, n_vehs]
    
    eligible_batch = ope_ma_veh_eligible.flatten(1).count_nonzero(dim=1) # [B,]
    is_ineligible = (eligible_batch == 0).sum(dim=0)
    if is_ineligible > 0:
        print("No eligible O-M-V pair!")
        return 
    # === get action_idx with masking the ineligible actions ===
    mask = ope_ma_veh_eligible.permute(0, 3, 2, 1).flatten(1) # [B, n_vehs, n_mas, n_jobs] -> [B, n_vehs * n_mas * n_jobs]
    rand_act[~mask] = float('-inf')
    rand_act_prob = F.softmax(rand_act, dim=1)  # [B, n_vehs * n_mas * n_jobs]
    dist = Categorical(rand_act_prob)
    elig_act_idx = dist.sample()    # [B,]
    # ===== transper action_idx to indexes of machine, job and operation =====
    veh, ma, job, ope = _act_2_objects(elig_act_idx, ope_step_batch, batch_idxes, num_mas, num_jobs)
    # print(f"elig_act_idx:{elig_act_idx} | veh:{veh} | ma:{ma} | job:{job} | ope:{ope}")

    return torch.stack((ope, ma, job, veh), dim=0)

def _act_2_objects(action, ope_step_batch, batch_idxes, num_mas, num_jobs):
        '''
        Input
            action: [B,]
            ope_ste: tensor: [B, num_jobs]
        Output:
            veh_idx: scalar
            ma_idx: scalar
            job_idx: scalar
            ope_idx: scalar
        '''
        
        veh_idx = torch.div(action, (num_mas * num_jobs), rounding_mode='floor')
        ma_job = torch.remainder(action, (num_mas * num_jobs))
        ma_idx = torch.div(ma_job, num_jobs, rounding_mode='floor')
        job_idx = torch.remainder(ma_job, num_jobs)
        
        # veh_idx = action // (num_mas * num_jobs)
        # ma_job = action % (num_mas * num_jobs)
        # ma_idx = ma_job // num_jobs
        # job_idx = ma_job % num_jobs
        
        ope_idx = ope_step_batch[batch_idxes, job_idx]
        return veh_idx, ma_idx, job_idx, ope_idx

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE
    # print(f'qkv:{qkv.shape}')
    batch_s = qkv.size(0)
    qkv_len = len(qkv.size())
    
    if qkv_len == 3:
        n = qkv.size(1)

        q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
        # shape: (batch, n, head_num, key_dim)

        q_transposed = q_reshaped.transpose(1, 2)
        # shape: (batch, head_num, n, key_dim)

        return q_transposed
    elif qkv_len == 4:
        n = qkv.size(1)
        m = qkv.size(2)
        q_reshaped = qkv.reshape(batch_s, n, m, head_num, -1)   # [B, n_opes, n_mas, H, key_dim]
        q_transposed =q_reshaped.permute(0, 3, 1, 2, 4) # [B, H, n_opes, n_mas, key_dim]
        
        return q_transposed
    else:
        raise ValueError("qkv_len error!")
    
def generate_trans_mat(
    trans_time_batch, state, transtime_btw_ma_max=None, job_embedding=False
    ):
    '''
    Input:
        trans_time_batch: [B, n_mas, n_mas]
    Output:
        trans_mat: [B, n_opes or n_jobs, n_mas, n_vehs]
        empty_trans_batch: [B, n_opes or n_jobs, n_vehs]
        travel_trans_batch: [B, n_opes or n_jobs, n_mas]
        MVpair_trans_batch: [B, n_vehs, n_mas]: from current vehicle locations to machines
    '''
    batch_size, num_vehs = state.mask_veh_procing_batch.size()
    num_mas = state.mask_ma_procing_batch.size(1)
    veh_loc_batch = state.veh_loc_batch # [B, n_vehs]
    prev_ope_locs_batch = state.prev_ope_locs_batch # [B, n_jobs]
    allo_ma_batch = state.allo_ma_batch # [B, n_opes]
    elig_vehs = ~state.mask_veh_procing_batch   # [B, n_vehs]
    
    if job_embedding:
        ma_loc = prev_ope_locs_batch    # [B, n_jobs]
    else:
        ma_loc = allo_ma_batch          # [B, n_opes]
    
    MVpair_trans_batch = trans_time_batch.gather(1, veh_loc_batch[:, :, None].expand(-1, -1, trans_time_batch.size(2)))    # [B, n_vehs, n_mas]
    empty_trans_batch = MVpair_trans_batch.gather(2, ma_loc[:, None, :].expand(-1, MVpair_trans_batch.size(1), -1))    # [B, n_vehs, n_opes]
    travel_trans_batch = trans_time_batch.gather(1, ma_loc[:, :, None].expand(-1, -1, trans_time_batch.size(2)))   # [B, n_opes, n_mas]
    trans_mat = empty_trans_batch[:, :, :, None].expand(-1, -1, -1, num_mas) + travel_trans_batch[:, None, :, :].expand(-1, num_vehs, -1, -1)   # [B, n_vehs, n_opes, n_mas]
    # === eligible check ===
    if transtime_btw_ma_max is not None:
        trans_mat[~elig_vehs] = transtime_btw_ma_max
        empty_trans_batch[~elig_vehs] = transtime_btw_ma_max
    
    return trans_mat.permute(0, 2, 3, 1), empty_trans_batch.permute(0, 2, 1), \
        travel_trans_batch, MVpair_trans_batch


# def generate_trans_mat(trans_time_batch, num_vehs, num_mas,
#                         veh_loc_batch, prev_ope_locs_batch, elig_vehs):
#     '''
#     Input:
#         trans_time_batch: [B, n_mas, n_mas]
#     Output:
#         trans_mat: [B, n_jobs, n_mas, n_vehs]
#     '''

#     empty_trans_batch = trans_time_batch.gather(1, veh_loc_batch[:, :, None].expand(-1, -1, trans_time_batch.size(2)))    # [B, n_vehs, n_mas]
#     empty_trans_batch = empty_trans_batch.gather(2, prev_ope_locs_batch[:, None, :].expand(-1, empty_trans_batch.size(1), -1))    # [B, n_vehs, n_jobs]
#     travel_trans_batch = trans_time_batch.gather(1, prev_ope_locs_batch[:, :, None].expand(-1, -1, trans_time_batch.size(2)))   # [B, n_jobs, n_mas]
#     trans_mat = empty_trans_batch[:, :, :, None].expand(-1, -1, -1, num_mas) + travel_trans_batch[:, None, :, :].expand(-1, num_vehs, -1, -1)   # [B, n_vehs, n_jobs, n_mas]
#     trans_mat[~elig_vehs] = 1000
    
#     return trans_mat.permute(0, 2, 3, 1)

def select_vehicle(state, select_ma, select_job):
    '''
    Input:
        state:
        select_ope: [B, 1]
        select_ma: [B, 1]
    Output:
        results = {
            'veh_id': [B,]
            'trans_time': [B,]
        }
    '''
    trans_times_batch = state.trans_times_batch   # [B, n_mas, n_mas]
    veh_loc_batch = state.veh_loc_batch # [B, n_vehs]
    prev_ope_locs_batch = state.prev_ope_locs_batch # [B, n_jobs]
    batch_size = trans_times_batch.size(0)
    
    # print(f'trans_times_batch:{trans_times_batch}')
    
    elig_vehs = ~state.mask_veh_procing_batch   # [B, n_vehs]
    
    prev_ope_locs = prev_ope_locs_batch.gather(1, select_job)   # [B, 1]
    
    
    # min_trans_time = torch.ones(size=(batch_size,)) * 1000000
    results = {
        'veh_id': torch.zeros(size=(batch_size,)),
        'trans_time': torch.zeros(size=(batch_size,)),
    }
    for b in range(batch_size):
        elig_veh_ids = torch.where(elig_vehs[b, :] == True)
        # print(f'elig_veh_ids:{elig_veh_ids}')
        
        veh_locs = veh_loc_batch[b, elig_veh_ids[0]]    # [elig_n_vehs]
        tmp_prev_ope_locs = prev_ope_locs[b].expand(veh_locs.size(0))
        tmp_select_ma = select_ma[b].expand(veh_locs.size(0))
        # print(f'veh_locs:{veh_locs} -> {tmp_prev_ope_locs} -> {tmp_select_ma}')
        
        
        empty_trans = trans_times_batch[b, veh_locs, tmp_prev_ope_locs]
        travel_trans = trans_times_batch[b, tmp_prev_ope_locs, tmp_select_ma]
        # print(f'empty_trans:{empty_trans}, travel_trans:{travel_trans}')
        trans_time = empty_trans + travel_trans
        min_value, min_idx = trans_time.min(dim=0, keepdim=True)
        results['veh_id'][b] = elig_veh_ids[0][min_idx]
        results['trans_time'][b] = min_value
    # print(f'results:{results}')
    return results

def select_vehicle_v2(state, select_ma, select_ope, job_embedding=True):
    '''
    Input:
        state:
        select_ope: [B, 1]
        select_ma: [B, 1]
    Output:
        results = {
            'veh_id': [B,]
            'trans_time': [B,]
        }
    '''
    trans_times_batch = state.trans_times_batch   # [B, n_mas, n_mas]
    batch_size = trans_times_batch.size(0)
    batch_idxes = torch.arange(batch_size)
    trans_time, offload_trans, onload_trans, _ = generate_trans_mat(trans_times_batch, state, None, job_embedding)
    
    sel_offload_trans = offload_trans[batch_idxes, select_ope.squeeze(1), :] # [B, n_vehs]
    output_onload_time = onload_trans[batch_idxes, select_ope.squeeze(1), select_ma.squeeze(1)]    # [B]
    output_offload_time, veh_id = sel_offload_trans.min(dim=1)   # [B]
    results = {
        'veh_id': torch.zeros(size=(batch_size,)),
        'trans_time': torch.zeros(size=(batch_size,)),
    }
    results['veh_id'] = veh_id
    results['trans_time'] = output_onload_time + output_offload_time
    
    return results
    
    
    


def feature_normalize(data):
    return (data - torch.mean(data)) / ((data.std() + 1e-5))

def get_normalized(raw_opes, raw_mas, raw_vehs, proc_time, trans_time, flag_sample=False, flag_train=False):
    '''
    :param raw_opes: Raw feature vectors of operation nodes
    :param raw_mas: Raw feature vectors of machines nodes
    :param raw_vehs:
    :param proc_time: Processing time
    :param trans_time:
    :param batch_idxes: Uncompleted instances
    :param nums_opes: The number of operations for each instance
    :param flag_sample: Flag for DRL-S
    :param flag_train: Flag for training
    :return: Normalized feats, including operations, machines and edges
    '''
    # batch_size = batch_idxes.size(0)  # number of uncompleted instances
    
    # There may be different operations for each instance, which cannot be normalized directly by the matrix
    if not flag_sample and not flag_train:
        raise Exception('Not described here yet!')
    else:
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
        mean_vehs = torch.mean(raw_vehs, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_veh]
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
        std_vehs = torch.std(raw_vehs, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_veh]
        
        proc_time_norm = feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        trans_time_norm = feature_normalize(trans_time)  # shape: [len(batch_idxes), n_opes, num_mas]
        
    return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5), \
        (raw_vehs - mean_vehs) / (std_vehs + 1e-5), proc_time_norm, trans_time_norm)

def get_mask_ope_ma(state):
    '''
    Output:
        mask: [B, n_jobs, n_mas]
        mask_ope_ma: [B, n_opes, n_mas]
    '''
    batch_idxes = state.batch_idxes
    batch_size = state.ope_ma_adj_batch.size(0)
    num_opes = state.ope_ma_adj_batch.size(1)
    num_mas = state.ope_ma_adj_batch.size(2)
    num_jobs = state.mask_job_procing_batch.size(1)
    
    ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                    state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
    opes_appertain_batch = state.opes_appertain_batch   # [B, n_opes]
    # machine mask
    mask_ma = ~state.mask_ma_procing_batch[batch_idxes] # [B, n_mas]
    
    # machine mask for each job
    eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                        ope_step_batch[..., None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])    # [B, n_jobs, n_mas]
    dummy_shape = torch.zeros(size=(len(batch_idxes), num_jobs, num_mas))
    ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(dummy_shape) # [B, n_jobs, n_mas]
    job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                        state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(dummy_shape)   # [B, n_jobs, n_mas]
    eligible = job_eligible & ma_eligible & (eligible_proc == 1)

    if (~(eligible)).all():
        print("No eligible J-M pair!")
        return
    mask = eligible  # [B, n_jobs, n_mas]
    
    # === operation mask ===
    # : masks current ordered operation for each job
    mask_ope_step = torch.full(size=(batch_size, num_opes), dtype=torch.bool, fill_value=False) 
    tmp_batch_idxes = batch_idxes.unsqueeze(-1).repeat(1, num_jobs) # [B, n_jobs]
    mask_ope_step[tmp_batch_idxes, ope_step_batch] = True
    
    # : mask jobs that have no available machine and are processing
    mask_job = torch.where(mask.sum(dim=-1) > torch.zeros(size=(batch_size, num_jobs)), True, False)  # [B, n_jobs]
    mask_ope_by_job = mask_job.gather(1, opes_appertain_batch)
    mask_ope = mask_ope_by_job & mask_ope_step  # [B, n_opes]
    
    # === operation-machine mask ===
    mask_ope_padd = mask_ope[:, :, None].expand(-1, -1, num_mas)    # [B, n_opes, n_mas]
    mask_ma_padd = mask_ma[:, None, :].expand(-1, num_opes, -1) # [B, n_opes, n_mas]
    ope_ma_adj = state.ope_ma_adj_batch[batch_idxes]
    mask_ope_ma = mask_ope_padd & mask_ma_padd & (ope_ma_adj==1)  # [B, n_opes, n_mas]
    
    if (~(eligible)).all():
        print("No eligible O-M pair!")
        return
    
    return  mask, mask_ope_ma

def norm_disc_rewards(memory_rewards, memory_is_terminals, gamma, device):
    '''
    Input:
        memory_rewards: [B, T]
        memory_is_terminals: [B, T]
    Output:
        rewards_batch: [B, T]
        discounted_reward_batch: [B,]
    '''
    batch_size = memory_rewards.size(0)
    rewards_batch = []
    discounted_rewards = 0
    discounted_reward_batch =[]
    for i in range(batch_size):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
            if is_terminal:
                discounted_rewards += discounted_reward
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        discounted_rewards += discounted_reward
        rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards_batch.append(rewards)
        discounted_reward_batch.append(discounted_reward)
    # rewards_batch = torch.cat(rewards_batch)
    rewards_batch = torch.stack(rewards_batch, dim=0)
    discounted_reward_batch = torch.stack(discounted_reward_batch, dim=0)   # [B,]
    discounted_reward_batch = (discounted_reward_batch - discounted_reward_batch.mean()) / (discounted_reward_batch.std() + 1e-5)

    return rewards_batch.float(), discounted_reward_batch.float()

def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs