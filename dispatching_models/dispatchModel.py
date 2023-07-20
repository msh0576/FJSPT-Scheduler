import torch
from copy import deepcopy
import math
import torch.nn.functional as F

class dispatchModel():
    def __init__(self,
                 rule='spt',
                 **model_paras
                 ):
        
        # rule: 
        # 'spt': operation with shortest processing time
        # 'lpt': operation with longest processing time
        self.rule = rule    
        self.model_paras = model_paras
    
    def init(self, state, dataset=None, loader=None):
        self.batch_size = state.ope_ma_adj_batch.size(0)
        self.num_opes = state.ope_ma_adj_batch.size(1)
        self.num_mas = state.ope_ma_adj_batch.size(2)
        self.num_jobs = state.mask_job_finish_batch.size(1)
        self.num_vehs = state.mask_veh_procing_batch.size(1)
    
    def act(self, state):
        # : O-M pair with shortest processing time
        if self.rule == 'spt':
            # : select ope, ma, job
            select_ope, select_ma, select_job = self._select_OMPair_on_ProcTime(state, proc_crit='short')
        # : O-M pair with longest processing time
        elif self.rule == 'lpt':    
            select_ope, select_ma, select_job = self._select_OMPair_on_ProcTime(state, proc_crit='long')
        # : select job/machine as fast as possible
        elif self.rule == 'fifo':
            # : among multiple eligible jobs/machines, select one randomly
            select_job, select_ope = self._select_fifo_job(state)
            select_ma = self._select_fifo_ma(state, select_job)
        # : select a machine with lowest utilization, and then select an operation with the shortest processing time
        elif self.rule == 'lum_spt':
            select_ma = self._select_ma_on_util(state, util_crit='low')
            select_ope, select_job = self._select_oper_given_ma(state, select_ma, proc_crit='short')
        # : select a machine with lowest utilization, and then select an operation with the longest processing time
        elif self.rule == 'lum_lpt':
            select_ma = self._select_ma_on_util(state, util_crit='low')
            select_ope, select_job = self._select_oper_given_ma(state, select_ma, proc_crit='long')
        else:
            raise Exception('dispatch rule error!')
        # : select nearest vehicle node
        veh_dict = self._select_nearest_veh(state, select_ope, select_ma, select_job)
        select_veh = veh_dict['veh_id'].long()
        
        # === formulate it with action ===
        action = torch.cat([select_ope, select_ma, select_job, select_veh], dim=1).transpose(1, 0)    # [4, B]
        
        return action, 0

    def _select_ma_on_util(self, state, util_crit='low'):
        '''
        Input:
            state: ,
            util_crit: 
                'low': lowest utilization machine
                'high': highest utilization machine
        Output:
            select_ma: [B, 1]
        '''
        mask, _ = self._get_mask_ope_ma(state)
        mask_ma = torch.where(mask.sum(dim=1) == 0, False, True)    # [B, n_mas]
        
        util = deepcopy(state.feat_mas_batch[:, 2, :])    # [B, n_mas]
        if util_crit == 'low':
            util[~mask_ma] = 1000
            select_ma = util.min(dim=1, keepdim=True)[1]    # [B, 1]
        elif util_crit == 'high':
            util[~mask_ma] = 0
            select_ma = util.max(dim=1, keepdim=True)[1]    # [B, 1]
        else:
            raise Exception('util_cirt error!')
        return select_ma
        
    def _select_oper_given_ma(self, state, select_ma, proc_crit='short'):
        '''
        Input:
            state:
            select_ma: [B, 1]
            proc_crit:
                'short': shortest processing time operation given the machine
                'long': longest processing time operation given the machine
        Output:
            select_ope: [B, 1]
            select_job: [B, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        
        # : get eligible O-M pairs
        mask, mask_ope_ma = self._get_mask_ope_ma(state)
        
        # : find shortest processing time operation given machine
        proc_time = deepcopy(state.proc_times_batch) # [B, n_opes, n_mas]
        if proc_crit == 'short':
            proc_time[torch.where(mask_ope_ma == False)] = 1000
            proc_time = proc_time.gather(2, select_ma[:, None, :].expand(-1, num_opes, -1)).squeeze(2)  # [B, n_opes]
            select_ope = proc_time.argmin(dim=1, keepdim=True)  # [B, 1]
        elif proc_crit == 'long':
            proc_time[torch.where(mask_ope_ma == False)] = 0
            proc_time = proc_time.gather(2, select_ma[:, None, :].expand(-1, num_opes, -1)).squeeze(2)  # [B, n_opes]
            select_ope = proc_time.argmax(dim=1, keepdim=True)  # [B, 1]
        
        select_job = self.from_ope_to_job(select_ope.squeeze(1), state).unsqueeze(1).long()    # [B, 1]
        return select_ope, select_job
        

    def _select_fifo_ma(self, state, select_job):
        '''
        among multiple eligible machines, select one randomly
        
        Input:
            state:
            select_job: [B, 1]
        Output:
            select_ma: [B, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        batch_idxes = state.batch_idxes
        
        mask, _ = self._get_mask_ope_ma(state)    # [B, n_jobs, n_mas]
        avail_mask_mas = mask.gather(1, select_job[:, :, None].expand(-1, -1, num_mas)).squeeze(1)  # [B, n_mas]
        avail_mas = torch.where(avail_mask_mas == True, 0., -math.inf)  # [B, n_mas]
        avail_ma_probs = F.softmax(avail_mas, dim=1)    # [B, n_mas]
        
        
        while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
            select_ma = avail_ma_probs.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1) # [B, 1]
            ma_prob = avail_ma_probs.gather(1, select_ma) # [B, 1]
            non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
            non_finish_batch[batch_idxes] = True
            finish_batch = torch.where(non_finish_batch == True, False, True)
            ma_prob[finish_batch] = 1  # do not backprob finished episodes
            if (ma_prob != 0).all():
                break
        return select_ma
    
    def _select_fifo_job(self, state):
        '''
        fifo: eligible jobs are processed as fast as possible
        
        when there are multiple eligible jobs, we select one randomly
        
        Input:
            state:
        Output:
            select_job: [B, 1]
            select_ope: [B, 1]
        '''
        batch_size = state.ope_ma_adj_batch.size(0)
        batch_idxes = state.batch_idxes
        
        # === select one job with uniform distribution among availble jobs ===
        mask, _ = self._get_mask_ope_ma(state)    # [B, n_jobs, n_mas]
        avail_jobs = torch.where(mask.sum(dim=2) > 0, 0., -math.inf)  # [B, n_jobs]
        avail_job_probs = F.softmax(avail_jobs, dim=1)  # [B, n_jobs]
        
        while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
            select_job = avail_job_probs.reshape(batch_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, 1) # [B, 1]
            job_prob = avail_job_probs.gather(1, select_job) # [B, 1]
            non_finish_batch = torch.full(size=(batch_size, 1), dtype=torch.bool, fill_value=False)
            non_finish_batch[batch_idxes] = True
            finish_batch = torch.where(non_finish_batch == True, False, True)
            job_prob[finish_batch] = 1  # do not backprob finished episodes
            if (job_prob != 0).all():
                break
        
        # === transform select_job to select_ope ===
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        select_ope = ope_step_batch.gather(1, select_job)   # [B, 1]
        
        return select_job, select_ope
    
    def _select_nearest_veh(self, state, select_ope, select_ma, select_job):
        '''
        Input:
            state:
            select_ope: [B, 1]
            select_ma: [B, 1]
            select_job: [B, 1]
        Output:
            results = {
                'veh_id': [B, 1]
                'trans_time': [B, 1]
            }
            
        '''
        trans_times_batch = state.trans_times_batch   # [B, n_mas, n_mas]
        veh_loc_batch = state.veh_loc_batch # [B, n_vehs]
        prev_ope_locs_batch = state.prev_ope_locs_batch # [B, n_jobs]
        batch_size = trans_times_batch.size(0)
        
        elig_vehs = ~state.mask_veh_procing_batch   # [B, n_vehs]
        
        prev_ope_locs = prev_ope_locs_batch.gather(1, select_job)   # [B, 1]
        
        
        # min_trans_time = torch.ones(size=(batch_size,)) * 1000000
        results = {
            'veh_id': torch.zeros(size=(batch_size, 1)),
            'trans_time': torch.zeros(size=(batch_size, 1)),
        }
        for b in range(batch_size):
            elig_veh_ids = torch.where(elig_vehs[b, :] == True)

            veh_locs = veh_loc_batch[b, elig_veh_ids[0]]    # [elig_n_vehs]
            tmp_prev_ope_locs = prev_ope_locs[b].expand(veh_locs.size(0))
            tmp_select_ma = select_ma[b].expand(veh_locs.size(0))
            
            empty_trans = trans_times_batch[b, veh_locs, tmp_prev_ope_locs]
            travel_trans = trans_times_batch[b, tmp_prev_ope_locs, tmp_select_ma]
            trans_time = empty_trans + travel_trans
            min_value, min_idx = trans_time.min(dim=0, keepdim=True)
            results['veh_id'][b] = elig_veh_ids[0][min_idx]
            results['trans_time'][b] = min_value
        # print(f'results:{results}')
        return results
    
    def _select_OMPair_on_ProcTime(self, state, proc_crit='short'):
        '''
        Input:
            prec_cirt: 
                'short': shortest processing time
                'long': longest processing time
        Ouput:
            ope: [B, 1]
            ma: [B, 1]
            job: [B, 1]
        '''
        batch_size, num_opes, num_mas = state.ope_ma_adj_batch.size()
        
        # : get eligible O-M pairs
        mask, mask_ope_ma = self._get_mask_ope_ma(state)
        
        # : find shortest processing time O-M pair
        proc_time = deepcopy(state.proc_times_batch) # [B, n_opes, n_mas]
        if proc_crit == 'short':
            proc_time[torch.where(mask_ope_ma == False)] = 1000
            proc_time_resh = proc_time.reshape(batch_size, -1) # [B, n_opes * n_mas]
            OM_idx = proc_time_resh.argmin(dim=1, keepdim=True)  # [B, 1]
        elif proc_crit == 'long':
            proc_time[torch.where(mask_ope_ma == False)] = 0
            proc_time_resh = proc_time.reshape(batch_size, -1) # [B, n_opes * n_mas]
            OM_idx = proc_time_resh.argmax(dim=1, keepdim=True)  # [B, 1]
        else:
            raise Exception('implement this!')
        # : reformulate operation and machine index from O-M idx
        num_mas_torch = torch.ones(size=(batch_size, 1)) * num_mas
        ma = torch.remainder(OM_idx, num_mas_torch).long()  # [B, 1]
        ope = torch.div(OM_idx, num_mas_torch).floor().long()   # [B, 1]
        job = self.from_ope_to_job(ope.squeeze(1), state).unsqueeze(1).long()    # [B, 1]
        # print(f'ope:{ope} | ma:{ma} | job:{job}')
        return ope, ma, job
        
    def _get_mask_ope_ma(self, state):
        '''
        Output:
            mask: [B, n_jobs, n_mas]
            mask_ope_ma: [B, n_opes, n_mas]
        '''
        batch_idxes = state.batch_idxes
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
        dummy_shape = torch.zeros(size=(len(batch_idxes), self.num_jobs, self.num_mas))
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
        mask_ope_step = torch.full(size=(self.batch_size, num_opes), dtype=torch.bool, fill_value=False) 
        tmp_batch_idxes = batch_idxes.unsqueeze(-1).repeat(1, num_jobs) # [B, n_jobs]
        mask_ope_step[tmp_batch_idxes, ope_step_batch] = True
        # for batch in batch_idxes:
        #     mask_ope_step[batch, ope_step_batch[batch]] = True  # [B, n_opes]
        
        # : mask jobs that have no available machine and are processing
        mask_job = torch.where(mask.sum(dim=-1) > torch.zeros(size=(self.batch_size, self.num_jobs)), True, False)  # [B, n_jobs]
        mask_ope_by_job = mask_job.gather(1, opes_appertain_batch)
        # : expand the mask_job to operation size, and then aggregate the operation order mask and job mask
        # mask_ope_by_job = torch.full(size=(self.batch_size, num_opes), dtype=torch.bool, fill_value=False)  # [B, n_opes]
        # for batch in batch_idxes:
        #     n_remain_opes = num_opes - sum(state.nums_ope_batch[batch])   # scalar
        #     padd_ = torch.full(size=(n_remain_opes,), dtype=torch.bool, fill_value=False)
        #     mask_ope_by_job[batch] = torch.cat([torch.repeat_interleave(mask_job[batch], state.nums_ope_batch[batch]), padd_], dim=-1)
        
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

    def from_ope_to_job(self, select_ope, state):
        '''
        Input:
            select_ope: [B,]: selected operation index
        Output:
            selecct_job: [B,]
        '''
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, state.end_ope_biases_batch, state.ope_step_batch)  # [B, n_jobs]
        select_job = torch.where(ope_step_batch == select_ope[:, None].expand(-1, self.num_jobs))[1] # [B,]
        return select_job