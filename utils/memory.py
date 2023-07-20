import numpy as np
import torch

from env.common_func import get_mask_ope_ma, generate_trans_mat



class ReplayBuffer():
    def __init__(self, 
                 device, max_size=int(1e5)):
        self.device = device
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.storage = dict()
        # self.storage['ope_ma_adj'] = np.zeros((max_size, num_opes, num_mas))
        # self.storage['ope_pre_adj'] = np.zeros((max_size, num_opes, num_opes))
        # self.storage['ope_sub_adj'] = np.zeros((max_size, num_opes, num_opes))
        # self.storage['raw_opes'] = np.zeros((max_size, num_opes, in_size_ope))
        # self.storage['raw_mas'] = np.zeros((max_size, num_mas, in_size_ma))
        # self.storage['proc_time'] = np.zeros((max_size, num_opes, num_mas))
        # self.storage['nums_ope'] = np.zeros((max_size,))
        # self.storage['jobs_gather'] = np.zeros((max_size, num_jobs, out_size_ope))
        # self.storage['eligible'] = np.zeros((max_size, num_jobs, num_mas))
        # self.storage['rewards'] = np.zeros((max_size,))
        # self.storage['is_terminals'] = np.zeros((max_size,))
        # self.storage['logprobs'] = np.zeros((max_size,))
        # self.storage['action_indexes'] = np.zeros((max_size,))
        
        self.storage['ope_ma_adj'] = []
        self.storage['ope_pre_adj'] = []
        self.storage['ope_sub_adj'] = []
        self.storage['raw_opes'] = []
        self.storage['raw_mas'] = []
        self.storage['raw_vehs'] = []
        self.storage['proc_time'] = []
        self.storage['trans_time'] = []
        self.storage['nums_opes'] = []
        self.storage['jobs_gather'] = []
        self.storage['ope_step_batch'] = []
        self.storage['eligible'] = []
        self.storage['eligible_ope_ma'] = []
        self.storage['rewards'] = []
        self.storage['is_terminals'] = []
        self.storage['logprobs'] = []
        self.storage['action_indexes'] = []
        self.storage['action'] = []
        self.storage['batch_idxes'] = []
    
    def add_action_info(self, action, action_idx, logprob):
        '''
        Input:
            action: np, [B, 4]
            action_indexes: np, [B,]
            logprobs: np, [B,]
        '''
        self.storage['action'].append(action)
        self.storage['action_indexes'].append(action_idx)
        self.storage['logprobs'].append(logprob)
        
    
           
    def add_state_info(self, state):
        '''
        Input:
            ope_ma_adj: np: [B, n_opes, n_mas]
            ope_pre_adj: np: [B, n_opes, n_opes]
            ope_sub_adj: np: [B, n_opes, n_opes]
            raw_opes: np: [B, n_opes, in_size_ope]
            raw_mas: np: [B, n_mas, in_size_ma]
            proc_time: np: [B, n_mas, in_size_ma]
            nums_ope: np: [B,]
            jobs_gather: np: [B, n_jobs, out_size_ope]
            eligible: np: [B, n_jobs, n_mas]
        '''
        raw_opes = state.feat_opes_batch.detach().cpu().numpy() # [B, n_opes, feat_ope_dim]
        raw_mas = state.feat_mas_batch.detach().cpu().numpy()   # [B, n_mas, feat_ma_dim]
        raw_vehs = state.feat_vehs_batch.detach().cpu().numpy() # [B, n_vehs, feat_veh_dim]
        proc_time = state.proc_times_batch.detach().cpu().numpy()   # [B, n_opes, n_mas]
        trans_time = state.trans_times_batch.detach().cpu().numpy() # [B, n_mas, n_mas]
        ope_ma_adj = state.ope_ma_adj_batch.detach().cpu().numpy()  # [B, n_opes, n_mas]
        ope_pre_adj = state.ope_pre_adj_batch.detach().cpu().numpy()
        ope_sub_adj = state.ope_sub_adj_batch.detach().cpu().numpy()
        batch_idxes = state.batch_idxes.detach().cpu().numpy()  # [B,]
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, 
                                     state.end_ope_biases_batch, state.ope_step_batch).detach().cpu().numpy()  # [B, n_jobs]
        nums_opes = state.nums_opes_batch.detach().cpu().numpy()
        
        
        mask, mask_ope_ma = get_mask_ope_ma(state)
        
        
        
        self.storage['ope_ma_adj'].append(ope_ma_adj)
        self.storage['ope_pre_adj'].append(ope_pre_adj)
        self.storage['ope_sub_adj'].append(ope_sub_adj)
        self.storage['raw_opes'].append(raw_opes)
        self.storage['raw_mas'].append(raw_mas)
        self.storage['raw_vehs'].append(raw_vehs)
        self.storage['proc_time'].append(proc_time)
        self.storage['trans_time'].append(trans_time)
        self.storage['eligible'].append(mask.cpu().numpy())
        self.storage['eligible_ope_ma'].append(mask_ope_ma.cpu().numpy())
        self.storage['ope_step_batch'].append(ope_step_batch)
        self.storage['nums_opes'].append(nums_opes)
        
        
        
        
    
    def add_reward_info(self, reward, is_terminal, batch_size):
        '''
        Input:
            reward: np: [B,]
            is_terminal: np: [B,]
        '''
        self.storage['rewards'].append(reward)
        self.storage['is_terminals'].append(is_terminal)
        
        self.size = min(self.size + batch_size, self.max_size)
        
    def all_sample(self):
        ope_ma_adj = torch.tensor(np.stack(self.storage['ope_ma_adj']), dtype=torch.long, device=self.device)
        ope_pre_adj = torch.tensor(np.stack(self.storage['ope_pre_adj']), dtype=torch.long, device=self.device)
        ope_sub_adj = torch.tensor(np.stack(self.storage['ope_sub_adj']), dtype=torch.long, device=self.device)
        raw_opes = torch.tensor(np.stack(self.storage['raw_opes']), dtype=torch.float, device=self.device)
        raw_mas = torch.tensor(np.stack(self.storage['raw_mas']), dtype=torch.float, device=self.device)
        raw_vehs = torch.tensor(np.stack(self.storage['raw_vehs']), dtype=torch.float, device=self.device)
        proc_time = torch.tensor(np.stack(self.storage['proc_time']), dtype=torch.float, device=self.device)
        trans_time = torch.tensor(np.stack(self.storage['trans_time']), dtype=torch.float, device=self.device)
        # jobs_gather = torch.tensor(np.stack(self.storage['jobs_gather']), dtype=torch.long, device=self.device)
        ope_step_batch = torch.tensor(np.stack(self.storage['ope_step_batch']), dtype=torch.long, device=self.device)
        eligible = torch.tensor(np.stack(self.storage['eligible']), dtype=torch.bool, device=self.device)
        rewards = torch.tensor(np.stack(self.storage['rewards']), dtype=torch.float, device=self.device)
        is_terminals = torch.tensor(np.stack(self.storage['is_terminals']), dtype=torch.float, device=self.device)
        logprobs = torch.tensor(np.stack(self.storage['logprobs']), dtype=torch.float, device=self.device)
        action_indexes = torch.tensor(np.stack(self.storage['action_indexes']), dtype=torch.long, device=self.device)
        
        
        return (ope_ma_adj, ope_pre_adj, ope_sub_adj,
                raw_opes, raw_mas, raw_vehs, 
                proc_time, trans_time, 
                ope_step_batch, eligible,
                rewards, is_terminals, 
                logprobs, action_indexes)
        
    def ramdom_sample(self, batch_size):
        '''
        Output:
            ope_ma_adj: [B, max_n_opes, n_mas] 
            ope_pre_adj: [B, max_n_opes, max_n_opes] 
            ope_sub_adj: [B, max_n_opes, max_n_opes]
            raw_opes: [B, max_n_opes, in_size_ope] 
            raw_mas: [B, n_mas, in_size_ma] 
            proc_time: [B, max_n_opes, n_mas]
            jobs_gather: [B, n_jobs, out_size_ope] 
            eligible: [B, n_jobs, n_mas, n_vehs] 
            rewards: [B, 1] 
            is_terminals: [B, 1] 
            logprobs: [B, 1] 
            action_indexes: [B, 1]
        '''
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.storage['ope_ma_adj'][ind], dtype=torch.long, device=self.device),
            torch.tensor(self.storage['ope_pre_adj'][ind], dtype=torch.long, device=self.device),
            torch.tensor(self.storage['ope_sub_adj'][ind], dtype=torch.long, device=self.device),
            torch.tensor(self.storage['raw_opes'][ind], dtype=torch.float, device=self.device),
            torch.tensor(self.storage['raw_mas'][ind], dtype=torch.float, device=self.device),
            torch.tensor(self.storage['proc_time'][ind], dtype=torch.float, device=self.device),
            torch.tensor(self.storage['jobs_gather'][ind], dtype=torch.long, device=self.device),
            torch.tensor(self.storage['eligible'][ind], dtype=torch.bool, device=self.device),
            torch.tensor(self.storage['rewards'][ind], dtype=torch.float, device=self.device),
            torch.tensor(self.storage['is_terminals'][ind], dtype=torch.bool, device=self.device),
            torch.tensor(self.storage['logprobs'][ind], dtype=torch.float, device=self.device),
            torch.tensor(self.storage['action_indexes'][ind], dtype=torch.long, device=self.device),
        )
    
    def clear(self):
        del self.storage['ope_ma_adj'][:]
        del self.storage['ope_pre_adj'][:]
        del self.storage['ope_sub_adj'][:]
        del self.storage['raw_opes'][:]
        del self.storage['raw_mas'][:]
        del self.storage['raw_vehs'][:]
        del self.storage['proc_time'][:]
        del self.storage['trans_time'][:]
        del self.storage['nums_opes'][:]
        del self.storage['jobs_gather'][:]
        del self.storage['ope_step_batch'][:]
        del self.storage['eligible'][:]
        del self.storage['rewards'][:]
        del self.storage['is_terminals'][:]
        del self.storage['logprobs'][:]
        del self.storage['action_indexes'][:]
        del self.storage['action'][:]
    
    def __len__(self):
        # return self.size
        return len(self.storage['ope_ma_adj'])


class ReplayBuffer_ppo(ReplayBuffer):
    def __init__(self, device, max_size=int(1e5)):
        super().__init__(device, max_size)
    
    def add_state_info(self, state):
        '''
        Input:
            ope_ma_adj: np: [B, n_opes, n_mas]
            ope_pre_adj: np: [B, n_opes, n_opes]
            ope_sub_adj: np: [B, n_opes, n_opes]
            raw_opes: np: [B, n_opes, in_size_ope]
            raw_mas: np: [B, n_mas, in_size_ma]
            proc_time: np: [B, n_mas, in_size_ma]
            nums_ope: np: [B,]
            jobs_gather: np: [B, n_jobs, out_size_ope]
            eligible: np: [B, n_jobs, n_mas]
        '''
        raw_opes = state.feat_opes_batch.detach().cpu().numpy() # [B, n_opes, feat_ope_dim]
        raw_mas = state.feat_mas_batch.detach().cpu().numpy()   # [B, n_mas, feat_ma_dim]
        raw_vehs = state.feat_vehs_batch.detach().cpu().numpy() # [B, n_vehs, feat_veh_dim]
        proc_time = state.proc_times_batch.detach().cpu().numpy()   # [B, n_opes, n_mas]
        trans_time = state.trans_times_batch # [B, n_mas, n_mas]
        ope_ma_adj = state.ope_ma_adj_batch.detach().cpu().numpy()  # [B, n_opes, n_mas]
        ope_pre_adj = state.ope_pre_adj_batch.detach().cpu().numpy()
        ope_sub_adj = state.ope_sub_adj_batch.detach().cpu().numpy()
        batch_idxes = state.batch_idxes.detach().cpu().numpy()  # [B,]
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch, 
                                     state.end_ope_biases_batch, state.ope_step_batch).detach().cpu().numpy()  # [B, n_jobs]
        nums_opes = state.nums_opes_batch.detach().cpu().numpy()
        
        
        mask, mask_ope_ma = get_mask_ope_ma(state)
        
        trans_mat = generate_trans_mat(
            trans_time, 
            num_vehs=state.mask_veh_procing_batch.size(1), 
            num_mas=state.mask_ma_procing_batch.size(1),
            veh_loc_batch=state.veh_loc_batch,  # [B, n_vehs] 
            prev_ope_locs_batch=state.prev_ope_locs_batch, # [B, n_jobs] 
            elig_vehs=~state.mask_veh_procing_batch   # [B, n_vehs]
        )    # [B, n_jobs, n_mas, n_vehs]
        trans_mat = trans_mat.detach().cpu().numpy()
        
        self.storage['ope_ma_adj'].append(ope_ma_adj)
        self.storage['ope_pre_adj'].append(ope_pre_adj)
        self.storage['ope_sub_adj'].append(ope_sub_adj)
        self.storage['raw_opes'].append(raw_opes)
        self.storage['raw_mas'].append(raw_mas)
        self.storage['raw_vehs'].append(raw_vehs)
        self.storage['proc_time'].append(proc_time)
        self.storage['trans_time'].append(trans_mat)
        self.storage['eligible'].append(mask.cpu().numpy())
        self.storage['eligible_ope_ma'].append(mask_ope_ma.cpu().numpy())
        self.storage['ope_step_batch'].append(ope_step_batch)
        self.storage['nums_opes'].append(nums_opes)
        self.storage['batch_idxes'].append(batch_idxes)
    
    def add_action_info(self, action, logprob):
        '''
        Input:
            action: np, [B, 4]
            action_indexes: np, [B,]
            logprobs: np, [B,]
        '''
        self.storage['action'].append(action)
        self.storage['logprobs'].append(logprob)
    
    def all_sample(self):
        ope_ma_adj = torch.tensor(np.stack(self.storage['ope_ma_adj']), dtype=torch.long, device=self.device)
        ope_pre_adj = torch.tensor(np.stack(self.storage['ope_pre_adj']), dtype=torch.long, device=self.device)
        ope_sub_adj = torch.tensor(np.stack(self.storage['ope_sub_adj']), dtype=torch.long, device=self.device)
        raw_opes = torch.tensor(np.stack(self.storage['raw_opes']), dtype=torch.float, device=self.device)
        raw_mas = torch.tensor(np.stack(self.storage['raw_mas']), dtype=torch.float, device=self.device)
        raw_vehs = torch.tensor(np.stack(self.storage['raw_vehs']), dtype=torch.float, device=self.device)
        proc_time = torch.tensor(np.stack(self.storage['proc_time']), dtype=torch.float, device=self.device)
        trans_time = torch.tensor(np.stack(self.storage['trans_time']), dtype=torch.float, device=self.device)
        # jobs_gather = torch.tensor(np.stack(self.storage['jobs_gather']), dtype=torch.long, device=self.device)
        ope_step_batch = torch.tensor(np.stack(self.storage['ope_step_batch']), dtype=torch.long, device=self.device)
        eligible = torch.tensor(np.stack(self.storage['eligible']), dtype=torch.bool, device=self.device)
        eligible_ope_ma = torch.tensor(np.stack(self.storage['eligible_ope_ma']), dtype=torch.bool, device=self.device)
        
        rewards = torch.tensor(np.stack(self.storage['rewards']), dtype=torch.float, device=self.device)
        is_terminals = torch.tensor(np.stack(self.storage['is_terminals']), dtype=torch.float, device=self.device)
        logprobs = torch.tensor(np.stack(self.storage['logprobs']), dtype=torch.float, device=self.device)
        # action_indexes = torch.tensor(np.stack(self.storage['action_indexes']), dtype=torch.long, device=self.device)
        actions = torch.tensor(np.stack(self.storage['action']), dtype=torch.long, device=self.device)
        batch_idxes = torch.tensor(np.stack(self.storage['batch_idxes']), dtype=torch.long, device=self.device)
        
        return (ope_ma_adj, ope_pre_adj, ope_sub_adj,
                raw_opes, raw_mas, raw_vehs, 
                proc_time, trans_time, 
                ope_step_batch, eligible, eligible_ope_ma,
                rewards, is_terminals, 
                logprobs, actions,
                batch_idxes)
    
    def clear(self):
        del self.storage['ope_ma_adj'][:]
        del self.storage['ope_pre_adj'][:]
        del self.storage['ope_sub_adj'][:]
        del self.storage['raw_opes'][:]
        del self.storage['raw_mas'][:]
        del self.storage['raw_vehs'][:]
        del self.storage['proc_time'][:]
        del self.storage['trans_time'][:]
        del self.storage['nums_opes'][:]
        del self.storage['jobs_gather'][:]
        del self.storage['ope_step_batch'][:]
        del self.storage['eligible'][:]
        del self.storage['eligible_ope_ma'][:]
        del self.storage['rewards'][:]
        del self.storage['is_terminals'][:]
        del self.storage['logprobs'][:]
        del self.storage['action_indexes'][:]
        del self.storage['action'][:]
        del self.storage['ope_step_batch'][:]
        del self.storage['batch_idxes'][:]
        
        