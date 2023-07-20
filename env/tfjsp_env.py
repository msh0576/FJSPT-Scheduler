import sys
import gym
import torch
from copy import deepcopy
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataclasses import dataclass
from env.load_data import load_tfjs
from utils.utils_fjspt import read_json, write_json


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None
    nums_ope_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    feat_vehs_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    trans_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    ma_veh_adj_batch: torch.Tensor = None
    ope_veh_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    mask_veh_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None
    ope_adj_batch: torch.Tensor = None
    dyn_ope_ma_adj_batch: torch.Tensor = None
    dyn_ope_veh_adj_batch: torch.Tensor = None
    
    prev_ope_locs_batch: torch.Tensor = None
    veh_loc_batch: torch.Tensor = None
    allo_ma_batch: torch.Tensor = None
    ope_status: torch.Tensor = None
    
    def update(self, batch_idxes, 
               feat_opes_batch, feat_mas_batch, feat_vehs_batch, 
               proc_times_batch, trans_times_batch, 
               ope_ma_adj_batch, ma_veh_adj_batch, ope_veh_adj_batch,
               prev_ope_locs_batch, veh_loc_batch, allo_ma_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, mask_veh_procing_batch,
               ope_step_batch, time_batch, ope_status, ope_adj_batch,
               dyn_ope_ma_adj_batch, dyn_ope_veh_adj_batch):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_vehs_batch = feat_vehs_batch
        self.proc_times_batch = proc_times_batch
        self.trans_times_batch = trans_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        self.ma_veh_adj_batch = ma_veh_adj_batch
        self.ope_veh_adj_batch = ope_veh_adj_batch
        
        self.prev_ope_locs_batch = prev_ope_locs_batch
        self.veh_loc_batch = veh_loc_batch
        self.allo_ma_batch = allo_ma_batch
        
        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.mask_veh_procing_batch = mask_veh_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time_batch
        self.ope_status = ope_status
        self.ope_adj_batch = ope_adj_batch
        self.dyn_ope_ma_adj_batch = dyn_ope_ma_adj_batch
        self.dyn_ope_veh_adj_batch = dyn_ope_veh_adj_batch

def convert_feat_job_2_ope(feat_job, opes_appertain):
    '''
    Convert job features into operation features (such as dimension)
    Ex)
        feat_job: [B, num_jobs]
        opes_appertain: [B, num_opes]
        
        output: [B, num_opes]
    '''
    return feat_job.gather(1, opes_appertain)


STATUS = 0
N_NEIGH_MA = 1
PROC_TIME = 2
N_UNSCHED_OPE = 3
JOB_COMP_TIME = 4
ESTI_START = 5        
ALLO_MA = 6
N_NEIGH_VEH = 7
        
class TFJSPEnv(gym.Env):
    '''
    Transportable FJSP environment
    '''
    
    
    def __init__(self, case, env_paras, data_source='case', new_job_dict=None):
        '''
        :param data_source: 'case' or 'benckmark'
        :param new_job_dict:
            {
                'new_job_idx': [B, n_jobs], bool, True if the new job index
                'release': [B, n_jobs], scalar value if the new job index, 0 otherwise
            }
        '''
        # === load paras ===
        self.show_mode = env_paras["show_mode"]
        # self.num_jobs = env_paras["num_jobs"]
        # self.num_mas = env_paras["num_mas"]
        # self.num_vehs = env_paras["num_vehs"]
        self.num_jobs = case.num_jobs
        self.num_opes_list = case.num_opes_list
        self.num_mas = case.num_mas
        self.num_vehs = case.num_vehs
        self.proctime_per_ope_max = case.proctime_per_ope_max
        self.transtime_btw_ma_max = case.transtime_btw_ma_max
        if env_paras["meta_rl"] is not None:
            self.batch_size = env_paras["meta_rl"]['minibatch']
        else:
            self.batch_size = env_paras["batch_size"]
        self.paras = env_paras
        self.device = env_paras["device"]
        self.new_job_dict = new_job_dict
        
        # load instance
        num_data = 8    # the number of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        
        # === generate case results ===
        case_results = []
        self.num_opes = 0
        for i in range(self.batch_size):
            case_results.append(case.get_case_for_transport())
            self.num_opes = max(self.num_opes, case.num_opes)
        num_data = len(case_results[0])
        tensors = [[] for _ in range(num_data)]
        for i in range(self.batch_size):
            load_data = load_tfjs(case_results[i], self.num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])
        
        # === generated environment ===
        self.proc_times_batch = torch.stack(tensors[0], dim=0)                # [B, num_opes, num_mas]
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()         # [B, num_opes, num_mas]
        self.ope_veh_adj_batch = torch.ones(size=(self.batch_size, self.num_opes, self.num_vehs)).long() # [B, n_opes, n_vehs]
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)               # [B, num_opes, num_opes]
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)               # [B, num_opes, num_opes]
        self.cal_cumul_adj_batch = torch.stack(tensors[4], dim=0).float()     # [B, num_opes, num_opes]
        self.nums_ope_batch = torch.stack(tensors[5], dim=0)                  # [B, num_jobs,]   
        self.num_ope_biases_batch = torch.stack(tensors[6], dim=0).long()     # [B, num_jobs,]
        self.end_ope_biases_batch = torch.stack(tensors[7], dim=0).long()     # [B, num_jobs,]
        self.opes_appertain_batch = torch.stack(tensors[8], dim=0).long()     # [B, num_opes,]
        self.trans_times_batch = torch.stack(tensors[9], dim=0)               # [B, num_mas, num_mas]
        self.ma_veh_adj_batch = torch.stack(tensors[10], dim=0).long()        # [B, num_mas, num_vehs]
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)
        self.dyn_ope_ma_adj_batch = deepcopy(self.ope_ma_adj_batch)   # [B, n_opes, n_mas]
        self.dyn_ope_veh_adj_batch = deepcopy(self.ope_veh_adj_batch)
        self.ope_status = torch.full(size=(self.batch_size, self.num_opes), dtype=torch.bool, fill_value=False)
        # print(f'---------------- self.nums_opes: {self.nums_opes} ----------------')
        
        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(size=(self.batch_size,))  # Current time of the environment
        self.N = torch.zeros(size=(self.batch_size,)).int()   # Count scheduled operations
        self.ope_step_batch = deepcopy(self.num_ope_biases_batch)   # the id of the current operation (be waiting to be processed) of each job
        self.veh_loc_batch = torch.zeros(size=(self.batch_size, self.num_vehs), dtype=torch.long) # current location of vehicles (machine 0 at initial)
        self.prev_ope_locs_batch = torch.zeros(size=(self.batch_size, self.num_jobs), dtype=torch.long)  # previous operation location (machines) for each job
        self.allo_ma_batch = torch.zeros(size=(self.batch_size, self.num_opes), dtype=torch.long) # allocated machine if the operation is scheduled
        aver_trans_time = self.trans_times_batch.flatten(1).mean()
        self.ope_trans_time_batch = torch.ones(size=(self.batch_size, self.num_opes)) * aver_trans_time
        # : operation adjacent
        self.ope_adj_batch = deepcopy(self.ope_pre_adj_batch)  # current operation neighbor matrix
        
        
        
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Number of neighboring vehicles
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                estimation of Start time
                allocated machine (curret job location)
                release time
            ma:
                Number of neighboring operations
                Available time
                Utilization
            veh:
                Number of neighboring operations
                Available time
                Utilization
                Current location
                transportation time of the scheduled operation to the scheduled machine at a time 
        '''
        
        # === generate raw feature vectors ===
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], self.num_mas))
        feat_vehs_batch = torch.zeros(size=(self.batch_size, self.paras["veh_feat_dim"], self.num_vehs))
        
        # == operation features ==
        # ope_ma_adj_padd = self.padd_on_nums_ope(self.ope_ma_adj, self.max_num_opes, dim_=0)
        feat_opes_batch[:, N_NEIGH_MA, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, PROC_TIME, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9) # for each operation, it has average process time for compatible machines
        feat_opes_batch[:, N_UNSCHED_OPE, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)  
        feat_opes_batch[:, ESTI_START, :] = torch.bmm(feat_opes_batch[:, PROC_TIME, :].unsqueeze(1), self.cal_cumul_adj_batch).squeeze()  # start time of each operation by using the cumulative average process time along the job path
        end_time_batch = (feat_opes_batch[:, ESTI_START, :] + feat_opes_batch[:, PROC_TIME, :]).gather(1, self.end_ope_biases_batch)   # [num_jobs,]: for each job, its estimated completion time
        feat_opes_batch[:, JOB_COMP_TIME, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_opes_batch[:, ALLO_MA, :] = torch.zeros(size=(self.batch_size, self.num_opes))
        feat_opes_batch[:, N_NEIGH_VEH, :] = torch.count_nonzero(self.ope_veh_adj_batch, dim=2)
        # feat_opes_batch[:, RELEASE, :] = torch.zeros(size=(self.batch_size, self.num_jobs))
        
        
        # : minimum completion time of operations
        non_zero_proc_times_batch = torch.where(self.proc_times_batch == 0, 1000, self.proc_times_batch)
        self.min_proc_times_batch = torch.min(non_zero_proc_times_batch, dim=2)[0]   # [B, n_opes]
        self.min_esti_start = torch.bmm(self.min_proc_times_batch.unsqueeze(1), self.cal_cumul_adj_batch).squeeze()   # [B, n_opes]
        self.min_esti_end = self.min_esti_start + self.min_proc_times_batch   # [B, n_opes]
        min_esti_end_of_job = self.min_esti_end.gather(1, self.end_ope_biases_batch)    # [B, n_jobs]
        self.prev_esti_makespan = torch.max(min_esti_end_of_job, dim=1)[0]    # [B, ]

        # == machine features ==
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        
        # == vehicle features ==
        # feat_vehs_batch[:, 0, :] = torch.count_nonzero(self.ma_veh_adj_batch, dim=1)
        feat_vehs_batch[:, 0, :] = torch.count_nonzero(self.ope_veh_adj_batch, dim=1)
        
        
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.feat_vehs_batch = feat_vehs_batch
        
        # === masks of current status ===
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.mask_veh_procing_batch = torch.full(size=(self.batch_size, self.num_vehs), dtype=torch.bool, fill_value=False)
        
        # === job insertion initial setup ===
        if self.new_job_dict:
            n_newJobs = self.new_job_dict['new_job_idx'].count_nonzero(dim=1)[0].item()
            tmp_end_time_batch = deepcopy(end_time_batch)
            tmp_end_time_batch[self.new_job_dict['new_job_idx']] = 0
            max_job_comp_times = tmp_end_time_batch.max(dim=1)[0].tolist()  # [B,]
            rand_release_time = [[random.randint(1, int(max_job_comp_time)) for _ in range(n_newJobs)] \
                for max_job_comp_time in max_job_comp_times]
            rand_release_time = torch.tensor(rand_release_time) # [B, n_newJobs]
            self.new_job_dict['release'][self.new_job_dict['new_job_idx']] = rand_release_time.reshape(-1)
            
            self.mask_job_procing_batch[self.new_job_dict['new_job_idx']] = True
        
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
            Allocated vehicles
            Transportation end time
        '''
        self.sched_ope_dim = 6
        self.sched_opes_batch = torch.zeros(size=(self.batch_size, self.num_opes, self.sched_ope_dim))
        self.sched_opes_batch[:, :, 2] = feat_opes_batch[:, ESTI_START, :]
        self.sched_opes_batch[:, :, 3] = feat_opes_batch[:, ESTI_START, :] + feat_opes_batch[:, PROC_TIME, :]
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
            using vehicle_id for each machine
        '''
        self.sched_mas_dim = 5
        self.sched_mas_batch = torch.zeros(size=(self.batch_size, self.num_mas, self.sched_mas_dim))
        self.sched_mas_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
        self.sched_mas_batch[:, :, 4] = torch.ones(size=(self.batch_size, self.num_mas)) * -1.
        '''
        Partial Schedule (state) of vehicles, dynamic
            idle
            available_time
            utilization_time
            ma_id
        '''
        self.sched_veh_dim = 4
        self.sched_vehs_batch = torch.zeros(size=(self.batch_size, self.num_vehs, self.sched_veh_dim))
        self.sched_vehs_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_vehs))
        
        self.makespan_batch = torch.max(self.feat_opes_batch[:, JOB_COMP_TIME, :], dim=1)[0]    # [B,]
        self.done_batch = self.mask_job_finish_batch.all(dim=1) # [B, ]
        
        self.state = EnvState(
            batch_idxes=self.batch_idxes,
            feat_opes_batch=self.feat_opes_batch, 
            feat_mas_batch=self.feat_mas_batch, 
            feat_vehs_batch=self.feat_vehs_batch,
            proc_times_batch=self.proc_times_batch, 
            trans_times_batch=self.trans_times_batch,
            ope_ma_adj_batch=self.ope_ma_adj_batch, 
            ope_veh_adj_batch=self.ope_veh_adj_batch,
            ope_pre_adj_batch=self.ope_pre_adj_batch, 
            ope_sub_adj_batch=self.ope_sub_adj_batch, 
            ma_veh_adj_batch=self.ma_veh_adj_batch,
            prev_ope_locs_batch=self.prev_ope_locs_batch,
            allo_ma_batch=self.allo_ma_batch,
            veh_loc_batch=self.veh_loc_batch,
            mask_job_procing_batch=self.mask_job_procing_batch,
            mask_job_finish_batch=self.mask_job_finish_batch,
            mask_ma_procing_batch=self.mask_ma_procing_batch, 
            mask_veh_procing_batch=self.mask_veh_procing_batch, 
            opes_appertain_batch=self.opes_appertain_batch,
            ope_step_batch=self.ope_step_batch, 
            end_ope_biases_batch=self.end_ope_biases_batch,
            time_batch=self.time, 
            nums_ope_batch=self.nums_ope_batch,
            nums_opes_batch=self.nums_opes,
            ope_status=self.ope_status,
            ope_adj_batch=self.ope_adj_batch,
            dyn_ope_ma_adj_batch=self.dyn_ope_ma_adj_batch,
            dyn_ope_veh_adj_batch=self.dyn_ope_veh_adj_batch
        )
        
        # === Save initial data for reset ===
        self.old_proc_times_batch = deepcopy(self.proc_times_batch)
        self.old_trans_times_batch = deepcopy(self.trans_times_batch)
        self.old_ope_ma_adj_batch = deepcopy(self.ope_ma_adj_batch)
        self.old_ope_veh_adj_batch = deepcopy(self.ope_veh_adj_batch)
        self.old_cal_cumul_adj_batch = deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = deepcopy(self.feat_mas_batch)
        self.old_feat_vehs_batch = deepcopy(self.feat_vehs_batch)
        self.old_state = deepcopy(self.state)
        self.old_ope_status = deepcopy(self.ope_status)
        self.old_ope_adj_batch = deepcopy(self.ope_adj_batch)
        self.old_dyn_ope_ma_adj_batch = deepcopy(self.dyn_ope_ma_adj_batch)
        self.old_dyn_ope_veh_adj_batch = deepcopy(self.dyn_ope_veh_adj_batch)
    
    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = deepcopy(self.old_proc_times_batch)
        self.trans_times_batch = deepcopy(self.old_trans_times_batch)
        self.ope_ma_adj_batch = deepcopy(self.old_ope_ma_adj_batch)
        self.ope_veh_adj_batch = deepcopy(self.old_ope_veh_adj_batch)
        self.cal_cumul_adj_batch = deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = deepcopy(self.old_feat_mas_batch)
        self.feat_vehs_batch = deepcopy(self.old_feat_vehs_batch)
        self.state = deepcopy(self.old_state)
        self.ope_status = deepcopy(self.old_ope_status)
        self.ope_adj_batch = deepcopy(self.old_ope_adj_batch)
        self.dyn_ope_ma_adj_batch = deepcopy(self.old_dyn_ope_ma_adj_batch)
        self.dyn_ope_veh_adj_batch = deepcopy(self.old_dyn_ope_veh_adj_batch)
        
        # === dynamic variables ===
        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        
        self.veh_loc_batch = torch.zeros(size=(self.batch_size, self.num_vehs), dtype=torch.long) 
        self.prev_ope_locs_batch = torch.zeros(size=(self.batch_size, self.num_jobs), dtype=torch.long)
        self.allo_ma_batch = torch.zeros(size=(self.batch_size, self.num_opes), dtype=torch.long)
        
        
        aver_trans_time = self.trans_times_batch.flatten(1).mean()
        self.ope_trans_time_batch = torch.ones(size=(self.batch_size, self.num_opes)) * aver_trans_time
        # ===
        
        self.ope_step_batch = deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.mask_veh_procing_batch = torch.full(size=(self.batch_size, self.num_vehs), dtype=torch.bool, fill_value=False)
        if self.new_job_dict:
            self.mask_job_procing_batch[self.new_job_dict['new_job_idx']] = True
            
        self.sched_opes_batch = torch.zeros(size=(self.batch_size, self.num_opes, self.sched_ope_dim))
        self.sched_opes_batch[:, :, 2] = self.feat_opes_batch[:, ESTI_START, :]
        self.sched_opes_batch[:, :, 3] = self.feat_opes_batch[:, ESTI_START, :] + self.feat_opes_batch[:, PROC_TIME, :]
        self.sched_mas_batch = torch.zeros(size=(self.batch_size, self.num_mas, self.sched_mas_dim))
        self.sched_mas_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas)) # idle: 1 is True (idle machine)
        self.sched_mas_batch[:, :, 4] = torch.ones(size=(self.batch_size, self.num_mas)) * -1.
        self.sched_vehs_batch = torch.zeros(size=(self.batch_size, self.num_vehs, self.sched_veh_dim))
        self.sched_vehs_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_vehs))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, JOB_COMP_TIME, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        
        # ===
        non_zero_proc_times_batch = torch.where(self.proc_times_batch == 0, 1000, self.proc_times_batch)
        self.min_proc_times_batch = torch.min(non_zero_proc_times_batch, dim=2)[0]   # [B, n_opes]
        self.min_esti_start = torch.bmm(self.min_proc_times_batch.unsqueeze(1), self.cal_cumul_adj_batch).squeeze()   # [B, n_opes]
        self.min_esti_end = self.min_esti_start + self.min_proc_times_batch   # [B, n_opes]
        min_esti_end_of_job = self.min_esti_end.gather(1, self.end_ope_biases_batch)    # [B, n_jobs]
        self.prev_esti_makespan = torch.max(min_esti_end_of_job, dim=1)[0]    # [B, ]
        
        return self.state
    
    def step(self, action, step_reward=None):
        '''
        Input:
            action: [ope, ma, job, veh]: [4, B]
        '''
        ope = action[0, :]
        ma = action[1, :]
        job = action[2, :]
        veh = action[3, :]
        self.N += 1
        self.ope_status[self.batch_idxes, ope] = True
        
        # print(f'------------------------------------')
        # print(f"time:{self.time} | ope:{ope} | ma:{ma} | job:{job} | veh:{veh}")
        
        
        
        # === Update current operation-operation adjacent ===
        self.ope_adj_batch[self.batch_idxes, ope, :] = False
        
        
        # === Update transportation time ===
        prev_ope_loc = self.prev_ope_locs_batch[self.batch_idxes, job].long()  # machine_id (location): [B, ]
        empty_trans_time = self.trans_times_batch[self.batch_idxes, self.veh_loc_batch[self.batch_idxes, veh], prev_ope_loc]    # [B,]
        travel_trans_time = self.trans_times_batch[self.batch_idxes, prev_ope_loc, ma]
        trans_times = empty_trans_time + travel_trans_time   # [B,] 
        
        # === remove unselected O-M arcs of the scheduled operations ===
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.long)
        remain_ope_ma_adj[self.batch_idxes, ma] = 1
        self.ope_ma_adj_batch[self.batch_idxes, ope] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch
        # === remove unselected O-V arcs of the scheduled operations ===
        remain_ope_veh_adj = torch.zeros(size=(self.batch_size, self.num_vehs), dtype=torch.long)
        remain_ope_veh_adj[self.batch_idxes, veh] = 1
        self.ope_veh_adj_batch[self.batch_idxes, ope] = remain_ope_veh_adj[self.batch_idxes, :]
        
        # === remove unselected M-V arcs of the scheduled machine ===
        remain_ma_veh_adj = torch.zeros(size=(self.batch_size, self.num_vehs), dtype=torch.long)
        remain_ma_veh_adj[self.batch_idxes, veh] = 1
        self.ma_veh_adj_batch[self.batch_idxes, ma] = remain_ma_veh_adj[self.batch_idxes, :]  # [n_mas, n_vehs]
        
        # === Update dynamic adjacent matrix ===
        self.dyn_ope_ma_adj_batch, self.dyn_ope_veh_adj_batch = self.get_dyn_adj_mat()   # [B, n_opes, n_mas], [B, n_opes, n_vehs]
        
        
        
        # if veh == 0:
        #     print(f'job_{job.item()} - ope_{ope.item()} from {self.veh_loc_batch[self.batch_idxes, veh].item()} to {prev_ope_loc.item()}, empty_trans_time{empty_trans_time.item()}')
        #     print(f'from {prev_ope_loc.item()} to {ma.item()}, travel_trans_time:{travel_trans_time.item()}')
        
        
        # === Update feature vectors of operations : for some O-M arcs, 'Status', 'Number of neighboring machines' and 'Processing time' ===
        proc_times = self.proc_times_batch[self.batch_idxes, ope, ma]
        self.feat_opes_batch[self.batch_idxes, :3, ope] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1) # [B, 3]
        
        
        # 이건 왜하는 거지?: 현재 oper 의 start estimation time은 바로 직전 operation 의 real start time + real process time 이다. 그 전의 operation 계산은 생략
        last_ope = torch.where(ope - 1 < self.num_ope_biases_batch[self.batch_idxes, job], self.num_opes - 1, ope - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_ope, :] = 0

        # = Update 'Number of unscheduled operations in the job' =
        start_ope = self.num_ope_biases_batch[self.batch_idxes, job]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, job]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], N_UNSCHED_OPE, start_ope[i]:end_ope[i]+1] -= 1 # unscheduled operation in this job decreases by 1
            self.feat_opes_batch[self.batch_idxes[i], ALLO_MA, start_ope[i]:end_ope[i]+1] = ma[i].float()
            # print(f'self.feat_opes_batch[self.batch_idxes[i], N_UNSCHED_OPE]:{self.feat_opes_batch[self.batch_idxes[i], N_UNSCHED_OPE, start_ope[i]:end_ope[i]+1]}')
            
        
        # = Update 'Start time' and 'Job completion time' =
        self.feat_opes_batch[self.batch_idxes, ESTI_START, ope] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, STATUS, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, PROC_TIME, :]   # [B, n_opes]
        start_times = self.feat_opes_batch[self.batch_idxes, ESTI_START, :] * is_scheduled    # real start time of scheduled ope
        
        # : set real transportation time
        self.ope_trans_time_batch[self.batch_idxes, ope] = trans_times
        mean_trans_time = self.ope_trans_time_batch[self.batch_idxes, :]    # [B, n_opes]
        # :
        un_scheduled = 1 - is_scheduled
        estimate_times = torch.bmm((start_times + mean_trans_time + mean_proc_time).unsqueeze(1), 
                                  self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() * un_scheduled    # [B, n_opes,]: estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, ESTI_START, :] = start_times + estimate_times
        end_times_batch = (self.feat_opes_batch[self.batch_idxes, ESTI_START, :] + mean_trans_time + self.feat_opes_batch[self.batch_idxes, PROC_TIME, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :]) # [B, n_jobs]
        self.feat_opes_batch[self.batch_idxes, JOB_COMP_TIME, :] = convert_feat_job_2_ope(end_times_batch, self.opes_appertain_batch[self.batch_idxes, :]).squeeze()
        
        # : Update the number of neighbor nodes
        self.feat_opes_batch[self.batch_idxes, N_NEIGH_MA, :] = torch.count_nonzero(self.dyn_ope_ma_adj_batch[self.batch_idxes, :, :], dim=2).float()    # [B, n_opes]
        self.feat_opes_batch[self.batch_idxes, N_NEIGH_VEH, :] = torch.count_nonzero(self.dyn_ope_veh_adj_batch[self.batch_idxes, :, :], dim=2).float()    # [B, n_opes]
        
        # === Update partial schedule (state) ===
        self.sched_opes_batch[self.batch_idxes, ope, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), ma), dim=1)   # Status, Allocated machines
        self.sched_opes_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, ESTI_START, :]    # Start time
        self.sched_opes_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, ESTI_START, :] + mean_trans_time + mean_proc_time # End time
        self.sched_opes_batch[self.batch_idxes, ope, 4] = veh.float() # allocated vehicle
        self.sched_opes_batch[self.batch_idxes, ope, 5] = self.feat_opes_batch[self.batch_idxes, ESTI_START, ope] + trans_times    # transportation end time
        
        self.sched_mas_batch[self.batch_idxes, ma, 0] = torch.zeros(self.batch_idxes.size(0))  # idle
        self.sched_mas_batch[self.batch_idxes, ma, 1] = self.time[self.batch_idxes] + trans_times + proc_times   # available_time
        self.sched_mas_batch[self.batch_idxes, ma, 2] += proc_times  # utilization_time
        self.sched_mas_batch[self.batch_idxes, ma, 3] = job.float() # id_ope
        self.sched_mas_batch[self.batch_idxes, ma, 4] = veh.float() # using vehicle on the machine
        
        self.sched_vehs_batch[self.batch_idxes, veh, 0] = torch.zeros(self.batch_idxes.size(0))
        self.sched_vehs_batch[self.batch_idxes, veh, 1] = self.time[self.batch_idxes] + trans_times  # available_time
        self.sched_vehs_batch[self.batch_idxes, veh, 2] += trans_times   # utilization_time
        self.sched_vehs_batch[self.batch_idxes, veh, 3] = ma.float()    # ma_id
        
        # === Update feature vectors of machines ===
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float() # [B, n_mas]
        self.feat_mas_batch[self.batch_idxes, 1, ma] = self.time[self.batch_idxes] + trans_times + proc_times
        utilize = self.sched_mas_batch[self.batch_idxes, :, 2]  # [B, n_mas]
        cur_time = self.time[self.batch_idxes, None].expand_as(utilize) # [B, n_mas]
        utilize = torch.minimum(utilize, cur_time)
        utilize = utilize.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utilize
        
        
        # === Update feature vectors of vehicles ===
        self.feat_vehs_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_veh_adj_batch[self.batch_idxes, :, :], dim=1).float()  # the number of connected operations on a vehicle
        self.feat_vehs_batch[self.batch_idxes, 1, veh] = self.time[self.batch_idxes] + trans_times # available time
        utilize = self.sched_vehs_batch[self.batch_idxes, :, 2]     # [n_vehs,]
        cur_time = self.time[self.batch_idxes, None].expand_as(utilize)
        utilize = torch.minimum(utilize, cur_time)
        utilize = utilize.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_vehs_batch[self.batch_idxes, 2, :] = utilize
        self.feat_vehs_batch[self.batch_idxes, 3, veh] = ma.float() # vehicle location (machine_id) to go
        self.feat_vehs_batch[self.batch_idxes, 4, veh] = trans_times # transportation time on the scheduled machine
        
        # === Update other variable according to actions ===
        self.ope_step_batch[self.batch_idxes, job] += 1
        self.mask_job_procing_batch[self.batch_idxes, job] = True
        self.mask_ma_procing_batch[self.batch_idxes, ma] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch+1, True, self.mask_job_finish_batch)
        self.mask_veh_procing_batch[self.batch_idxes, veh] = True
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()
        
        if step_reward is not None:
            if step_reward == 'version1':
                max = torch.max(self.feat_opes_batch[self.batch_idxes, JOB_COMP_TIME, :], dim=1)[0] # scalar: job completion time
                self.reward_batch = self.makespan_batch - max   # scalar
                self.makespan_batch = max
            elif step_reward == 'version2':
                self.reward_batch = self.get_step_reward(ope, proc_times)
            else:
                raise Exception('step_reward error!')
        else:
            if self.done:
                self.reward_batch = -self.get_makespan()
            else:
                self.reward_batch = torch.zeros(size=(self.batch_size,), dtype=torch.float)
            
        # === Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time ===
        flag_trans_2_next_time = self.if_no_eligible()  # [B,]
        while ((flag_trans_2_next_time==0) & (~self.done_batch)).any(dim=0):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()
        
        
        # === update previous operation location for the next step() ===
        self.prev_ope_locs_batch[self.batch_idxes, job] = ma
        self.veh_loc_batch[self.batch_idxes, veh] = ma
        self.allo_ma_batch[self.batch_idxes, ope] = ma
        
        # === Update state of the environment ===
        self.state.update(
                        self.batch_idxes,
                        self.feat_opes_batch, self.feat_mas_batch, self.feat_vehs_batch, 
                        self.proc_times_batch, self.trans_times_batch, 
                        self.ope_ma_adj_batch, self.ma_veh_adj_batch, self.ope_veh_adj_batch,
                        self.prev_ope_locs_batch, self.veh_loc_batch, self.allo_ma_batch,
                        self.mask_job_procing_batch, self.mask_job_finish_batch,
                        self.mask_ma_procing_batch, self.mask_veh_procing_batch, self.ope_step_batch, self.time,
                        self.ope_status, self.ope_adj_batch,
                        self.dyn_ope_ma_adj_batch, self.dyn_ope_veh_adj_batch)        
        return self.state, self.reward_batch, self.done_batch
    
    def get_makespan(self):
        # === makespane: only at finished time step ===
        mask_non_finish = (self.N+1) <= self.nums_opes  # if all operations are completed, then it will be False : [B,]
        makespan = torch.zeros(size=(self.batch_size,))
        makespan[~mask_non_finish] = self.time[~mask_non_finish]
        
        return makespan
    
    def get_step_reward(self, ope, proc_times):
        # === Estimate minimum job completion time ===
        is_scheduled = self.feat_opes_batch[self.batch_idxes, STATUS, :]
        un_scheduled = 1 - is_scheduled
        self.min_esti_start[self.batch_idxes, ope] = self.time[self.batch_idxes]    
        min_esti_start_sched = self.min_esti_start[self.batch_idxes, :] * is_scheduled    # [B, n_opes]
        self.min_proc_times_batch[self.batch_idxes, ope] = proc_times
        min_esti_start_unsched = torch.bmm((min_esti_start_sched + self.min_proc_times_batch).unsqueeze(1), self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() * un_scheduled
        min_esti_start = min_esti_start_sched + min_esti_start_unsched
        self.min_esti_end = min_esti_start + self.min_proc_times_batch  # [B, n_opes]
        
        # === estimate makespan of job: not consider transportation time ===
        min_esti_end_of_job = self.min_esti_end.gather(1, self.end_ope_biases_batch)    # [B, n_jobs]
        esti_makespan = torch.max(min_esti_end_of_job, dim=1)[0]    # [B, ]
        
        reward = esti_makespan - self.prev_esti_makespan
        # reward = self.prev_esti_makespan - esti_makespan
        self.prev_esti_makespan = esti_makespan
        return reward
        
    
    def if_no_eligible(self):
        '''
        Check if there are still O-M-V pairs to be processed
        Output:
            flag_tarns_2_next_time: [B,]
        '''
        dummy_shape = torch.zeros(size=(self.batch_size, self.num_jobs, self.num_mas, self.num_vehs))
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch, self.end_ope_biases_batch, self.ope_step_batch) # [B, n_jobs]
        
        # === operation-machine eligible ===
        ma_eligible = ~self.mask_ma_procing_batch[:, None, :].expand(-1, self.num_jobs, -1)       # [B, n_jobs, n_mas]
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[..., None].expand(-1, -1, self.num_mas)
        ope_ma_adj = self.ope_ma_adj_batch.gather(1, ope_step_batch[..., None].expand(-1, -1, self.num_mas)).bool()   # [B, n_jobs, n_mas]
        ope_ma_eligible = job_eligible & ma_eligible & ope_ma_adj   # [B, n_jobs, n_mas]
        # === operation-machine-vehicle eligible ===
        veh_eligible = ~self.mask_veh_procing_batch[:, None, None, :].expand_as(dummy_shape)
        ope_ma_veh_eligible = ope_ma_eligible[..., None].expand_as(dummy_shape) & veh_eligible

        # If there is no eligible O-M arc, 'flag_trans_2_next_time' will be 0
        # if it is 0, then repeatedly call next_time() until it will not be 0
        flag_trans_2_next_time = torch.where(ope_ma_veh_eligible == True, 1.0, 0.0)
        flag_trans_2_next_time = torch.sum(flag_trans_2_next_time, dim=[1, 2, 3])
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time
    
    def get_dyn_adj_mat(self):
        '''
        Output:
            dyn_ope_ma_adj: [B, n_opes, n_mas]
            dyn_ope_veh_adj: [B, n_opes, n_vehs]
        '''
        # === operation-machine eligible ===
        ope_eligible = ~self.ope_status[..., None].expand(-1, -1, self.num_mas)       # [B, n_opes, n_mas]
        ma_eligible = ~self.mask_ma_procing_batch[:, None, :].expand(-1, self.num_opes, -1)       # [B, n_opes, n_mas]
        ope_ma_adj = self.ope_ma_adj_batch.bool()   # [B, n_opes, n_mas]
        dyn_ope_ma_adj = ope_eligible & ma_eligible & ope_ma_adj   # [B, n_opes, n_mas]
        # === operation-vehicle eligible ===
        ope_eligible = ~self.ope_status[..., None].expand(-1, -1, self.num_vehs)       # [B, n_opes, n_vehs]
        veh_eligible = ~self.mask_veh_procing_batch[:, None, :].expand(-1, self.num_opes, -1)   # [B, n_opes, n_vehs]
        # ope_veh_adj = self.ope_veh_adj_batch.bool() # [B, n_opes, n_vehs]
        dyn_ope_veh_adj = ope_eligible & veh_eligible    # [B, n_opes, n_vehs]
        return dyn_ope_ma_adj.long(), dyn_ope_veh_adj.long()
        

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        Input:
            flag_trans_2_next_time: [B,]
        '''
        # === check whether need to transit ===
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done)
        # === available_time of machines ===
        avail_time = self.sched_mas_batch[:, :, 1]   # [B, n_mas]
        # === remain elements of available_time, greater than current time ===
        job_comp_time_max = torch.max(self.feat_opes_batch[:, JOB_COMP_TIME, :])    # [scalar] : this is an absurdly high value
        remain_avail_time = torch.where(avail_time > self.time[:, None], avail_time, job_comp_time_max + 1.0)    # [B, n_mas] : there is no meaning about 'job_comp_time_max'
        # === Return the minimum value of available_time (the time to transit to) ===
        transit_time = torch.min(remain_avail_time, dim=1)[0]   # [B,]
        # === detect the machines that completed (at above time) ===
        comp_mas = torch.where((avail_time == transit_time[:, None]) & (self.sched_mas_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)  # [B, n_mas]
        # === the time for each batch to transit to or stay in ===
        transit_time_on_batch = torch.where(flag_need_trans, transit_time, self.time)
        # === update current time ===
        self.time = transit_time_on_batch
        # print(f'remain_avail_time:{remain_avail_time}')
        # print(f'transit_time_on_batch:{transit_time_on_batch}')
        
        
        # === Update partial schedule (state), variables and feature vectors ===
        sched_mas_batch_trans = self.sched_mas_batch.transpose(1, 2)    # [B, 4(D_sched_mas), n_mas]
        sched_mas_batch_trans[comp_mas, 0] = 1  # 1 = idle (not processing now): [n_mas, 4 (=D_sched_mas))]
        sched_mas_batch_trans[comp_mas, 4] = -1    # -1 = not using vehicles
        self.sched_mas_batch = sched_mas_batch_trans.transpose(1, 2)    # [B, n_mas, D_sched_mas]
        
        utilize = self.sched_mas_batch[:, :, 2] # [B, n_mas]
        cur_time = self.time[:, None].expand_as(utilize)
        utilize = torch.minimum(utilize, cur_time)
        utilize = utilize.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utilize

        jobs = torch.where(comp_mas == True, self.sched_mas_batch[:, :, 3].double(), -1.0).float() # [B, n_mas]: job_idx on the completed machine
        jobs_idx = np.argwhere(jobs.cpu() >= 0).to(self.device)
        jobs_idxes = jobs[jobs_idx[0], jobs_idx[1]].long()  # [B, len of jobs that are on the completed machine]
        batch_idxes = jobs_idx[0]   # batches that have complete machines
        self.mask_job_procing_batch[batch_idxes, jobs_idxes] = False
        self.mask_ma_procing_batch[comp_mas] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1, True, self.mask_job_finish_batch)

        # === find complete vehicles at the changed time ===
        # : veh can be used at the transportation finished time, not until the process finished time.
        avail_time_veh = self.sched_vehs_batch[:, :, 1] # [B, n_vehs]
        comp_vehs = torch.where(avail_time_veh <= self.time[:, None], True, False)  # [B, n_vehs]
        
        # === Update variables of vehicle: mask the vehicle process and set the current location ===
        self.veh_loc_batch = torch.where(self.sched_vehs_batch[:, :, 3] == comp_vehs, self.sched_vehs_batch[:, :, 3], self.veh_loc_batch).long()   # current location of vehicle
        veh_locs = self.sched_vehs_batch[:, :, 0]
        veh_locs[comp_vehs] = 1   # idle
        self.mask_veh_procing_batch[comp_vehs] = False
        # print(f'self.mask_job_procing_batch:{self.mask_job_procing_batch}')
        # print(f'self.mask_ma_procing_batch:{self.mask_ma_procing_batch}')
        # print(f'self.mask_veh_procing_batch:{self.mask_veh_procing_batch}')
        
        # === check new job insertion ===
        if self.new_job_dict is not None:
            self._check_newJobInsert(self.time)

    def _check_newJobInsert(self, time):
        time_resh = time[:, None].expand(-1, self.new_job_dict['release'].size(1))  # [B, n_jobs]
        new_jobs = (self.new_job_dict['release'] <= time_resh) & (self.new_job_dict['release'] > 0) # [B, n_jobs]
        curr_new_jobs = new_jobs & self.new_job_dict['new_job_idx']
        self.mask_job_procing_batch[curr_new_jobs] = False
        self.new_job_dict['new_job_idx'][curr_new_jobs] = False
        

    
    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        veh_gantt_batch = [[[] for _ in range(self.num_vehs)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.sched_opes_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i] # row: opers | col: [status, machine, job_start_time, job_end_time, veh, trans_end_time]
                
                # : ma_gantt_batch[batch_id][ma_id][ope_id, proc_start_time, job_end_time]
                proc_start_time = step[5]
                job_end_time = step[3]
                ma_gantt_batch[batch_id][int(step[1])].append([i, proc_start_time.item(), job_end_time.item()])  
                # : veh_gantt_batch[batch_id][veh_id][ope_id, trans_start_time, trans_end_time, destination machine]
                trans_start_time = step[2]
                trans_end_time = step[5]
                dest_ma = int(step[1])
                veh_gantt_batch[batch_id][int(step[4])].append([i, trans_start_time.item(), trans_end_time.item(), dest_ma])  
        proc_time_batch = self.proc_times_batch     # [B, n_opes, n_mas]
        trans_time_batch = self.trans_times_batch   # [B, n_mas, n_mas]

        # Check whether there are overlaps and correct processing times on the machine, and check transportation time overlaps
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag_trans_time = 0
        flag_veh_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            veh_gantt = veh_gantt_batch[k]
            trans_time = trans_time_batch[k]
            # : processing time overlap
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):   # for all operation on a specific machine
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1
            # : transportation time overlap
            for i in range(self.num_vehs):
                veh_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(veh_gantt[i])):
                    num_alloc_ope = len(veh_gantt[i])
                    if (num_alloc_ope <= 1) or (j == num_alloc_ope-1):
                        break
                    if veh_gantt[i][j][2] > veh_gantt[i][j+1][1]:
                        flag_veh_overlap += 1
                        

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.sched_opes_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1
        
        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.sched_opes_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add
        
        # print(f'flag_ma_overlap:{flag_ma_overlap} | flag_ope_overlap:{flag_ope_overlap} | ', \
        #     f'flag_proc_time:{flag_proc_time} | flag_trans_time:{flag_trans_time} | flag_unscheduled:{flag_unscheduled}')
        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_trans_time + flag_unscheduled != 0:
            return False, self.sched_opes_batch
        else:
            return True, self.sched_opes_batch

    def render(self, spec_batch_id=None):
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            num_vehs = self.num_vehs
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            if spec_batch_id is not None:
                self._draw_fig(spec_batch_id, num_mas, num_jobs, num_vehs, color)
            else:
                raise Exception('define "spec_batch_id" !!')
        return
    
    def _draw_fig(self, batch_id, num_mas, num_jobs, num_vehs, color):
        # row: opers | col: [status, machine, job_start_time, job_end_time, veh, trans_end_time]
        schedules = self.sched_opes_batch[batch_id].to('cpu')
        fig = plt.figure(figsize=(8, 5))
        fig.canvas.set_window_title('Visual_gantt')
        # axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
        
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.8, top=0.9, wspace=0.1, hspace=0.4)
        
        # : machine y_ticks
        y_ticks_ma = []
        y_ticks_loc_ma = []
        for i in range(num_mas):
            y_ticks_ma.append('Machine {0}'.format(i))
            # y_ticks_loc.insert(0, i + 1)
            y_ticks_loc_ma.insert(0, i)
        # : vehicle y_ticks
        y_ticks_veh = []
        y_ticks_loc_veh = []
        for i in range(num_vehs):
            y_ticks_veh.append('Vehicle {0}'.format(i))
            y_ticks_loc_veh.insert(0, i)
        
        labels = [''] * num_jobs
        for j in range(num_jobs):
            labels[j] = "job {0}".format(j + 1)
        patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
        
        # === set axis ==
        self._set_axis(ax1, num_mas, title='Machine Schedule', xlabel='', ylabel='Machine',
                       y_ticks_loc=y_ticks_loc_ma, y_ticks=y_ticks_ma, patches=patches)
        self._set_axis(ax2, num_vehs, title='Vehicle Schedule', xlabel='Time', ylabel='Vehicle',
                       y_ticks_loc=y_ticks_loc_veh, y_ticks=y_ticks_veh, patches=patches)
        ax1.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
        
        # === draw figure ===
        for i in range(int(self.nums_opes[batch_id])):
            id_ope = i
            idx_job, idx_ope = self.get_idx(id_ope, batch_id)
            id_machine = schedules[id_ope][1]
            id_veh = schedules[id_ope][4]
            proc_start = schedules[id_ope][5]
            job_end = schedules[id_ope][3]
            job_start = schedules[id_ope][2]    # jobs_start = trans_start
            trans_end = schedules[id_ope][5]
            # : machine gantt_chart
            ax1.barh(id_machine,
                        0.1,    # width
                        left=proc_start,  # The x coordinates of the left side of the bars
                        color='#b2b2b2',
                        height=0.5)
            ax1.barh(id_machine,
                        job_end - proc_start - 0.1,
                        left=proc_start+0.1,
                        color=color[idx_job],
                        height=0.5)
            if trans_end > job_start:
                # : vehicle gantt_chart
                ax2.barh(id_veh,
                            0.1,    # width
                            left=job_start,  # The x coordinates of the left side of the bars
                            color='#b2b2b2',
                            height=0.5)
                ax2.barh(id_veh,
                            trans_end - job_start - 0.1,
                            left=job_start+0.1,
                            color=color[idx_job],
                            height=0.5)
            
        plt.show()
    
    def _set_axis(self, ax, num_item, title, xlabel, ylabel,\
        y_ticks_loc, y_ticks, patches):
        ax.cla()
        ax.set_title(title)
        ax.grid(linestyle='-.', color='gray', alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yticks(y_ticks_loc, y_ticks)
        # ax.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
        ax.set_ybound(1 - 1 / num_item, num_item + 1 / num_item)
    
    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope    
        
    def close(self):
        pass    
        
    