
import random
import time
import torch
import numpy as np
from copy import deepcopy

class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, num_jobs, num_opes, num_mas, num_vehs, device,
                opes_per_job_min, opes_per_job_max,
                proctime_per_ope_mas=20, transtime_btw_ma_max=10,
                dynamic=None, job_centric=True,
                data_source='case', case_config=None
    ):
        '''
        :param data_source: 'case' or 'benchmark'
        '''
        self.num_jobs = num_jobs
        self.num_opes = num_opes
        self.num_mas = num_mas
        self.num_vehs = num_vehs
        self.device = device
        self.data_source = data_source
        
        if data_source == 'case':
            if not job_centric:
                # static case: operations are divided equality
                if dynamic is None:
                    self.num_jobs = self.num_opes//self.num_mas
                    self.num_opes_list = self._get_num_opes_list(self.num_opes, self.num_jobs, None)
                # dynamic case: operation sequnce are generated randomly
                else:
                    self.num_opes_list = self._get_num_opes_list(self.num_opes, None, dynamic)
                    self.num_jobs = len(self.num_opes_list)
            else:
                self.num_opes_list = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(self.num_jobs)]    # the number of operations for each job
                self.num_opes = sum(self.num_opes_list)
        elif data_source == 'benchmark':
            self.num_jobs = case_config['num_jobs']
            self.num_mas = case_config['num_mas']
            self.num_vehs = case_config['num_vehs']
            self.num_opes_list = case_config['num_opes_list']
            self.num_opes = case_config['num_opes']
            self.ope_ma_adj = case_config['ope_ma_adj']
            self.proc_time_mat = case_config['proc_time']
            self.trans_time_mat = case_config['trans_time']
        else:
            raise Exception('data_source error')

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = self.num_mas
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = proctime_per_ope_mas
        self.proctime_dev = 0.2
        
        self.transtime_btw_ma_min = 1   # average transportation time
        self.transtime_btw_ma_max = transtime_btw_ma_max
        self.transtime_dev = 0.2
        
    def _get_num_opes_list(self, num_opes, num_jobs, dynamic=None):
        '''
        Output:
            num_opes_list: [number of operations per job, ...], len = num_jobs
        '''
        num_opes_list = []
        if dynamic is None:
            rest_opes = num_opes%num_jobs
            for i in range(num_jobs):
                if i < rest_opes:
                    num_opes_list.append(num_opes//num_jobs + 1)
                else:
                    num_opes_list.append(num_opes//num_jobs)
        else:
            min_ope_per_job = dynamic['min_ope_per_job']
            # max_ope_per_job = dynamic['max_ope_per_job']
            max_ope_per_job = max(num_opes//3, min_ope_per_job+1)
            total_opes = 0
            while True:
                tmp_num = np.random.randint(min_ope_per_job, max_ope_per_job)
                if total_opes + tmp_num > num_opes:
                    tmp_num = num_opes - total_opes
                    if tmp_num > 0:
                        num_opes_list.append(tmp_num)
                    break
                else:
                    num_opes_list.append(tmp_num)
                    total_opes += tmp_num
        return num_opes_list
    
    def get_case_for_transport(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        
        Output:
            matrix_proc_time: tensor: [num_opes, num_mas]
            matrix_ope_ma_adj: tensor: [num_opes, num_mas]
            matrix_pre_ope_adj: tensor: [num_opes, num_opes]
            matrix_suc_ope_adj: tensor: [num_opes, num_opes]
            matrix_cal_cumul: tensor: [num_opes, num_opes]
            nums_ope: tensor: [num_jobs,]: each element is the number of operation on a job
            num_ope_biases: tensor: [num_jobs,]
            end_ope_biases: tensor: [num_jobs,]
            opes_appertain: tensor: [num_opes,]
            matrix_trans_time: tensor: [num_mas, num_mas]
            matrix_ma_veh_adj: tensor: [num_mas, num_mas]
            
        '''
        
        # print(f'self.num_jobs:{self.num_jobs}')
        # print(f'self.num_opes_list:{self.num_opes_list}')
        assert self.num_jobs <= self.num_opes
        assert len(self.num_opes_list) == self.num_jobs
        assert self.num_opes == sum(self.num_opes_list)
        
        # the number of compatible machine list
        self.num_cpt_mas_list = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]   # for each ope, assign the number of compatible machines
        self.num_cpt_mas = sum(self.num_cpt_mas_list)
        
        
        self.num_ope_biases = [sum(self.num_opes_list[0:i]) for i in range(self.num_jobs)]
        self.end_ope_biases = [val + self.num_opes_list[i] - 1 for i, val in enumerate(self.num_ope_biases)]
        self.num_ma_biases = [sum(self.num_cpt_mas_list[0:i]) for i in range(self.num_opes)]
        
        
        # === For each operation idx, sample machine indexes as the number of compatible machines ===
        self.ope_ma = []
        for val in self.num_cpt_mas_list:
            sample_mas = sorted(random.sample(range(1, self.num_mas+1), val))
            sample_mas_idx = [val - 1 for val in sample_mas]
            self.ope_ma.append(sample_mas_idx)
        
        # === operation-machine adjacent matrix ===
        matrix_ope_ma_adj = torch.zeros(size=(self.num_opes, self.num_mas))
        for ope_idx in range(self.num_opes):
            matrix_ope_ma_adj[ope_idx, self.ope_ma[ope_idx]] = 1
        
        
        # === set a process time for each operation-machine pair ===
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)] # list of proc_time_mean for each operation
        for i in range(len(self.num_cpt_mas_list)):  # len: the number of operations that exist
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.num_cpt_mas_list[i])] # for an operation, it has different proc_time for available machines
            # self.proc_time = self.proc_time + proc_time_ope 
            self.proc_time.append(proc_time_ope)
        
        
        # === process time matrix ===
        matrix_proc_time = torch.zeros(size=(self.num_opes, self.num_mas))
        for ope_idx in range(self.num_opes):
            for n, ma_idx in enumerate(self.ope_ma[ope_idx]):
                matrix_proc_time[ope_idx, ma_idx] = self.proc_time[ope_idx][n]
        
        # === set a transporation time for each machine-vehicle pair ===
        self.trans_time = []
        self.trans_time_mean = [random.randint(self.transtime_btw_ma_min, self.transtime_btw_ma_max) for _ in range(self.num_mas)]
        for i in range(self.num_mas):
            low_bound = max(self.transtime_btw_ma_min, round(self.trans_time_mean[i]*(1-self.transtime_dev)))
            high_bound = min(self.transtime_btw_ma_max, round(self.trans_time_mean[i]*(1+self.transtime_dev)))
            trans_time_ma = [random.randint(low_bound, high_bound) for _ in range(self.num_mas)]
            self.trans_time.append(trans_time_ma)
        
        # === transportation time matrix ===
        matrix_trans_time = torch.zeros(size=(self.num_mas, self.num_mas), dtype=torch.float)
        for from_ma in range(self.num_mas):
            for to_ma in range(self.num_mas):
                if from_ma == to_ma:
                    matrix_trans_time[from_ma, to_ma] = 0
                elif from_ma < to_ma:   # symmetric matrix
                    matrix_trans_time[from_ma, to_ma] = self.trans_time[from_ma][to_ma]
                    matrix_trans_time[to_ma, from_ma] = self.trans_time[from_ma][to_ma]
        # print(f"self.trans_time:{self.trans_time}")
        # print(f"matrix_trans_time:{matrix_trans_time}")
        
        # === machine-vehicle adjacent matrix ===
        matrix_ma_veh_adj = torch.ones(size=(self.num_mas, self.num_vehs))
        
        
        # === predecessor operation matrix ===
        matrix_pre_ope_adj_np = np.eye(self.num_opes, k=1, dtype=np.bool)
        matrix_pre_ope_adj_np[self.end_ope_biases, :] = False
        matrix_pre_ope_adj = torch.from_numpy(matrix_pre_ope_adj_np).to(self.device)

        # === successor operation matrix ===
        matrix_suc_ope_adj_np = np.eye(self.num_opes, k=-1, dtype=np.bool)
        matrix_suc_ope_adj_np[self.num_ope_biases, :] = False
        matrix_suc_ope_adj = torch.from_numpy(matrix_suc_ope_adj_np).to(self.device)
        
        # === cumulative matrix for an estimation of start time of each operation ===
        matrix_cal_cumul = torch.zeros(size=(self.num_opes, self.num_opes)).float()
        job_idx = 0
        cunt_ope = 0
        for col in range(self.num_opes):
            if col not in self.num_ope_biases:
                vector = torch.zeros(size=(self.num_opes,))
                vector[self.num_ope_biases[job_idx] + cunt_ope - 1] = 1
                matrix_cal_cumul[:, col] = matrix_cal_cumul[:, col-1] + vector
            
            if cunt_ope == self.num_opes_list[job_idx]-1:
                job_idx += 1        
                cunt_ope = 0
            else:
                cunt_ope += 1

        # === job_idx that each operation appertain to ===
        opes_appertain = torch.zeros(size=(self.num_opes,), dtype=torch.long)
        for job_idx in range(self.num_jobs):
            opes_appertain[self.num_ope_biases[job_idx] : self.num_ope_biases[job_idx] + self.num_opes_list[job_idx]] = job_idx
        
        
        # ===
        num_opes_list = torch.tensor(self.num_opes_list, dtype=torch.long)
        num_ope_biases = torch.tensor(self.num_ope_biases, dtype=torch.long)
        end_ope_biases = torch.tensor(self.end_ope_biases, dtype=torch.long)
        
        # === for data_source == 'benchmark' ===
        if self.data_source == 'benchmark':
            matrix_ope_ma_adj = self.ope_ma_adj
            matrix_proc_time = self.proc_time_mat
            matrix_trans_time = self.trans_time_mat
        return (matrix_proc_time, matrix_ope_ma_adj, matrix_pre_ope_adj, matrix_suc_ope_adj, matrix_cal_cumul, \
            num_opes_list, num_ope_biases, end_ope_biases, opes_appertain, matrix_trans_time, matrix_ma_veh_adj)
        