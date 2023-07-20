import random
import time
import torch
import numpy as np
from copy import deepcopy

class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, num_vehs, device, 
                 proctime_per_ope_mas=20, transtime_btw_ma_max=10,
                 nums_ope=None, path='../data/',
                 flag_same_opes=False, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas
        self.num_vehs = num_vehs
        self.device = device

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = proctime_per_ope_mas
        self.proctime_dev = 0.2
        # === vehicle variables ===
        # assume: a vehicle can move to all machines
        self.transtime_btw_ma_min = 1
        self.transtime_btw_ma_max = transtime_btw_ma_max
        self.transtime_dev = 0.2

    

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
        self.num_jobs = self.job_init
        # if not self.flag_same_opes:
        #     self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]    # the number of operations for each job
        self.num_opes = sum(self.nums_ope)  # the number of all operations in the environment
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]   # for each ope, assign the number of compatible machines
        self.num_options = sum(self.nums_option)
        
        self.num_ope_biases = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.end_ope_biases = [val + self.nums_ope[i] - 1 for i, val in enumerate(self.num_ope_biases)]
        self.num_ma_biases = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        
        # print(f"self.nums_ope:{self.nums_ope}")
        # print(f"self.num_ope_biass:{self.num_ope_biases}")
        # print(f"self.nums_option:{self.nums_option}")
        # print(f"self.num_ma_biass:{self.num_ma_biases}")
        
        
        # === For each operation idx, sample machine indexes as the number of compatible machines ===
        self.ope_ma = []
        for val in self.nums_option:
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
        for i in range(len(self.nums_option)):  # len: the number of operations that exist
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])] # for an operation, it has different proc_time for available machines
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
            
            if cunt_ope == self.nums_ope[job_idx]-1:
                job_idx += 1        
                cunt_ope = 0
            else:
                cunt_ope += 1

        # === job_idx that each operation appertain to ===
        opes_appertain = torch.zeros(size=(self.num_opes,), dtype=torch.long)
        for job_idx in range(self.num_jobs):
            opes_appertain[self.num_ope_biases[job_idx] : self.num_ope_biases[job_idx] + self.nums_ope[job_idx]] = job_idx
        
        
        # ===
        nums_ope = torch.tensor(self.nums_ope, dtype=torch.long)
        num_ope_biases = torch.tensor(self.num_ope_biases, dtype=torch.long)
        end_ope_biases = torch.tensor(self.end_ope_biases, dtype=torch.long)
        return (matrix_proc_time, matrix_ope_ma_adj, matrix_pre_ope_adj, matrix_suc_ope_adj, matrix_cal_cumul, \
            nums_ope, num_ope_biases, end_ope_biases, opes_appertain, matrix_trans_time, matrix_ma_veh_adj)
        
        
    
    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]    # the number of operations for each job
        self.num_opes = sum(self.nums_ope)  # the number of all operations in the environment
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]   # for each ope, assign the number of compatible machines
        self.num_options = sum(self.nums_option)
        
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas+1), val))    # list of available machine idxes for each operation
        
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)] # list of proc_time_mean for each operation
        for i in range(len(self.nums_option)):  # len: the number of operations that exist
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])] # for an operation, it has different proc_time for available machines
            self.proc_time = self.proc_time + proc_time_ope 
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        
        
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs
    
