import json
import torch
import os
import random

def read_json(path:str) -> dict:
    with open(path+".json","r",encoding="utf-8") as f:
        config = json.load(f)
    return config

def write_json(data:dict, path:str):
    with open(path+".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))


def build_caseConfig(parameters, benchmark_path_base):
    '''
    :param parameters:
        {'machinesNb':0, 'jobs':[], 'num_opes_list':[]}
    
    :return case_config:
        {
            'num_jobs':,
            'num_mas':,
            'num_vehs':,
            'num_opes':,
            'ope_ma_adj': ,
            'proc_time': ,
            'trans_time': ,
            'num_opes_list': ,
            'proctime_per_ope_max':,
            'transtime_btw_ma_max':,
        }
    '''
    # num_jobs, num_opes, num_mas, num_vehs, device,
    # opes_per_job_min, opes_per_job_max,
    # proctime_per_ope_mas=20, transtime_btw_ma_max=10,
    # data_source='case', num_opes_list
    # proc_time, trans_time
    case_config = {}
    num_mas = parameters['machinesNb']
    num_opes_list = parameters['num_opes_list']
    num_opes = sum(num_opes_list)
    num_jobs = len(num_opes_list)
    proc_time = torch.zeros(size=(num_opes, num_mas), dtype=torch.float)
    jobs = parameters['jobs']
    
    # === build proc_time matrix ===
    for job_idx, job in enumerate(jobs):
        # print(f'job[{job_idx}]: {job}')
        for ope_idx, ope in enumerate(job):
            if job_idx == 0:
                job_bias = 0
            else:
                job_bias = sum(num_opes_list[:job_idx])
            total_ope_idx = job_bias + ope_idx
            
            for comp_mas in ope:
                ma_idx = comp_mas['machine'] - 1
                proc = comp_mas['processingTime']
                proc_time[total_ope_idx, ma_idx] = proc
    ope_ma_adj = torch.where(proc_time>0, 1, 0).long()
    # === machine layout file path ===
    file_name = f'{num_mas}_machine_layout.txt'
    layout_path = os.path.join(benchmark_path_base, file_name)
    trans_time_list = parse_machine_layout(layout_path)
    trans_time = torch.tensor(trans_time_list, dtype=torch.float)   # [n_mas+1, n_mas+1]
    trans_time = trans_time[1:, 1:] # [n_mas, n_mas]: ignore L/U positions
    
    case_config['num_jobs'] = num_jobs
    case_config['num_mas'] = num_mas
    min_num_vehs = max(int(round(num_mas * 0.8)), 2)
    max_num_vehs = int(round(num_mas*1.2))
    case_config['num_vehs'] = random.randint(min_num_vehs, max_num_vehs)
    case_config['num_opes'] = num_opes
    case_config['ope_ma_adj'] = ope_ma_adj
    case_config['proc_time'] = proc_time
    case_config['num_opes_list'] = num_opes_list
    case_config['trans_time'] = trans_time
    case_config['proctime_per_ope_max'] = proc_time.max()
    case_config['transtime_btw_ma_max'] = trans_time.max()
    return case_config
    

def parse(path):
    file = open(path, 'r')

    firstLine = file.readline()
    firstLineValues = list(map(int, firstLine.split()[0:2]))

    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    jobs = []
    num_opes_list = []
    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split()))

        operations = []
        num_opes_list.append(currentLineValues[0])
        
        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j]
            j = j+1

            operation = []

            for ik in range(k):
                machine = currentLineValues[j]
                j = j+1
                processingTime = currentLineValues[j]
                j = j+1

                operation.append({'machine': machine, 'processingTime': processingTime})

            operations.append(operation)

        jobs.append(operations)

    file.close()

    return {'machinesNb': machinesNb, 'jobs': jobs, 'num_opes_list': num_opes_list}

def parse_machine_layout(path):
    '''
    machine_layout:
        [num_mas+1, num_mas+1], where '1' is a Load/Unload position. 
        we ignore the L/U data (first row and column)
    
    :return trans_time_list: list [num_mas+1, num_mas+1]
    '''
    trans_time_list = []
    with open(path, 'r') as file:
        lines = file.readlines()
        
        for line_idx, line in enumerate(lines):
            line_list = list(map(float, line.split()))
            if line_list:
                trans_time_list.append(line_list)
    return trans_time_list
            
        