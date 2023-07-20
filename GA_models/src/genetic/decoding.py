#!/usr/bin/env python

import sys
import torch

def split_ms(pb_instance, ms):
    jobs = []
    current = 0
    for index, job in enumerate(pb_instance['jobs']):
        jobs.append(ms[current:current+len(job)])
        current += len(job)
    return jobs


def get_processing_time(op_by_machine, machine_nb):
    for op in op_by_machine:
        if op['machine'] == machine_nb:
            return op['processingTime']
    print("[ERROR] Machine {} doesn't to be able to process this task.".format(machine_nb))
    sys.exit(-1)


def is_free(tab, start, duration):
    for k in range(start, start+duration):
        if not tab[k]:
            return False
    return True


def find_first_available_place(start_ctr, duration, machine_jobs):
    max_duration_list = []
    max_duration = start_ctr + duration

    # max_duration is either the start_ctr + duration or the max(possible starts) + duration
    if machine_jobs:
        for job in machine_jobs:
            max_duration_list.append(job[3] + job[1] + job[4])  # start + process time + trans_time

        max_duration = max(max(max_duration_list), start_ctr) + duration

    usedTime = [True] * max_duration

    # Updating array with used places
    for job in machine_jobs:
        start = job[3]  
        long = job[1] + job[4]  # proc_time + trans_time
        for k in range(start, start + long):
            usedTime[k] = False

    # Find the first available place that meets constraint
    for k in range(start_ctr, len(usedTime)):
        if is_free(usedTime, k, duration):
            return k

def find_first_available_place_vehicle(start_ctr, procTime, transTime, machine_jobs, vehicle_jobs):
    max_duration_list = []
    max_duration = start_ctr + procTime + transTime
    # print(f'machine_jobs:{machine_jobs}')
    # print(f'vehicle_jobs:{vehicle_jobs}')

    # max_duration is either the start_ctr + duration or the max(possible starts) + duration
    if machine_jobs:
        for job in machine_jobs:
            max_duration_list.append(job[3] + job[1])  # start + proc_time
    if vehicle_jobs:
        for job in vehicle_jobs:
            max_duration_list.append(job[3] + job[1])  # start + proc_time
    
    if max_duration_list:    
        max_duration = max(max(max_duration_list), start_ctr) + procTime + transTime
    else:
        max_duration = max(0, start_ctr) + procTime + transTime
        
    # print(f'max_duration_list:{max_duration_list} | max_duration:{max_duration}')
    usedTime = [True] * max_duration
    # print(f"start_ctr:{start_ctr} | len(usedTime):{len(usedTime)} | procTime:{procTime} | transTime:{transTime}")

    # Updating array with used places
    for job in machine_jobs:
        start = job[3]  
        long = job[1]  # proc_time + trans_time
        for k in range(start, start + long):
            usedTime[k] = False
    for job in vehicle_jobs:
        start = job[3]  
        long = job[1]  #  trans_time
        for k in range(start, start + long):
            usedTime[k] = False
    # Find the first available place that meets constraint
    for k in range(start_ctr, len(usedTime)):
        # print(f'k:{k}')
        if is_free(usedTime, k, procTime+transTime):
            return k

def get_transTime(pb_instance, os, ms, trans_mat, print_=False):
    '''
    :param trans_mat: tensor [n_mas, n_mas]
    
    :return vs_s: list of vehicle index: [[veh_idxes for job1], [], ...]
    :return transTime_s: list of transportation time for the corresponding operation
    '''
    o = pb_instance['jobs']
    # initial vehicle location is ma0
    veh_loc = torch.zeros(size=(pb_instance['vehiclesNb'],), dtype=torch.long) 
    
    ms_s = split_ms(pb_instance, ms)  # machine for each operations
    indexes = [0] * len(ms_s)
    vs_s = [[] for _ in range(len(ms_s))]
    transTime_s = [[] for _ in range(len(ms_s))]
    prev_ope_loc_s = [0 for _ in range(len(ms_s))]
    
    for job in os:
        index_machine = ms_s[job][indexes[job]]
        machine = o[job][indexes[job]][index_machine]['machine']
        
        prev_ope_loc = prev_ope_loc_s[job]
        tmp_trans_mat = trans_mat.gather(0, veh_loc[:, None].expand(-1, trans_mat.size(1))) # [n_vehs, n_mas]
        offload_trans_s = tmp_trans_mat[:, prev_ope_loc]  # [n_vehs]
        offload_trans, veh_idx = offload_trans_s.min(dim=0)

        onload_trans = trans_mat[prev_ope_loc, machine-1]   # scalar
        trans_time = offload_trans + onload_trans
        
        veh_loc[veh_idx] = machine-1
        prev_ope_loc_s[job] = machine-1
        vs_s[job].append(veh_idx.item())
        transTime_s[job].append(trans_time.long().item())
        indexes[job] += 1
    
    return vs_s, transTime_s


def decode(pb_instance, os, ms, print_=False):
    o = pb_instance['jobs']
    machine_operations = [[] for i in range(pb_instance['machinesNb'])]
    vehicle_operations = [[] for i in range(pb_instance['vehiclesNb'])]
    # initial vehicle location is ma0
    trans_mat = pb_instance['trans_mat']    # list: [n_mas, n_mas]

    ms_s = split_ms(pb_instance, ms)  # machine for each operations
    
    vs_s, transTime_s = get_transTime(pb_instance, os, ms, trans_mat, print_)

    indexes = [0] * len(ms_s)
    start_task_cstr = [0] * len(ms_s)
    
    # Iterating over OS to get task execution order and then checking in
    # MS to get the machine
    for job in os:
        index_machine = ms_s[job][indexes[job]] # job-ope의 index_machine 번째 있는 compatible machine data를 pointing 하는 var
        machine = o[job][indexes[job]][index_machine]['machine']
        prcTime = o[job][indexes[job]][index_machine]['processingTime']
        start_cstr = start_task_cstr[job]
        
        veh = vs_s[job][indexes[job]]
        transTime = transTime_s[job][indexes[job]]

        # Getting the first available place for the operation
        # start_ma = find_first_available_place(start_cstr, prcTime, machine_operations[machine - 1])
        start = find_first_available_place_vehicle(
            start_cstr, prcTime, transTime, machine_operations[machine - 1], vehicle_operations[veh]
        )
        
        name_task = "{}-{}".format(job, indexes[job]+1)

        machine_operations[machine - 1].append((name_task, prcTime, start_cstr, start+transTime, transTime))
        vehicle_operations[veh].append((name_task, transTime, start_cstr, start))
        
        # Updating indexes (one for the current task for each job, one for the start constraint
        # for each job)
        indexes[job] += 1
        start_task_cstr[job] = (start + transTime + prcTime)
    
    return machine_operations, vehicle_operations

def translate_decoded_to_gantt(machine_operations):
    data = {}
    # print(f"machine_operations:{machine_operations}")
    for idx, machine in enumerate(machine_operations):
        machine_name = "Machine-{}".format(idx + 1)
        operations = []
        
        for operation in machine:
            start = operation[3]
            end = operation[3] + operation[1]
            
            # for machine_oprations, not vehicle_operations
            # if len(operation) > 4:
            #     start += operation[4]   # transTime
            #     end += operation[4]
                
            operations.append([start, end, operation[0]])

        data[machine_name] = operations
    return data

def translate_veh_decoded_to_gantt(vehicle_operations):
    data = {}
    for idx, vehicle in enumerate(vehicle_operations):
        vehicle_name = "Vehicle-{}".format(idx + 1)
        operations = []
        for operation in vehicle:
            start = operation[3]
            end = operation[3] + operation[1]
            
            operations.append([start, end, operation[0]])

        data[vehicle_name] = operations
    return data


def tmp_decode(pb_instance, os, ms):
    o = pb_instance['jobs']
    machine_operations = [[] for i in range(pb_instance['machinesNb'])]

    ms_s = split_ms(pb_instance, ms)  # machine for each operations

    indexes = [0] * len(ms_s)
    start_task_cstr = [0] * len(ms_s)

    # Iterating over OS to get task execution order and then checking in
    # MS to get the machine
    for job in os:
        index_machine = ms_s[job][indexes[job]]
        machine = o[job][indexes[job]][index_machine]['machine']
        prcTime = o[job][indexes[job]][index_machine]['processingTime']
        start_cstr = start_task_cstr[job]

        # Getting the first available place for the operation
        start = find_first_available_place(start_cstr, prcTime, machine_operations[machine - 1])
        name_task = "{}-{}".format(job, indexes[job]+1)

        machine_operations[machine - 1].append((name_task, prcTime, start_cstr, start))

        # Updating indexes (one for the current task for each job, one for the start constraint
        # for each job)
        indexes[job] += 1
        start_task_cstr[job] = (start + prcTime)

    return machine_operations


def tmp_translate_decoded_to_gantt(machine_operations):
    data = {}

    for idx, machine in enumerate(machine_operations):
        machine_name = "Machine-{}".format(idx + 1)
        operations = []
        for operation in machine:
            operations.append([operation[3], operation[3] + operation[1], operation[0]])

        data[machine_name] = operations

    return data
