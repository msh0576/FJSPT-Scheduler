import torch
import time

from GA_models.src.utils import parser, gantt
from GA_models.src.genetic import encoding, decoding, genetic, termination
from GA_models.src import config


def parser_from_case(
    proc_time, trans_time, nums_ope, num_vehs,
):
    '''
    :param proc_time: [n_opes, n_mas]
    :param trans_time: [n_mas, n_mas]
    :param nums_ope: [n_jobs]
    
    :return parameters
    '''
    parameters = {}
    # print(f'proc_time:{proc_time}')
    # print(f'nums_ope:{nums_ope}')
    num_opes, num_mas = proc_time.size()
    num_jobs = nums_ope.size(0)
    
    nums_ope_list = nums_ope.tolist()
    
    parameters["machinesNb"] = num_mas
    jobs_list = []
    for job_idx in range(num_jobs):
        job_opes_list = []
        for ope_idx in range(nums_ope[job_idx]):
            tmp_ope_idx = nums_ope[:job_idx].sum() + ope_idx
            ma_idxes = torch.where(proc_time[tmp_ope_idx, :]>0)[0].tolist()
            proc_values = proc_time[[tmp_ope_idx]*len(ma_idxes), ma_idxes].long().tolist()
            ope_comp_ma_list = []
            for ma, proc in zip(ma_idxes, proc_values):
                ope_comp_ma_list.append({'machine': ma+1, 'processingTime': proc})
            job_opes_list.append(ope_comp_ma_list)
        jobs_list.append(job_opes_list)
    parameters["jobs"] = jobs_list
    parameters["vehiclesNb"] = num_vehs
    parameters['trans_mat'] = trans_time
    
    return parameters
                
            
            
    
    
    
    

def GAtest(parameters, print_=True):
    
    
    t0 = time.time()
    inter_time = time.time()
    # Initialize the Population
    population = encoding.initializePopulation(parameters)
    gen = 1
    # Evaluate the population
    while not termination.shouldTerminate(population, gen, inter_time-t0):
        # Genetic Operators
        population = genetic.selection(population, parameters)
        population = genetic.crossover(population, parameters)
        population = genetic.mutation (population, parameters)

        gen = gen + 1
        inter_time = time.time()

    sortedPop = sorted(population, key=lambda cpl: genetic.timeTaken(cpl, parameters))

    t1 = time.time()
    total_time = t1 - t0
    # print("Finished in {0:.2f}s".format(total_time))

    # Termination Criteria Satisfied ?
    machine_operations, vehicle_operations = decoding.decode(parameters, sortedPop[0][0], sortedPop[0][1], print_=True)
    ma_gantt_data = decoding.translate_decoded_to_gantt(machine_operations)
    veh_gantt_data = decoding.translate_veh_decoded_to_gantt(vehicle_operations)
    makespan = gantt.get_makespan(ma_gantt_data)
    
    # print(f"ma_gantt_data:{ma_gantt_data}")
    # print(f"veh_gantt_data:{veh_gantt_data}")
    if print_ is True:
        if config.latex_export:
            gantt.export_latex(ma_gantt_data)
            gantt.export_latex(veh_gantt_data)
        else:
            gantt.draw_chart(ma_gantt_data, veh_gantt_data)
        print(f'makespan:{makespan:.2f}, run_time:{total_time:.2f}')
    
    return makespan, total_time