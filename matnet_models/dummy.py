
import random
import torch
import time

from env.case_generator import CaseGenerator
from env.tfjsp_env import TFJSPEnv

def validate(env, model, logger=None):
    if logger is not None:
        logger.info('========== validating ==========')
    # start = time.time()
    
    state = env.reset()
    model.init(state)
    
    done = False
    dones = env.done_batch
    start_time = time.time()
    while not done:
        with torch.no_grad():
            action, _ = model.act(state)   # [3, B] | [B, 1]
        state, rewards, dones = env.step(action)
        done = dones.all()
    spand_time = time.time() - start_time
    
    score = -rewards.mean() 
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        if logger is not None:
            logger.info("Scheduling Error!!!")
        else:
            print("Scheduling Error!!!")
    # logger.info('validating time: ', time.time() - start)
    if logger is not None:
        logger.info('score (the lower the better): {}'.format(score))
    return score.item(), spand_time

def validate_multi_models(env, models, model_names, logger, result_folder, result_log):
    for i, model in enumerate(models):
        score, spand_time = validate(env, model)
        logger.info('{} Score: {:0.2f} | SpandTime: {:0.2f} '.format(model_names[i], score, spand_time))
        result_log.append('{}_score'.format(model_names[i]), score)
        result_log.append('{}_SpandTime'.format(model_names[i]), spand_time)
    
    # print(f'result_log.get_raw_data():{result_log.get_raw_data()}')
    
    # === save log results of validation ===
    # result_dict = {
    #     'result_log': result_log.get_raw_data()
    # }
    # torch.save(result_dict, '{}/test_results.pt'.format(result_folder))
        
    

def generate_vali_env(test_env_paras, logger, device, proctime_per_ope_mas, transtime_btw_ma_max):
    num_jobs = test_env_paras['num_jobs']
    num_mas = test_env_paras['num_mas']
    num_vehs = test_env_paras['num_vehs']
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
        
    nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
    case = CaseGenerator(
        num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope, num_vehs=num_vehs, device=device,
        proctime_per_ope_mas=proctime_per_ope_mas, transtime_btw_ma_max=transtime_btw_ma_max
    )
    env = TFJSPEnv(case=case, env_paras=test_env_paras)


    logger.info('vali_env info: num_jobs:{}, num_mas:{}, num_veh:{}, batch_size:{}'.format(num_jobs, num_mas, num_vehs, test_env_paras['batch_size']))
    return env, case


def restore_model(model, device, **model_load):
    '''
    Input:
        model_load: {
            'path': ~~
            'epoch': ~~
        }
    '''
    checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model