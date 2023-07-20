
import random
import torch
import time

from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv
from SAHJS_models.temporal_graph_dataset import set_GraphData
from sat_models.sat.data import GraphDataset
from torch_geometric.loader.dataloader import DataLoader
from dispatching_models.dispatchModel import dispatchModel

def validate(env, model, logger=None, test_dataset=None, test_loader=None):
    if logger is not None:
        logger.info('========== validating ==========')
    # start = time.time()
    if not isinstance(model, dispatchModel):
        model.eval()
    state = env.reset()
    model.init(state, dataset=test_dataset, loader=test_loader)
    
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
    batch_scores = -rewards # [B,]
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        if logger is not None:
            logger.info("Scheduling Error!!!")
        else:
            print("Scheduling Error!!!")
    # logger.info('validating time: ', time.time() - start)
    if logger is not None:
        logger.info('score (the lower the better): {}'.format(score))
    return score.item(), spand_time, batch_scores

def validate_multi_models(
    env_list, models, model_names, logger, result_folder, result_log, test_len=100,
    test_dataset_list=None, test_loader_list=None,
):
    
    for i, model in enumerate(models):
        avg_score = 0
        avg_spand_time = 0
        
        for test_idx in range(test_len):
            if test_dataset_list is not None:
                score, spand_time, rewards = validate(
                    env_list[test_idx], model, 
                    test_dataset=test_dataset_list[test_idx],
                    test_loader=test_loader_list[test_idx]
                )
            else:
                score, spand_time, batch_scores = validate(env_list[test_idx], model)
            avg_score += score
            avg_spand_time += spand_time
        avg_score /= test_len
        avg_spand_time /= test_len
        
        logger.info('{} Score: {:0.2f} | SpandTime: {:0.2f} '.format(model_names[i], avg_score, avg_spand_time))
        result_log.append('{}_score'.format(model_names[i]), avg_score)
        result_log.append('{}_SpandTime'.format(model_names[i]), avg_spand_time)
        result_log.append('{}_batch_scores'.format(model_names[i]), batch_scores.cpu().tolist())
        
    
    # print(f'result_log.get_raw_data():{result_log.get_raw_data()}')
    
    # === save log results of validation ===
    # result_dict = {
    #     'result_log': result_log.get_raw_data()
    # }
    # torch.save(result_dict, '{}/test_results.pt'.format(result_folder))
        
    

def generate_vali_env(
    test_env_paras, logger, device, 
    opes_per_job_min, opes_per_job_max,
    proctime_per_ope_max, transtime_btw_ma_max, 
    job_centric=False,
    new_job_flag=False,
    test_len=100):
    num_jobs = test_env_paras['num_jobs']
    num_opes = test_env_paras['num_opes']
    num_mas = test_env_paras['num_mas']
    num_vehs = test_env_paras['num_vehs']
    dynamic = test_env_paras['dynamic']
    batch_size = test_env_paras["batch_size"]
    case_list = []
    env_list = []
    
    test_dataset_list = []
    test_loader_list = []
    for i in range(test_len):
        case = CaseGenerator(
            num_jobs, num_opes, num_mas, num_vehs, device,
            opes_per_job_min, opes_per_job_max,
            proctime_per_ope_max, transtime_btw_ma_max,
            dynamic, job_centric
        )
        new_job_dict = None
        if new_job_flag:
            new_job_dict = {
                'new_job_idx': torch.full(size=(test_env_paras['batch_size'], num_jobs), fill_value=False),
                'release': torch.full(size=(test_env_paras['batch_size'], num_jobs), fill_value=0)
            }
            n_newJobs = test_env_paras['num_newJobs']
            newJob_idxes = [random.randint(0, num_jobs-1) for _ in range(n_newJobs)]
            new_job_dict['new_job_idx'][:, newJob_idxes] = True
            print(f"new_job_dict['new_job_idx']:{new_job_dict['new_job_idx']}")
        env = TFJSPEnv(case=case, env_paras=test_env_paras, new_job_dict=new_job_dict)
        case_list.append(case)
        env_list.append(env)
        logger.info('vali_env info: num_jobs:{}, num_opes:{} num_mas:{}, num_veh:{}, batch_size:{}'\
            .format(env.num_jobs, env.num_opes, env.num_mas, env.num_vehs, test_env_paras['batch_size']))
        
        # === generate test graph dataset ===
        # graph_dataset = []
        # for batch in range(batch_size):
        #     data = set_GraphData(
        #         env.num_opes, env.num_mas, env.num_vehs, env.nums_ope_batch[batch],
        #         env.ope_ma_adj_batch[batch], env.proc_times_batch[batch], env.trans_times_batch[batch],
        #         node_feat_dim=8, edge_feat_dim=1
        #     )
        #     graph_dataset.append(data)

        # test_dataset = GraphDataset(graph_dataset, degree=True, 
        #     k_hop=3, se='khopgnn',
        #     use_subgraph_edge_attr=True
        # )
        # test_loader = DataLoader(test_dataset, batch_size=batch_size,
        #     shuffle=False,
        #     generator=torch.Generator(device=device)
        # )
        # test_dataset_list.append(test_dataset)
        # test_loader_list.append(test_loader)


    
    return env_list, case_list, test_dataset_list, test_loader_list


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