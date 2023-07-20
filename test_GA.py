
from email import parser
import logging
import torch
import numpy as np

from options import get_options
from utils.utils import *
from logging import getLogger
from train import setup_seed

from GA_models.GATester import GAtest, parser_from_case
from DHJS_models.TFJSPTester import generate_vali_env, restore_model, validate_multi_models

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

def _print_config():
    logger = logging.getLogger('root')
    # logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    # logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def main(opts, num_opes=None, num_mas=None, num_vehs=None):
    # ========= Pytorch initialization ==========
    # === load config ===
    with open("./configs/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    optimizer_paras = load_dict["optimizer_paras"]
    logger_paras = load_dict["logger_paras"]
    test_paras = load_dict["test_paras"]
    
    
    # === modify config ===    
    model_paras['sqrt_embedding_dim'] = model_paras['embedding_dim']**(1/2)
    model_paras['sqrt_qkv_dim'] = model_paras['qkv_dim']**(1/2)
    model_paras['ms_layer1_init'] = (1/2)**(1/2)
    model_paras['ms_layer2_init'] = (1/16)**(1/2)
    
    logger_paras['log_file']['desc'] = opts.log_file_desc
    logger_paras['log_file']['filepath'] = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    # ===
    setup_seed(seed=opts.test_seed)
    create_logger(**logger_paras)
    _print_config()
    
    
    # PyTorch initialization
    device = torch.device("cpu")
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    
    
    env_paras["device"] = device
    test_paras["device"] = device
    model_paras["device"] = device
    
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    test_paras["batch_size"] = env_paras["batch_size"]
    model_paras["batch_size"] = env_paras["batch_size"]
    model_paras["proctime_per_ope_max"] = env_paras["proctime_per_ope_max"]
    model_paras["transtime_btw_ma_max"] = env_paras["transtime_btw_ma_max"]
    
    model_paras["checkpoint_encoder"] = opts.checkpoint_encoder
    model_paras["algorithm"] = opts.algorithm
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras['num_opes'] = test_paras["num_opes"]
    model_paras['num_mas'] = test_paras["num_mas"]
    model_paras['num_vehs'] = test_paras["num_vehs"]
        
    test_paras["dynamic"]['max_ope_per_job'] = test_paras['num_opes'] // test_paras['num_mas']
    if opts.static:
        test_paras['dynamic'] = None
    
    # =====================================================
    # === TFJSP AI algorithms ===
    # : logger 
    logger = getLogger(name='tester')
    result_folder = get_result_folder()
    result_log = LogData()
    
    # === generate validate environments ===
    opes_per_job_min = int(test_paras['num_mas'] * 0.8)
    opes_per_job_max = int(test_paras['num_mas'] * 1.2)
    vali_env, vali_case, test_dataset_list, test_loader_list = generate_vali_env(
        test_paras, logger, device, 
        opes_per_job_min, opes_per_job_max,
        env_paras["proctime_per_ope_max"], env_paras["transtime_btw_ma_max"],
        job_centric=model_paras['job_centric'],
        test_len=test_paras['num_test']
    )
    print_graph_att=[]
    
    
    
    
    

    # === setting genetic algorithm parameters ===
    # parameters = 
    for idx, env in enumerate(vali_env):
        print_graph_att.append((env.num_jobs, env.num_opes, env.num_mas, env.num_vehs))
        
        makespan_batch = 0
        runtime_batch = 0
        batch_size = env.batch_size
        print(f'env.nums_ope_batch[0]:{env.nums_ope_batch[0]}')
        print(f'env.proc_times_batch[0]:{env.proc_times_batch[0]}')
        print(f'env.trans_times_batch[0]:{env.trans_times_batch[0]}')
        
        parameters = parser_from_case(
            env.proc_times_batch[0], env.trans_times_batch[0], env.nums_ope_batch[0],
            env.num_vehs
        )
        makespan, runtime = GAtest(parameters, print_=True)
    print(f'makespan:{makespan:.2f}, runtime:{runtime:.2f}')




if __name__ == '__main__':
    args = get_options()
    main(args)
    