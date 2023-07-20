import logging
import torch
import numpy as np

from options import get_options
from utils.utils import *
from logging import getLogger
from train import setup_seed

from DHJS_models.TFJSPTester import generate_vali_env, restore_model, validate_multi_models
from DHJS_models.TFJSPTrainer_dhjs import TFJSPModel_DHJS
from matnet_models.TFJSPTrainer_matnet import TFJSPModel_matnet
from SAHJS_models.TFJSPModel_sahjs import TFJSPModel_SAHJS
from DTrans_models.TFJSPModel_dtrans import TFJSPModel_DTrans
from dispatching_models.dispatchModel import dispatchModel
from hgnn_models.TFJSPModel_hgnn import TFJSPModel_hgnn
from GTrans_models.TFJSPModel_gtrans import TFJSPModel_GTrans

from SAHJS_models.temporal_graph_dataset import set_GraphData
from sat_models.sat.data import GraphDataset
from torch_geometric.loader.dataloader import DataLoader
from sat_models.TFJSPModel_sat import GraphTransformer
import torch_geometric.utils as utils
from sat_models.sat.position_encoding import POSENCODINGS
from GA_models.GATester import GAtest, parser_from_case

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

def _print_config():
    logger = logging.getLogger('root')
    # logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    # logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

def main(opts, num_jobs=None, num_mas=None, num_vehs=None):
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
    
    
    # === multi test ===
    if num_jobs is not None and num_mas is not None and num_vehs is not None:
        test_paras['num_jobs'] = num_jobs
        test_paras['num_mas'] = num_mas
        test_paras['num_vehs'] = num_vehs
        opts.log_file_desc = f'test_{num_jobs}_{num_mas}_{num_vehs}'
    
    
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
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:"+str(opts.cuda) if torch.cuda.is_available() else "cpu")
    # opts.device = device
    
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
        new_job_flag=opts.new_job,
        test_len=test_paras['num_test'],
    )
    print_graph_att=[]
    for idx, env in enumerate(vali_env):
        print_graph_att.append((env.num_jobs, env.num_opes, env.num_mas, env.num_vehs))
        
    
    model_paras['encoder_layer_num'] = 2
    # : 20230426_144354_matnet_jobcentric_10_6_6
    matnet_10_6_6 = TFJSPModel_matnet(
        embedding_dim_=model_paras["embedding_dim"],
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        **model_paras
    ).to(device)
    
    # : 20230427_094750_hgnn_jobcentric_10_6_6
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras["action_dim"] = 1
    hgnn_10_6_6 = TFJSPModel_hgnn(env_paras, model_paras)
    
    
    hgs_selfatten_10_6_6 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=1,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    
    
    # : 20230504_165911_gtrans_jobcentric_10_6_6_EncV2_DecV5
    hgs_10_6_6 = TFJSPModel_GTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=2,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230508_135053_gtrans_jobcentric_5_3_3_EncV2_DecV5
    hgs_5_3_3 = TFJSPModel_GTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=2,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230508_135451_gtrans_jobcentric_10_6_3_EncV2_DecV5
    hgs_10_6_3 = TFJSPModel_GTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=2,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230508_135332_gtrans_jobcentric_10_3_6_EncV2_DecV5
    hgs_10_3_6 = TFJSPModel_GTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=2,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230515_081059_dtrans_jobcentric_5_3_3_EncV0_DecV0
    hgs_nograph = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=0,
        decoder_version=0,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    
    models = [
        matnet_10_6_6,
        hgnn_10_6_6,
        hgs_selfatten_10_6_6,
        hgs_10_6_6,
        hgs_5_3_3,
        hgs_10_6_3,
        hgs_10_3_6,
        hgs_nograph
    ]
    # : load saved model paths
    model_names = []
    model_loads = []
    for key, val in test_paras['models'].items():
        model_names.append(val['name'])
        model_loads.append(val)
    
    # : load model state_dict
    for i, model in enumerate(models):
        print(f'model_loads[{i}]:{model_loads[i]["name"]}')
        restore_model(model, device, **model_loads[i])
    
    # === validate dispatch rule ===
    if opts.test_dispatch:
        dispatch_spt = dispatchModel(
            rule='spt',
            **model_paras
        )
        models.append(dispatch_spt)
        model_names.append('dispatch_spt')

        dispatch_lpt = dispatchModel(
            rule='lpt',
            **model_paras
        )
        models.append(dispatch_lpt)
        model_names.append('dispatch_lpt')

        dispatch_fifo = dispatchModel(
            rule='fifo',
            **model_paras
        )
        models.append(dispatch_fifo)
        model_names.append('dispatch_fifo')
        
        # dispatch_lum_spt = dispatchModel(
        #     rule='lum_spt',
        #     **model_paras
        # )
        # models.append(dispatch_lum_spt)
        # model_names.append('dispatch_lum_spt')
        
        # dispatch_lum_lpt = dispatchModel(
        #     rule='lum_lpt',
        #     **model_paras
        # )
        # models.append(dispatch_lum_lpt)
        # model_names.append('dispatch_lum_lpt')
    
    
    
    
    # === validate AI algorithms ===
    validate_multi_models(
        vali_env, models, model_names, logger, 
        result_folder, result_log, test_len=test_paras['num_test'],
        test_dataset_list=None,
        test_loader_list=None
    )
        
    # === validate Genetic algorithm ===
    if opts.test_GA:
        for idx, env in enumerate(vali_env):
            batch_size = env.batch_size
            avg_makespan = 0
            avg_runtime = 0
            cunt = min(batch_size, 1)
            parameters = parser_from_case(
                env.proc_times_batch[0], env.trans_times_batch[0], env.nums_ope_batch[0],
                env.num_vehs
            )
            makespan, runtime = GAtest(parameters, print_=False)
            # avg_makespan += makespan
            # avg_runtime += runtime
            # avg_makespan /= cunt
            # avg_runtime /= cunt
        logger.info('{} Score: {:0.2f} | SpandTime: {:0.2f} '.format("GA", makespan, runtime))
    
    
    # === validate meta-heuristic algorithms ===
    # ortools = ORtoolsModel(test_paras, logger, result_folder, result_log)
    # ortools.pre_process(vali_env)
    # ortools.solve_problem_batch()
    
    
    # === save log results of validation ===
    result_dict = {
        'result_log': result_log.get_raw_data()
    }
    torch.save(result_dict, '{}/test_results.pt'.format(result_folder))
    
    
if __name__ == '__main__':
    args = get_options()
    if args.multi_test:
        num_jobs = []
        num_mas = []
        num_vehs = []
        # === multi-eval ===
        # num_jobs += [5, 10, 10]
        # num_mas += [3, 6, 3]
        # num_vehs += [3, 3, 6]
        # ===
        num_jobs += [10, 20, 30, 40, 50]
        num_mas += [6, 10, 15, 20, 25]
        num_vehs += [6, 10, 15, 20, 25]
        # num_jobs += [50]
        # num_mas += [25]
        # num_vehs += [25]
        # ===
        # num_jobs += [50, 50, 50, 50]
        # num_mas += [25, 20, 15, 10]
        # num_vehs += [25, 25, 25, 25]
        for n_jobs, n_mas, n_vehs in zip(num_jobs, num_mas, num_vehs):
            main(args, n_jobs, n_mas, n_vehs)
    else:
        main(args)
    
    