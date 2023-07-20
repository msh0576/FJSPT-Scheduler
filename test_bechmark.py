import torch
import os

from utils.utils import *
from utils.utils_fjspt import build_caseConfig, parse
from options import get_options
from logging import getLogger
from train import setup_seed

from test import _print_config
from DHJS_models.TFJSPTester import generate_vali_env, restore_model, validate_multi_models
from GA_models.GATester import GAtest, parser_from_case

from env.case_generator_v2 import CaseGenerator
from env.tfjsp_env import TFJSPEnv

from DTrans_models.TFJSPModel_dtrans import TFJSPModel_DTrans
from matnet_models.TFJSPTrainer_matnet import TFJSPModel_matnet
from hgnn_models.TFJSPModel_hgnn import TFJSPModel_hgnn
from dispatching_models.dispatchModel import dispatchModel
from GTrans_models.TFJSPModel_gtrans import TFJSPModel_GTrans


def main(opts, benchmark_file=None):
    # === load config ===
    with open("./configs/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    optimizer_paras = load_dict["optimizer_paras"]
    logger_paras = load_dict["logger_paras"]
    test_paras = load_dict["test_paras"]
    
    
    # === setting benchmark dataset path ===
    benchmark_path_base = './BenchmarkDataset/dataset/'
    if benchmark_file is not None:
        benchmark_path = os.path.join(benchmark_path_base, benchmark_file)
        opts.log_file_desc = f'test_{benchmark_file}'
    else:
        benchmark_path = os.path.join(benchmark_path_base, opts.benchmark_file)
        opts.log_file_desc = f'test_{opts.benchmark_file}'

    # === runtime config change ===
    model_paras['sqrt_embedding_dim'] = model_paras['embedding_dim']**(1/2)
    model_paras['sqrt_qkv_dim'] = model_paras['qkv_dim']**(1/2)
    model_paras['ms_layer1_init'] = (1/2)**(1/2)
    model_paras['ms_layer2_init'] = (1/16)**(1/2)
    
    logger_paras['log_file']['desc'] = opts.log_file_desc
    logger_paras['log_file']['filepath'] = './result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'
    
    setup_seed(seed=opts.test_seed)
    create_logger(**logger_paras)
    _print_config()
    
    device = torch.device("cuda:"+str(opts.cuda) if torch.cuda.is_available() else "cpu")
    
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
    env_paras["batch_size"] = 1 # for test_benchmark
    
    test_paras["batch_size"] = env_paras["batch_size"]
    model_paras["batch_size"] = env_paras["batch_size"]
    model_paras["checkpoint_encoder"] = opts.checkpoint_encoder
    model_paras["algorithm"] = opts.algorithm
    
    
        
    
    # === load benchmark dataset ===
    parameters = parse(benchmark_path)
    case_config = build_caseConfig(parameters, benchmark_path_base)
    
    # === transform datset into case_generator format ===
    case = CaseGenerator(
        num_jobs=None, num_opes=None, num_mas=None, num_vehs=None, device=device,
        opes_per_job_min=None, opes_per_job_max=None,
        data_source='benchmark', case_config=case_config
    )
    env_paras["proctime_per_ope_max"] = 30
    env_paras["transtime_btw_ma_max"] = 20
    test_paras["num_jobs"] = case_config['num_jobs']
    test_paras["num_mas"] = case_config['num_mas']
    test_paras["num_vehs"] = case_config['num_vehs']
    test_paras["num_opes"] = case_config['num_opes']
    
    model_paras["proctime_per_ope_max"] = env_paras["proctime_per_ope_max"]
    model_paras["transtime_btw_ma_max"] = env_paras["transtime_btw_ma_max"]
    
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras['num_opes'] = test_paras["num_opes"]
    model_paras['num_mas'] = test_paras["num_mas"]
    model_paras['num_vehs'] = test_paras["num_vehs"]
    
    logger = getLogger(name='tester')
    result_folder = get_result_folder()
    result_log = LogData()
    
    env = TFJSPEnv(case=case, env_paras=test_paras)
    vali_env = [env]
    
    logger.info('{} | job_{}, ma_{}, veh_{}'.
                format(opts.log_file_desc, case_config['num_jobs'],case_config['num_mas'], case_config['num_vehs']))
    
    
    model_paras['encoder_layer_num'] = 2
    # : 20230426_144354_matnet_jobcentric_10_6_6
    matnet_jobcentric_10_6_6 = TFJSPModel_matnet(
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
    hgnn_jobcentric_10_6_6 = TFJSPModel_hgnn(env_paras, model_paras)
    
    # : 20230428_083724_dtrans_jobcentric_10_6_6_EncV5_DecV5
    dtrans_jobcentric_10_6_6_EncV5_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=5,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230428_084322_dtrans_jobcentric_10_6_6_EncV0_DecV5
    dtrans_jobcentric_10_6_6_EncV0_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=0,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230428_083853_dtrans_jobcentric_10_6_6_EncV5_DecV0
    dtrans_jobcentric_10_6_6_EncV5_DecV0 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=5,
        decoder_version=0,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230428_172447_dtrans_jobcentric_10_6_6_EncV1_DecV5
    # : self-node transformer 가 제안기법이랑 성능이 어떻게 다른지 비교하기 위해
    # : 모든 모드에게 MHA 적용, edge 고려 없이
    dtrans_jobcentric_10_6_6_EncV1_DecV5 = TFJSPModel_DTrans(
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
    
    # : 20230429_101909_dtrans_jobcentric_10_6_6_EncV3_DecV5
    model_paras['encoder_layer_num'] = 3
    dtrans_jobcentric_10_6_6_EncV3_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=3,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    model_paras['encoder_layer_num'] = 2
    
    # : 20230501_091050_dtrans_jobcentric_10_6_6_EncV6_DecV5
    dtrans_jobcentric_10_6_6_EncV6_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=6,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230501_091135_dtrans_jobcentric_10_6_6_EncV4_DecV5
    dtrans_jobcentric_10_6_6_EncV4_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=4,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230502_072132_dtrans_jobcentric_10_6_6_EncV7_DecV5
    dtrans_jobcentric_10_6_6_EncV7_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=7,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230503_095120_dtrans_jobcentric_10_6_6_EncV8_DecV5
    dtrans_jobcentric_10_6_6_EncV8_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=8,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230503_165454_gtrans_jobcentric_10_6_6_EncV1_DecV5
    gtrans_jobcentric_10_6_6_EncV1_DecV5 = TFJSPModel_GTrans(
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
    gtrans_jobcentric_10_6_6_EncV2_DecV5 = TFJSPModel_GTrans(
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
    gtrans_jobcentric_5_3_3_EncV2_DecV5 = TFJSPModel_GTrans(
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
    gtrans_jobcentric_10_6_3_EncV2_DecV5 = TFJSPModel_GTrans(
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
    gtrans_jobcentric_10_3_6_EncV2_DecV5 = TFJSPModel_GTrans(
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
    
    # : 20230510_084311_dtrans_jobcentric_10_6_6_EncV4_DecV5
    dtrans_jobcentric_10_6_6_EncV4_DecV5_sec = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=4,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    # : 20230510_132332_dtrans_jobcentric_5_3_3_EncV4_DecV5
    dtrans_jobcentric_5_3_3_EncV4_DecV5 = TFJSPModel_DTrans(
        embedding_dim_=model_paras["embedding_dim"],
        hidden_dim_=model_paras["hidden_dim"],
        problem=None,
        ope_feat_dim=model_paras["in_size_ope"],
        ma_feat_dim=model_paras["in_size_ma"],
        veh_feat_dim=model_paras["in_size_veh"],
        mask_inner=True,
        mask_logits=True,
        encoder_version=4,
        decoder_version=5,
        meta_rl=train_paras['meta_rl'] if train_paras['meta_rl']['enable'] else None,
        **model_paras
    ).to(device)
    
    
    models = [
        matnet_jobcentric_10_6_6,
        hgnn_jobcentric_10_6_6,
        dtrans_jobcentric_10_6_6_EncV5_DecV5,
        dtrans_jobcentric_10_6_6_EncV0_DecV5,
        dtrans_jobcentric_10_6_6_EncV5_DecV0,
        dtrans_jobcentric_10_6_6_EncV1_DecV5,
        dtrans_jobcentric_10_6_6_EncV3_DecV5,
        dtrans_jobcentric_10_6_6_EncV6_DecV5,
        dtrans_jobcentric_10_6_6_EncV4_DecV5,
        dtrans_jobcentric_10_6_6_EncV7_DecV5,
        dtrans_jobcentric_10_6_6_EncV8_DecV5,
        gtrans_jobcentric_10_6_6_EncV1_DecV5,
        gtrans_jobcentric_10_6_6_EncV2_DecV5,
        gtrans_jobcentric_5_3_3_EncV2_DecV5,
        gtrans_jobcentric_10_6_3_EncV2_DecV5,
        gtrans_jobcentric_10_3_6_EncV2_DecV5,
        dtrans_jobcentric_10_6_6_EncV4_DecV5_sec,
        dtrans_jobcentric_5_3_3_EncV4_DecV5
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
    
    
    
    # === save log results of validation ===
    result_dict = {
        'result_log': result_log.get_raw_data()
    }
    torch.save(result_dict, '{}/test_results.pt'.format(result_folder))
    

if __name__ == '__main__':
    args = get_options()
    
    if args.multi_test:
        # === multi-eval ===
        benchmark_files = ['mt10c1.fjs', 'mt10cc.fjs', 'mt10x.fjs', 'mt10xx.fjs', 'mt10xxx.fjs',
                           'mt10xy.fjs', 'mt10xyz.fjs', 'setb4c9.fjs', 'setb4cc.fjs', 'setb4x.fjs',
                           'setb4xx.fjs', 'setb4xxx.fjs', 'setb4xy.fjs', 'setb4xyz.fjs', 'seti5c12.fjs',
                           'seti5cc.fjs', 'seti5x.fjs', 'seti5xx.fjs', 'seti5xxx.fjs', 'seti5xxx.fjs',
                           'seti5xy.fjs', 'seti5xyz.fjs']
        # benchmark_files = ['Mk01.fjs', 'Mk02.fjs', 'Mk03.fjs', 'Mk04.fjs', 'Mk05.fjs',
        #                    'Mk06.fjs', 'Mk07.fjs', 'Mk08.fjs', 'Mk09.fjs', 'Mk10.fjs']
        for benchmark_file in benchmark_files:
            main(args, benchmark_file)
    else:
        main(args)
    