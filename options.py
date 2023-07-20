import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")
    parser.add_argument('--algorithm', type=str, default="matnet", help="one of matnet, hgnn, attention, matnet_attention")
    parser.add_argument('--cuda', type=int, default=0, help="cuda device number")
    parser.add_argument('--log_file_desc', type=str, default='heteronet_train', help='file name of results to store')
    parser.add_argument('--enable_change_paras', action='store_true', help='train the model over the changing environment')
    parser.add_argument('--test_seed', default=11, type=int, help='test seed')
    parser.add_argument('--CPoptimizer', action='store_true')
    parser.add_argument('--static', action='store_true', help='generate env parameters with static method')
    parser.add_argument('--metarl', action='store_true', help='use metarl training')
    parser.add_argument('--metarl_subgraphs', action='store_true', help='use metarl training')
    parser.add_argument('--subprob', action='store_true', help='use metarl training')
    parser.add_argument('--multi_test', action='store_true', help='run multiple tests')
    parser.add_argument('--job_centric', action='store_true', help='use job instead of ope')
    parser.add_argument('--test_GA', action='store_true', help='run GA algorithm tests')
    parser.add_argument('--test_dispatch', action='store_true', help='run dispatch rule tests')
    parser.add_argument('--benchmark_file', type=str, default='', help='benchpark dataset file name')
    parser.add_argument('--new_job', action='store_true', help='new job insertion')

    # Environment
    # parser.add_argument('--num_jobs', type=int, default=10, help="The number of jobs")
    # parser.add_argument('--nums_ope', type=list, default=[4, 4, 6, 5, 6, 4, 6, 4, 5, 4], help="The number of operations for each job")
    # parser.add_argument('--num_mas', type=int, default=5, help="The number of machines")
    # parser.add_argument('--batch_size', type=int, default=50, help='Number of instances')
    # parser.add_argument('--ope_feat_dim', type=int, default=6, help='opeartion feature dimension')
    # parser.add_argument('--ma_feat_dim', type=int, default=3, help='machine feature dimension')
    
    # # Model
    # parser.add_argument('--in_size_ma', type=int, default=3, help='should be same with ma_feat_dim')
    # parser.add_argument('--out_size_ma', type=int, default=10, help='')
    # parser.add_argument('--in_size_ope', type=int, default=6, help='should be same with ope_feat_dim')
    # parser.add_argument('--out_size_ope', type=int, default=10, help='')
    # parser.add_argument('--embedding_dim', type=int, default=128, help='')
    # parser.add_argument('--hidden_dim', type=int, default=128, help='')
    # parser.add_argument('--n_encode_layers', type=int, default=3, help='')
    # parser.add_argument('--tanh_clipping', type=int, default=10, help='')
    # parser.add_argument('--normalization', type=str, default="batch", help='')
    # parser.add_argument('--action_dim', type=int, default=1, help='')
    
    
    
    # # Data
    # parser.add_argument('--problem', default='pdp', help="The problem to solve, default 'pdp'")
    # parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    # parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    # parser.add_argument('--val_size', type=int, default=10000,
    #                     help='Number of instances used for reporting validation performance')
    # parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # # Model
    # parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    # parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    # parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    # parser.add_argument('--n_encode_layers', type=int, default=3,
    #                     help='Number of layers in the encoder/critic network')
    # parser.add_argument('--tanh_clipping', type=float, default=10.,
    #                     help='Clip the parameters to within +- this value using tanh. '
    #                          'Set to 0 to not perform any clipping.')
    # parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # # Training
    # parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    # parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    # parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    # parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    # parser.add_argument('--n_epochs', type=int, default=800, help='The number of epochs to train')
    # parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    # parser.add_argument('--max_grad_norm', type=float, default=1.0,
    #                     help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    # parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    # parser.add_argument('--exp_beta', type=float, default=0.8,
    #                     help='Exponential moving average baseline decay (default 0.8)')
    # parser.add_argument('--baseline', default=None,
    #                     help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    # parser.add_argument('--bl_alpha', type=float, default=0.05,
    #                     help='Significance in the t-test for updating rollout baseline')
    # parser.add_argument('--bl_warmup_epochs', type=int, default=None,
    #                     help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
    #                          'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    # parser.add_argument('--eval_batch_size', type=int, default=1024,
    #                     help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    # parser.add_argument('--shrink_size', type=int, default=None,
    #                     help='Shrink the batch size if at least this many instances in the batch are finished'
    #                          ' to save memory (default None means no shrinking)')
    # parser.add_argument('--data_distribution', type=str, default=None,
    #                     help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    # parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    # parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    # parser.add_argument('--run_name', default='run', help='Name to identify the run')
    # parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    # parser.add_argument('--epoch_start', type=int, default=0,
    #                     help='Start at epoch # (relevant for learning rate decay)')
    # parser.add_argument('--checkpoint_epochs', type=int, default=1,
    #                     help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    # parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    # parser.add_argument('--resume', help='Resume from previous checkpoint file')
    # parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    # parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args(args)

    # opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    # opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    # opts.save_dir = os.path.join(
    #     opts.output_dir,
    #     "{}_{}".format(opts.problem, opts.graph_size),
    #     opts.run_name
    # )
    # if opts.bl_warmup_epochs is None:
    #     opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    # assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    # assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
