import argparse
import os


def get_generative_args(
        model='gan', vocab_size=None, dec_hsz=500, depth=3,
        bsz=128, max_len=51, dropout=0.2, use_cuda=True,
        data= 'flickr30k', encoder = 'resnet', sample_num = 5000,
        save_dir='models', sample_dir='samples', dataset_dir='coco',
        log_dir='tb_logdir/', latent_dim = 128, input_dim=100,
        output_dim=100, input_size=100, g_num=5, pretrained = False,
        g_conv_dim=64, d_conv_dim=64, epochs = 10, version=1,
        mixture=None, mix_noise=0.0, mix_direction=0, mix_lb=0.0,
        mix_ub=0.2, reward='knn', reward_reg = 0.5, lr=5e-3, nsample=10,
        input_file = '', emb_path = '', model_name = '', optim = 'adam'
    ):

    gan_choices = ['acgan', 'qgan', 'began', 'biggan', 'lsgan', 'gan',
     'ebgan', 'infogan', 'rsgan', 'dragan',
     'cgan', 'sagan', 'wgan', 'wgan_gp']

    mixture_choices = ['exp', 'linear', 'sigmoid', 'gumbel', 'static']
    reward_choices = ['knn', 'mmd', 'emd', 'inception', 'frechet']

    parser = argparse.ArgumentParser()

    root = 'D:/data/{}/pretrained_models/{}/'.format(data, model)
    results_path = 'D:/data/{}/results/{}/'.format(data, model)

    # gas params
    parser.add_argument('--reward_reg', type=float, default=reward_reg)
    parser.add_argument('--reward', type=str, default=reward, choices=reward_choices)
    parser.add_argument('--mixture', type=str, default=mixture, choices=mixture_choices)
    parser.add_argument('--mix_noise', type=float, default=mix_noise)
    parser.add_argument('--mix_direction', type=int, default=mix_direction,
                        help="0: discriminator keeps transferred features"
                        "1: discriminator loses transferred features"
                        "2: discriminator and generator features are swapped")
    parser.add_argument('--mix_lb', type=float, default=mix_lb, help='when using a curriculum, lower bound rate')
    parser.add_argument('--mix_ub', type=float, default=mix_ub, help='when using a curriculum, lower bound rate')

    # using pretrained
    parser.add_argument('--root', type=str, default=root, help='root')
    parser.add_argument('--pretrained_model', type=bool, default=pretrained) # used for sagan
    parser.add_argument('--vocab_size', type=int, default=vocab_size,
                        help='vocab size if not already specified')
    parser.add_argument('--dec_hsz', type=int, default=dec_hsz)
    parser.add_argument('--encoder', type=str, default=encoder, help='resnet or gan')
    parser.add_argument('--depth', type=int, default=depth)
    parser.add_argument('--batch_size', type=int, default=bsz) # 64 for sagan
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=max_len)
    parser.add_argument('--dropout', type=float, default=dropout)
    parser.add_argument('--use_cuda', type=bool, default=use_cuda)
    parser.add_argument('--seed', type=int, default=1111)
    # parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                        help='num iterations (default: 120000)')
    parser.add_argument('--grad_clip', type=float, default=1.)
    #parser.add_argument('--n_', type=float, default=1.)

    parser.add_argument('--model', type=str, default=model, choices=gan_choices)
    parser.add_argument('--optim', type=str, default=optim)
    parser.add_argument('--gan_type', type= str, default=model, choices=gan_choices,
                        help='The type of GAN')
    parser.add_argument('--class_num', type=int, default=10)
    # parser.add_argument('--gen_input_dim', type=int, default=256)
    # parser.add_argument('--gen_output_dim', type=int, default=256)
    # parser.add_argument('--gen_input_size', type=int, default=256)

    # Model hyper-parameters
    parser.add_argument('--adv_loss', type=str, default='wgan-gp', choices=['wgan-gp', 'hinge'])
    parser.add_argument('--imsize', type=int, default=input_size)
    parser.add_argument('--g_num', type=int, default=g_num)
    parser.add_argument('--z_dim', type= int, default= latent_dim)
    parser.add_argument('--g_conv_dim', type=int, default=g_conv_dim)
    parser.add_argument('--d_conv_dim', type=int, default=d_conv_dim)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str, default=model+'_'+str(version))

    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--beta1', type=float, default=0.5) # beta 1 0.0 for sagan
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--benchmark_mode', type=bool, default=True)

    # Training setting
    # Step size
    parser.add_argument('--log_step', type=int, default=100) # sagan
    parser.add_argument('--sample_step', type=int, default=100) # sagan
    parser.add_argument('--model_save_step', type=float, default=1.0) # sagan
    parser.add_argument('--results_save_step', type=float, default=50.0)
    parser.add_argument('--results_path', type=str, default=results_path)
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)

    parser.add_argument('--input_dim', type=int, default=input_dim)
    parser.add_argument('--output_dim', type=int, default=output_dim)
    parser.add_argument('--input_size', type=int, default=input_size)
    parser.add_argument('--epoch', type= int, default=epochs)
    parser.add_argument('--sample_num', type= int, default=sample_num)

    # Misc
    parser.add_argument('--dataset', type=str, default=dataset_dir, choices=['lsun', 'celeb', 'coco', 'flickr',
                                                                             'lsun-bed', 'stl10', 'svhn', 'cifar10',
                                                                             'fashion-mnist', 'mnist'])
    parser.add_argument('--data', type=str, default=data)
    parser.add_argument('--image_path', type=str, default=data)
    parser.add_argument('--sample_path', type=str, default=data)
    parser.add_argument('--embedding_path', type=str, default=emb_path, help="only used when using "
                                                                       "pretrained embeddings in npy file")
    parser.add_argument('--save_dir', type=str, default=root+save_dir)
    parser.add_argument('--sample_dir', type=str, default=root+sample_dir)
    parser.add_argument('--result_dir', type=str, default=results_path)
    parser.add_argument('--log_dir', type=str, default=root+log_dir)
    parser.add_argument('--gpu_mode', type= str, default=True)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # evaluation params
    parser.add_argument('--subset_size', type=int, default=5000,
                        help="used when computing kid metric to choose sub size")
    parser.add_argument('--ret_var', type=bool, default=True,
                        help="whether to retain variable or not")

    # rvae params - https://raw.githubusercontent.com/kefirski/pytorch_RVAE/master/train.py
    parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                        help='use cuda (default: True)')
    parser.add_argument('--learning-rate', type=float, default=lr, metavar='LR',
                        help='learning rate (default: 0.00005)')
    parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                        help='load pretrained model (default: False)')
    parser.add_argument('--ce-result', default='', metavar='CE',
                        help='ce result path (default: '')')
    parser.add_argument('--kld-result', default='', metavar='KLD',
                        help='ce result path (default: '')')

    # configs for recurrent VAE in rvae_train.py (originally for paraphrasing)
    parser.add_argument('--num-sample', type=int, default=nsample, metavar='NS',
                        help='num samplings (default: 10)')
    parser.add_argument('--input-file', type=str, default='input.txt', metavar='IF',
                        help='input file with source phrases (default: "input.txt")')
    parser.add_argument('--model-name', default=model_name, metavar='MN',
                        help='name of model to save (default: '')')

    args = parser.parse_args()

    if args.mixture is not None:
        ext = "_" + args.mixture
        args.root += ext
        args.results_path += ext
        args.save_dir += ext
        args.sample_dir += ext
        args.result_dir += ext
        args.log_dir += ext

    return args


"""checking arguments"""
def check_args(args):

    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


def str2bool(v):
    return v.lower() in ('true')


