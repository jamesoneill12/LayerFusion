import argparse


def get_segmentation_config(
        cuda=True,
        model_name = 'resnet_101',fname="ade20k-resnet50dilated-ppm_deepsup.yaml", opts=None, bsize = 64,
        train_bsize=16,  val_bsize=16, w_thresh = None, all_prune = False, task_name = None,
        train_file = None, nbits = 5, exp_name='fcn8s', prune_type = None, prune_perc = 0.5,
        output_dir = None, prune_global = False, lr = 1e-8, weight_decay= 5e-4, input_size= (256, 512),
        momentum=0.95, lr_patience= 100, snapshot='', print_freq=20, val_save_to_img_file=False,
        val_img_sample_rate=0.05, base_size = 513, crop_size = 513, seed = 1234,
        dataset = 'cityscapes', epochs=50
                            ):

    # for cityscapes-fcn/train originally, epochs = 500 and lr = 1e-10

    root = "C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/configs/tasks/segmentation/"
    mod_path = fname.replace(".yaml", "/")
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Validation")
    parser.add_argument( "--cfg", default= mod_path, metavar="FILE", help="path to config file", type=str)
    parser.add_argument( "--gpus", default="0", help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument( "--opts",  help="Modify config options using the command-line", default=opts, nargs=argparse.REMAINDER)
    parser.add_argument( "--cuda", default=cuda)
    parser.add_argument( "--ckpt_path", default='D:/data/{}/mods/ckpt/'.format(dataset))
    parser.add_argument( "--exp_name", default=exp_name)

    parser.add_argument( "--train_batch_size", default=train_bsize)
    parser.add_argument( "--val_batch_size", default=val_bsize)
    parser.add_argument( "--epoch_num", default=epochs)
    parser.add_argument( "--lr", default=lr)
    parser.add_argument( "--weight_decay", default=weight_decay)
    parser.add_argument( "--input_size", default=input_size)
    parser.add_argument( "--momentum", default=momentum)
    parser.add_argument( "--lr_patience", default=lr_patience)
    parser.add_argument( "--snapshot", default=snapshot, help='empty string denotes no snapshot')
    parser.add_argument( "--print_freq", default=print_freq)
    parser.add_argument( "--val_save_to_img_file", default=val_save_to_img_file, type=bool)
    parser.add_argument( "--val_img_sample_rate", default=val_img_sample_rate,
                         help="randomly sample some validation results to display")

    parser.add_argument( "--base_size", default=base_size, help="")
    parser.add_argument( "--crop_size", default=crop_size, help="")
    parser.add_argument( "--bsize", default=bsize, help="")
    parser.add_argument( "--seed", default=seed, help="")
    parser.add_argument( "--dataset", default=dataset, choices=["cityscapes", "pascal", "ade"])

    parser.add_argument("--prune_type", default=prune_type, type=str,
    help = "Picks between ", choices = ['prune', 'svd', 'autoencode', 'merge', 'weighted_merge'])
    parser.add_argument("--prune_perc", default=prune_perc, type=float, help="Picks how to to prune, when used with "
                                                                             "svd or autoencode, this is the percentage"
                                                                             " to reduce the dimensionality of each weight type")
    parser.add_argument("--merge_measure", default='cov', type=str,
                        help="which similarity metric to use when merging layers")
    parser.add_argument("--prune_threshold", default=w_thresh, type=float,
                        help="instead of pruning a % weights, sets a threshold"
    " for weights to be over, else pruned")
    parser.add_argument("--epochs", default=epochs, type=float)
    parser.add_argument("--all_prune", default=all_prune, type=bool,
                        help="if set True, even layer norm and attention is pruned along with dense layers")
    parser.add_argument("--prune_global", default=prune_global, type=bool,
                        help="if set True, threshold chose according to percentage for all "
    "weights combined")
    parser.add_argument("--num_bits", default=nbits, type=bool,
    help = "if set True, threshold chose according to percentage for all "
    "weights combined")

    parser.add_argument("--train_file", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--model", default=model_name, type=str, help="The pretrained model",
                        choices=["resnet_101", "resnet_50", "resnet_18", "u_net", "mobile_net"])
    # {'fcn_resnet50_coco': None,
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    # 'deeplabv3_resnet50_coco': None,
    # 'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'}
    parser.add_argument("--data_dir", default=None, type=str, help="The input data dir."
    " Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=output_dir, type=str,
    help = "The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
    help = "Where do you want to store the pre-trained mods downloaded from s3")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--task_name", default=task_name, type=str, help="The name of the task to train.")

    args = parser.parse_args()
    return args