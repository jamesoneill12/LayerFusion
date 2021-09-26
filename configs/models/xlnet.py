import argparse

def get_xlnet_args(
                        vocab_size_or_config_json_file = 267735,
                        cutoffs = [20000, 40000, 200000],
                        d_model = 1024,
                        d_embed = 1024,
                        n_head = 16,
                        d_head = 64,
                        d_inner = 4096,
                        div_val = 4,
                        pre_lnorm = False,
                        n_layer = 18,
                        tgt_len = 128,
                        ext_len = 0,
                        mem_len = 1600,
                        clamp_len = 1000,
                        same_length = True,
                        proj_share_all_but_first = True,
                        attn_type = 0,
                        sample_softmax = -1,
                        adaptive = True,
                        tie_weight = True,
                        dropout = 0.1,
                        dropatt = 0.0,
                        untie_r = True,
                        init = "normal",
                        init_range = 0.01,
                        proj_init_std = 0.01,
                        init_std = 0.02
                       ):

    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_checkpoint_path", default=None, type=str, required=True,
                        help="Path to the TensorFlow checkpoint path.")
    parser.add_argument("--xlnet_config_file", default=None, type=str,required=True,
                        help="The config json file corresponding to the pre-trained XLNet model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, required=True,
                        help="Path to the folder to store the PyTorch model or dataset/vocab.")
    parser.add_argument("--finetuning_task", default=None, type=str,
                        help="Name of a task on which the XLNet TensorFloaw model was fine-tuned")
    parser.add_argument("--vocab_size_or_config_json_file", default = vocab_size_or_config_json_file)
    parser.add_argument("--cutoffs", default = cutoffs)
    parser.add_argument("--d_model", default = d_model)
    parser.add_argument("--d_embed", default = d_embed)
    parser.add_argument("--n_head", default = n_head)
    parser.add_argument("--d_head", default = d_head)
    parser.add_argument("--d_inner", default = d_inner)
    parser.add_argument("--div_val", default = div_val)
    parser.add_argument("--pre_lnorm", default = pre_lnorm)
    parser.add_argument("--n_layer", default = n_layer)
    parser.add_argument("--tgt_len", defualt = tgt_len)
    parser.add_argument("--ext_len", default = ext_len)
    parser.add_argument("--mem_len", default = mem_len)
    parser.add_argument("--clamp_len", default = clamp_len)
    parser.add_argument("--same_length", default = same_length)
    parser.add_argument("--proj_share_all_but_first", default = proj_share_all_but_first)
    parser.add_argument("--attn_type", default = attn_type)
    parser.add_argument("--sample_softmax", default = sample_softmax)
    parser.add_argument("--adaptive", default = adaptive)
    parser.add_argument("--tie_weight", default = tie_weight)
    parser.add_argument("--dropout", default = dropout)
    parser.add_argument("--dropatt", default = dropatt)
    parser.add_argument("--untie_r", default = untie_r)
    parser.add_argument("--init", default = init)
    parser.add_argument("--init_range", default = init_range)
    parser.add_argument("--proj_init_std", default = proj_init_std)
    parser.add_argument("--init_std", default = 0.02)

    args = parser.parse_args()
    return args