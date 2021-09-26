import argparse


def get_trans_config():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    # https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    # new transformer model configs
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--submission_dir', type=str, default='submission/')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default='model/encoder_bpe_40000.json')
    parser.add_argument('--bpe_path', type=str, default='model/vocab_40000.bpe')
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)


    # transformer model configs
    parser.add_argument('--total_key_depth', type=int, default=8)
    parser.add_argument('--total_value_depth', type=int, default=8)
    parser.add_argument('--filter_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--in_dropout', type=float, default=0.1)
    parser.add_argument('--layer_dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--relu_dropout', type=float, default=0.)

    parser.add_argument('--vocab_size_or_config_json_file', type=int, default=50000)
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--use_labels', type=bool, default=True)

    parser.add_argument('--mem_len', type=int, default=800, help='length of the retained previous heads')
    parser.add_argument('--clamp_len', type=int, default=500, help='max positional embedding index')
    parser.add_argument('--d_model', type=int, default=512, help='hidden dimension of the model')
    parser.add_argument('--max_ngram_size', type=int, default=4)
    parser.add_argument('--d_embed', type=int, default=256)
    parser.add_argument('--d_head', type=int, default=32)
    parser.add_argument('--d_inner', type=int, default=2048)
    parser.add_argument('--div_val', type=int, default=2) # might need to be 2
    parser.add_argument('--scope', default=None)
    parser.add_argument('--return_probs', default=False)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('--attn_type', type=int, default=0)
    parser.add_argument('--sample_softmax', type=int, default=-1)
    parser.add_argument('--adaptive', type=bool, default=True)
    parser.add_argument('--tie_weight', type=int, default=True)
    parser.add_argument('--dropout', type=int, default=.1)
    parser.add_argument('--dropatt', type=int, default=0.)
    parser.add_argument('--untie_r', type=int, default=True)
    parser.add_argument('--init_range', type=int, default=0.01)
    parser.add_argument('--proj_init_std', type=int, default=0.01)
    parser.add_argument('--init_std', type=int, default=0.02)

    parser.add_argument('--model_name', type=str, default='transfo-xl-wt103',
                        help='pretrained model name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
    parser.add_argument('--tgt_len', type=int, default=40,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--pre_lnorm', type=bool, default=False,
                        help='max positional embedding index')
    parser.add_argument('--same_length', type=bool, default=True,
                        help='set same length attention with masking')
    parser.add_argument('--proj_share_all_but_first', type=bool, default=True)
    parser.add_argument('--no_cuda', type=bool, default=True,
                        help='Do not use CUDA even though CUA is available')
    parser.add_argument('--work_dir', type=str, default=True, help='path to the work_dir')
    parser.add_argument('--no_log', default=True, help='do not log the eval result')
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    return args
