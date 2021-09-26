import argparse


def get_transxl_config(model_name='transfo-xl-wt103', vsize_json_file=50000, w_thresh=None, all_prune=False,
                       do_retrain=False, compress_step=None,
                       mem_len=500, clamp_len=200, cutoffs=[1000, 30000, 20000], task_name=None,
                       d_model=512, max_ngram_size=4, d_embed=100, n_head=1, d_head=16, epochs=2.0,
                       d_inner=256, div_val=2, n_layer=18, is_training=True, use_labels=True, no_cuda=False,
                       attn_type=0, samp_smx = -1, adaptive_smx=True, tie_weights=True, train_file=None, nbits=5,
                       dropout=0.1, dropatt=0.0, prune_type=None, prune_perc=0.5, output_dir=None, prune_global=False):

    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')

    parser.add_argument("--do_retrain", default=do_retrain, help='if')
    parser.add_argument("--compress_step", default=compress_step)

    parser.add_argument("--prune_type", default=prune_type, type=str,
                        help="Picks between ", choices=['prune', 'svd', 'autoencode', 'merge', 'weighted_merge'])
    parser.add_argument("--prune_perc", default=prune_perc, type=float, help="Picks how to to prune, when used with"
                                                                             "svd or autoencode, this is the percentage"
                                                                             " to reduce the dimensionality of each weight type")
    parser.add_argument("--merge_measure", default='cov', type=str, help="which similarity metric to use when merging layers")
    parser.add_argument("--prune_threshold", default=w_thresh, type=float, help="instead of pruning a % weights, sets a threshold"
                                                                                " for weights to be over, else pruned")
    parser.add_argument("--epochs", default=epochs, type=float)
    parser.add_argument("--all_prune", default=all_prune, type=bool, help="if set True, even layer norm and attention is pruned along"
                                                                       "with dense layers")
    parser.add_argument("--prune_global", default=prune_global, type=bool, help="if set True, threshold chose according to percentage for all "
                                                                                "weights combined")
    parser.add_argument("--num_bits", default=nbits, type=bool,
                        help="if set True, threshold chose according to percentage for all "
                                                                                "weights combined")

    ## Required parameters

    parser.add_argument("--train_file", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--train_corpus", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--data_dir", default=None,  type=str, help="The input data dir."
                                                                    " Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained mods downloaded from s3")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--task_name", default=task_name, type=str, help="The name of the task to train.")


    # transformer model configs
    parser.add_argument('--vocab_size_or_config_json_file', type=int, default=vsize_json_file)
    parser.add_argument('--is_training', type=bool, default=is_training)
    parser.add_argument('--use_labels', type=bool, default=use_labels)

    # parser.add_argument('--mem_len', type=int, default=30)
    # parser.add_argument('--clamp_len', type=int, default=15)
    parser.add_argument('--mem_len', type=int, default=mem_len, help='length of the retained previous heads')
    parser.add_argument('--clamp_len', type=int, default=clamp_len, help='max positional embedding index')
    parser.add_argument('--cutoffs', type=list, default = cutoffs)
    parser.add_argument('--d_model', type=int, default=d_model)
    parser.add_argument('--max_ngram_size', type=int, default=max_ngram_size)
    parser.add_argument('--d_embed', type=int, default=d_embed)
    parser.add_argument('--n_head', type=int, default=n_head)
    parser.add_argument('--d_head', type=int, default=d_head)
    parser.add_argument('--d_inner', type=int, default=d_inner) # 2048)
    parser.add_argument('--div_val', type=int, default=div_val) # might need to be 2
    parser.add_argument('--n_layer', type=int, default=n_layer) # 1 but originally 18 layers
    parser.add_argument('--scope', default=None)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--attn_type', type=int, default=attn_type)
    parser.add_argument('--sample_softmax', type=int, default=samp_smx)
    parser.add_argument('--adaptive', type=bool, default=adaptive_smx)
    parser.add_argument('--tie_weight', type=bool, default=tie_weights)
    parser.add_argument('--dropout', type=int, default=dropout)
    parser.add_argument('--dropatt', type=int, default=dropatt)
    parser.add_argument('--untie_r', type=int, default=True)
    parser.add_argument('--init_range', type=int, default=0.01)
    parser.add_argument('--proj_init_std', type=int, default=0.01)
    parser.add_argument('--init_std', type=int, default=0.02)

    # taken from the run_transfo_xl.py in examples of bert_pretrained
    parser.add_argument('--model_name', type=str, default=model_name,  help='pretrained model name')
    parser.add_argument('--split', type=str, default='test',   choices=['all', 'valid', 'test'],
                        help='which split to evaluate')
    parser.add_argument('--tgt_len', type=int, default=56,
                        help='number of tokens to predict')
    parser.add_argument('--ext_len', type=int, default=0,
                        help='length of the extended context')
    parser.add_argument('--pre_lnorm', type=bool, default=False,
                        help='max positional embedding index')
    parser.add_argument('--same_length', type=bool, default=True,
                        help='set same length attention with masking')
    parser.add_argument('--proj_share_all_but_first', type=bool, default=True)
    parser.add_argument('--no_cuda', type=bool, default=no_cuda,
                        help='Do not use CUDA even though CUA is available')
    parser.add_argument('--work_dir', type=str, default=True, help='path to the work_dir')
    parser.add_argument('--no_log', default=True, help='do not log the eval result')
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    assert args.ext_len >= 0, 'extended context length must be non-negative'
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    return args


def get_transfol_text8_config(base=True):
    args = get_transxl_config()
    args.data = 'data/text8/'
    if base:
        args.adaptive = True
        args.div_val = 4
        args.n_layer = 12
        args.d_model = 512
        args.n_head = 8
        args.d_head = 64
        args.d_inner = 2048
        args.dropout = 0.1
        args.dropatt = 0.
        args.optim = 'adam'
        args.lr = 0.00025
        args.warmup_step = 0
        args.max_step = 4000000
        args.tgt_len = 512
        args.mem_len = 512
        args.eval_tgt_len = 128
        args.batch_size = 22
        args.multi_gpu = False
    else:
        args.adaptive = True
        args.div_val = 4
        args.n_layer = 24
        args.d_model = 1024
        args.n_head = 8
        args.d_head = 128
        args.d_inner = 3072
        args.dropout = 0.15
        args.dropatt = 0.15
        args.optim = 'adam'
        args.lr = 0.00025
        args.tgt_len = 768
        args.mem_len = 768
        args.eval_tgt_len = 128
        args.batch_size = 64
        args.max_step = 400000
    return args


def get_transfol_enwik8_config(base=True):
    args = get_transxl_config()
    args.data = 'data/enwik8/'

    if base:
        args.data = True
        args.dataset = 'enwik8'
        args.div_val = 4
        args.n_layer = 24
        args.d_model = 1024
        args.n_head = 8
        args.d_head = 128
        args.d_inner = 3072
        args.dropout = 0.15
        args.dropatt = 0.15
        args.optim = 'adam'
        args.lr = 0.00025
        args.warmup_step = 4000
        args.max_step = 4000000
        args.tgt_len = 768
        args.mem_len = 768
        args.eval_tgt_len = 128
        args.batch_size = 64
        args.multi_gpu = False
    else:
        args.adaptive = True
        args.div_val = 4
        args.n_layer = 24
        args.d_model = 1024
        args.n_head = 8
        args.d_head = 128
        args.d_inner = 3072
        args.dropout = 0.15
        args.dropatt = 0.15
        args.optim = 'adam'
        args.lr = 0.00025
        args.tgt_len = 768
        args.mem_len = 768
        args.eval_tgt_len = 128
        args.batch_size = 64
        args.max_step = 400000
    return args


def get_transfol_wiki103_config(args=None, base=True):

    if args is None: args = get_transxl_config()
    args.data = 'data/wikitext-103/'

    if base:
        args.adaptive = True
        args.div_val = 4
        args.n_layer = 16
        args.d_model = 410
        args.n_head = 10
        args.d_head = 41
        args.d_inner = 2100
        args.dropout = 0.1
        args.dropatt = 0.
        args.optim = 'adam'
        args.lr = 0.00025
        args.warmup_step = 0
        args.max_step = 2000000
        args.tgt_len = 150
        args.mem_len = 150
        args.eval_tgt_len = 150
        args.batch_size = 60
        args.multi_gpu = False
    else:
        args.adaptive = True
        args.div_val = 4
        args.n_layer = 18
        args.d_model = 1024
        args.n_head = 16
        args.d_head = 64
        args.d_inner = 4096
        args.dropout = 0.2
        args.dropatt = 0.2
        args.optim = 'adam'
        args.lr = 0.00025
        args.warmup_step = 16000
        args.max_step = 4000000
        args.tgt_len = 384
        args.mem_len = 384
        args.eval_tgt_len = 128
        args.batch_size = 128
    return args


def wiki_german_config(compress_type=None, prune_perc=0.0, epochs=2.0):
    from macros import WIKITEXT3_TRAIN
    out_dir = "C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/models/networks/compress/results/"
    wiki_ex = WIKITEXT3_TRAIN.replace("wikitext-3/wiki.train.tokens", "wikitext_alt/train.txt")
    no_cuda = False
    args = get_transxl_config(train_file=wiki_ex, epochs=epochs, no_cuda=no_cuda,
                              prune_type=compress_type, output_dir=out_dir,
                              task_name='language modelling: WikiText-103',
                              prune_perc=prune_perc)
    args = get_transfol_wiki103_config(args)
    return args



"""Constructs TransfoXLConfig.
    Args:
        vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TransfoXLModel` or a configuration json file.
        cutoffs: cutoffs for the adaptive softmax
        d_model: Dimensionality of the model's hidden states.
        d_embed: Dimensionality of the embeddings
        d_head: Dimensionality of the model's heads.
        div_val: divident value for adapative input and softmax
        pre_lnorm: apply LayerNorm to the input instead of the output
        d_inner: Inner dimension in FF
        n_layer: Number of hidden layers in the Transformer encoder.
        n_head: Number of attention heads for each attention layer in
            the Transformer encoder.
        tgt_len: number of tokens to predict
        ext_len: length of the extended context
        mem_len: length of the retained previous heads
        same_length: use the same attn length for all tokens
        proj_share_all_but_first: True to share all but first projs, False not to share.
        attn_type: attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
        clamp_len: use the same pos embeddings after clamp_len
        sample_softmax: number of samples in sampled softmax
        adaptive: use adaptive softmax
        tie_weight: tie the word embedding and softmax weights
        dropout: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        dropatt: The dropout ratio for the attention probabilities.
        untie_r: untie relative position biases           
        embd_pdrop: The dropout ratio for the embeddings.
        init: parameter initializer to use
        init_range: parameters initialized by U(-init_range, init_range).
        proj_init_std: parameters initialized by N(0, init_std)
        init_std: parameters initialized by N(0, init_std)
    """