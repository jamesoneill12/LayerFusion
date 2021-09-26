import argparse
# model_name_or_path 'openai-gpt2' did not work, just gpt2 now


def get_gpt2_config(model_name='gpt2', do_train=True, do_eval=True, do_test=True, context_length=1024,
                    do_retrain=False, compress_step = 0, start_from_pretrained=True,
                    distributed=True, vsize_json_file=50000, w_thresh=None, all_prune=False, seed=42,
                    num_train_epochs=3, train_batch_size=8, eval_batch_size=16, test_batch_size=16, batch_size=-1,
                    max_grad_norm=1, learning_rate=6.25e-5, warmup_proportion=0.002, lr_schedule='warmup_linear',
                    weight_decay=0.01, lm_coef=0.9, n_valid=374, length=-1, temp=1.0, nsamples=1,
                    task_name=None, epochs=2.0, prune_type=None, prune_perc=0.5, out_dir=None,
                    prune_global=False, train_file=None, val_file=None, test_file=None, nbits=5):

    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument("--start_from_pretrained", default=start_from_pretrained)
    parser.add_argument("--do_retrain", default=do_retrain, help='if')
    parser.add_argument("--compress_step", default=compress_step)
    parser.add_argument("--prune_type", default=prune_type, type=str,
                        help="Picks between ", choices=['prune', 'svd', 'autoencode', 'merge', 'weighted_merge'])
    parser.add_argument("--prune_perc", default=prune_perc, type=float, help="Picks how to to prune, when used with"
                                                                             "svd or autoencode, this is the percentage"
                                                                             " to reduce the dimensionality of each weight type")
    parser.add_argument("--merge_measure", default='cov', type=str, help="which similarity metric to use when merging layers")
    parser.add_argument("--prune_threshold", default=w_thresh, type=float,
                        help="instead of pruning a % weights, sets a threshold for weights to be over, else pruned")
    parser.add_argument("--epochs", default=epochs, type=float)
    parser.add_argument("--all_prune", default=all_prune, type=bool,
                        help="if set True, even layer norm and attention is pruned along with dense layers")
    parser.add_argument("--prune_global", default=prune_global, type=bool,
                        help="if set True, threshold chose according to percentage for all weights combined")
    parser.add_argument("--num_bits", default=nbits, type=bool,
                        help="if set True, threshold chose according to percentage for all weights combined")


    ## Required parameters
    parser.add_argument("--train_dataset", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--val_dataset", default=val_file, type=str, help="The input val corpus.")
    parser.add_argument("--test_dataset", default=test_file, type=str, help="The input test corpus.")
    parser.add_argument("--train_corpus", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--data_dir", default=None,  type=str, help="The input data dir."
                                                                    " Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=out_dir, type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--task_name", default=task_name, type=str, help="The name of the task to train.")

    parser.add_argument('--model_name', type=str, default=model_name, choices=['gpt2', 'openai-gpt'],
                        help='pretrained model name OR path to local checkpoint')
    parser.add_argument("--do_train", default=do_train, help="Whether to run training.")
    parser.add_argument("--do_eval", default=do_eval, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=do_test, help="Whether to run eval on the dev set.")
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--num_train_epochs', type=int, default=num_train_epochs)
    parser.add_argument('--train_batch_size', type=int, default=train_batch_size)
    parser.add_argument('--eval_batch_size', type=int, default=eval_batch_size)
    parser.add_argument('--test_batch_size', type=int, default=test_batch_size)
    parser.add_argument('--max_grad_norm', type=int, default=max_grad_norm)
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--warmup_proportion', type=float, default=warmup_proportion)
    parser.add_argument('--lr_schedule', type=str, default=lr_schedule)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--lm_coef', type=float, default=lm_coef)
    parser.add_argument('--n_valid', type=int, default=n_valid)
    parser.add_argument("--nsamples", type=int, default=nsamples)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--length", type=int, default=length)
    parser.add_argument("--temperature", type=float, default=temp)
    parser.add_argument('--context_length', type=int, default=context_length)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--distributed', default=distributed, help='Run distributed training')
    parser.add_argument('--run_name', type=str, default='', help="Name of this run for easier tensorboard analysis")
    parser.add_argument('--logdir',type=str, default='/tmp/runs', help="location of logging directory")

    args = parser.parse_args()
    return args


def wiki_config(
        model_name, compress_type=None, prune_perc=0.0, epochs=2.0,
            do_train=True, do_eval=True, do_test=True, start_from_pretrained=True,
                train_batch_size=8, eval_batch_size=8, test_batch_size = 16,
                ):
    from macros import WIKITEXT2_TRAIN, WIKITEXT2_VALID, WIKITEXT2_TEST
    out_dir = "C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/models/networks/compress/results/"
    context_length = 512 if 'openai' in model_name else 1024

    args = get_gpt2_config(
        train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, test_batch_size=test_batch_size,
        start_from_pretrained=start_from_pretrained,
        model_name=model_name, train_file=WIKITEXT2_TRAIN, val_file=WIKITEXT2_VALID, test_file=WIKITEXT2_TEST,
        epochs=epochs, prune_type=compress_type, out_dir=out_dir, task_name='language modelling: WikiText-103',
        prune_perc=prune_perc, do_train=do_train, do_eval=do_eval, do_test=do_test, context_length=context_length
    )
    return args


class Args:

    def __init__(self, model_name, compress_type = 'prune', epochs = 2.0,
                 prune_perc = 30, compress_steps = 10, prune_global=True):
        from macros import WIKITEXT2_TRAIN, WIKITEXT2_VALID, WIKITEXT2_TEST

        self.prune_perc = prune_perc
        self.context_length = 1024
        self.prune_perc = 30.0  # [10.0, 20.0, 30.0, 50.0, 70.0]
        self.all_prune = False
        self.prune_global = prune_global
        self.start_from_pretrained = True
        self.do_train = True
        self.do_retrain = True
        self.compress_step = compress_steps
        self.train_dataset = WIKITEXT2_TRAIN
        self.val_dataset = WIKITEXT2_VALID
        self.test_dataset = WIKITEXT2_TEST
        self.train_batch_size = 1
        self.test_batch_size = 4
        self.output_dir = "C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/" \
                          "models/networks/compress/results/"
        self.model_name = model_name
        self.epochs = epochs
        self.num_train_epochs = epochs
        self.prune_type = compress_type
        self.learning_rate = 6.25e-5
        self.warmup_proportion = 0.002
        self.weight_decay = 0.01
        self.max_grad_norm = 1
        self.lm_coef = 0.9
        self.n_valid = 374
        self.length = -1
        self.temp = 1.0
        self.nsamples = 1