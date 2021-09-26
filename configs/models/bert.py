"""
https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
BERT, however, wants data to be in a tsv file with a specific format as given below (Four columns, and no header row).
Column 0: An ID for the row
Column 1: The label for the row (should be an int)
Column 2: A column of the same letter for all rows. BERT wants this so we’ll give it, but we don’t have a use for it.
Column 3: The text for the row
Let’s make things a little BERT-friendly.
"""

import argparse


def get_bert_config(train_file=None, bert_model = 'bert-base-cased', task_name=None, compress_step=20,
                    do_retrain = False,
                    max_seq_len = 128, train_bsize = 128, eval_bsize=8, num_train_epochs=5.0,
                    fp=False, is_train=True, input_mask = True, v_size = 100, all_prune=False,
                    hidden_size = 100, num_hid_layers= 12, num_att_heads = 4, no_cuda=True,
                    inter_size = 37, h_act = "gelu", h_drop_prob = 0.1, output_dir=None,
                    att_prob_drop_prob = 0.1, max_p_emb = 512, type_vsize = 16, nbits=5,
                    type_sequence_label_size = 2, init_range = 0.02, n_classes=100,
                    lr=3e-5, prune_type=None, prune_perc=0.5, w_thresh=None, on_mem=False):

    parser = argparse.ArgumentParser()

    # prune parameters
    parser.add_argument("--prune_type", default=prune_type, type=str,
                        help="Picks between ", choices=['prune', 'svd', 'autoencode', 'merge', 'weighted_merge'])

    parser.add_argument("--do_retrain", default=do_retrain, help='if')
    parser.add_argument("--compress_step", default=compress_step)
    parser.add_argument("--prune_perc", default=prune_perc, type=float, help="Picks how to to prune, when used with"
                                                                             "svd or autoencode, this is the percentage"
                                                                             " to reduce the dimensionality of each weight type")
    parser.add_argument("--merge_measure", default='cov', type=str, help="which similarity metric to use when merging layers")
    parser.add_argument("--prune_threshold", default=w_thresh, type=float, help="instead of pruning a % weights, sets a threshold"
                                                                                " for weights to be over, else pruned")
    parser.add_argument("--all_prune", default=all_prune, type=bool, help="if set True, even layer norm and attention is pruned along"
                                                                       "with dense layers")
    parser.add_argument("--distilled", type=bool, help="if chosen, we pick distilBERT")
    parser.add_argument("--num_bits", default=nbits, type=bool,
                        help="if set True, threshold chose according to percentage for all "
                                                                                "weights combined")

    ## Required parameters
    parser.add_argument("--train_file", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--train_corpus", default=train_file, type=str, help="The input train corpus.")
    parser.add_argument("--data_dir", default=None,  type=str, help="The input data dir."
                                                                    " Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=bert_model, type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained mods downloaded from s3")

    parser.add_argument("--task_name", default=task_name, type=str, help="The name of the task to train.")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=max_seq_len, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")

    parser.add_argument("--train_bsize", default=train_bsize, type=int, help="Total batch size for training.")
    parser.add_argument("--train_batch_size", default=train_bsize, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=eval_bsize, type=int,  help="Total batch size for eval.")

    parser.add_argument("--learning_rate",default=lr, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs", default=num_train_epochs, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", default=no_cuda, help="Whether not to use CUDA when available")

    parser.add_argument("--on_memory", default=on_mem, help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case", default=True, help="Whether to lower case the input text. True "
                                                              "for uncased mods, False for cased mods.")
    parser.add_argument("--local_rank",type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16', default=fp, help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--vocab_size',  type=int,  default=v_size)
    parser.add_argument('--batch_size',  type=int,  default=train_bsize)
    parser.add_argument('--hidden_size',  type=int,  default=hidden_size)
    parser.add_argument('--num_hidden_layers',  type=int,  default=num_hid_layers)
    parser.add_argument('--num_attention_heads',  type=int,  default=num_att_heads)
    parser.add_argument('--intermediate_size',  type=int,  default=inter_size)
    parser.add_argument('--hidden_act',  type=str,  default=h_act)
    parser.add_argument('--hidden_dropout_prob',  type=int,  default=h_drop_prob)
    parser.add_argument('--attention_probs_dropout_prob',  type=int,  default=att_prob_drop_prob)
    parser.add_argument('--max_position_embeddings',  type=int,  default=max_p_emb)
    parser.add_argument('--type_vocab_size',  type=int,  default=type_vsize)
    parser.add_argument('--initializer_range',  type=int,  default=init_range)
    parser.add_argument('--n_classes',  type=int,  default=n_classes)
    parser.add_argument('--is_training',  type=bool,  default=is_train)
    parser.add_argument('--use_input_mask',  type=bool,  default=input_mask)
    parser.add_argument('--type_sequence_label_size',  type=bool,  default=type_sequence_label_size)

    parser.add_argument('--overwrite_output_dir', default=True, help="Overwrite the content of the output directory")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()
    return args


def wiki_german_config(compress_type=None, prune_perc=0.0, num_train_epochs=2.0):
    from macros import WIKITEXT3_TRAIN
    out_dir = "C:/Users/jamesoneill/Projects/NLP/GOLM/golm/golm/golm_hil/models/networks/compress/results/"
    wiki_ex = WIKITEXT3_TRAIN.replace("wikitext-3/wiki.train.tokens", "wikitext_alt/train.txt")
    bsize = 16
    no_cuda = False
    args = get_bert_config(train_file=wiki_ex, train_bsize=bsize, no_cuda=no_cuda, num_train_epochs=num_train_epochs,
                           fp=False, prune_type=compress_type, output_dir=out_dir,
                           task_name='language modelling: WikiText-103', prune_perc=prune_perc)
    return args













