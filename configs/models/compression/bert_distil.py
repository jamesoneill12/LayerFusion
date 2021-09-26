import os
import argparse


def get_bert_distil_args(student_type='distil_bert', dump_path='D:/data/ucsd/'
                         , filename = 'aggressive_dedup.json.gz'):

    data_file = dump_path + filename
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--force", action='store_true',
                        help="Overwrite dump_path if it already exists.")

    parser.add_argument("--dump_path", type=str, default=dump_path,
                        help="The output directory (log, checkpoints, parameters, etc.)")
    parser.add_argument("--data_file", type=str, default=data_file,
                        help="The binarized file (tokenized + tokens_to_ids) and grouped by sequence.")

    parser.add_argument("--student_type", type=str, default=student_type, choices=["distilbert", "roberta", "gpt2"],
                        help="The student type (DistilBERT, RoBERTa).")
    parser.add_argument("--student_config", type=str, required=True,
                        help="Path to the student configuration.")
    parser.add_argument("--student_pretrained_weights", default=None, type=str,
                        help="Load student initialization checkpoint.")

    parser.add_argument("--teacher_type", choices=["bert", "roberta", "gpt2"], required=True,
                        help="Teacher type (BERT, RoBERTa).")
    parser.add_argument("--teacher_name", type=str, required=True,
                        help="The teacher model.")

    parser.add_argument("--temperature", default=2., type=float,
                        help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=0.5, type=float,
                        help="Linear weight for the distillation loss. Must be >=0.")
    parser.add_argument("--alpha_mlm", default=0.0, type=float,
                        help="Linear weight for the MLM loss. Must be >=0. Should be used in coonjunction with `mlm` flag.")
    parser.add_argument("--alpha_clm", default=0.5, type=float,
                        help="Linear weight for the CLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.0, type=float,
                        help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument("--alpha_cos", default=0.0, type=float,
                        help="Linear weight of the cosine embedding loss. Must be >=0.")

    parser.add_argument("--mlm", action="store_true",
                        help="The LM step: MLM or CLM. If `mlm` is True, the MLM is used over CLM.")
    parser.add_argument("--mlm_mask_prop", default=0.15, type=float,
                        help="Proportion of tokens for which we need to make a prediction.")
    parser.add_argument("--word_mask", default=0.8, type=float,
                        help="Proportion of tokens to mask out.")
    parser.add_argument("--word_keep", default=0.1, type=float,
                        help="Proportion of tokens to keep.")
    parser.add_argument("--word_rand", default=0.1, type=float,
                        help="Proportion of tokens to randomly replace.")
    parser.add_argument("--mlm_smoothing", default=0.7, type=float,
                        help="Smoothing parameter to emphasize more rare tokens (see XLM, similar to word2vec).")
    parser.add_argument("--token_counts", type=str,
                        help="The token counts in the data_file for MLM.")

    parser.add_argument("--restrict_ce_to_mask", action='store_true',
                        help="If true, compute the distilation loss only the [MLM] prediction distribution.")
    parser.add_argument("--freeze_pos_embs", action="store_true",
                        help="Freeze positional embeddings during distillation. For student_type in ['roberta', 'gpt2'] only.")
    parser.add_argument("--freeze_token_type_embds", action="store_true",
                        help="Freeze token type embeddings during distillation if existent. For student_type in ['roberta'] only.")

    parser.add_argument("--n_epoch", type=int, default=3,
                        help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size (for each process).")
    parser.add_argument("--group_by_size", action='store_false',
                        help="If true, group sequences that have similar length into the same batch. Default is true.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=50,
                        help="Gradient accumulation for larger training batches.")
    parser.add_argument("--warmup_prop", default=0.05, type=float,
                        help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float,
                        help="Random initialization range.")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--n_gpu", type=int, default=1,
                        help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=56,
                        help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500,
                        help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=4000,
                        help="Checkpoint interval.")
    args = parser.parse_args()
    return args


def sanity_checks(args):
    """
    A bunch of args sanity checks to perform even starting...
    """
    assert (args.mlm and args.alpha_mlm > 0.) or (not args.mlm and args.alpha_mlm == 0.)
    assert (args.alpha_mlm > 0. and args.alpha_clm == 0.) or (args.alpha_mlm == 0. and args.alpha_clm > 0.)
    if args.mlm:
        assert os.path.isfile(args.token_counts)
        assert (args.student_type in ['roberta', 'distilbert']) and (args.teacher_type in ['roberta', 'bert'])
    else:
        assert (args.student_type in ['gpt2']) and (args.teacher_type in ['gpt2'])

    assert args.teacher_type == args.student_type or (args.student_type == 'distilbert' and args.teacher_type == 'bert')
    assert os.path.isfile(args.student_config)
    if args.student_pretrained_weights is not None:
        assert os.path.isfile(args.student_pretrained_weights)

    if args.freeze_token_type_embds: assert args.student_type in ['roberta']

    assert args.alpha_ce >= 0.
    assert args.alpha_mlm >= 0.
    assert args.alpha_clm >= 0.
    assert args.alpha_mse >= 0.
    assert args.alpha_cos >= 0.
    assert args.alpha_ce + args.alpha_mlm + args.alpha_clm + args.alpha_mse + args.alpha_cos > 0.