"""Mostly consists of getters for testing particular configurations"""

from loaders import data


def get_ecoc_corpus(text, args, ec=True, count=False):
    if 'wikitext-2' in text:
        args.epochs = 20
        args.batch_size = 56
        corpus = data.Corpus(args.data, emb_type=args.pretrained,
                             error_coding=ec, count=count)

    elif 'wikitext-3' in text:
        args.epochs = 20
        args.batch_size = 200
        args.vocab_limit = 50000
        corpus = data.Corpus(args.data, emb_type=args.pretrained,
                             limit=args.vocab_limit, lcs=args.lcs,
                             error_coding=ec, count=count)
    elif 'ptb' in text:
        # args.epochs = 1
        args.epochs = 40
        corpus = data.Corpus(args.data, emb_type=args.pretrained,
                             error_coding=ec, count=count)
    return args, corpus