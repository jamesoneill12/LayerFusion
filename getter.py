from models.networks.compress.prune.fischer import Pruner
from models.networks.compress.merge.merger import layer_merge_by_type
from models.networks.compress.layer_similarity import compute_merge
from models.networks.compress.prune.basic import weight_prune, global_weight_prune
from models.networks.compress.quantize.compressor import svd_compress, pca_compress
from models.networks.compress.quantize.huffman import huffman_encode_model
from models.networks.compress.quantize.cluster import apply_weight_sharing


def compress_net(model, args):

    # layer merging
    if args.prune_type == 'merge':
        sims, param_by_type, x_names, _ = layer_merge_by_type(
            model, perc=args.prune_perc, metric=args.merge_measure)
        model = compute_merge(model, sims, x_names)
    elif args.prune_type == 'weighted_merge':
        sims, param_by_type, x_names, _ = layer_merge_by_type(
            model, perc=args.prune_perc, metric=args.merge_measure)
        model = compute_merge(model, sims, x_names)

    # pruning
    elif 'prune' in args.prune_type:
        if args.prune_global:
            model = global_weight_prune(model, pruning_perc=args.prune_perc, all_weights=args.all_prune)
        elif 'fisher_prune' in args.prune_type:
            pruner = Pruner()
            print('Pruning')
            if 'random' not in args.prune_type:
                if 'l1_prune' in args.prune_type:
                    print('fisher compress')
                    pruner.fisher_prune(model, prune_every=args.prune_every)
                else:
                    print('l1 compress')
                    pruner.l1_prune(model, prune_every=args.prune_every)
            else:
                print('random compress')
                pruner.random_prune(model, )
        else:
            model = weight_prune(model, pruning_perc=args.prune_perc, all_weights=args.all_prune)

    # tensor decomposition methods
    elif args.prune_type == 'svd':
        model = svd_compress(model, args.prune_perc)
    elif args.prune_type == 'pca':
        model = pca_compress(model, args.prune_perc)
    elif args.prune_type == 'autoencode':
        model = svd_compress(model, args.prune_perc)

    # the quantization methods
    elif args.prune_type == 'huffman':
        model = huffman_encode_model(model)#, args.prune_perc)
    elif args.prune_type == 'kmeans':
        model = apply_weight_sharing(model, args.num_bits)#, args.prune_perc)
    return model
