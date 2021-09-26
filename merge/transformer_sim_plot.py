from merge.merger import layer_merge_by_type, layer_merge
from merge.plotter import plot_layer_sims
from layer_similarity import compute_merge
from prune.basic import prune_attention
from quantize.compressor import pca_compress
from helpers import get_prune_n_params


def get_bert():
    from transformers import BertForSequenceClassification, BertConfig
    from configs.models.bert import get_bert_config
    config = get_bert_config()
    bert_config = BertConfig(
        vocab_size_or_config_json_file=config.vocab_size,
        hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                max_position_embeddings=config.max_position_embeddings,
                type_vocab_size=config.type_vocab_size,
                initializer_range=config.initializer_range
                        )
    model = BertForSequenceClassification(bert_config, config.n_classes)
    return model


def get_transfoxl():
    from transformers import TransfoXLLMHeadModel
    from configs.models.transformerxl import get_transxl_config
    args = get_transxl_config()
    model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    return model


def get_gpt():
    from configs.models.gpt2 import get_gpt2_config
    from transformers import OpenAIGPTLMHeadModel
    args = get_gpt2_config('openai-gpt')
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    return model


def get_gpt2():
    from configs.models.gpt2 import get_gpt2_config
    from transformers import GPT2LMHeadModel
    args = get_gpt2_config('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model


def test_prune_bert():
    model = get_bert()
    print("Testing Prune ...")
    prune_attention(model.to('cuda'))
    # print("Testing Network SVD Compression...")
    # svd_compress(model.to('cuda'))
    print("Testing Network PCA Compression...")
    pca_compress(model.to('cuda'))


def get_model(model):
    print(model)
    if model == 'bert': return get_bert()
    elif model == 'gpt' or model == 'openai-gpt': return get_gpt()
    elif model == 'gpt2': return get_gpt2()
    elif model == 'transfoxl': return get_transfoxl()
    elif model == 'transformer': return get_transfoxl()


def test_plot(by_type= False, metric='cov', dist_mat = None, cn=False, mod='bert'):
    """Test methods to merge layers in bert for a smaller and more efficient network"""
    model = get_model(mod)
    print(f"Testing Layer Merging with {metric} Similarity ...")
    if by_type:
        sims, param_by_type, x_names, y_names = \
            layer_merge_by_type(model, perc=0.5, metric=metric, cn=cn)
        print("Hello,", model.__class__.__name__)
        plot_layer_sims(sims, param_by_type, metric=metric, mod_name=mod.upper())
    else:
        print(layer_merge(model, perc = 0.3, metric=metric))


def test_merge(mod='bert', by_type= False, metric='cov', cn=False):
    model = get_model(mod)
    print("Testing Layer Merging with Covariance Similarity ...")
    if by_type:
        sims, param_by_type, x_names, y_names =\
            layer_merge_by_type(model, perc=0.5, metric=metric, cn=cn)
        merged_model = compute_merge(model, sims, x_names)
    else:
        print(layer_merge(model, perc = 0.3, metric=metric))

    assert get_prune_n_params(merged_model) != 1


if __name__ == "__main__":

    # test_prune_bert()
    pretrain_mods = ['openai-gpt', 'gpt2', 'transfoxl']
    metrics = ['euclidean'] # , 'cov', 'emd', 'kl', 'cka', 'cos']
    for pretrain_mod in pretrain_mods:
        for metric in metrics:
            test_plot(by_type= True, metric=metric,
                  dist_mat = None, cn=False, mod=pretrain_mod)

        #test_merge(
        #    pretrain_mod, by_type=True,
        #           metric='euclidean', cn=False
        #)
