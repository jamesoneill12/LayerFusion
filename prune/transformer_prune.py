from models.networks.compress.merge.plotter import plot_layer_sims
from models.networks.compress.merge.merger import layer_merge_by_type, layer_merge
from models.networks.compress.layer_similarity import compute_merge
from models.networks.compress.prune.basic_prune import prune_attention
from models.networks.compress.dim_reduction.compressor import pca_compress


def get_bert(model):
    from transformers import BertForSequenceClassification, BertConfig
    from configs.models.bert import get_bert_config
    config = get_bert_config()
    bert_config = BertConfig(
        config.vocab_size,
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
    # bert_config, config.n_classes
    model = BertForSequenceClassification.from_pretrained(model)
    return model


def test_prune_bert(model, device='cuda'):
    model = get_bert(model).to(device)
    print("Testing Prune ...")
    prune_attention(model.to(device))
    # print("Testing Network SVD Compression...")
    # svd_compress(model.to('cuda'))
    print("Testing Network PCA Compression...")
    pca_compress(model.to(device))


def test_merge_bert(
        model,
        by_type= False,
        metric='cov',
        dist_mat=None,
        cn=False,
        perc=0.5,
        device = 'cuda'
):
    """Test methods to merge layers in bert for a smaller and more efficient network"""
    model = get_bert(model).to(device)
    print("Testing Layer Merging with Covariance Similarity ...")
    if by_type:
        sims, param_by_type, x_names, y_names = layer_merge_by_type(
            model,
            perc=perc,
            metric=metric,
            cn=cn
        )
        merged_model = compute_merge(model, sims, x_names)
        plot_layer_sims(sims, param_by_type)
    else:
        print(layer_merge(model, perc=0.3, metric=metric))


if __name__ == "__main__":

    model_name = 'distilbert-base-uncased'
    device = 'cpu'
    # test_prune_bert(model)
    test_merge_bert(model_name, True, metric='kl', cn=False, device=device)