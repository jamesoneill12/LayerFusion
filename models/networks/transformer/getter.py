from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path
from pytorch_pretrained_bert.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, GPT2LMHead, GPT2DoubleHeadsModel
from models.networks.transformer import transformer
from pytorch_pretrained_bert import TransfoXLModel, TransfoXLConfig, TransfoXLLMHeadModel
from pytorch_pretrained_bert import BertForSequenceClassification, BertConfig
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transformer(args, model_args, critic = False,
                    train_data=None, weight_decay = 0.01, vocab_len=None):
    """model is a string for which """
    """ if critic true, then we return critic model, otherwise we are choosing the decoder (policy pap) """
    mod = args.reward if critic else args.rnn_type

    if mod == "gpt":
        # Load tokenizer and model
        # This loading functions also add new tokens and embeddings called `special tokens`
        # These new embeddings will be fine-tuned on the RocStories dataset
        special_tokens = ['_start_', '_delimiter_', '_classify_']
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_args.model_name, special_tokens=special_tokens)
        model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
        model.to(device)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': weight_decay}
        ]

        num_train_optimization_steps = len(train_data) * args.epochs // args.batch_size
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=model_args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)
        return model, optimizer

    elif mod == "gpt2":
        """pretrained on different text (not to be confused with pretrained on image cap dataset and loaded back in)"""
        if args.pretrained_transformer:
            """
                    Instantiate a GPT2PreTrainedModel from a pre-trained model file or a pytorch state dict.
                    Download and cache the pre-trained model file if needed.
                    Params:
                        pretrained_model_name_or_path: either:
                            - a str with the name of a pre-trained model to load selected in the list of:
                                . `openai-gpt`
                            - a path or url to a pretrained model archive containing:
                                . `gpt2_config.json` a configuration file for the model
                                . `pytorch_model.bin` a PyTorch dump of a GPT2Model instance
                            - a path or url to a pretrained model archive containing:
                                . `bert_config.json` a configuration file for the model
                                . a TensorFlow checkpoint with trained weights
                        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
                        cache_dir: an optional path to a folder in which the pre-trained mods will be cached.
                        state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of pre-trained mods
                        *inputs, **kwargs: additional input for the specific Bert class
                            (ex: num_labels for BertForSequenceClassification)
            """
            # Load pre-trained model (weights)
            model = GPT2Model.from_pretrained('gpt2')
            model.to('cuda')
            model.eval()
            return model
        else:
            "finish"
            pass

    elif mod == "gpt2_lmhead":
        if args.pretrained_transformer:
            # Load pre-trained model (weights)
            model = GPT2LMHead.from_pretrained('gpt2')
            model.to('cuda')
            model.eval()
            return model
        else:
            "finish"
            pass

    elif mod == "gpt2_double_lmhead":
        if args.pretrained_transformer:
            # Load pre-trained model (weights)
            model = GPT2LMHead.from_pretrained('gpt2')
            model.to('cuda')
            model.eval()
            return model
        else:
            "finish"
            pass

    elif mod == "bert":
        if args.pretrained_transformer:
            # Load pre-trained model (weights)

            config = BertConfig(
                vocab_size_or_config_json_file=model_args.vocab_size,
                hidden_size=model_args.hidden_size,
                num_hidden_layers=model_args.num_hidden_layers,
                num_attention_heads=model_args.num_attention_heads,
                intermediate_size=model_args.intermediate_size,
                hidden_act=model_args.hidden_act,
                hidden_dropout_prob=model_args.hidden_dropout_prob,
                attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
                max_position_embeddings=model_args.max_position_embeddings,
                type_vocab_size=model_args.type_vocab_size,
                initializer_range=model_args.initializer_range)

            model = BertForSequenceClassification(config, config.num_classes)
            model.to('cuda')
            model.eval()
            return model
        else:
            "finish"
            pass

    elif mod == "transformer":
        model = transformer.Decoder(embedding_size=vocab_len, hidden_size=model_args.d_embed, num_layers=model_args.n_layer,
                                     num_heads=model_args.n_head, total_key_depth=model_args.total_key_depth,
                                     total_value_depth=model_args.total_value_depth, filter_size=model_args.filter_size, vocab_size=args.vocab_size,
                                     max_length=model_args.max_len, input_dropout=model_args.in_dropout, layer_dropout=0.0,
                                     attention_dropout=model_args.attention_dropout, relu_dropout=0.0)
        model.to('cuda')
        return model
    elif mod == "transformerxl":
        config = TransfoXLConfig(
            vocab_size_or_config_json_file=vocab_len,
            mem_len=model_args.mem_len,
            clamp_len=model_args.clamp_len,
            cutoffs=model_args.cutoffs,
            d_model=model_args.d_model,
            d_embed=model_args.d_embed,
            n_head=model_args.n_head,
            d_head=model_args.d_head,
            d_inner=model_args.d_inner,
            div_val=model_args.div_val,
            n_layer=model_args.n_layer
        )
        model = TransfoXLModel(config)
        model.to('cuda')
        model.eval()
        return model
