""" BERT for language modelling where the pretrained model is trained by predicting masked words"""
import torch
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig

# Already been converted into WordPiece token ids
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
config = BertConfig(vocab_size_or_config_json_file=100000, hidden_size=768,
                    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
model = BertForMaskedLM(config)
masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)