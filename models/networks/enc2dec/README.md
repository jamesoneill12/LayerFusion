# Encoder-Decoder Package

This package provides all combinations of encoders with decoder sequence mods.

The sub-module enc2dec contains an extensive range of various modules for encoder-decoder
architectures that can be used for a wide range of problems such as image captioning, 
machine translation, question-answering and many more.

## Github for Sequence-To-Sequence PyTorch
https://github.com/Mjkim88/Pytorch-Torchtext-Seq2Seq/blob/master/trainer.py

## Output Shape

Make sure output shape is always (sent_len, batch_size, vocab_size) prior 
to flattening with .view(-1, ntarg_word)


