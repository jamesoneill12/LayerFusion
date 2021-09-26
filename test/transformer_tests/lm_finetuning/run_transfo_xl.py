# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Transformer XL model evaluation script.
    Adapted from https://github.com/kimiyoung/transformer-xl.
    In particular https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/eval.py

    This script with default values evaluates a pretrained Transformer-XL on WikiText 103
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import time
import torch
from models.networks.compress.getter import compress_net
from util.printers import get_fps, display_loss
from pytorch_pretrained_bert import TransfoXLLMHeadModel,\
    TransfoXLCorpus, TransfoXLTokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main(args):

    assert args.ext_len >= 0, 'extended context length must be non-negative'
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("device: {}".format(device))

    # Load a pre-processed dataset
    # You can also build the corpus yourself using TransfoXLCorpus methods
    # The pre-processing involve computing word frequencies to prepare the Adaptive input and SoftMax
    # and tokenizing the dataset
    # The pre-processed corpus is a convertion (using the conversion script )
    tokenizer = TransfoXLTokenizer.from_pretrained(args.model_name)
    corpus = TransfoXLCorpus.from_pretrained(args.model_name)
    ntokens = len(corpus.vocab)

    va_iter = corpus.get_iterator('valid', args.batch_size, args.tgt_len,
        device=device, ext_len=args.ext_len)
    te_iter = corpus.get_iterator('test', args.batch_size, args.tgt_len,
        device=device, ext_len=args.ext_len)

    # loaded_model_file = args.output_dir + args.model_name + "_" + WEIGHTS_NAME
    # loaded_config_file = args.output_dir + args.model_name + "_" + CONFIG_NAME
    # loaded_vocab_file = args.output_dir + args.model_name + "_" + "vocab.txt"

    output_model_file, output_config_file = get_fps(args)

    """
    # Load a pre-trained model if not using default pretrained args.model_name
    if os.path.exists(output_model_file) and os.path.exists(output_config_file) and args.prune_type is not None:
        logger.info("** ** *Loading {} configuration file** ** * ".format(loaded_config_file))
        transfo_xl_config = TransfoXLConfig.from_json_file(loaded_config_file)
        model = TransfoXLLMHeadModel.from_pretrained(transfo_xl_config)
        logger.info("** ** *Loading {} fine - tuned model ** ** * ".format(loaded_model_file))
        model.load_state_dict(torch.load(loaded_model_file))
        tokenizer = TransfoXLTokenizer.from_pretrained(args.transfo_xl_model, do_lower_case=args.do_lower_case)
        tokenizer.from_pretrained(loaded_vocab_file)
    else:    
    """

    def get_model(args):
        model = TransfoXLLMHeadModel.from_pretrained(args.model_name)
        model = model.to(device)
        logger.info('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
            args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))
        model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
        return model

    model = get_model(args)

    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
    if args.same_length:
        model.same_length = True

    ###############################################################################
    # Evaluation code
    ###############################################################################
    def evaluate(eval_iter):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_len, total_loss, perc = 0, 0., 0.
        start_time = time.time()
        with torch.no_grad():
            mems = None
            for idx, (data, target, seq_len) in enumerate(eval_iter):
                ret = model(data, target, mems)
                loss, mems = ret
                loss = loss.mean()
                total_loss += seq_len * loss.item()
                total_len += seq_len
                if idx % eval_iter.n_batch // 100 == 0:
                    print("{} % eval computed".format(perc))
                    perc += 1.0
            total_time = time.time() - start_time
        logger.info('Time : {:.2f}s, {:.2f}ms/segment'.format(
                total_time, 1000 * total_time / (idx+1)))
        return total_loss / total_len

    log_str = ''

    if type(args.prune_perc) == list:
        percentages = args.prune_perc
        for perc_prune in percentages:
            args.prune_perc = perc_prune
            model = compress_net(model, args)
            # Run on test data.
            if args.split == 'all':
                test_loss = evaluate(te_iter)
                valid_loss = evaluate(va_iter)
            elif args.split == 'valid':
                valid_loss = evaluate(va_iter)
                test_loss = None
            elif args.split == 'test':
                test_loss = evaluate(te_iter)
                valid_loss = None

            log_str = display_loss(log_str, valid_loss, test_loss,
                                   args.prune_type, args.prune_perc)

            logger.info('=' * 100)
            logger.info(log_str)
            logger.info('=' * 100)

            # reset model to make sure already pruned model is
            # not pruned further with same percentage
            model = get_model(args)

    else:
        model = compress_net(model, args)
        # Run on test data.
        if args.split == 'all':
            test_loss = evaluate(te_iter)
            valid_loss = evaluate(va_iter)
        elif args.split == 'valid':
            valid_loss = evaluate(va_iter)
            test_loss = None
        elif args.split == 'test':
            test_loss = evaluate(te_iter)
            valid_loss = None
        # log_str = display_loss(log_str, valid_loss, test_loss)
        import math
        print(math.exp(test_loss))

    # write validation and test perplexity results to file
    performance_path = args.output_dir + "_" + args.model_name + "_"
    if args.prune_type is not None:
        performance_path += args.prune_type + ".txt"
    else:
        performance_path += ".txt"

    result_writer = open(performance_path, "w+")
    result_writer.write(log_str)

    # Save a trained model
    logger.info("** ** * Saving fine - tuned model ** ** * ")
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    if args.prune_type is not None and 'merge' in args.prune_type:
            output_model_file = "{}_{}_{}_{}".format(args.prune_type, args.prune_perc,
                                                     args.merge_measure, output_model_file)
            output_config_file = "{}_{}_{}_{}".format(args.prune_type, args.prune_perc,
                                                      args.merge_measure, output_config_file)

    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        if args.prune_type is None:
            VOCAB_NAME = args.output_dir + "_" + args.model_name + "_" + "vocab.txt"
            tokenizer.save_vocabulary(VOCAB_NAME)


if __name__ == '__main__':

    # 'prune'
    from configs.models.transformerxl import wiki_german_config
    args = wiki_german_config(compress_type='kmeans', prune_perc=10, epochs=2.0)
    # perc refers to amount to reduce the model by
    args.all_prune = False
    args.prune_global = True
    args.batch_size = 40 # 40 apart from svd
    # problem is it is requires num_samp > prune_perc
    main(args)
