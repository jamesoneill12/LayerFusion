# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
"""
Preprocessing script before distillation.
"""
import argparse
import types
import pickle
import gzip
import random
import time
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer, DistilBertTokenizer
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main(
        file_path= 'D:/data/ucsd/aggressive_dedup.json.gz', tokenizer_type='distilbert',
         tokenizer_name='bert-large-uncased'
         ):

    parser = argparse.ArgumentParser(description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids).")
    parser.add_argument('--file_path', type=str, default=file_path,
                        help='The path to the data.')

    if tokenizer_type == 'distilbert':
        tokenizer_name = 'distilbert-base-uncased-distilled-squad'

    parser.add_argument('--tokenizer_type', type=str, default=tokenizer_type, choices=['bert', 'roberta', 'gpt2'])
    parser.add_argument('--tokenizer_name', type=str, default=tokenizer_name,
                        help="The tokenizer to use.")
    parser.add_argument('--dump_file', type=str, default=file_path.rsplit("/",1 )[0],
                        help='The dump file prefix.')
    args = parser.parse_args()

    logger.info(f'Loading Tokenizer ({args.tokenizer_name})')
    if args.tokenizer_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `[CLS]`
        sep = tokenizer.special_tokens_map['sep_token'] # `[SEP]`
    if args.tokenizer_type == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `[CLS]`
        sep = tokenizer.special_tokens_map['sep_token'] # `[SEP]`
    elif args.tokenizer_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `<s>`
        sep = tokenizer.special_tokens_map['sep_token'] # `</s>`
    elif args.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['bos_token'] # `<|endoftext|>`
        sep = tokenizer.special_tokens_map['eos_token'] # `<|endoftext|>`

    logger.info(f'Loading text from {args.file_path}')

    if 'txt' in args.file_path:
        with open(args.file_path, 'r', encoding='utf8') as fp:
            data = fp.readlines()

    elif 'gz' in args.file_path:
        def parse(path):
            g = gzip.open(path, 'r')
            for l in g:
                yield eval(l)
        data = parse(args.file_path)

    rslt = []
    iter = 0
    interval = 10000
    start = time.time()

    logger.info(f'Start encoding')
    if not isinstance(data, types.GeneratorType):
        logger.info(f'{len(data)} examples to process.')
        for text in data:
            text = f'{bos} {text.strip()} {sep}'
            token_ids = tokenizer.encode(text)
            rslt.append(token_ids)
            iter += 1
            if iter % interval == 0:
                end = time.time()
                logger.info(f'{iter} examples processed. - {(end-start)/interval:.2f}s/expl')
                start = time.time()
    else:
        for text in data:
            text = text['reviewText']
            text = f'{bos} {text.strip()} {sep}'
            if len(text.split()) > 300:
                chunks = divide_chunks(text, 300)
                for t in chunks:
                    token_ids = tokenizer.encode(t)
                    rslt.append(token_ids)
                    iter += 1
            else:
                token_ids = tokenizer.encode(text)
                rslt.append(token_ids)
                iter += 1
            if iter % interval == 0:
                end = time.time()
                logger.info(f'{iter} examples processed. - {(end-start)/interval:.2f}s/expl')
                start = time.time()

    if not isinstance(data, types.GeneratorType):
        logger.info(f'{len(data)} examples to process.')
        logger.info('Finished binarization')
        logger.info(f'{len(data)} examples processed.')

    dp_file = f'{args.dump_file}.{args.tokenizer_name}.pickle'
    rslt_ = [np.uint16(d) for d in rslt]
    random.shuffle(rslt_)

    if not isinstance(data, types.GeneratorType):
        logger.info(f'{len(data)} examples to process.')
        logger.info(f'Dump to {dp_file}')

    with open(dp_file, 'wb') as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()