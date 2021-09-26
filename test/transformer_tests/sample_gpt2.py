#!/usr/bin/env python3

import logging
import torch
import numpy as np
from util.samplers import sample_sequence
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def sample_model(args):

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    while True:
        context_tokens = []
        if not args.unconditional:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:, len(context_tokens):].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
        else:
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=None,
                    start_token=enc.encoder['<|endoftext|>'],
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:,1:].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


if __name__ == '__main__':

    from configs.models.gpt2 import get_gpt2_config
    args = get_gpt2_config()
    sample_model(args)


