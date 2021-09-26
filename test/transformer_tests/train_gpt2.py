#!/usr/bin/env python
# coding: utf-8

import tqdm
from tensorboardX import SummaryWriter
from tqdm import trange
from loaders.models.gpt_loader import get_data_loader
from util.samplers import print_samples
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam, \
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from configs.models.gpt2 import wiki_config
from models.networks.compress.getter import compress_net
import os
from util.printers import *


def main(args):

    cache_path = f'{args.output_dir}/{args.model_name}_from_pretrained{args.start_from_pretrained}' \
        f'_retrain{args.do_retrain}_trainEpochs{args.epochs}_pruneGlobal{args.prune_global}__pruneAll{args.all_prune}' \
        f'{args.prune_type}Compression_percentage{args.prune_perc}_compressStep{args.compress_step}' \
        f'_trainBsize{args.train_batch_size}.json'

    global global_example_count, event_writer
    assert args.do_train or args.do_eval or args.do_test, "Specify at least one of do_train, do_eval or do_test"
    args.logdir = f'{args.logdir}/{args.run_name}-{current_timestamp()}'
    os.system(f'mkdir -p {args.logdir}')

    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    output_model_file, output_config_file = get_fps(args)

    def evaluate(data_loader, val=True, prune_type=None, prune_perc=0.0):
        global global_example_count, event_writer
        print(event_writer)
        split = 'val' if val else 'test'

        model.eval()
        nb_steps, nb_examples, eval_loss, exp_average_loss = 0, 0, 0, None
        with torch.no_grad():
            tqdm_bar = tqdm.tqdm(data_loader, desc="Eval")
            for step, batch in enumerate(tqdm_bar):
                # Put model in training mode.
                batch = batch.to(device).type(torch.cuda.LongTensor)
                # print(batch.size())
                # input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None
                # if lm_labels, outputs loss
                loss = model(batch, lm_labels=batch)
                eval_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None\
                    else 0.7*exp_average_loss+0.3*loss.item()
                nb_steps += 1
                nb_examples += batch.size(0)
                tqdm_bar.desc = f"{split} loss: {exp_average_loss:.4e} " \
                    f" {split} ppl: {math.exp(exp_average_loss):.4e}"
                if val:
                    global_example_count += args.eval_batch_size
                else:
                    global_example_count += args.test_batch_size

        tp, tp_exp, tp_lr = '{}_ppl'.format(split), \
                            '{}_ppl_exp_average'.format(split), \
                            '{}_lr'.format(split)
        t = time.time()
        eval_loss /= nb_steps
        exp_average_ppl = math.exp(exp_average_loss/nb_steps)
        ppl = math.exp(eval_loss)
        p_out = 'Final {} ppl: {}'.format(ppl, split)
        if prune_type is not None: p_out += "| {}".format(prune_type)
        if prune_perc != 0.0: p_out += "| {}".format(prune_perc)
        event_writer._SummaryWriter__append_to_scalar_dict(tp, ppl,  global_example_count, t)
        event_writer._SummaryWriter__append_to_scalar_dict(tp_exp, exp_average_ppl, global_example_count, t)

    def get_model(model_name):
        if 'openai' in args.model_name:
            # model_name = args.model_name.replace("openai-", "")
            enc = OpenAIGPTTokenizer.from_pretrained(model_name)
            model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
        else:
            enc = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(device)
        return enc, model

    enc, model = get_model(args.model_name)

    # setup TensorBoard logging
    global_example_count = 0
    print(f"Logging to {args.logdir}")
    event_writer = SummaryWriter(args.logdir)

    if args.do_eval: val_loader = get_data_loader(args.val_dataset, enc, args.eval_batch_size, args)
    if args.do_test: test_loader = get_data_loader(args.test_dataset,  enc, args.test_batch_size, args)

    if args.do_train:
        data_loader = get_data_loader(args.train_dataset, enc, args.train_batch_size, args)

        # ## Prep optimizer with OpenAIAdam because that's what run_openai_gpt used
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = len(data_loader) * args.num_train_epochs

        optimizer = OpenAIAdam(
            optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            max_grad_norm=args.max_grad_norm,
                            weight_decay=args.weight_decay,
                            t_total=num_train_optimization_steps
                    )

        # ## Train loop, based on `run_openai_gpt.py`
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None

        # Reset all model weights so we can train from scratch.
        if not args.start_from_pretrained: model.apply(model.init_weights)

        # setup compression steps
        if args.do_retrain:
            perc_inc = args.prune_perc/args.compress_step
            total_batch_iters = int(args.num_train_epochs * len(data_loader))
            prune_step = int(total_batch_iters / args.compress_step)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss, nb_tr_steps = 0, 0
            tqdm_bar = tqdm.tqdm(data_loader, desc="Training")

            for step, batch in enumerate(tqdm_bar):
                model.train()
                batch = batch.type(torch.cuda.LongTensor)
                loss = model(batch, lm_labels=batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None \
                    else 0.7*exp_average_loss+0.3*loss.item()

                nb_tr_steps += 1

                if args.do_retrain:
                    if step % prune_step == 0 and step != 0 and args.prune_perc > 0:
                        batch_num = step * (epoch+1) # if epoch != 0 else step
                        args.prune_perc = perc_inc
                        print("\n compressing at step {}/{} by {} % \n "
                              .format(batch_num, total_batch_iters, args.prune_perc))
                        model = compress_net(model, args)
                    # we test the model every 20 % throughout retraining
                    # since there is a lot of batches in 1 epoch
                    if step % (int(prune_step * (args.compress_step// 5))) == 0 and step != 0:
                        if args.do_eval: evaluate(val_loader)
                        evaluate(test_loader, val=False)

                    t = time.time()
                    event_writer._SummaryWriter__append_to_scalar_dict(
                        'retrain_ppl_exp_average', math.exp(exp_average_loss), global_example_count, t)
                    event_writer._SummaryWriter__append_to_scalar_dict(
                        'retrain_ppl', math.exp(loss.item()), global_example_count, t)
                    event_writer._SummaryWriter__append_to_scalar_dict(
                        'retrain_lr', optimizer.get_lr()[0], global_example_count, t)

                tqdm_bar.desc = f"Training loss: {exp_average_loss:.2e}" \
                    f" lr: {optimizer.get_lr()[0]:.2e} ppl: {math.exp(exp_average_loss):.2e}"

                t = time.time()
                event_writer._SummaryWriter__append_to_scalar_dict(
                    'train_ppl', math.exp(loss.item()),  global_example_count, t)
                event_writer._SummaryWriter__append_to_scalar_dict(
                    'train_ppl_exp_average', math.exp(exp_average_loss), global_example_count, t)
                event_writer._SummaryWriter__append_to_scalar_dict(
                    'train_lr', optimizer.get_lr()[0], global_example_count, t)
                global_example_count += args.train_batch_size

    # we don't want to be compressing the model before testing
    # if we've already done so during retraining
    if args.do_train is False and args.do_retrain is False:
        if type(args.prune_perc) == list:
            percentages = args.prune_perc
            for perc_prune in percentages:
                args.prune_perc = perc_prune
                model = compress_net(model, args)
                # Run on test data.
                evaluate(test_loader,
                         prune_type=args.prune_type, prune_perc=args.prune_perc)
                # reset model
                enc, model = get_model(args.model_name)
        elif args.prune_perc != 0.0:
            model = compress_net(model, args)
            evaluate(test_loader, args.prune_type, args.prune_perc)
    else:
        evaluate(test_loader, val=False)

    sample = print_samples(
        model, enc, args,
        # Context is a random sample from the dataset.
        context_tokens=next(iter(test_loader)),
        batch_size=1, length=20, nsamples=1,
        temperature=1, top_k=40)
    event_writer.add_text('sample', sample, global_example_count)
    checkpoint(model, output_model_file)

    event_writer.export_scalars_to_json(cache_path)
    event_writer.close()


def test_all(epochs = 2.0):
    sfp = [True, False]
    models = ['gpt2', 'openai-gpt']
    prune_percs = [0.0, 10.0, 20., 30., 50., 70.0]
    prune_types = ['prune', 'merge', 'huffman', 'svd', 'fisher_prune']
    merge_dists = ['cov', 'cov_eig', 'cov_kl', 'cov_eig_kl', 'euclidean',
                   'manhattan', 'cka', 'cos', 'emd', 'emd_flow',
                   'sinkhorn', 'wasserstein', 'kl']
    all_prunes = [False, True]
    prune_global = False
    compress_step = 20  # retrains 20 times over num_train_epochs

    for start_from_pretrained in sfp:
        for prune_type in prune_types:
            for mod in models:
                for all_prune in all_prunes:
                    for prune_perc in prune_percs:

                        if prune_type == 'merge':
                            test_bsize, val_bsize = 1, 1
                        else:
                            test_bsize, val_bsize = 4, 4

                        args = wiki_config( model_name=mod, compress_type=prune_type,
                        prune_perc=0.0, epochs=epochs, do_train=False,
                        do_eval=False, do_test=True, train_batch_size=1,
                        eval_batch_size=val_bsize, test_batch_size=test_bsize )
                        args.prune_perc = prune_perc # 30.0 # [10.0, 20.0, 30.0, 50.0, 70.0]
                        args.all_prune = all_prune
                        args.prune_global = prune_global
                        args.start_from_pretrained = start_from_pretrained
                        args.do_train = True
                        args.do_retrain = True
                        args.do_eval = True
                        args.compress_step = compress_step

                        if prune_type == 'prune':
                            main(args)
                            args.prune_global = True
                            main(args)

                        main(args)


def single_run_test(train=True, pt='prune', deval=True):
    sfp = True
    mod = 'gpt2'
    prune_perc = 10.0
    prune_type = pt # 'prune'
    all_prune = False
    prune_global = False
    compress_step = 20  # retrains 20 times over num_train_epochs

    if prune_type == 'merge':
        test_bsize, val_bsize = 1, 1
    else:
        test_bsize, val_bsize = 4, 4

    args = wiki_config(model_name=mod, compress_type=prune_type,
    prune_perc=0.0, epochs=1.0, do_train=False,
    do_eval=False, do_test=True, train_batch_size=1,
    eval_batch_size=val_bsize, test_batch_size=test_bsize)
    args.prune_perc = prune_perc #30.0 # [10.0, 20.0, 30.0, 50.0, 70.0]
    args.all_prune = all_prune
    args.prune_global = prune_global
    args.start_from_pretrained = True
    args.do_train = train
    args.do_retrain = True
    args.do_eval = deval
    args.compress_step = compress_step

    main(args)


if __name__ == '__main__':

    # single_run_test(train=True, pt='merge', deval=False)
    epochs =  2.0
    test_all(epochs)