from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, OpenAIAdam, \
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from configs.models.gpt2 import Args
import torch, time, math
from loaders.models.gpt_loader import get_data_loader
from models.networks.compress.prune.basic import global_weight_prune


def test_retrained_prune(model_name='gpt2', pruning_perc=50., compress_type=None,
                         epochs=2.0, compress_steps = 20):

    results = []

    if model_name == 'gpt2':
        enc = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif 'gpt' in model_name:
        enc = OpenAIGPTTokenizer.from_pretrained(model_name)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_name)

    model.to('cuda')

    def evaluate(data_loader, val=True, compress_type=None, prune_perc=0.0):
        model.eval()
        split = 'val' if val else 'test'
        nb_steps, nb_examples, eval_loss, exp_average_loss = 0, 0, 0, None
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                batch = batch.to('cuda').type(torch.cuda.LongTensor)
                loss = model(batch, lm_labels=batch)
                eval_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss\
                                                  is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_steps += 1
                nb_examples += batch.size(0)
                # no need to see every step
                # print( f"{split} loss: {exp_average_loss:.4e}
                # {split} ppl: {math.exp(exp_average_loss):.4e}")
        eval_loss /= nb_steps
        exp_average_ppl = math.exp(exp_average_loss/nb_steps)
        ppl = math.exp(eval_loss)
        p_out = 'test ppl: {:.4e} \t test exp_average ppl {:.4e}'.format(ppl, exp_average_ppl)
        if compress_type is not None: p_out += "| {}".format(compress_type)
        if prune_perc > 0: p_out += "| {}".format(prune_perc)
        results.append(p_out)
        print(p_out)

    args = Args(model_name, compress_type=compress_type, epochs=epochs, prune_perc=pruning_perc,
                compress_steps=compress_steps)

    cache_path = f'{args.output_dir}/{args.model_name}_from_pretrained{args.start_from_pretrained}' \
        f'_retrain{args.do_retrain}_trainEpochs{args.epochs}_pruneGlobal{args.prune_global}__pruneAll{args.all_prune}' \
        f'{args.prune_type}Compression_percentage{args.prune_perc}_compressStep{args.compress_step}' \
        f'_trainBsize{args.train_batch_size}.txt'

    train_loader = get_data_loader(args.train_dataset, enc, 1, args)
    test_loader = get_data_loader(args.test_dataset,  enc, 4, args)

    # ## Prep optimizer with OpenAIAdam because that's what run_openai_gpt used
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = len(train_loader) * args.num_train_epochs

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
    print(args.prune_perc, args.compress_step)
    if args.prune_perc > 0 and args.compress_step is not None:
        perc_inc = args.prune_perc // args.compress_step
        total_batch_iters = int(args.num_train_epochs * len(train_loader))
        prune_step = int(total_batch_iters / args.compress_step)

    t_ppl, t_ppl_exp = 0, 0
    for epoch in range(int(args.num_train_epochs)):
        tr_loss, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_loader):
            model.train()
            batch = batch.type(torch.cuda.LongTensor)
            loss = model(batch, lm_labels=batch)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for l, p in enumerate(model.parameters()):
                pruned_inds = pruned_inds_by_layer[l]
                if type(pruned_inds) is not str:
                    p.grad.data[pruned_inds] = 0.

            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None \
                else 0.7 * exp_average_loss + 0.3 * loss.item()
            nb_tr_steps += 1
            t_ppl += math.exp(loss.item())
            t_ppl_exp += math.exp(exp_average_loss)

            if args.prune_perc > 0 and args.compress_step is not None:
                if step % prune_step == 0 and step != 0:
                    batch_num = step * (epoch + 1)  # if epoch != 0 else step
                    print("\n compressing at step {}/{} by {} % \n "
                          .format(batch_num, total_batch_iters, args.prune_perc))
                    model = global_weight_prune(model, perc_inc, return_masks=False, all_weights=False)

                    print('train_ppl {} \t train_ppl_exp_average {}'.
                          format(t_ppl/prune_step, t_ppl_exp/prune_step))
                    t_ppl, t_ppl_exp = 0, 0

                if step % (int(prune_step * (args.compress_step // 5))) == 0 and step != 0:
                    evaluate(test_loader, val=False)

    evaluate(test_loader, val=False)

    with open(cache_path, 'w+') as f:
        for result in results:
            f.write("%s\n" % result)


if __name__ == "__main__":

    # test_retrained_prune(model_name='gpt2', pruning_perc=0., compress_type=None)
    test_retrained_prune(model_name='gpt2', pruning_perc=50., compress_type='prune')