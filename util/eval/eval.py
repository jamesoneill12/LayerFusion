from util.eval.bleu import *
from  util.eval.cider import *
from  util.eval.cosine import *
from  util.eval.rouge import *
from util.metrics.dist import csim_np
from loaders.dictionary import id2sent, reorder_sents


def get_reward(pred_sents=None, targ_sents=None, method=None,
               pred_inds=None, targ_inds=None, embs=None,
               lengths=None, model=None, n = 2, alpha = 0.5,
               bsize=256, dist='cosine', mean=True):

    """don't confuse model methods with decoder, they are just evaluators"""
    if method == 'cosine' or method == 'cos':
        return torch_cos_sim(pred_sents, targ_sents, embs)
    elif method == 'bleu':
        return convert2bleu(pred_sents, targ_sents, lengths)
    elif method == 'wmd':
         return sum([model.wmd(p_sent, t_sent) for (p_sent, t_sent) in zip(pred_sents, targ_sents)])/len(pred_sents)
    elif method in ['infersent', 'uniskip', 'biskip', 'bert', 'elmo',
                    'gpt', 'gpt2', 'transformer', 'transformerxl'] and model is not None:

        # try:
        # except:
        # print(targ_sents)
        # ValueError("This batch caused problems CUDA error: unspecified launch failure")

        if method == 'infersent':
            pred_embs = model.encode(pred_sents, bsize)
            targ_embs = model.encode(targ_sents, bsize)
        elif method == 'uniskip' or method == 'biskip':
            pred_embs = model(pred_inds)
            targ_embs = model(targ_inds)

        if type(pred_embs) == torch.Tensor:
            pred_embs = F.normalize(pred_embs, p=2, dim=1)
            targ_embs = F.normalize(targ_embs, p=2, dim=1)
            if dist == 'cosine':
                sim = F.cosine_similarity(pred_embs, targ_embs)
            elif dist == 'euclidean':
                y_norm = torch.norm(pred_embs - targ_embs, 2, dim=1)
                sim = 1/(1+y_norm)
            elif dist == 'manhattan':
                y_norm = torch.norm(pred_embs - targ_embs, 1, dim=1)
                sim = 1/(1+y_norm)
            return sim.mean() if mean else sim

        elif type(pred_embs) == np.ndarray:
            sim = csim_np(pred_embs, targ_embs)
            if mean: sim = np.mean(sim)
            print(sim)
            print()
            return sim
    elif method == 'meta':
        model()
    elif method == 'rouge':
        return rouge_n(targ_sents, pred_sents, n, alpha)
    elif method == 'cider':
        return get_cider(pred_sents ,targ_sents, lengths)
    elif method == 'meteor':
        pass


def get_task_score(words, labels, args, pre_emb=None, vocab=None, sent_mod=None):

    if args.critic_reward == 'rouge':
        policy_values = rouge_l(words, labels)

    elif args.critic_reward == 'bleu':
        words_c, labels_c = words.cpu().numpy(), labels.cpu().numpy()
        policy_values = get_bleu(words_c, labels_c, seq_len=4, bleu_len=2)
        policy_values = torch.Tensor([policy_values]).cuda()

    elif args.critic_reward == 'cider':
        policy_values = get_reward()

    elif args.critic_reward == 'wmd':
        if type(words) == torch.Tensor:
            words = words.cpu().numpy()
        if type(labels) == torch.Tensor:
            labels = labels.cpu().numpy()
        pred_sents = reorder_sents([id2sent(vocab, words[:, i]) for i in range(words.shape[1])])
        targ_sents = reorder_sents([id2sent(vocab, labels[:, i]) for i in range(labels.shape[1])])
        # print(targ_sents)
        policy_values = torch.from_numpy(np.asarray([1 - pre_emb.wmdistance(pred_sent, targ_sent) for
                                                                (pred_sent, targ_sent) in
                                                                zip(pred_sents, targ_sents)])).cuda()
        # print(policy_values)
        # print("policy values shape {}".format(policy_values.size()))
    elif args.critic_reward == 'cosine' or args.critic_reward == 'cos':
        policy_values = torch_cos_sim(words, labels, vocab.id2vec, mean=False)
    elif args.critic_reward in ['infersent', 'bert', 'elmo', 'gpt', 'gpt2', 'transformer', 'transformerxl']:
        pred_sents = reorder_sents([id2sent(vocab, words[:, i]) for i in range(words.shape[1])])
        targ_sents = reorder_sents([id2sent(vocab, labels[:, i]) for i in range(labels.shape[1])])
        policy_values = get_reward(pred_sents, targ_sents, method=args.critic_reward, model=sent_mod,
                                   bsize=args.batch_size, dist=args.reward_dist, mean=False)
        policy_values = np.asarray(policy_values, dtype=np.float32)

    elif args.critic_reward in ['uniskip', 'biskip']:
        policy_values = get_reward(pred_inds=words, targ_inds=labels, method=args.critic_reward, model=sent_mod,
                                   bsize=args.batch_size, dist=args.reward_dist, mean=False)
    else:
        policy_values = rouge_l(words, labels)
    return policy_values


if __name__ == "__main__":
    words = torch.randint(1, 5, (80, 24)).type(torch.LongTensor)
    labels = torch.randint(1, 5, (80, 24)).type(torch.LongTensor)

    print(rouge_l(words, labels))