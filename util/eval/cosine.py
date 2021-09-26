import torch
import numpy as np
import torch.nn.functional as F

def np_cos_sim(preds, targets, vocab):
    """preds and targets both gpu tensors, vocab maps inds to pretrain vecs"""
    preds, targets = list(preds.cpu().numpy()), list(targets.cpu().numpy())
    total_cos = 0.
    for p, t in zip(preds, targets):
        total_cos += float(np.dot(vocab[p], vocab[p])/(np.linalg.norm(p) * np.linalg.norm(p)))
    return total_cos/len(preds)


def torch_cos_sim(preds, targets, embs, mean=True):
    """vocab is gpu tensor in this case corresponding to embd"""
    if type(preds) is not torch.LongTensor:
        preds = preds.type(torch.LongTensor)
    if type(targets) is not torch.LongTensor:
        targets = targets.type(torch.LongTensor)
    pred_emb = embs[preds.squeeze()]; targ_embs = embs[targets]
    assert pred_emb.shape == targ_embs.shape
    if pred_emb.dim() == 3:
        csims = F.cosine_similarity(
                pred_emb.view(-1, pred_emb.size(2)),
                targ_embs.view(-1, targ_embs.size(2)), 1).view(pred_emb.size(0), pred_emb.size(1)
                                                               ).mean(1)
    else:
        csims = F.cosine_similarity(pred_emb, targ_embs, 1)
    return csims.mean() if mean else csims


if __name__ == "__main__":

    batch_size = 30
    images = torch.randn(torch.Size([40, 3, 224, 224])).cuda()
    labels = torch.randn(torch.Size([40, 45])).type(torch.cuda.LongTensor)

    if labels.size(0) != batch_size:
        pad_tensor = torch.zeros((batch_size - labels.size(0), labels.size(1))).type(torch.cuda.LongTensor)
        labels = torch.cat([labels, pad_tensor], 0)
        labels = torch.autograd.Variable(labels, requires_grad=False)
        # in_labels = torch.cat([in_labels, pad_tensor], 0)
    if images.size(0) != batch_size:
        pad_tensor = torch.zeros((batch_size - images.size(0), images.size(1),
                                  images.size(2), images.size(3))).type(torch.cuda.FloatTensor)
        images = torch.cat([images, pad_tensor], 0)

    print(labels.size())
    print(images.size())