import torch
from models.networks.autoencoders.ae import Autoencoder
from torch.autograd import Variable


def ae_distil(model, perc = 0.5, num_epochs=20):
    perc = 1- perc / 100 if perc > 1 else 100 - perc
    """percent: 0.5 means reduce dimensionality to 50%"""
    criterion = torch.nn.MSELoss()
    for p_name, p in model.named_parameters():
        if 'weight' in p_name.lower() and 'layernorm' not in p_name.lower() \
                and 'word_emb' not in p_name.lower() and len(p.data.size()) != 1:

            m, n = p.size()
            k = int(n * perc)  # keep top k hidden dims
            model = Autoencoder(n, k)
            opt = torch.optim.Adam(model.parameters())
            for _ in num_epochs:
                opt.zero_grad()
                p_hat = model(p)
                loss = criterion(p_hat, p)
                loss.backward()
                opt.step()

            p_emb = model.get_embedding(p)
            p_emb = torch.cat([torch.zeros(p_emb.size(0), abs(n - p_emb.size(1))).cuda(), p_emb], 1)
            p_emb = Variable(p_emb, requires_grad=True).cuda()
            model.state_dict()[p_name].data.copy_(p_emb)

    return model.cuda()