import torch
import random
import string

src_vocab_size = 10000
trg_vocab_size = 5000

emb_size = 400
hidden_size = 400

batch_size = 20
sent_len = 40
nlayers = 2
dropout_rate = 0.2
str_len = 5
h2h = True  # highway2highway
x_src = torch.randint(0, src_vocab_size, (batch_size, sent_len)).type(torch.LongTensor)
x_trg = torch.randint(0, trg_vocab_size, (batch_size, sent_len)).type(torch.LongTensor)

strings = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(str_len)) for _ in
           range(trg_vocab_size)]
# thought it was
# dict(zip(strings, list(range(trg_vocab_size)))
trg_soi = trg_vocab_size - 1