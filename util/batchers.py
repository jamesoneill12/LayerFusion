# coding: utf-8
import collections
import re
from six.moves import cPickle
from util.misc import *


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

# why not first create batches by grouping ones of similar length then ?
# gets rid of stochasticity but I think it would be better ?

def check_nan(tensor):
    check = torch.isnan(tensor.cpu()).type(torch.ByteTensor).any()
    return check


def check_gradients(model):
    for name, p in model.named_parameters():
        print("{} : Gradient {}".format(name, p.grad is not None))


def get_mt_batches(batch, corpus, joint=False, src='german'):
    """By default German is the source language. """
    if src == 'german':
        source_vocab = corpus.de_vocab
        if joint:
            target_vocab = corpus.en_vocab
    else:
        source_vocab = corpus.en_vocab
        if joint:
            target_vocab = corpus.de_vocab

    x_src = batch.src
    # src_length = batch.src.size(0)
    x_trg = batch.trg[:, :-1]
    x_start = (torch.zeros((x_trg.size(0), 1)) *
               source_vocab[corpus.BOS_WORD]).type(torch.cuda.LongTensor)
    x_trg = torch.cat([x_start, x_trg], dim=1)
    trg_output = batch.trg[:, 1:]
    x_end = (torch.zeros((trg_output.size(0), 1)) *
             source_vocab[corpus.EOS_WORD]).type(torch.cuda.LongTensor)
    trg_output = torch.cat([trg_output, x_end], dim=1)

    if joint:
        """
        Needed when building language model on the source LM. When would we need
        a language model on the source side ? When we want to predict multiple steps
        ahead to create context when only given some words at the beginning.
        """
        x_src_output = batch.src[:, :-1]
        x_src_output_start = (torch.zeros((x_src_output.size(0), 1)) *
                   target_vocab[corpus.BOS_WORD]).type(torch.cuda.LongTensor)
        x_src_output = torch.cat([x_src_output_start, x_src_output], dim=1)
        return x_src, x_trg, trg_output, x_src_output

    return x_src, x_trg, trg_output


def get_seq_batch(x_src, corpus, task='mt'):
    if task == 'mt':
        bos = corpus.BOS_WORD; eos = corpus.EOS_WORD
        x_start = (torch.zeros((x_src.size(0), 1)) *
                   corpus.en_vocab[bos]).type(torch.cuda.LongTensor)
        x_trg = torch.cat([x_start, x_src], dim=1)
        src_output = x_src[:, 1:]
        x_end = (torch.zeros((src_output.size(0), 1)) *
                 corpus.en_vocab[eos]).type(torch.cuda.LongTensor)
    # corpus is actually the vocab for image captioning
    elif task == 'ic':
        # make sure start and end tokens are the same for both flickr30k and mscoco
        # x_start = (torch.zeros((x_src.size(0), 1)) * corpus['<start>']).type(torch.cuda.LongTensor)
        x_trg = x_src # torch.cat([x_start, x_src], dim=1)
        src_output = x_src[:, 1:]
        x_end = (torch.zeros((src_output.size(0), 1)) * corpus['<end>']).type(torch.cuda.LongTensor)
    trg_output = torch.cat([src_output, x_end], dim=1)
    # x_trg = in_captions, trg_output = captions
    return x_trg, trg_output


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.



"""
This is mainly evaluation metrics for machine translation
"""
"""
def get_pred_target_split(outputs, targets, lengths):
    output_p, output_pind = outputs.topk(1)
    assert len(output_pind) == sum(lengths)
    output_pind = np.split(np.squeeze(output_pind.cpu().numpy()), lengths)
    targets = np.split(targets.cpu().numpy(), lengths)
    return output_pind, targets
"""


def get_pred_target_split(outputs, targets, lengths):
    # used in image captioning where pack_padded is used (lengths comes from this)
    output_p, output_pind = outputs.topk(1)
    sq_out = np.squeeze(output_pind.cpu().numpy())
    #print(lengths)
    lengths = np.cumsum(lengths)
    output_pind = np.split(sq_out, lengths)
    # print(lengths)
    #print("targets {}".format(targets.shape))
    #print("Max length {}".format(max(lengths)))
    targets = np.split(targets.cpu().numpy(), lengths)
    return output_pind, targets


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def batchify_context(data, args):
    """Truncate corpus so remaining data can be split into batches evenly."""
    nbatch = data.size(0) // args.batch_size
    data = data.narrow(0, 0, nbatch * args.batch_size)

    print('Number of tokens after processing: %d' % data.size(0))

    if args.cuda:
        data = data.cuda()
    return data


def get_context_batch(source, i, args):
    """
    For restricted context size, the hidden state is not copied across targets, where almost
     every token serves as a target. The amount of data used depends on the sequence length.

    Examples of (context, target) pairs for the corpus "The cat sat on the mat to play with yarn" and sequence length 5:
        ("The cat sat on the", "mat")
        ("cat sat on the mat", "to")
        ("sat on the mat to", "play")
        ...
    """

    data_ = []
    target_ = []
    for j in range(args.batch_size):
        start = i * args.batch_size + j
        end = start + args.seq_len
        data_ += [source[start:end]]
        target_ += [source[start+1:end+1]]

    # No training, so volatile always True
    data = Variable(torch.stack(data_), volatile=True)
    target = Variable(torch.stack(target_))

    # sequence length x batch size for consistency with Merity et al.
    # Since each example corresponds to 1 target, only the last row of the targets variable are relevant,
    # but passing the whole tensor for complete info.
    return data.transpose(1,0), target.transpose(1,0)


def get_vocab_all_pos(pos_datafile, corpus_dict):
    """
    Generate a map.
    Keys = POS tag
    Values = a list of words with that POS tag, sorted by frequency
    """
    pos_ = {}
    with open(pos_datafile, 'r') as f:
        for line in f:
            line = line.strip().split(' ') + ['<eos>_<eos>'] if len(line.strip()) > 0 else ['<eos>_<eos>']
            for word_pair in line:
                w, p = word_pair.split('_')
                if p not in pos_:
                    pos_[p] = {}
                token_id = corpus_dict.word2idx[w]
                pos_[p][token_id] = corpus_dict.counter[token_id]

    for tag in pos_:
        # sort dictionary by rank and throw away the frequencies
        pos_[tag] = sorted(pos_[tag], key=pos_[tag].get)

    return pos_


def make_pos_cond(T, B, lengths, max_T):
    device = lengths.device

    p_plus_int = torch.arange(T, device=device)[:, None].repeat(1, B)[:, :, None]
    p_plus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_plus_oh.scatter_(2, p_plus_int, 1)

    p_minus_int = lengths[None, :] - 1 - torch.arange(T, device=device)[:, None]
    p_minus_int[p_minus_int < 0] = max_T - 1
    p_minus_oh = torch.empty(T, B, max_T, device=device).zero_()
    p_minus_oh.scatter_(2, p_minus_int[:, :, None], 1)

    pos_cond = torch.cat((p_plus_oh, p_minus_oh), -1)  # [T, B, max_T*2]

    return pos_cond


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    if batch_first:
        inputs = inputs.transpose(0, 1)

    if inputs.size(1) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')

    reversed_inputs = inputs.data.clone()
    for i, length in enumerate(lengths):
        time_ind = torch.LongTensor(list(reversed(range(length))))
        reversed_inputs[:length, i] = inputs[:, i][time_ind]

    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)

    return reversed_inputs


# BatchLoader for Recurrent VAE
class BatchLoader:
    def __init__(self, path='../../'):

        '''
            :properties

                data_files - array containing paths to data sources

                idx_files - array of paths to vocabulury files

                tensor_files - matrix with shape of [2, target_num] containing paths to files
                    with data represented as tensors
                    where first index in shape corresponds to types of representation of data,
                    i.e. word representation and character-aware representation

                blind_symbol - special symbol to fill spaces in every word in character-aware representation
                    to make all words be the same lenght
                pad_token - the same special symbol as blind_symbol, but in case of lines of words
                go_token - start of sequence symbol
                end_token - end of sequence symbol

                chars_vocab_size - number of unique characters
                idx_to_char - array of shape [chars_vocab_size] containing ordered list of inique characters
                char_to_idx - dictionary of shape [chars_vocab_size]
                    such that idx_to_char[char_to_idx[some_char]] = some_char
                    where some_char is such that idx_to_char contains it

                words_vocab_size, idx_to_word, word_to_idx - same as for characters

                max_word_len - maximum word length
                max_seq_len - maximum sequence length
                num_lines - num of lines in data with shape [target_num]

                word_tensor -  tensor of shape [target_num, num_lines, line_lenght] c
                    ontains word's indexes instead of words itself

                character_tensor - tensor of shape [target_num, num_lines, line_lenght, max_word_len].
                    Rows contain character indexes for every word in data

            :methods

                build_character_vocab(self, data) -> chars_vocab_size, idx_to_char, char_to_idx
                    chars_vocab_size - size of unique characters in corpus
                    idx_to_char - array of shape [chars_vocab_size] containing ordered list of inique characters
                    char_to_idx - dictionary of shape [chars_vocab_size]
                        such that idx_to_char[char_to_idx[some_char]] = some_char
                        where some_char is such that idx_to_char contains it

                build_word_vocab(self, sentences) -> words_vocab_size, idx_to_word, word_to_idx
                    same as for characters

                preprocess(self, data_files, idx_files, tensor_files) -> Void
                    preprocessed and initialized properties and then save them

                load_preprocessed(self, data_files, idx_files, tensor_files) -> Void
                    load and and initialized properties

                next_batch(self, batch_size, target_str) -> encoder_word_input, encoder_character_input, input_seq_len,
                        decoder_input, decoder_output
                    randomly sampled batch_size num of sequences for target from target_str.
                    fills sequences with pad tokens to made them the same lenght.
                    encoder_word_input and encoder_character_input have reversed order of the words
                        in case of performance
        '''

        self.data_files = [path + 'train.txt',
                           path + 'test.txt']

        self.idx_files = [path + 'words_vocab.pkl',
                          path + 'characters_vocab.pkl']

        self.tensor_files = [[path + 'train_word_tensor.npy',
                              path + 'valid_word_tensor.npy'],
                             [path + 'train_character_tensor.npy',
                              path + 'valid_character_tensor.npy']]

        self.blind_symbol = ''
        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'

        idx_exists = fold(f_and,
                          [os.path.exists(file) for file in self.idx_files],
                          True)

        tensors_exists = fold(f_and,
                              [os.path.exists(file) for target in self.tensor_files
                               for file in target],
                              True)

        if idx_exists and tensors_exists:
            self.load_preprocessed(self.data_files,
                                   self.idx_files,
                                   self.tensor_files)
            print('preprocessed data was found and loaded')
        else:
            self.preprocess(self.data_files,
                            self.idx_files,
                            self.tensor_files)
            print('data have preprocessed')

        self.word_embedding_index = 0

    def clean_whole_data(self, string):
        string = re.sub('^[\d\:]+ ', '', string, 0, re.M)
        string = re.sub('\n\s{11}', ' ', string, 0, re.M)
        string = re.sub('\n{2}', '\n', string, 0, re.M)

        return string.lower()

    def clean_str(self, string):
        '''
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        '''

        string = re.sub(r"[^가-힣A-Za-z0-9(),!?:;.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r":", " : ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def build_character_vocab(self, data):

        # unique characters with blind symbol
        chars = list(set(data)) + [self.blind_symbol, self.pad_token, self.go_token, self.end_token]
        chars_vocab_size = len(chars)

        # mappings itself
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return chars_vocab_size, idx_to_char, char_to_idx

    def build_word_vocab(self, sentences):

        # Build vocabulary
        word_counts = collections.Counter(sentences)

        # Mapping from index to word
        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = list(sorted(idx_to_word)) + [self.pad_token, self.go_token, self.end_token]

        words_vocab_size = len(idx_to_word)

        # Mapping from word to index
        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        return words_vocab_size, idx_to_word, word_to_idx

    def preprocess(self, data_files, idx_files, tensor_files):

        data = [open(file, "r").read() for file in data_files]
        merged_data = data[0] + '\n' + data[1]

        self.chars_vocab_size, self.idx_to_char, self.char_to_idx = self.build_character_vocab(merged_data)

        with open(idx_files[1], 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

        data_words = [[line.split() for line in target.split('\n')] for target in data]
        merged_data_words = merged_data.split()

        self.words_vocab_size, self.idx_to_word, self.word_to_idx = self.build_word_vocab(merged_data_words)
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        with open(idx_files[0], 'wb') as f:
            cPickle.dump(self.idx_to_word, f)

        self.word_tensor = np.array(
            [[list(map(self.word_to_idx.get, line)) for line in target] for target in data_words])
        print(self.word_tensor.shape)
        for i, path in enumerate(tensor_files[0]):
            np.save(path, self.word_tensor[i])

        self.character_tensor = np.array(
            [[list(map(self.encode_characters, line)) for line in target] for target in data_words])
        for i, path in enumerate(tensor_files[1]):
            np.save(path, self.character_tensor[i])

        self.just_words = [word for line in self.word_tensor[0] for word in line]

    def load_preprocessed(self, data_files, idx_files, tensor_files):

        data = [open(file, "r").read() for file in data_files]
        data_words = [[line.split() for line in target.split('\n')] for target in data]
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]

        [self.idx_to_word, self.idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]

        [self.words_vocab_size, self.chars_vocab_size] = [len(idx) for idx in [self.idx_to_word, self.idx_to_char]]

        [self.word_to_idx, self.char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                                [self.idx_to_word, self.idx_to_char]]

        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])

        [self.word_tensor, self.character_tensor] = [np.array([np.load(target) for target in input_type])
                                                     for input_type in tensor_files]

        self.just_words = [word for line in self.word_tensor[0] for word in line]

    def next_batch(self, batch_size, target_str):
        target = 0 if target_str == 'train' else 1

        indexes = np.array(np.random.randint(self.num_lines[target], size=batch_size))

        encoder_word_input = [self.word_tensor[target][index] for index in indexes]
        encoder_character_input = [self.character_tensor[target][index] for index in indexes]
        input_seq_len = [len(line) for line in encoder_word_input]
        max_input_seq_len = np.amax(input_seq_len)

        encoded_words = [[idx for idx in line] for line in encoder_word_input]
        decoder_word_input = [[self.word_to_idx[self.go_token]] + line for line in encoder_word_input]
        decoder_character_input = [[self.encode_characters(self.go_token)] + line for line in encoder_character_input]
        decoder_output = [line + [self.word_to_idx[self.end_token]] for line in encoded_words]

        # sorry
        for i, line in enumerate(decoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_word_input[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        for i, line in enumerate(decoder_character_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_character_input[i] = line + [self.encode_characters(self.pad_token)] * to_add

        for i, line in enumerate(decoder_output):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            decoder_output[i] = line + [self.word_to_idx[self.pad_token]] * to_add

        for i, line in enumerate(encoder_word_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_word_input[i] = [self.word_to_idx[self.pad_token]] * to_add + line[::-1]

        for i, line in enumerate(encoder_character_input):
            line_len = input_seq_len[i]
            to_add = max_input_seq_len - line_len
            encoder_character_input[i] = [self.encode_characters(self.pad_token)] * to_add + line[::-1]

        return np.array(encoder_word_input), np.array(encoder_character_input), \
               np.array(decoder_word_input), np.array(decoder_character_input), np.array(decoder_output)

    def next_embedding_seq(self, seq_len):
        """
        :return:
            tuple of input and output for word embedding learning,
            where input = [b, b, c, c, d, d, e, e]
            and output  = [a, c, b, d, d, e, d, g]
            for line [a, b, c, d, e, g] at index i
        """

        words_len = len(self.just_words)
        seq = [self.just_words[i % words_len]
               for i in np.arange(self.word_embedding_index, self.word_embedding_index + seq_len)]

        result = []
        for i in range(seq_len - 2):
            result.append([seq[i + 1], seq[i]])
            result.append([seq[i + 1], seq[i + 2]])

        self.word_embedding_index = (self.word_embedding_index + seq_len) % words_len - 2

        # input and target
        result = np.array(result)

        return result[:, 0], result[:, 1]

    def go_input(self, batch_size):
        go_word_input = [[self.word_to_idx[self.go_token]] for _ in range(batch_size)]
        go_character_input = [[self.encode_characters(self.go_token)] for _ in range(batch_size)]

        return np.array(go_word_input), np.array(go_character_input)

    def encode_word(self, idx):
        result = np.zeros(self.words_vocab_size)
        result[idx] = 1
        return result

    def decode_word(self, word_idx):
        word = self.idx_to_word[word_idx]
        return word

    def sample_word_from_distribution(self, distribution):
        ix = np.random.choice(range(self.words_vocab_size), p=distribution.ravel())
        x = np.zeros((self.words_vocab_size, 1))
        x[ix] = 1
        return self.idx_to_word[np.argmax(x)]

    def encode_characters(self, characters):
        word_len = len(characters)
        to_add = self.max_word_len - word_len
        characters_idx = [self.char_to_idx[i] for i in characters] + to_add * [self.char_to_idx['']]
        return characters_idx

    def decode_characters(self, characters_idx):
        characters = [self.idx_to_char[i] for i in characters_idx]
        return ''.join(characters)
