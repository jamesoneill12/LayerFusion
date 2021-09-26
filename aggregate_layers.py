import torch


def get_layers_by_type(model, ignore_bias=True, layer_norm=False, embeddings=False, clean_names=False):
    """This returns a dictionary with number of layers
    distinct to type of parameter e.g dense layer, intermediate or attention layer"""
    model_name = model.__class__.__name__
    # for name, p in model.named_parameters(): print(name)
    nlayers = get_layer_dict(model_name)
    if layer_norm: nlayers = update_dict_ln(nlayers, model_name)
    if embeddings: nlayers = update_dict_emb(nlayers, model_name)
    att_tag = get_att_tag(model_name)
    for p_name, param in model.named_parameters():
        if clean_names: p_name = clean_param_name(p_name)
        print(p_name)
        for lname in nlayers.keys():
            if lname in p_name:
                if ignore_bias and 'bias' in p_name:
                    continue
                elif att_tag in p_name and att_tag not in lname:
                    continue
                nlayers[lname][p_name] = param
    return nlayers


def get_n_layers_per_type(model, ignore_bias=True, layer_norm=False, embeddings=False):
    """This returns the number of layers aggregated by type of parameter"""
    mod_name = model.__class__.__name__.lower()
    print(mod_name)
    nlayers = get_layer_dict(mod_name).keys()
    nlayers = dict(zip(list(nlayers), [0]*len(nlayers)))
    if layer_norm: nlayers = update_dict_ln(nlayers, mod_name, True)
    if embeddings: nlayers = update_dict_emb(nlayers, mod_name, True)
    att_tag = get_att_tag(mod_name)
    for p_name, param in model.named_parameters():
        for lname in nlayers.keys():
            if lname in p_name:
                if ignore_bias and 'bias' in p_name:
                    pass
                elif att_tag in p_name and att_tag not in lname:
                    pass
                else:
                    nlayers[lname] += 1
    return nlayers


def get_layer_dict(model_name):
    print("this is ", model_name)
    model_name = model_name.lower()
    print(model_name)
    if 'bert' in model_name:
        nlayers = {'intermediate.dense': {}, 'attention.self.query': {},
                   'attention.self.value': {}, 'attention.self.key': {},
                   'attention.output.dense': {}, 'output.dense': {},
                   }
    elif 'gpt2' in model_name:
        nlayers = {
            'ln_1': {}, 'ln_2': {},
            'attn.c_attn': {}, 'attn.c_proj': {},
            'mlp.c_fc': {}, 'mlp.c_proj': {},
                   }
    elif 'gpt' in model_name or 'openai-gpt' in model_name:
        nlayers = {'ln_1': {}, 'attn.c_attn': {},
                   'attn.c_proj': {}, 'ln_2': {},
                   'mlp.c_fc': {}, 'mlp.c_proj': {},
                   }
    elif model_name.strip() == 'transfoxllmheadmodel' or 'transfoxl' in model_name:
        nlayers = {'dec_attn.qkv_net': {}, 'dec_attn.o_net': {},
                   'dec_attn.layer_norm': {}, 'dec_attn.r_net': {},
                   'pos_ff.CoreNet.0': {}, 'pos_ff.CoreNet.3': {},
                   }
    return nlayers


def update_dict_ln(nlayers, model_name, init=False):
    if 'bert' in model_name:
        if init:
            nlayers['output.LayerNorm'] = 0
            nlayers['attention.output.LayerNorm'] = 0
        else:
            nlayers['output.LayerNorm'] =  {}
            nlayers['attention.output.LayerNorm'] = {}
    elif 'openai-gpt' in model_name:
        if init:
            nlayers['ln_2'] = 0
            nlayers['ln_1'] = 0
        else:
            nlayers['ln_2']  = {}
            nlayers['ln_1'] = {}
    elif 'gpt2' in model_name:
        if init:
            nlayers['ln_2']  = 0
            nlayers['ln_1'] =  0
        else:
            nlayers['ln_2']  = {}
            nlayers['ln_1'] =  {}
    elif model_name.strip() == 'transfoxllmheadmodel':
        if init:
            nlayers['output.LayerNorm'] = 0
            nlayers['attention.output.LayerNorm'] = 0
        else:
            nlayers['output.LayerNorm'] =  {}
            nlayers['attention.output.LayerNorm'] = {}
    return nlayers


def update_dict_emb(nlayers, model_name, init=False):
    if 'bert' in model_name:
        if init:
            nlayers['embeddings.weight'] = 0
        else:
            nlayers['embeddings.weight'] = {}
    elif 'openai-gpt' in model_name:
        if init:
            nlayers['wte.weight'] = 0
            nlayers['wpe.weight'] = 0
        else:
            nlayers['wte.weight'] = {}
            nlayers['wpe.weight'] = {}
    elif 'gpt2' in model_name:
        if init:
            nlayers['wte.weight'] = 0
            nlayers['wpe.weight'] = 0
        else:
            nlayers['wte.weight'] = {}
            nlayers['wpe.weight'] = {}
    elif model_name.strip() == 'transfoxllmheadmodel':
        if init:
            nlayers['embeddings.weight'] = 0
        else:
            nlayers['embeddings.weight'] = {}
    return nlayers


def get_att_tag(model_name):
    if 'gpt' in model_name: return 'attn'
    elif 'transfoxl' in model_name: return 'attention'
    elif 'bert' in model_name: return 'attention'
    else: return 'attention'


def clean_param_name(s, model_name):
    model_name = model_name.lower()
    if 'bert' in model_name: return s.replace("bert.encoder.layer.", "").replace("bert.embeddings.", "")
    elif 'gpt2' in model_name: return s.replace("transformer.h.", "").replace("transformer.", "")
    elif 'gpt' in model_name: return s.replace("transformer.h.", "").replace("transformer.", "")
    elif model_name.strip() == 'transfoxllmheadmodel': return s
    else: return s


