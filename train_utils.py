import os
import torch
import numpy as np
import json
from models.icd_model import IcdModel


def create_all_config(args, train_dataset):
    word_config = create_word_config(args, train_dataset)
    combine_config = create_combine_config(args)
    decoder_config = create_decoder_config(args, train_dataset)
    label_config = create_label_config(args, train_dataset)
    loss_config = create_loss_config(args, train_dataset)
    return word_config, combine_config, decoder_config, label_config, loss_config

def short_name(path):
    return path.split('/')[-1]
    

def short(x):
    if x.startswith("["):
        return x[1:-1]
    return x

def generate_output_folder_name(args):
    word_lst = ['word', args.word_dp]
    if args.word_frz:
        word_lst.append('frz')

    combine_lst = [args.combiner]
    if args.combiner == "lstm":
        combine_lst.extend([args.num_layers, args.rnn_dim, args.lstm_dp])

    
    decoder_lst = [args.decoder, args.attention_dim, args.attention_head_dim]
    if args.xavier:
        decoder_lst.append('xav')

    if args.decoder in ["CoRelationV2", "CoRelationV3", 'CoRelationV4']:
        decoder_lst.extend([args.rep_dropout, args.attention_head])
        decoder_lst.append(args.att_dropout)
        decoder_lst.append(args.head_pooling)
        decoder_lst.append(args.text_pooling)
        decoder_lst.append(args.act_fn_name)
        decoder_lst.append(args.use_graph)
        decoder_lst.append(str(args.alpha_weight))
        if args.use_graph:
            decoder_lst.append(f"graph{args.use_graph_rate}")
    
    label_lst = [args.label_pooling]
    if args.label_num_layers > 0:
        label_lst.append(args.label_num_layers)
    if args.label_dropout > 0:
        label_lst.append(args.label_dropout)
    label_lst.append(f'est{args.est_cls}')
    label_lst.append(args.topk_num)
    label_lst.append(args.sample_num)
    label_lst.append(args.break_loss)
        
    if args.term_count > 1:
        label_lst.extend([args.term_count, args.sort_method])
        
    loss_lst = [args.kl_type, args.loss_name, args.with_rdrop_weight, args.code_loss_weight, args.code_loss_weight]
    if args.loss_name == 'sigfocal':
        loss_lst.append(args.focal_gamma)

    bsz = args.batch_size * args.gradient_accumulation_steps * args.n_gpu
    train_lst = [f'bsz{bsz}', args.optimizer, args.train_epoch, args.truncate_length, f'warm{args.warmup_ratio}', f'wd{args.weight_decay}']
    if args.optimizer in ["Adam", "SGD", "AdamW"]:
        train_lst.append(args.learning_rate)
    if args.rdrop_alpha > 0.0:
        train_lst.append(f"rdrop{args.rdrop_alpha}")
    if args.without_bias:
        train_lst.append("nobias")
    if args.dual_rdrop_alpha:
        train_lst.append("dualrdrop")
    #if args.scheduler != "linear":
    #    train_lst.append(args.scheduler)
    if args.use_multihead>2:
        train_lst.append("mh{}".format(args.use_multihead))
                  
    all_lst = [[args.version], word_lst, combine_lst, decoder_lst, label_lst, loss_lst, train_lst]
    folder_name = "_".join(["-".join([str(y) for y in x]) for x in all_lst if x])
    if args.debug:
        folder_name = "debug_" + folder_name
    if args.tag:
        folder_name = folder_name + "-" + args.tag
    folder_name += f"-{args.round}"
        
    return folder_name
                  
def create_word_config(args, train_dataset):
    word_config = {}
    
    try:
        padding_idx = train_dataset.word2id['**PAD**']
    except BaseException:
        padding_idx = None
    word_config['padding_idx'] = padding_idx
    word_config['count'] = len(train_dataset.word2id)
    word_config['dropout'] = args.word_dp
    word_config['word_embedding_path'] = args.word_embedding_path
    word_config['dim'] = args.word_dim
    word_config['frz'] = args.word_frz

    
    return word_config


def create_combine_config(args):
    combine_config = {}
    combine_config['input_dim'] = args.word_dim
    
    combine_config['models'] = args.combiner
    if args.combiner == "lstm":
        combine_config['lstm_dropout'] = args.lstm_dp
        combine_config['rnn_dim'] = args.rnn_dim
        combine_config['num_layers'] = args.num_layers
        if args.num_layers <= 1:
            combine_config['lstm_dropout'] = 0.0
    if args.combiner == "reformer":
        combine_config['rnn_dim'] = args.rnn_dim
        combine_config['num_layers'] = args.num_layers
        reformer_config = {'reformer_head':args.reformer_head,
                           'n_hashes':args.n_hashes,
                           'local_attention_head':args.local_attention_head,
                           'pkm_layers':()}
        if args.pkm_layers:
            reformer_config['pkm_layers'] = tuple([int(n) for n in args.pkm_layers.split(',')])
        combine_config.update(reformer_config)
        combine_config['pos_embed'] = args.pos_embed
        if args.pos_embed == "axial":
            combine_config['axial'] = args.axial

    combine_config['dim'] = args.attention_dim

    return combine_config

def create_decoder_config(args, train_dataset):
    decoder_config = {}
    decoder_config['models'] = args.decoder
    decoder_config['input_dim'] = args.attention_dim
    decoder_config['attention_dim'] = args.attention_dim
    decoder_config['attention_head'] = args.attention_head
    decoder_config['attention_head_dim'] = args.attention_head_dim
    decoder_config['label_count'] = train_dataset.code_count
    decoder_config['code_embedding_path'] = args.code_embedding_path
    decoder_config['ind2c'] = train_dataset.ind2c
    decoder_config['c2ind'] = train_dataset.c2ind
    decoder_config['mc2ind'] = train_dataset.mc2ind
    decoder_config['ind2mc'] = train_dataset.ind2mc
    decoder_config['xavier'] = args.xavier
    decoder_config['est_cls'] = args.est_cls
    decoder_config['att_dropout'] = args.att_dropout
    decoder_config['term_count'] = args.term_count
    decoder_config['random_sub'] = args.random_sub
    decoder_config['use_graph'] = args.use_graph
    decoder_config['edge_dim'] = args.edge_dim
    decoder_config['without_bias'] = args.without_bias
    decoder_config['use_multihead'] = args.use_multihead
    if args.version in ['mimic3', 'mimic3-50']:
        MIMIC_DIR = 'mimic3'
    if args.version in ['mimic4', 'mimic4-50']:
        MIMIC_DIR = 'mimic4_9'
    if args.version in ['mimic4_10', 'mimic4_10-50']:
        MIMIC_DIR = 'mimic4_10'
    decoder_config['MIMIC_DIR'] = MIMIC_DIR

    if args.decoder in ['CoRelationV3', 'CoRelationV4']:
        decoder_config['attention_head'] = args.attention_head
        decoder_config['rep_dropout'] = args.rep_dropout
        decoder_config['head_pooling'] = args.head_pooling
        decoder_config['text_pooling'] = args.text_pooling
        decoder_config['act_fn_name'] = args.act_fn_name
        
    return decoder_config


def create_label_config(args, train_dataset):
    label_config = {}
    label_config['input_dim'] = args.rnn_dim
    label_config['num_layers'] = args.label_num_layers
    label_config['dropout'] = args.label_dropout
    label_config['pooling'] = args.label_pooling
    return label_config


def create_loss_config(args, train_dataset):
    loss_dict = {}
    if args.loss_name == "ce":
        loss_dict = {'name':'ce'}
    if args.loss_name == "focal":
        loss_dict = {'name':'focal', 'gamma':args.focal_gamma, 'alpha':args.focal_alpha}
    if args.loss_name == 'sigce' or args.decoder == 'CoRelationV4':
        loss_dict['name'] = 'sigce'
    if args.loss_name == "sigfocal":
        loss_dict = {'name':'sigfocal', 'gamma':args.focal_gamma, 'alpha':args.focal_alpha}

    if args.loss_name == "asy":
        if args.able_torch_grad_focal_loss:
            disable = False
        else:
            disable = True
        loss_dict = {'name':'asy', 'gamma_neg':args.asy_gamma_neg, 'gamma_pos':args.asy_gamma_pos,
                     'clip':args.asy_clip, 'disable_torch_grad_focal_loss':disable}
    if args.loss_name == "ldam":
        loss_dict = {'name':'ldam', 'ldam_c':args.ldam_c}
        total_label_count = np.array([0] * train_dataset.code_count)
        for i in range(len(train_dataset)):
            label = np.array(train_dataset[i][2])
            total_label_count += label
        loss_dict['label_count'] = total_label_count

    loss_dict['rdrop_alpha'] = args.rdrop_alpha

    loss_dict['main_code_loss_weight'] = args.main_code_loss_weight
    loss_dict['code_loss_weight'] = args.code_loss_weight
    loss_dict['alpha_weight'] = args.alpha_weight

    return loss_dict

def generate_model(args, train_dataset):
    word_config, combine_config, decoder_config, label_config, loss_config = \
        create_all_config(args, train_dataset)
    model = IcdModel(word_config, combine_config,
                     decoder_config, label_config, loss_config, args) 
    return model
