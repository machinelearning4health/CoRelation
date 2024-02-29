import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from preprocess.coder_init_code_embedding import coder_init, load_code_embedding
from models.mlp import MLP
from torch.nn.init import xavier_uniform_ as xavier_uniform
from opt_einsum import contract
import pickle

class Decoder(nn.Module):
    def __init__(self, decoder_config={}):
        super(Decoder, self).__init__()
        self.decoder_config = decoder_config
        self.input_dim = self.decoder_config['input_dim']
        self.attention_dim = self.decoder_config['attention_dim']
        # self.label_count = self.decoder_config['label_count']
        self.code_embedding_path = self.decoder_config['code_embedding_path']
        self.ind2c = self.decoder_config['ind2c']
        self.ind2mc = self.decoder_config['ind2mc']
        
        # self.final = nn.Linear(self.input_dim, 1)
        self.final = nn.Linear(self.input_dim, len(self.ind2c))
        self.ignore_mask = False

        self.est_cls = self.decoder_config['est_cls']
        if self.est_cls > 0:
            # self.w_linear = nn.Linear(self.attention_dim, self.attention_dim)
            # self.b_linear = nn.Linear(self.attention_dim, 1)
            self.w_linear = MLP(self.attention_dim, self.attention_dim, self.attention_dim, self.est_cls)
            self.b_linear = MLP(self.attention_dim, self.attention_dim, 1, self.est_cls)
        elif self.est_cls == -1:
            self.w_linear = nn.Identity()
            self.b_linear = self.zero_first_dim

    def zero_first_dim(self, x):
        return torch.zeros_like(x)[:,0]

    def forward(self, h, word_mask, label_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat)
        
        if hasattr(self, 'w_linear'):
            w = self.w_linear(label_feat) # label * hidden
            b = self.b_linear(label_feat) # label * 1
            logits = self.get_logits(m, w, b)
        else:
            logits = self.get_logits(m)
        return logits
    
    def get_logits(self, m, w=None, b=None):
        # logits = self.final(m).squeeze(-1)
        # m: batch * label * hidden
        if w is None:
            # logits = self.final(m).squeeze(-1)
            logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            if len(w.shape) == 2:
                logits = contract('blh,lh->bl', m, w)
            else:
                logits = contract('blh,blh->bl', m, w)
        if b is not None:
            logits += b.squeeze(-1)
        return logits
    
    def get_label_queried_features(self, h, word_mask, label_feat):
        raise NotImplementedError
    
    def _code_emb_init(self, layer, version=None, init_code='ind2c'):
        if not self.code_embedding_path:
            return
        dim = layer.weight.shape[1]
        # if version is None:
        #     if self.label_count == 50 or self.label_count > 7000: 
        #         version = "mimic3"
        version = "mimic3"

        try:
            if init_code == "ind2c":
                ind2c = self.ind2c
            if init_code == "ind2mc":
                ind2c = self.ind2mc
            label_count = len(ind2c)
            code_embs = coder_init(self.code_embedding_path, ind2c, dim, 'cuda:0', version)
        except BaseException:
            code_embs = load_code_embedding(self.code_embedding_path)

        print(f'Init Layer {layer} using {self.code_embedding_path}')
            
        weights = np.zeros(layer.weight.size())
        for i in range(label_count):
            code = ind2c[i]
            weights[i] = code_embs[code]
        layer.weight.data[0:weights.shape[0]] = torch.Tensor(weights).clone()


class LAAT(Decoder):
    def __init__(self, decoder_config={}):
        super(LAAT, self).__init__(decoder_config)
        self.W = nn.Linear(self.input_dim, self.attention_dim)
        self.U = nn.Linear(self.attention_dim, len(self.ind2mc))
        
        self.xavier = self.decoder_config['xavier']
        
        if self.xavier:
            xavier_uniform(self.W.weight)
        self._code_emb_init(self.U, init_code='ind2mc')
    
    def get_label_queried_features(self, h, word_mask=None, label_feat=None):
        if word_mask is not None and not self.ignore_mask:
            l = word_mask.shape[-1]
            h = h[:,0:l]
        z = torch.tanh(self.W(h))
        if label_feat is None:
            label_feat = self.U.weight
        score = label_feat.matmul(z.transpose(1,2))
        if word_mask is not None and not self.ignore_mask:
            word_mask = word_mask.bool()
            score = score.masked_fill(mask=~word_mask[:,0:score.shape[-1]].unsqueeze(1).expand_as(score), value=float('-1e6'))
        alpha = F.softmax(score, dim=2)
        m = alpha.matmul(h) # Batch * Label * Hidden
        return m

    
ACT2FN = {'tanh':torch.tanh,
          'relu':torch.relu}

import math
from models.graph_models import KGCodeReassign
class CoRelationV3(Decoder):
    def __init__(self, decoder_config={}):
        super(CoRelationV3, self).__init__(decoder_config)
        self.attention_head = self.decoder_config['attention_head']
        self.attention_head_dim = self.decoder_config['attention_head_dim']
        self.W = nn.Linear(self.input_dim, self.attention_head*self.attention_head_dim)
        self.V = nn.Linear(self.input_dim, self.attention_dim, bias=False)
        self.u_reduce = nn.Linear(self.attention_dim, self.attention_head*self.attention_head_dim)
        self.code_bias = nn.Parameter(torch.zeros(1, len(self.ind2c)))

        self.random_sub = decoder_config['random_sub']
        self.xavier = self.decoder_config['xavier']
        self.without_bias = decoder_config['without_bias']
        MIMIC_DIR = decoder_config['MIMIC_DIR']

        if self.xavier:
            xavier_uniform(self.W.weight)
            xavier_uniform(self.V.weight)
            xavier_uniform(self.u_reduce.weight)
        # assert self.input_dim % self.attention_head == 0
        # assert self.attention_dim % self.attention_head == 0
        self.rep_dropout = nn.Dropout(decoder_config['rep_dropout'])

        self.head_pooling = decoder_config['head_pooling']
        self.text_pooling = decoder_config['text_pooling'] if decoder_config['text_pooling'] != 'default' else decoder_config['head_pooling']
        if self.head_pooling == "concat":
            assert self.attention_dim % self.attention_head == 0
            self.reduce = nn.Linear(self.attention_dim, self.attention_head*self.attention_head_dim)
        # elif self.head_pooling == "attention":
        # TODO
        if decoder_config.get('att_dropout') > 0.0:
            self.att_dropout_rate = decoder_config['att_dropout']
            self.att_dropout = nn.Dropout(self.att_dropout_rate)

        self.act_fn_name = decoder_config['act_fn_name']
        self.act_fn = ACT2FN[self.act_fn_name]
        self.use_graph = decoder_config['use_graph']
        if decoder_config['use_graph'] == True:
            self.alpha_graph_linear = MLP(self.attention_dim, self.attention_head_dim, 1, 2, act='relu')
            self.w_graph_linear = MLP(self.attention_dim, self.attention_dim, self.attention_dim, self.est_cls)
            if len(decoder_config['c2ind']) == 50:
                edges_dict = pickle.load(open('./mimicdata/{}/50_relation.pkl'.format(MIMIC_DIR), 'rb'))
            else:
                edges_dict = pickle.load(open('./mimicdata/{}/full_relation.pkl'.format(MIMIC_DIR), 'rb'))
            self.graph_encoder = KGCodeReassign(decoder_config, edges_dict, decoder_config['c2ind'], decoder_config['mc2ind'])
            #print(self.code_alphas[0:10])


    def get_logits(self, m, w=None):
        # logits = self.final(m).squeeze(-1)
        # m: batch * label * hidden
        if w is None:
            # logits = self.final(m).squeeze(-1)
            logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            if len(w.shape) == 2:
                logits = contract('blh,lh->bl', m, w)
            elif len(w.shape) == 3 and (len(m.shape) == 3):
                logits = contract('blh,blh->bl', m, w)
            elif len(w.shape) == 3 and (len(m.shape) == 4):
                logits = contract('blsh,lsh->bls', m, w)
        return logits

    def transform_label_feats(self, label_feat, term_count):
        if self.head_pooling == "max":
            label_count = label_feat.shape[0] // term_count
            if len(label_feat.shape) == 2:
                label_feat = label_feat.reshape(label_count, term_count, -1).max(1)[0]
            else:
                label_feat = label_feat.max(2)[0]
        elif self.head_pooling == 'mean':
            label_count = label_feat.shape[0] // term_count
            if len(label_feat.shape) == 2:
                label_feat = label_feat.reshape(label_count, term_count, -1).mean(dim=1)
            else:
                label_feat = label_feat.mean(2)
        elif self.head_pooling == "concat":
            label_count = label_feat.shape[0] // term_count
            label_feat = self.reduce(label_feat)  # (label * head) * (hidden // head)
            label_feat = label_feat.reshape(label_count, term_count, -1)  # label * head * hidden
            label_feat = label_feat.reshape(label_count, -1)
        elif self.head_pooling == "full":
            label_count = label_feat.shape[0] // term_count
            label_feat = label_feat.reshape(label_count, term_count, -1)
        return label_feat

    def forward(self, h, word_mask, label_feat=None, term_count=None, indices=None, test=False, mlabel_feat=None):
        m = self.get_label_queried_features(h, word_mask, label_feat, term_count)
        label_feat = self.transform_label_feats(label_feat, term_count)
        w = self.w_linear(label_feat)  # label * hidden
        b = self.b_linear(label_feat) if not self.without_bias else None# label * 1
        logits = self.get_logits(m, w)
        if b is not None:
            logits += b.squeeze(-1)
        if self.use_graph:
            mm = self.get_label_queried_features(h, word_mask, mlabel_feat, 1)
            m_updated = self.graph_encoder(m, mm, indices)
            w_graph = self.w_graph_linear(label_feat)
            logits_updated = self.get_logits(m_updated, w_graph)
            code_alphas = self.alpha_graph_linear(m*w).squeeze(-1)
            code_alphas = torch.tanh(torch.nn.functional.softplus(code_alphas, beta=5))
            self.code_alphas = code_alphas
            logits = logits + code_alphas*logits_updated
        else:
            code_alphas = None
        if self.head_pooling == "full":
            logits = logits.max(2)[0]
        if test:
            return logits, b
        else:
            return logits, code_alphas

    def get_label_queried_features(self, h, word_mask=None, label_feat=None, term_count=None):
        if self.random_sub:
            label_count = label_feat.shape[0] // term_count
            label_feat = label_feat.reshape(label_count, term_count, -1).mean(dim=1)
            term_count = 1
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                l = word_mask.shape[-1]
                h = h[:, 0:l]
        z = self.W(h)  # batch_size * seq_length * att_dim
        v = self.V(h)
        batch_size, seq_length, att_dim = h.size()
        if not hasattr(self, 'attention_head_dim'):
            self.attention_head_dim = att_dim//self.attention_head
        #z_reshape = z_reshape.expand(batch_size, seq_length, self.attention_head, att_dim_sub)
        z_reshape = z.reshape(batch_size, seq_length, self.attention_head, self.attention_head_dim)
        v_reshape = v.reshape(batch_size, seq_length, self.attention_head, att_dim//self.attention_head)
        # batch_size, seq_length, att_head, sub_dim
        if label_feat is None:
            label_feat = self.U.weight
        #[B, L, H, E] [B, C, 512]-> [C, S, H, E]
        if len(label_feat.shape) == 2:
            label_count = label_feat.size(0) // term_count
            u_reshape = self.u_reduce(label_feat.reshape(label_count, term_count, self.attention_dim)).reshape(label_count, term_count, self.attention_head,  self.attention_head_dim)
            score = contract('blhe,cshe->bcshl', z_reshape, u_reshape)
        else:
            label_count = label_feat.size(0)
            u_reshape = self.u_reduce(label_feat).reshape(label_count, term_count, self.attention_head,  self.attention_head_dim)
            score = contract('blhe,cshe->bcshl', z_reshape, u_reshape)
        score = score / math.sqrt(self.attention_head_dim)
        if word_mask is not None:
            if not hasattr(self, 'ignore_mask') or not self.ignore_mask:
                word_mask = word_mask.bool()
                score = score.masked_fill(
                    mask=~word_mask[:, 0:score.shape[-1]].unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(score),
                    value=float('-1e4'))
        alpha = F.softmax(score, dim=-1)  # softmax on seq_length # batch_size, label_count, seq_length, att_head
        if hasattr(self, 'att_dropout'):
            alpha = self.att_dropout(alpha)
            if self.training:
                alpha *= (1 - self.att_dropout_rate)

        m = contract('blhe,bcshl->bcshe', v_reshape, alpha)
        m = m.reshape(batch_size, label_count, term_count, -1)

        if self.text_pooling == 'max':
            m = m.max(dim=2)[0]
        elif self.text_pooling == 'mean':
            m = m.mean(dim=2)
        elif self.text_pooling == "concat":
            m = self.reduce(m.permute(0, 1, 3, 2))  # batch * label * hidden // head * head
            m = m.reshape(batch_size, -1, self.attention_dim)
        elif self.text_pooling == "full":
            m = m

        # batch_size, label_count, attention_head, hidden // attention_head
        m = self.rep_dropout(m)
        return m

class CoRelationV4(CoRelationV3):
    def __init__(self, decoder_config={}):
        super(CoRelationV4, self).__init__(decoder_config)

    def get_logits(self, m, w=None):
        # logits = self.final(m).squeeze(-1)
        # m: batch * label * hidden
        if w is None:
            # logits = self.final(m).squeeze(-1)
            logits = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        else:
            if len(w.shape) == 2:
                logits = contract('blh,lh->bl', m, w)
            elif len(w.shape) == 3 and (len(m.shape) == 3):
                logits = contract('blh,blh->bl', m, w)
            elif len(w.shape) == 3 and (len(m.shape) == 4):
                logits = contract('blsh,lsh->bls', m, w)
        return logits

    def forward(self, h, word_mask, label_feat=None, term_count=None, indices=None, mlabel_feat=None, return_attentions=False):
        m = self.get_label_queried_features(h, word_mask, label_feat, term_count)
        label_feat = self.transform_label_feats(label_feat, term_count)
        w = self.w_linear(label_feat)  # label * hidden
        b = self.b_linear(label_feat) if not self.without_bias else None# label * 1
        logits = self.get_logits(m, w)
        if b is not None:
            logits += b.squeeze(-1)
        scores = torch.sigmoid(logits)
        if self.use_graph and mlabel_feat is not None:
            mm = self.get_label_queried_features(h, word_mask, mlabel_feat, 1)
            if return_attentions:
                m_updated, att = self.graph_encoder(m, mm, indices, return_attentions)
            else:
                m_updated = self.graph_encoder(m, mm, indices)
            w_graph = self.w_graph_linear(label_feat)
            #print(m_updated.shape, w_graph)
            logits_updated = self.get_logits(m_updated, w_graph)
            scores_updated = torch.sigmoid(logits_updated)
            code_alphas = self.alpha_graph_linear(m*w).squeeze(-1)
            code_alphas = torch.sigmoid(code_alphas)
            self.code_alphas = code_alphas
            scores = (1-code_alphas)*scores + code_alphas*scores_updated
        else:
            code_alphas = None
        if self.head_pooling == "full":
            scores = scores.max(2)[0]
        if return_attentions:
            return scores, code_alphas, att
        else:
            return scores, code_alphas


class MultiLabelMultiHeadLV4(CoRelationV4):
    def __init__(self, decoder_config={}):
        super(MultiLabelMultiHeadLV4, self).__init__(decoder_config)


def create_decoder(decoder_config):
    if decoder_config['models'] == 'LAAT':
        decoder = LAAT(decoder_config)
    if decoder_config['models'] == 'CoRelationV3':
        decoder = CoRelationV3(decoder_config)
    if decoder_config['models'] == 'CoRelationV4':
        decoder = CoRelationV4(decoder_config)
    return decoder
