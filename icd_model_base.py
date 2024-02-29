import torch
from torch import nn
import torch.nn.functional as F

base_contribution = 0.02530090886
from models.text_encoder import TextEncoder, TextEncoderV2, TextEncoderBERT
from models.decoder import create_decoder
from models.label_encoder import LabelEncoder
from models.losses import loss_fn
from evaluation import all_metrics
import random


def compute_kl_loss(p, q, label_avg_num, require_activation=False, with_ori_format=False):
    p = p.contiguous()
    q = q.contiguous()
    if require_activation:
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    else:
        p_loss = F.kl_div(torch.log(p), q, reduction='none')
        q_loss = F.kl_div(torch.log(q), p, reduction='none')

    if label_avg_num is not None:
        p_loss = (p_loss.sum(dim=1) / label_avg_num).mean()
        q_loss = (q_loss.sum(dim=1) / label_avg_num).mean()
    elif with_ori_format:
        p_loss = p_loss
        q_loss = q_loss
    else:
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def compute_kl_loss_v2(p, q, label_avg_num, require_activation=False, with_ori_format=False):
    if require_activation:
        p_loss = F.kl_div(F.logsigmoid(p), torch.sigmoid(q), reduction='none')
        q_loss = F.kl_div(F.logsigmoid(q), torch.sigmoid(p), reduction='none')
    else:
        p_loss = F.kl_div(torch.log(p), q, reduction='none')
        q_loss = F.kl_div(torch.log(q), p, reduction='none')
    if label_avg_num is not None:
        p_loss = (p_loss.sum(dim=1) / label_avg_num).mean()
        q_loss = (q_loss.sum(dim=1) / label_avg_num).mean()
    else:
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class IcdModel(nn.Module):
    def __init__(self, word_config={}, combine_config={},
                 decoder_config={}, label_config={}, loss_config={}, args=None):
        super().__init__()
        if args.combiner == 'clinicalbert':
            self.encoder = TextEncoderBERT(word_config, combine_config)
        else:
            self.encoder = TextEncoder(word_config, combine_config)
        self.decoder = create_decoder(decoder_config)
        self.label_encoder = LabelEncoder(label_config)
        self.loss_config = loss_config
        self.args = args
        self.topk_num = args.topk_num
        if args.kl_type == 'softmax':
            self.kl_loss_fn = compute_kl_loss
        else:
            self.kl_loss_fn = compute_kl_loss_v2

    def calculate_text_hidden(self, input_word, word_mask):
        hidden = self.encoder(input_word, word_mask)
        return hidden

    def calculate_label_hidden(self):
        label_hidden = self.calculate_text_hidden(self.c_input_word,
                                                  self.c_word_mask)
        self.label_feats = self.label_encoder(label_hidden, self.c_word_mask)

    def calculate_label_hidden_m(self):
        label_hidden = self.calculate_text_hidden(self.mc_input_word,
                                                  self.mc_word_mask)
        self.mlabel_feats = self.label_encoder(label_hidden, self.mc_word_mask)

    def select_traget_codes(self, yhat, topk, label=None, indices_n=None):
        # select the codes that need to be trained
        # [B, Y]
        yhat = yhat.detach()
        yhat = torch.sigmoid(yhat)
        if label is not None:
            label = label.detach().bool()
            # yhat[label] = 1e10
            index_label = torch.where(label)[1]
        else:
            index_label = None
        p_log, index_p = torch.topk(yhat.sum(dim=0), k=topk, dim=0)
        index_p = index_p.flatten()
        index_p = torch.unique(index_p)
        if index_label is not None:
            index_p = torch.cat([index_p, index_label], dim=0).unique()
        if indices_n is not None:
            index_p = torch.cat([index_p, indices_n], dim=0).unique()
        return index_p

    def select_random_codes(self, topk, label=None, indices_top=None):
        topk = 1 if topk < 0 else topk
        label = label.detach()
        pool = torch.arange(label.shape[1], device=label.device)
        if indices_top is not None:
            pool = pool[~torch.isin(pool, indices_top)]

        shuffled_pool = torch.randperm(pool.shape[0], device=label.device)
        index_f = torch.unique(pool[shuffled_pool[0:topk]])
        if indices_top is not None:
            index_f = torch.cat([index_f, indices_top], dim=0).unique()
        return index_f

    def train_calculate_text_hidden(self, indices=None):
        c_input_word = self.c_input_word
        c_word_mask = self.c_word_mask
        if indices is not None:
            label_count = c_input_word.shape[0] // self.args.term_count
            c_input_word = c_input_word.reshape(label_count, self.args.term_count, -1)
            c_input_word = c_input_word[indices].reshape(-1, c_input_word.shape[-1])
            c_word_mask = c_word_mask.reshape(label_count, self.args.term_count, -1)
            c_word_mask = c_word_mask[indices].reshape(-1, c_word_mask.shape[-1])

        if self.args.random_sub:
            c_input_word = c_input_word.reshape(-1, self.args.term_count, self.c_input_word.shape[1])
            c_word_mask = c_word_mask.reshape(-1, self.args.term_count, self.c_word_mask.shape[1])
            tid = random.randint(0, c_input_word.shape[1] - 1)
            label_hidden = self.calculate_text_hidden(c_input_word[:, tid],
                                                      c_word_mask[:, tid])
            label_feats = self.label_encoder(label_hidden, c_word_mask[:, tid])
        else:
            label_hidden = self.calculate_text_hidden(c_input_word,
                                                      c_word_mask)
            label_feats = self.label_encoder(label_hidden, c_word_mask)
        return label_feats

    def train_calculate_text_hidden_m(self):
        c_input_word = self.mc_input_word
        c_word_mask = self.mc_word_mask
        if self.args.random_sub:
            c_input_word = c_input_word.reshape(-1, self.args.term_count, self.c_input_word.shape[1])
            c_word_mask = c_word_mask.reshape(-1, self.args.term_count, self.c_word_mask.shape[1])
            tid = random.randint(0, c_input_word.shape[1] - 1)
            label_hidden = self.calculate_text_hidden(c_input_word[:, tid],
                                                      c_word_mask[:, tid])
            label_feats = self.label_encoder(label_hidden, c_word_mask[:, tid])
        else:
            label_hidden = self.calculate_text_hidden(c_input_word,
                                                      c_word_mask)
            label_feats = self.label_encoder(label_hidden, c_word_mask)
        return label_feats

    def forward(self, batch, rdrop=False, indices=None):
        if rdrop:
            return self.forward_rdrop(batch, indices)
        else:
            return self.forward_normal(batch)

    def forward_normal(self, batch):
        input_word, word_mask = batch[0:2]
        indices_next = None
        hidden = self.calculate_text_hidden(input_word, word_mask)
        if self.args.text_block_rate > 0:
            if torch.rand(1) < self.args.text_block_rate:
                hidden = hidden.detach()
        # mc_logits = self.decoder(hidden, word_mask)
        label_feats = self.train_calculate_text_hidden()
        if self.args.use_graph:
            label_feats_m = self.train_calculate_text_hidden_m()
        else:
            label_feats_m = None
        if self.topk_num > 0:
            with torch.no_grad():
                if self.args.random_sub:
                    c_logits, c_alphas = self.decoder(hidden, word_mask, label_feats, 1)
                else:
                    c_logits, c_alphas = self.decoder(hidden, word_mask, label_feats, self.args.term_count)
            indices = self.select_traget_codes(c_logits, self.topk_num, batch[-2])
            # [V*T, E]
            label_count = label_feats.shape[0] // self.args.term_count
            label_feats = label_feats.reshape(1, label_count, self.args.term_count, -1)
            label_feats = label_feats.expand(input_word.shape[0], label_count, self.args.term_count, -1)
            label_feats = label_feats[indices[0], indices[1]]
            if self.args.random_sub:
                c_logits, c_alphas = self.decoder(hidden, word_mask, label_feats, 1, indices)
            else:
                c_logits, c_alphas = self.decoder(hidden, word_mask, label_feats, self.args.term_count, indices)
            c_label = batch[-2][indices[0], indices[1]]
        else:
            if self.args.random_sub:
                c_logits, c_alphas = self.decoder(hidden, word_mask, label_feats, 1, mlabel_feat=label_feats_m)
            else:
                c_logits, c_alphas = self.decoder(hidden, word_mask, label_feats, self.args.term_count,
                                                  mlabel_feat=label_feats_m)
            # mc_label = batch[-1]
            c_label = batch[-2]
        # mc_loss = loss_fn(mc_logits, mc_label, self.loss_config)
        mc_loss = 0.0

        c_loss = loss_fn(c_logits, c_label, self.loss_config)
        loss = mc_loss * self.loss_config['main_code_loss_weight'] + \
               c_loss * self.loss_config['code_loss_weight']
        if 'alpha_weight' in self.loss_config and self.loss_config['alpha_weight'] != 0.0:
            a_loss = c_alphas.mean()
            loss += self.loss_config['alpha_weight'] * a_loss
        else:
            a_loss = torch.FloatTensor([0])
        return {'mc_loss': mc_loss, 'c_loss': c_loss * self.loss_config['code_loss_weight'], 'loss': loss,
                'alpha_loss': self.loss_config['alpha_weight'] * a_loss, 'indices_next': indices_next}

    def forward_rdrop(self, batch, indices):
        input_word, word_mask = batch[0:2]
        if self.args.use_graph:
            if self.topk_num > 0:
                if indices is not None:
                    label_feats_m = self.train_calculate_text_hidden_m()
                else:
                    label_feats_m = None
            else:
                label_feats_m = self.train_calculate_text_hidden_m()
        else:
            label_feats_m = None

        hidden0 = self.calculate_text_hidden(input_word, word_mask)
        hidden1 = self.calculate_text_hidden(input_word, word_mask)
        a_loss = None
        indices_next = None
        if self.args.random_sub:
            term_count = 1
        else:
            term_count = self.args.term_count
        if self.topk_num > 0:
            if indices is None:
                with torch.no_grad():
                    label_feats = self.train_calculate_text_hidden()
                    c_logits, c_alphas = self.decoder(hidden0.detach(), word_mask, label_feats, term_count)
                topk_num = self.args.sample_num if label_feats_m is None else self.topk_num
                indices_t = self.select_traget_codes(c_logits, topk_num, batch[-2])
                indices_all = self.select_random_codes(2 * topk_num - indices_t.shape[0], batch[-2],
                                                       indices_top=indices_t)
                if self.args.use_graph:
                    if self.args.break_loss:
                        indices_next = self.select_traget_codes(c_logits, self.topk_num, batch[-2])
                        # indices_next = self.select_random_codes(self.topk_num - indices_next.shape[0], batch[-2],
                        #                                       indices_top=indices_next)
                    else:
                        indices_next = self.select_traget_codes(c_logits, self.topk_num)
                        # indices_next = self.select_random_codes(self.topk_num - indices_next.shape[0], batch[-2],
                        #                                       indices_top=indices_next)
                    # keep the unique indices in indices_all
                    indices = indices_all[~torch.isin(indices_all, indices_next)]
                else:
                    indices = indices_all
                del c_logits, c_alphas, label_feats
            # [V*T, E]
            label_feats = self.train_calculate_text_hidden(indices)
            c_logits0, c_alphas0 = self.decoder(hidden0, word_mask, label_feats, term_count, indices,
                                                mlabel_feat=label_feats_m)
            c_logits1, c_alphas1 = self.decoder(hidden1, word_mask, label_feats, term_count, indices,
                                                mlabel_feat=label_feats_m)
            c_label = batch[-2][:, indices]
            if self.args.break_loss:
                c_loss = (loss_fn(c_logits0, c_label, self.loss_config) + \
                          loss_fn(c_logits1, c_label, self.loss_config)) * 0.5
                kl_loss = self.kl_loss_fn(c_logits0, c_logits1, self.avg_label_num,
                                          self.args.decoder != 'MultiLabelMultiHeadLAATV4')
                if 'alpha_weight' in self.loss_config and self.loss_config[
                    'alpha_weight'] != 0.0 and c_alphas0 is not None and c_alphas1 is not None:
                    a_loss = (c_alphas0.mean() * 0.5 + c_alphas1.mean() * 0.5)
            else:
                avg_label_num = self.args.sample_num * 2
                c_loss = (loss_fn(c_logits0, c_label, self.loss_config, avg_label_num=avg_label_num) +
                          loss_fn(c_logits1, c_label, self.loss_config, avg_label_num=avg_label_num)) * 0.5
                kl_loss = self.kl_loss_fn(c_logits0, c_logits1, avg_label_num,
                                          self.args.decoder != 'MultiLabelMultiHeadLAATV4')
                if 'alpha_weight' in self.loss_config and self.loss_config[
                    'alpha_weight'] != 0.0 and c_alphas0 is not None and c_alphas1 is not None:
                    a_loss = (c_alphas0.sum() * 0.5 + c_alphas1.sum() * 0.5) / c_alphas0.shape[0] / avg_label_num

        else:
            label_feats = self.train_calculate_text_hidden()
            # ignore mc_logits
            c_logits0, c_alphas0 = self.decoder(hidden0, word_mask, label_feats, term_count,
                                                mlabel_feat=label_feats_m)
            c_logits1, c_alphas1 = self.decoder(hidden1, word_mask, label_feats, term_count,
                                                mlabel_feat=label_feats_m)
            c_label = batch[-2]
            c_loss = (loss_fn(c_logits0, c_label, self.loss_config) + \
                      loss_fn(c_logits1, c_label, self.loss_config)) * 0.5
            kl_loss = self.kl_loss_fn(c_logits0, c_logits1, self.avg_label_num,
                                      self.args.decoder != 'MultiLabelMultiHeadLAATV4')
            if 'alpha_weight' in self.loss_config and self.loss_config[
                'alpha_weight'] != 0.0 and c_alphas0 is not None and c_alphas1 is not None:
                a_loss = (c_alphas0.mean() * 0.5 + c_alphas1.mean() * 0.5)
        if hasattr(self.args, 'dual_rdrop_alpha') and self.args.dual_rdrop_alpha:
            rdrop_alpha = self.loss_config['rdrop_alpha'] if c_alphas0 is not None else 7.5
            loss = rdrop_alpha * kl_loss + \
                   c_loss * self.loss_config['code_loss_weight']
        else:
            loss = self.loss_config['rdrop_alpha'] * kl_loss + \
                   c_loss * self.loss_config['code_loss_weight']
        if a_loss is not None:
            loss += self.loss_config['alpha_weight'] * a_loss
        else:
            a_loss = torch.FloatTensor([0])
        return {'kl_loss': self.loss_config['rdrop_alpha'] * kl_loss,
                'c_loss': c_loss * self.loss_config['code_loss_weight'], 'loss': loss,
                'alpha_loss': self.loss_config['alpha_weight'] * a_loss,
                'indices_next': indices_next}

    def predict(self, batch, threshold=None, return_attentions=False):
        attentions = None
        if self.topk_num > 0 and self.args.use_graph:
            input_word, word_mask = batch[0:2]
            hidden = self.calculate_text_hidden(input_word, word_mask)
            assert hasattr(self, 'label_feats')
            yhat_raw, alphas = self.decoder(hidden, word_mask, self.label_feats, self.args.term_count)
            indices = self.select_traget_codes(yhat_raw, self.topk_num)
            label_count = self.label_feats.shape[0] // self.args.term_count
            label_feats = self.label_feats.reshape(label_count, self.args.term_count, -1)
            label_feats = label_feats[indices].reshape(-1, label_feats.shape[-1])
            if return_attentions:
                yhat_raw_fine, alphas, attentions = self.decoder(hidden, word_mask, label_feats, self.args.term_count,
                                                                 indices,
                                                                 mlabel_feat=self.mlabel_feats,
                                                                 return_attentions=return_attentions)
            else:
                yhat_raw_fine, alphas = self.decoder(hidden, word_mask, label_feats, self.args.term_count, indices,
                                                     mlabel_feat=self.mlabel_feats)
            yhat_raw[:, indices] = yhat_raw_fine
            if isinstance(yhat_raw, tuple):
                yhat_raw = yhat_raw[0]
            if threshold is None:
                threshold = self.args.prob_threshold
            yhat = yhat_raw.cpu().detach().numpy() >= threshold
            y = batch[-2].cpu().detach().numpy()
            yhat_raw = yhat_raw.cpu().detach().numpy()
            return {"yhat_raw": yhat_raw, "yhat": yhat, "y": y, "attentions": attentions}
        else:
            input_word, word_mask = batch[0:2]
            hidden = self.calculate_text_hidden(input_word, word_mask)
            assert hasattr(self, 'label_feats')
            if return_attentions:
                yhat_raw, alphas, attentions = self.decoder(hidden, word_mask, self.label_feats, self.args.term_count,
                                                            mlabel_feat=self.mlabel_feats,
                                                            return_attentions=return_attentions)
            else:
                yhat_raw, alphas = self.decoder(hidden, word_mask, self.label_feats, self.args.term_count,
                                                mlabel_feat=self.mlabel_feats)
            if isinstance(yhat_raw, tuple):
                yhat_raw = yhat_raw[0]
            if threshold is None:
                threshold = self.args.prob_threshold
            yhat = yhat_raw.cpu().detach().numpy() >= threshold
            y = batch[-2].cpu().detach().numpy()
            yhat_raw = yhat_raw.cpu().detach().numpy()
            return {"yhat_raw": yhat_raw, "yhat": yhat, "y": y, "attentions": attentions}

    def configure_optimizers(self, train_dataloader=None):
        args = self.args
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
            return [optimizer], [None]
        if self.args.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=args.learning_rate)
            return [optimizer], [None]
        if self.args.optimizer == "AdamW":
            no_decay = ["bias", "LayerNorm.weight"]
            params = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                    "lr": args.learning_rate
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": args.learning_rate
                },
            ]

            optimizer = AdamW(params, eps=args.adam_epsilon)

            self.total_steps = len(train_dataloader) * args.train_epoch

            if not hasattr(self.args, 'scheduler') or self.args.scheduler == "linear":
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio),
                    num_training_steps=self.total_steps,
                )
            elif self.args.scheduler == "constant":
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio)
                )
            elif self.args.scheduler == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(self.total_steps * args.warmup_ratio),
                    num_training_steps=self.total_steps,
                )
            return [optimizer], [scheduler]
