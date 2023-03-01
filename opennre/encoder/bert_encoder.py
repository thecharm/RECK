import logging
import torch
import torch.nn as nn
from transformers import BertTokenizer
import transformers
from .base_encoder import BaseEncoder
import math
from torch.nn import functional as F
import json
from torch.nn.functional import gelu, relu, tanh
from ..module.nn import GA
import numpy as np
from torch import nn
from torchvision.models import resnet50
import timm

class BERTEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.pic_feat = 1024
        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        # self.word_embedding = BaseEncoder.word_embedding(self.num_token, self.word_size)

        # attention
        self.linear_q = nn.Linear(self.pic_feat, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)

        # fusion
        self.linear_final = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask, pic):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        x, sentence = self.bert(token, attention_mask=att_mask)
        output = self.linear_final(x)
        return output

    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask


class BERTEntityEncoder(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.word_size = 50
        self.hidden_size = 768 * 2
        self.word_embed = 300
        self.pic_feat = 2048 * 2
        self.pic_feat2 = 2048

        self.nfeat = 300
        self.nhid = 512  # 512 #
        self.nhead = 4  # 1 #
        self.nclass = 300  # 768 #
        # 512
        self.alpha = 0.01
        self.dropout = 0.05

        self.model_resnet50 = timm.create_model('resnet50', pretrained=True)


        for param in self.model_resnet50.parameters():
            param.requires_grad = True

        self.mask_entity = mask_entity
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_label = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.linear_pic = nn.Linear(2048, self.hidden_size // 2)
        self.linear_pic_combine = nn.Linear(768, self.hidden_size // 2)

        self.linear_hidden = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.linear_x = nn.Linear(self.word_embed, self.nfeat)

        self.linear_final = nn.Linear(self.hidden_size*2 + self.hidden_size // 2, self.hidden_size)

        # pic_attention
        self.linear_q1 = nn.Linear(768, self.hidden_size // 2)
        self.linear_k1 = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v1 = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        self.linear_q2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        self.linear_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_s = nn.Linear(self.hidden_size, self.hidden_size)

        # GAT
        self.gat = GAT(nfeat=self.nfeat, nclass=self.hidden_size // 2, nhid=self.nhid, dropout=self.dropout,
                       alpha=self.alpha,
                       nheads=self.nhead)
        self.gat_rev = GAT(nfeat=self.nfeat, nclass=self.hidden_size // 2, nhid=self.nhid, dropout=self.dropout,
                           alpha=self.alpha,
                           nheads=self.nhead)
        self.gat_img = GAT(nfeat=self.nfeat, nclass=self.nclass, nhid=self.nhid, dropout=self.dropout,
                           alpha=self.alpha,
                           nheads=self.nhead)

        self.predict = MLP(input_sizes=(self.word_embed, self.word_embed, 1))
        self.linear_weights = nn.Linear(self.hidden_size * 3, 3)

        self.dropout_linear = nn.Dropout(0.5)
        self.weight_linear = nn.Linear(self.hidden_size, 1)
        self.weight_linear2 = nn.Linear(self.hidden_size, 1)
        self.linear_entity = nn.Linear(self.hidden_size, self.hidden_size // 2)

    def forward(self, token, att_mask, pos1, pos2, pic, A, W, A_rev, W_rev):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """

        # graph
        W = self.gat(W, A)
        W_rev = self.gat_rev(W_rev, A_rev)

        W = torch.sum(W, dim=1)
        W_rev = torch.sum(W_rev, dim=1)

        W = torch.cat([W, W_rev], dim=1)
        W = self.linear_w(W)

        img_feature = self.model_resnet50.forward_features(pic)
        img_feature = torch.reshape(img_feature, (-1, 2048, 49))
        img_feature = torch.transpose(img_feature, 1, 2)
        img_feature = torch.reshape(img_feature, (-1, 4, 49, 2048))
        img_feature = torch.sum(img_feature, dim=2)
        img_feature = self.linear_pic(img_feature)

        output = self.bert(token, attention_mask=att_mask)
        hidden = output[0]

        # semantics alignment by attention
        pic_1 = img_feature
        pic_2 = img_feature

        hidden_k = self.linear_k1(hidden)
        hidden_v = self.linear_v1(hidden)
        pic_q = self.linear_q1(pic_1)
        pic_1 = torch.tanh(self.att(pic_q, hidden_k, hidden_v))
        pic = torch.sum(pic_1, dim=1)
        #
        pic_k = self.linear_k2(pic_2)
        pic_v = self.linear_v2(pic_2)
        hidden_q = self.linear_q2(hidden)
        hidden = torch.tanh(self.att(hidden_q, pic_k, pic_v))

        # Get entity start hidden state
        onehot_head = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_tail = torch.zeros(hidden.size()[:2]).float().to(hidden.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden, pic, W], -1)  # (B, 2H)
        x = self.linear_final(x)

        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        if len(item) == 5 and not isinstance(item, dict):
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(item)
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

            return indexed_tokens
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        elif 'token' in item:
            sentence = item['token']
            is_token = True

        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1


        return indexed_tokens, att_mask, pos1, pos2

    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def glove(self, ):
        f = open('ConceptNet_vocab.json', 'r')
        return json.load(f)


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)  # shape [N, out_features]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(h.size()[0], N * N, -1), h.repeat(1, N, 1)], dim=2).view(
            h.size()[0], N, -1, 2 * self.out_features)  # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [N,N,1] -> [N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # file_handle = open('first_case__.json', mode='a')  # w 写入模式
        x = attention.cpu().detach().numpy()
        x = x.tolist()
        import json
        # json.dump(x, file_handle)

        h_prime = torch.matmul(attention, h)  # [N,N], [N, out_features] --> [N, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                # x = gelu(x)
                x = relu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x


