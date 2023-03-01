import torch
from torch import nn, optim
from .base_model import SentenceRE
from torch.nn.functional import gelu,relu
from torch.nn import functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput

class SoftmaxNN_NER(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.linear = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, 10)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.crf = CRF(self.num_class, batch_first=True)
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, attention_mask, labels, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args)  # (B, H)
        rep = self.drop(rep)

        #logits = F.relu(self.linear(rep))
        emissions = self.fc(rep) # (B, N)
        # print(logits.shape)
        # print(emissions.shape)
        # print(attention_mask.shape)
        # print(labels.shape)
        #logits = F.dropout(logits, 0.2)
        #logits = F.softmax(logits, dim=1)
        # attention_mask = rep[1]
        # print(attention_mask.shape)
        logits = self.crf.decode(emissions, attention_mask.byte())
        # print(labels)
        # print(logits)
        # labels = torch.tensor(labels).long()
        loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
        # return logits, rep


class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.linear = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self,*args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args)  # (B, H)
        rep = self.drop(rep)
        #logits = F.relu(self.linear(rep))
        logits = self.fc(rep) # (B, N)
        #logits = F.dropout(logits, 0.2)
        #logits = F.softmax(logits, dim=1)
        return logits, rep
