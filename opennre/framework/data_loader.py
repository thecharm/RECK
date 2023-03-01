import torch
import torch.utils.data as data
import os, random, json, logging
import numpy as np
import sklearn.metrics
import timm
import cv2
from tqdm import tqdm, trange
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """

    def __init__(self, text_path, rel_path, pic_path, rel2id,
                 tokenizer, glove, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """

        super().__init__()
        self.text_path = text_path
        if 'train' in text_path:
            mode = 'train'
        elif 'val' in text_path:
            mode = 'val'
        else:
            mode = 'test'

        num_ = 5
        self.img_aux_path = text_path.replace('ours_%s.txt'%mode, 'mre_%s_dict.pth'%mode)
        self.pic_path_whole = pic_path
        self.pic_path_object = pic_path.replace('_org', '_vg')
        # concept for entity
        self.concept_path = './opennre/data/%s_0-%s_t2v_entity.json' % (mode, str(num_))
        self.concept_path_rev = './opennre/data/%s_0-%s_v2t_entity.json' % (mode, str(num_))

        self.rel_path = rel_path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs
        self.glove = glove
        # Load the text file
        f = open(text_path, encoding='UTF-8')
        self.data = []
        self.img_aux = {}
        self.transform = transforms.Compose([  # Compose 负责将后面几步整合到一起成为transform，后面只要调用transform就可以
            transforms.Resize(256),  # transforms.Resize (256) 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩
            transforms.CenterCrop(224),  # 截取中间224*224的区域
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化，image=(image-mean)/std，分别对应image的三个维度，一张图初始抽取就是三个维度的
                                 std=[0.229, 0.224, 0.225])])

        # self.data
        f_lines = f.readlines()
        self.state_dict = torch.load(self.img_aux_path)
        for i1 in tqdm(range(len(f_lines))):
            line = f_lines[i1].rstrip()
            if len(line) > 0:
                dic1 = eval(line)
                self.data.append(dic1)
        f.close()
        logging.info(
            "Loaded sentence RE dataset {} with {} lines and {} relations.".format(text_path, len(self.data),
                                                                                   len(self.rel2id)))

        with open(self.concept_path, 'r') as f:
            self.concept_data = json.load(f)
        with open(self.concept_path_rev, 'r') as f:
            self.concept_rev_data = json.load(f)

        self.tokenizer_ = BertTokenizer.from_pretrained('bert-base-cased')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        aux_imgs = []

        for i in range(3):
            if len(self.state_dict[index])>i:
                image = Image.open(os.path.join(self.pic_path_object, 'crops/' + self.state_dict[index][i])).convert(
                'RGB')
                img_features = self.transform(image).tolist()
                aux_img = img_features
                aux_imgs.append(aux_img)

        for i in range(3 - len(aux_imgs)):
            aux_imgs.append(torch.zeros((3, 224, 224)).tolist())

        assert len(aux_imgs) == 3
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))

        img_id = item['img_id']
        h = item['h']
        t = item['t']
        i_h_t = img_id + str(h) + str(t)

        image = cv2.imread((os.path.join(self.pic_path_whole, self.data[index]['img_id'])))
        size = (224, 224)
        img_features = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        img_features = torch.tensor(img_features)
        img_features = img_features.transpose(1, 2).transpose(0, 1)
        img = torch.reshape(img_features, (3, 224, 224)).to(torch.float32).tolist()
        pic = [img] + aux_imgs

        np_pic = np.array(pic).astype(np.float32)
        A, W = self.concept2emb(self.concept_data, i_h_t, img_id)
        A_rev, W_rev = self.concept2emb(self.concept_rev_data, i_h_t, img_id)
        pic = torch.tensor(np_pic).unsqueeze(0)

        list_p = list(pic)
        res = [self.rel2id[item['relation']]] + [img_id] + seq + list_p + [torch.Tensor(A).unsqueeze(0)] + [torch.Tensor(W).unsqueeze(0)] + [torch.Tensor(A_rev).unsqueeze(0)] + [torch.Tensor(W_rev).unsqueeze(0)]
        return res  # label, seq1, seq2, ...,pic

    def padding(self, item, rel, pos1, pos2):
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
        if not rev:
            pos1_len = pos_min[1] - pos_min[0]
            pos2_len = pos_max[1] - pos_max[0]
        else:
            pos2_len = pos_min[1] - pos_min[0]
            pos1_len = pos_max[1] - pos_max[0]
        pad_pos = [0, pos1, pos1 + pos1_len + 1, pos2, pos2_len + pos2 + 1]
        rel_zero_list = list(np.zeros([self.obj_num]))
        for pos in pad_pos:
            rel.insert(pos, rel_zero_list)
        rel = rel[:self.length]
        return rel

    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        img_id = data[1]
        seqs = data[2:]
        batch_labels = torch.tensor(labels).long()  # (B)
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(torch.cat(seq, 0))  # (B, L)
        return [batch_labels] + [img_id] + batch_seqs

    def token2emb(self, word, vocab):
        if word in vocab.keys():
            return vocab[word]
        else:
            return torch.FloatTensor(300).normal_(0, 1).tolist()

    def concept2emb(self, data, id_h_t, img_id):
        glv = self.glove
        # graph
        A = [[0 for i in range(110)] for j in range(110)]
        Word_Embedding = []
        concepts = set()
        for triplet in data[id_h_t]['noedes_and_edges_pos']:
            concepts.add(triplet['begin'])
            concepts.add(triplet['end'])

        concept2idx = {}
        concepts = list(concepts)
        for j in range(len(concepts)):
            concept2idx[concepts[j]] = j
            Word_Embedding.append(self.token2emb(concepts[j], glv))
        for j in range(len(concepts), 110):
            Word_Embedding.append([0 for n in range(300)])
        for triplet in data[id_h_t]['noedes_and_edges_pos']:
           A[concept2idx[triplet['begin']]][concept2idx[triplet['end']]] = 1

        return A, Word_Embedding


    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        correct_category = np.zeros([31, 1])
        org_category = np.zeros([31, 1])
        n_category = np.zeros([31, 1])
        data_with_pred_T = []
        data_with_pred_F = []
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        y_pred = []
        y_gt = []
        for i in range(total):
            y_pred.append(pred_result[i])
            y_gt.append(self.rel2id[self.data[i]['relation']])
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]  # Ground Truth Label
                n_category[golden] += 1
            data_with_pred = (str(self.data[i]) + str(pred_result[i]))
            if golden == pred_result[i]:
                correct += 1
                data_with_pred_T.append(data_with_pred)
                if golden != neg:
                    correct_positive += 1  # 预测正确的正样本数量
                    correct_category[golden] += 1
                else:
                    correct_category[0] += 1
            else:
                data_with_pred_F.append(data_with_pred)
            if golden != neg:
                gold_positive += 1  # 所有的正样本数量
                org_category[golden] += 1
            else:
                org_category[0] += 1
            if pred_result[i] != neg:
                pred_positive += 1  # 预测的正样本数量
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0
        # from sklearn.metrics import classification_report
        # target_names = ['class'+str(i) for i in range(18)]
        # print(classification_report(y_gt, y_pred, target_names=target_names,  digits=4))
        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))
        return result, correct_category, org_category, n_category, data_with_pred_T, data_with_pred_F


def SentenceRELoader(text_path, rel_path, pic_path, rel2id, tokenizer,
                     batch_size, glove,
                     shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn, **kwargs):
    dataset = SentenceREDataset(text_path=text_path, rel_path=rel_path, pic_path=pic_path,
                                rel2id=rel2id,
                                tokenizer=tokenizer,
                                glove=glove,
                                kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

