
# coding:utf-8
import torch
import numpy as np
import json
import opennre.encoder, opennre.model, opennre.framework
import sys
import os
import argparse
import random
import logging
from tqdm import tqdm

def set_seed(seed=1997):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_path', default='bert-base-cased',
                    help='Pre-trained ckpt path / model name (hugginface)')
parser.add_argument('--ckpt', default='',
                    help='Checkpoint name')
parser.add_argument('--pooler', default='entity', choices=['cls', 'entity', 'rel'],
                    help='Sentence representation pooler')
parser.add_argument('--only_test', action='store_true',
                    help='Only run test')
parser.add_argument('--mask_entity', action='store_true',
                    help='Mask entity mentions')

# Data
parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                    help='Metric for picking up best checkpoint')
parser.add_argument('--dataset', default='none', choices=['none', 'semeval', 'wiki80', 'tacred', 'nyt10', 'ours'],
                    help='Dataset. If not none, the following args can be ignored')
parser.add_argument('--train_file', default='', type=str,
                    help='Training data file')
parser.add_argument('--val_file', default='', type=str,
                    help='Validation data file')
parser.add_argument('--test_file', default='', type=str,
                    help='Test data file')
parser.add_argument('--rel2id_file', default='', type=str,
                    help='Relation to ID file')

# Hyper-parameters
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Learning rate')
parser.add_argument('--max_length', default=128, type=int,
                    help='Maximum sentence length')
parser.add_argument('--max_epoch', default=8, type=int,
                    help='Max number of training epochs')
parser.add_argument('--concept_num', default=100, type=int,
                    help='Max number of training epochs')
parser.add_argument('--img_concept_num', default=5, type=int,
                    help='Max number of training epochs')
args = parser.parse_args()

# Some basic settings
root_path = '/home/thecharm/re/RE/'
sys.path.append(root_path)
if not os.path.exists('ckpt'):
    os.mkdir('ckpt')
if len(args.ckpt) == 0:
    args.ckpt = '{}_{}_{}'.format(args.dataset, args.pretrain_path, args.pooler)
ckpt = 'ckpt/{}.pth.tar'.format(args.ckpt)

if args.dataset != 'none':
    # opennre.download(args.dataset, root_path=root_path)
    args.train_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt/{}_train.txt'.format(args.dataset))
    args.val_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt/{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt/{}_test.txt'.format(args.dataset))
    args.pic_train_file = os.path.join(root_path, 'benchmark', args.dataset, 'img_org/train')
    args.pic_val_file = os.path.join(root_path, 'benchmark', args.dataset, 'img_org/val')
    args.pic_test_file = os.path.join(root_path, 'benchmark', args.dataset, 'img_org/test')
    args.rel_train_file = os.path.join(root_path, 'benchmark', args.dataset,
                                       'rel/train'.format(args.dataset))
    args.rel_val_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel/val'.format(args.dataset))
    args.rel_test_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel/test'.format(args.dataset))
    args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))
    if not os.path.exists(args.test_file):
        logging.warn("Test file {} does not exist! Use val file instead".format(args.test_file))
        args.test_file = args.val_file
    if args.dataset == 'wiki80':
        args.metric = 'acc'
    else:
        args.metric = 'micro_f1'
else:
    if not (os.path.exists(args.train_file) and os.path.exists(args.val_file) and os.path.exists(
            args.test_file) and os.path.exists(args.rel2id_file)):
        raise Exception(
            '--train_file, --val_file, --test_file and --rel2id_file are not specified or files do not exist. Or specify --dataset')

logging.info('Arguments:')
for arg in vars(args):
    logging.info('    {}: {}'.format(arg, getattr(args, arg)))

rel2id = json.load(open(args.rel2id_file))
id2rel = {v: k for k, v in rel2id.items()}

# Define the sentence encoder
if args.pooler == 'entity':
    sentence_encoder = opennre.encoder.BERTEntityEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
elif args.pooler == 'cls':
    sentence_encoder = opennre.encoder.BERTEncoder(
        max_length=args.max_length,
        pretrain_path=args.pretrain_path,
        mask_entity=args.mask_entity
    )
else:
    raise NotImplementedError

# Define the model
model = opennre.model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)

# Define the whole training framework
framework = opennre.framework.SentenceRE(
    train_path=args.train_file,
    train_pic_path=args.pic_train_file,
    train_rel_path=args.rel_train_file,
    val_path=args.val_file,
    val_pic_path=args.pic_val_file,
    val_rel_path=args.rel_val_file,
    test_path=args.test_file,
    test_pic_path=args.pic_test_file,
    test_rel_path=args.rel_test_file,
    model=model,
    ckpt=ckpt,
    batch_size=args.batch_size,
    max_epoch=args.max_epoch,
    lr=args.lr,
    opt='adamw'
)

# Train the model
set_seed(2000)
if not args.only_test:
    framework.train_model('micro_f1')

# Test
for loader in [framework.test_loader]:
    framework.load_state_dict(torch.load(ckpt)['state_dict'])
    result, correct_category, org_category, n_category, data_pred_t, data_pred_f, id_list, feature_list = framework.eval_model(
        loader)
    acc_category = correct_category / org_category
    # Print the result
    logging.info('Test set results:\n')
    logging.info('Accuracy: {}\n'.format(result['acc']))
    logging.info('Micro precision: {}\n'.format(result['micro_p']))
    logging.info('Micro recall: {}\n'.format(result['micro_r']))
    logging.info('Micro F1: {}'.format(result['micro_f1']))
    with open('./results/test/' + args.ckpt + '_result3123.txt', 'w') as f:
        for i in range(len(rel2id)):
            f.write(str(id2rel[i]) + ':' + str(acc_category[i]) + 'the count is ' + str(n_category[i]) + '\n')
        f.write('Test set results: \n')
        f.write('Accuracy: {}\n'.format(result['acc']))
        f.write('Micro precision: {}\n'.format(result['micro_p']))
        f.write('Micro recall: {}\n'.format(result['micro_r']))
        f.write('Micro F1: {}\n'.format(result['micro_f1']))
    with open('./results/test/' + args.ckpt + '_data_with_pred_t123124.json', 'w', encoding='UTF-8') as f1:
        for i in range(len(data_pred_t)):
            f1.write(data_pred_t[i] + "\n")
    with open('./results/test/' + args.ckpt + '_data_with_pred_f412312.json', 'w', encoding='UTF-8') as f2:
        for i in range(len(data_pred_f)):
            f2.write(data_pred_f[i] + "\n")
