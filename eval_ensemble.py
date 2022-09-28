from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

'''
To test the ensemble model
'''

# Input arguments and options
parser = argparse.ArgumentParser()

# Model settings
parser.add_argument('--ids', nargs='+', required=True, help='id of the models to ensemble')
parser.add_argument('--rela_gnn_type', type=int, default=0,
                help='rela gcn type')
parser.add_argument('--sg_label_embed_size', type=int, default=512,
                help='graph embedding_size of obj, rela')
parser.add_argument('--num_obj_label_use', type=int, default=1,
                help='number of object labels to use')
parser.add_argument('--num_attr_label_use', type=int, default=3,
                help='number of attribute labels to use')
parser.add_argument('--geometry_rela_feat_dim', type=int, default=8,
                help='dim of geometry relationship features')

# Input paths
parser.add_argument('--sg_vocab_path', type=str, default='data/coco_pred_sg_rela.npy',
                help='path to the vocab file, containing vocabularies of object, attribute, relationships')
parser.add_argument('--sg_data_dir', type=str, default='data/coco_cmb_vrg_final/',
                help='path to the scene graph data directory, containing numpy files about the '
                         'labels of object, attribute, and semantic relationships for each image')
parser.add_argument('--sg_geometry_dir', type=str, default='data/geometry-iou0.2-dist0.5-undirected/',
                help='directory of geometry edges and features')
parser.add_argument('--sg_box_info_path', type=str, default='data/vsua_box_info.pkl',
                help='path to the pickle file containing the width and height infos of images')
parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc',
                help='path to the COCO bu fc feature')
parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att',
                help='path to the COCO bu attention feature')
parser.add_argument('--input_box_dir', type=str, default='data/cocobu_box',
                help='path to the COCO bu boxs ')
parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='data/cocotalk_final.json',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='',
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
parser.add_argument('--input_rel_box_dir',type=str, default='data/cocobu_adaptive_box_relative',
                help="this directory contains the bboxes in relative coordinates for the corresponding image features in --input_att_dir")
# Sampling options
parser.add_argument('--batch_size', type=int, default=0,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--max_ppl', type=int, default=0,
                help='beam search by max perplexity or max probability.')
parser.add_argument('--beam_size', type=int, default=3,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--group_size', type=int, default=1,
                help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
parser.add_argument('--diversity_lambda', type=float, default=0.5,
                help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
parser.add_argument('--decoding_constraint', type=int, default=0,
                help='If 1, not allowing same word in a row')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                help='In case the image paths have to be preprended with a root path to an image folder')
# misc
parser.add_argument('--id', type=str, default='ensemble_model',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--verbose_beam', type=int, default=1,
                help='if we need to print out all beam search beams.')
parser.add_argument('--verbose_loss', type=int, default=0,
                help='if we need to calculate loss.')

parser.add_argument('--seq_length', type=int, default=40,
                help='maximum sequence length during sampling')

opt = parser.parse_args()

model_infos = [cPickle.load(open('log_%s/infos_%s-best.pkl' %(id, id), 'rb'), encoding='utf-8') for id in opt.ids]
model_paths = ['log_%s/model-best.pth' %(id) for id in opt.ids]

# Load one infos, assume these models are generated according to different seeds
infos = model_infos[0]

# override and collect parameters
if len(opt.input_fc_dir) == 0:
    opt.input_fc_dir = infos['opt'].input_fc_dir
    opt.input_att_dir = infos['opt'].input_att_dir
    opt.input_box_dir = infos['opt'].input_box_dir
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id

vars(opt).update({k: vars(infos['opt'])[k] for k in vars(infos['opt']).keys() if k not in vars(opt)}) # copy over options from model

opt.use_box = max([getattr(infos['opt'], 'use_box', 0) for infos in model_infos])
assert max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_att_feat', 0) for infos in model_infos]), 'Not support different norm_att_feat'
assert max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]) == max([getattr(infos['opt'], 'norm_box_feat', 0) for infos in model_infos]), 'Not support different norm_box_feat'

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
from models.TransformerEnsemble import TransformerEnsemble

_models = []
for i in range(len(model_infos)):
    model_infos[i]['opt'].start_from = None
    tmp = models.setup(model_infos[i]['opt'])
    tmp.load_state_dict(torch.load(model_paths[i]))
    tmp.cuda()
    tmp.eval()
    _models.append(tmp)

model = TransformerEnsemble(_models,opt)
model.seq_length = opt.seq_length
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']

# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))
