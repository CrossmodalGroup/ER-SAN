# ER-SAN

## Introduction

ER-SAN: Enhanced-Adaptive Relation Self-Attention Network for Image Captioning — IJCAI22

In this paper, we propose to enhance the correlations between objects from a comprehensive view that jointly considers explicit semantic and geometric relations, generating plausible captions with accurate relationship predictions

![TripletTransformer](media/TripletTransformer.png)

## Requirements
* Cuda-enabled GPU
* ...


## Prepare Data
You can see [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md) for more details

### 1. Download and preprocess the COCO captions

Download the preprocessed COCO captions from this [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) of Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it into `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then run:

```
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

Next run:
```
$ python scripts/prepro_ngrams.py --input_json data/dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

This will preprocess the dataset and get the cache for calculating cider score.

### 2. Download Bottom-Up features
The pre-extracted [bottom-up](https://github.com/peteanderson80/bottom-up-attention) image features are used. Download pre-extracted feature from this [link](https://github.com/peteanderson80/bottom-up-attention#pretrained-features) (our paper uses the adaptive features).
Then:

```bash
mkdir data/bu_data; cd data/bu_data
wget https://storage.googleapis.com/bottom-up-attention/trainval.zip
unzip trainval.zip
```
```bash
python script/make_bu_data.py --output_dir data/cocobu
```
This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.

### 3. Download sematic graph data
The sematic graph data we use is from [WeakVRD-Captioning](https://github.com/Gitsamshi/WeakVRD-Captioning). First download the file `coco_cmb_vrg_final.zip` from this [link](https://drive.google.com/drive/folders/1Xt3ZSajATEkKb2RggkzRcgax_hVgfgZv) and unzip it in folder `data`, the semantic graph data of each image are stored in `coco_cmb_vrg_final`,  including the object label of each detected box and the relationship labels between boxes. Then download `cocotalk_final.json` for an extended vocabulary dictionary. More information can be obtained from [this](https://github.com/Gitsamshi/WeakVRD-Captioning).

### 4. Preprocess geometry graph data 
You need to generate the relative bounding box coordinates for geometry features, please run the following:
```
python scripts/prepro_bbox_relative_coords.py --input_json data/dataset_coco.json --input_box_dir data/cocobu_box --output_dir data/cocobu_box_relative --image_root $IMAGE_ROOT
```
One method of pre-obtaining geometric graph data is following [VSUA-Captioning](https://github.com/ltguo19/VSUA-Captioning) after downloading [vsua_box_info.pkl](https://drive.google.com/file/d/1G9_ZdjyIprl2wyWCExslWTWOimJf3x8G/view), run:
```bash
python scripts/cal_geometry_feats.py
python scripts/build_geometry_graph.py
```
Generated geometric graph data will be stored in `data/geometry-iou0.2-dist0.5-undirected`.

### 5. Overall 
All in all, to run this model, you need to prepare following files/folders:
 ```bash
cocotalk_final.json        # file containing additional info and vocab information
cocotalk_label.h5          # captions groudtruth
coco-train-idxs.p          # cached token file for cider
cocobu_att                 # bottom-up feature
cocobu_box                 # bottom-up detected boxes information
coco_cmb_vrg_final         # sematic graph data
cocobu_adaptive_box_relative  # relative bounding box coordinates
vsua_box_info.pkl          # boxes and width and height of images
geometry-iou0.2-dist0.5-undirected  # geometry graph data
 ```
## Training and Evaluation
### Cross entropy Training
Run the script train_triplet.sh or use the following code to train the model: 
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --id transformer_triplet --caption_model transformer_triplet --checkpoint_path log_transformer_triplet --label_smoothing 0.0 --batch_size 10 --learning_rate 3e-4 --num_layers 4 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 3 --learning_rate_decay_rate 0.5 --scheduled_sampling_start 0 --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --max_epochs 18 --noamopt_warmup 33000 --use_box 1 --loader_num_workers 4 --sg_label_embed_size 512 --seq_per_img 5 --use_warmup
```
The train script will dump checkpoints into the folder specified by `--checkpoint_path`. We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

For more options, see `opts.py`.

You can run test_triplet.sh or following code for evaluation:
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet/model-best.pth --infos_path log_transformer_triplet/infos_transformer_triplet-best.pkl --input_json data/cocotalk_final.json --language_eval 1 --beam_size 1 --sg_label_embed_size 512
```

### Self-critical RL training
First, copy the model from the pretrained model using cross entropy. 
```bash
$ bash scripts/copy_model.sh transformer_triplet transformer_triplet_rl
```

Then run train_triplet_rl.sh or following code:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --id transformer_triplet_rl --caption_model transformer_triplet --checkpoint_path log_transformer_triplet_rl --label_smoothing 0.0 --batch_size 10 --learning_rate 4e-5 --num_layers 4 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 17  --learning_rate_decay_rate 0.8  --scheduled_sampling_start 0 --save_checkpoint_every 3000 --language_eval 1 --val_images_use 5000 --self_critical_after 17 --max_epochs 58 --loader_num_workers 4 --start_from log_transformer_triplet_rl  --sg_label_embed_size 512 --seq_per_img 5 --use_box 1
```
For evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet_rl/model-best.pth --infos_path log_transformer_triplet_rl/infos_transformer_triplet_rl-best.pkl --language_eval 1 --beam_size 1
```

### Ensemble model testing
Our code allows different models trained from different random seeds to be combined to form an ensemble model.
The eval_ensemble.py assumes the model saving under log_transformer_triplet_$seed. Run following code in test_triplet_ensemble.sh:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_ensemble.py --dump_images 0 --num_images 5000 --input_json data/cocotalk_final.json --language_eval 1 --beam_size 1 --sg_label_embed_size 512 --ids transformer_triplet_2022 transformer_triplet_42 transformer_triplet_5201314 transformer_triplet_901 --id ensemble_model --verbose_loss 1
```
## Acknowledgement
Our code is mainly modified from [yahoo/object_relation_transformer](https://github.com/yahoo/object_relation_transformer). We use the visual features provided by Bottom-Up [peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), and the semantic graph data provided by [WeakVRD-Captioning](https://github.com/Gitsamshi/WeakVRD-Captioning), the geometry graph data provided by [VSUA-Captioning](https://github.com/ltguo19/VSUA-Captioning). If you think this code is helpful, please consider to cite the corresponding papers and our IJCAI paper.

