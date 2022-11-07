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
path/to/data/
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
