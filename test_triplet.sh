CUDA_VISIBLE_DEVICES=4 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet/model-best.pth --infos_path log_transformer_triplet/infos_transformer_triplet-best.pkl --input_json data/cocotalk_final.json --language_eval 1 --beam_size 1 --sg_label_embed_size 512

CUDA_VISIBLE_DEVICES=4 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet/model-best.pth --infos_path log_transformer_triplet/infos_transformer_triplet-best.pkl --input_json data/cocotalk_final.json --language_eval 1 --beam_size 2 --sg_label_embed_size 512

CUDA_VISIBLE_DEVICES=4 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet/model-best.pth --infos_path log_transformer_triplet/infos_transformer_triplet-best.pkl --input_json data/cocotalk_final.json --language_eval 1 --beam_size 3 --sg_label_embed_size 512
