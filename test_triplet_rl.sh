CUDA_VISIBLE_DEVICES=7 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet_rl/model-best.pth --infos_path log_transformer_triplet_rl/infos_transformer_triplet_rl-best.pkl --language_eval 1 --beam_size 1

CUDA_VISIBLE_DEVICES=7 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet_rl/model-best.pth --infos_path log_transformer_triplet_rl/infos_transformer_triplet_rl-best.pkl --language_eval 1 --beam_size 2

CUDA_VISIBLE_DEVICES=7 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet_rl/model-best.pth --infos_path log_transformer_triplet_rl/infos_transformer_triplet_rl-best.pkl --language_eval 1 --beam_size 3

CUDA_VISIBLE_DEVICES=7 python eval.py --dump_images 0 --num_images 5000 --model log_transformer_triplet_rl/model-best.pth --infos_path log_transformer_triplet_rl/infos_transformer_triplet_rl-best.pkl --language_eval 1 --beam_size 5