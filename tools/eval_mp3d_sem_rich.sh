#!/usr/bin/env bash

python eval.py --dataset_name=mp3d \
               --dataset_root=data/mp3d \
               --eval_set=test \
               --checkpoint=checkpoints/roomformer_stru3d_semantic_rich.pth \
               --output_dir=eval_mp3d_sem_rich \
               --num_queries=2800 \
               --num_polys=70 \
               --semantic_classes=19
