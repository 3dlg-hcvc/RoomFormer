#!/usr/bin/env bash

python eval.py --dataset_name=mp3d \
               --dataset_root=data/mp3d \
               --eval_set=test \
               --checkpoint=checkpoints/roomformer_stru3d.pth \
               --output_dir=eval_mp3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 
