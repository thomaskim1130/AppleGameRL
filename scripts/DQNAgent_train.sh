#!/bin/bash

python DQNAgent.py --train \
  --width 17 \
  --height 10 \
  --render_mode agent \
  --time_limit 120 \
  --seed 42 \
  --episodes 1000 \
  --patience 50 \
  --min_delta 1e-3 \
  --learning_starts 20000 \
  --eps_decay 1000000 \
  --target_update_steps 500 \
  --save_freq 100 \
  --ckpt_path "" \
  --lr 0.001 \
  --hidden_dim 256 \
  --gamma 0.99 \
  --buffer_size 50000 \
  --batch_size 32 \
  --epsilon_start 1.0 \
  --epsilon_end 0.01 \
  --device cpu
