#!/bin/bash

python apple_game_dqn.py \
  --width 17 \
  --height 10 \
  --render_mode human \
  --time_limit 120 \
  --target_update_steps 1000 \
  --learning_starts 5000 \
  --eps_decay 10000 \
  --frame_skip 4 \
  --max_frames 10000 \
  --train \
  --lr 0.001 \
  --hidden_dim 256 \
  --gamma 0.99 \
  --buffer_size 50000 \
  --batch_size 32 \
  --epsilon_start 1.0 \
  --epsilon_end 0.01 \
  --device cpu \
  --save_freq 5000

