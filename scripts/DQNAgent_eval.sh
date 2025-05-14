#!/bin/bash

python DQNAgent.py --eval \
  --width 17 \
  --height 10 \
  --render_mode human \
  --time_limit 120 \
  --seed 42 \
  --ckpt_path /Users/sangohkim/ksodev/cs377/Project/AppleGameRL/logs/20250514_160322_DQN/dqn_final.pth \
  --episodes 10 \
  --eval_episodes 10 \
  --lr 0.001 \
  --hidden_dim 256 \
  --gamma 0.99 \
  --buffer_size 50000 \
  --batch_size 32 \
  --epsilon_start 0.8 \
  --epsilon_end 0.8 \
  --device cpu
