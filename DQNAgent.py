#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from envs.AppleGame import AppleGameEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Apple Game DQN Train/Eval with Early Stopping')
    # Environment parameters
    parser.add_argument('--width',         type=int,   default=17,     help='Grid width')
    parser.add_argument('--height',        type=int,   default=10,     help='Grid height')
    parser.add_argument('--render_mode',   choices=['human','agent'], default='human', help='Render mode')
    parser.add_argument('--time_limit',    type=int,   default=120,    help='Time limit per episode (s)')
    parser.add_argument('--seed',          type=int,   default=None,   help='Random seed')

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true', help='Run training')
    mode.add_argument('--eval',  action='store_true', help='Run evaluation')

    # Training control
    parser.add_argument('--episodes',    type=int,   default=1000,   help='Maximum number of episodes')
    parser.add_argument('--patience',    type=int,   default=20,     help='Early-stop patience (episodes)')
    parser.add_argument('--min_delta',   type=float, default=1e-3,   help='Minimum improvement to reset patience')
    parser.add_argument('--learning_starts',     type=int,   default=50000, help='Replay warm-up frames')
    parser.add_argument('--eps_decay',           type=int,   default=10000, help='Epsilon decay frames')
    parser.add_argument('--target_update_steps', type=int,   default=500,   help='Target sync freq')
    parser.add_argument('--save_freq',           type=int,   default=100,   help='Save checkpoint every N episodes')
    parser.add_argument('--ckpt_path',           type=str,   default=None,  help='Resume checkpoint path')

    # Evaluation control
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of evaluation episodes')

    # Agent hyperparameters
    parser.add_argument('--lr',            type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim',    type=int,   default=256,   help='Hidden layer size')
    parser.add_argument('--gamma',         type=float, default=0.99,  help='Discount factor')
    parser.add_argument('--buffer_size',   type=int,   default=50000, help='Replay buffer capacity')
    parser.add_argument('--batch_size',    type=int,   default=32,    help='Batch size')
    parser.add_argument('--epsilon_start', type=float, default=1.0,   help='Initial epsilon')
    parser.add_argument('--epsilon_end',   type=float, default=0.01,  help='Final epsilon')
    parser.add_argument('--device',        type=str,   default='cpu',  help='Torch device')
    return parser.parse_args()

def setup_logging(run_id: str):
    log_dir = os.path.join("logs", f"{run_id}_DQN")
    os.makedirs(log_dir, exist_ok=True)
    # File-only logger
    logger = logging.getLogger("AppleGameDQN")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    return logger, tb_writer, log_dir

def evaluate(agent, env, episodes, seed=None):
    scores = []
    pbar = tqdm(range(1, episodes + 1), desc="Eval Episodes")
    for ep in pbar:
        obs, _ = env.reset(seed=seed)
        done, total_reward = False, 0.0
        while not done:
            action = agent.get_action(obs, is_training=False)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            env.render()
        final_score = info.get("score", total_reward)
        scores.append(final_score)
        pbar.set_postfix({"score": f"{final_score:.2f}", "avg": f"{np.mean(scores):.2f}", "std": f"{np.std(scores):.2f}"})
    avg, std = np.mean(scores), np.std(scores)
    print(f"\nEvaluation over {episodes} episodes → avg: {avg:.2f}, std: {std:.2f}")
    return scores

def main():
    args = parse_args()
    # seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, tb_writer, log_dir = setup_logging(run_id)
    mode = "TRAIN" if args.train else "EVAL"
    logger.info(f"Mode: {mode} | Config: {args}")
    print(f"Running in {mode} mode")

    # create env & agent
    env = AppleGameEnv(
        width=args.width, height=args.height,
        render_mode=args.render_mode, time_limit=args.time_limit
    )
    agent = DQNAgent(
        env,
        lr=args.lr, hidden_dim=args.hidden_dim, gamma=args.gamma,
        buffer_size=args.buffer_size, learning_starts=(args.learning_starts if args.train else 0),
        batch_size=args.batch_size,
        epsilon_start=(args.epsilon_start if args.train else 0.0),
        epsilon_end=(args.epsilon_end if args.train else 0.0),
        epsilon_decay=(args.eps_decay if args.train else 1),
        target_update_freq=(args.target_update_steps if args.train else 0),
        device=args.device
    )
    if args.ckpt_path:
        agent.load(args.ckpt_path)
        logger.info(f"Loaded checkpoint from {args.ckpt_path}")

    if args.train:
        best_avg = -np.inf
        no_improve = 0
        scores = []

        pbar = tqdm(range(1, args.episodes + 1), desc="Train Episodes", unit="ep")
        for ep in pbar:
            obs, _ = env.reset(seed=args.seed)
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, done, _, info = env.step(action)
                # store & update
                agent.store_transition(obs, action, reward, next_obs, done)
                agent.update()
                obs = next_obs
                env.render()

            scores.append(info.get("score"))
            logger.info(f"[TRAIN] Episode {ep:03d}: score = {info.get('score', 0):.2f}")
            tb_writer.add_scalar("episode_score", info.get('score', 0), ep)
            tb_writer.add_scalar("epsilon", agent.epsilon, ep)

            # early stopping check
            recent_avg = np.mean(scores[-args.patience:])
            if recent_avg > best_avg + args.min_delta:
                best_avg = recent_avg
                no_improve = 0
            else:
                no_improve += 1

            # update pbar postfix with best_avg & epsilon
            pbar.set_postfix(
                {
                    "best_avg": f"{best_avg:.2f}",
                    "eps": f"{agent.epsilon:.2f}",
                }
            )

            if no_improve >= args.patience:
                logger.info(f"Early stopping at episode {ep}: no improvement for {args.patience} eps")
                break

            # checkpoint
            if ep % args.save_freq == 0:
                ckpt = os.path.join(log_dir, f"dqn_ep{ep:04d}.pth")
                agent.save(ckpt)
                logger.info(f"Saved checkpoint at episode {ep}")

        # final save
        final_ckpt = os.path.join(log_dir, "dqn_final.pth")
        agent.save(final_ckpt)
        logger.info(f"Training complete at episode {ep}, best_avg = {best_avg:.2f}, model → {final_ckpt}")

    else:
        # evaluation
        agent.epsilon = 0.8
        print(f'Agent epsilon: {agent.epsilon}')
        evaluate(agent, env, args.eval_episodes, seed=args.seed)

    tb_writer.close()
    env.close()

if __name__ == "__main__":
    main()
