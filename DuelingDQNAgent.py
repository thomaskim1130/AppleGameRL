import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# AppleGameEnv이 아래 import 경로에 있다고 가정합니다.
# 실제 파일 경로에 맞춰 조정하세요.
from AppleGame import AppleGameEnv


class DuelingDQNNetwork(nn.Module):
    def __init__(self, height, width, max_actions=1000):
        super(DuelingDQNNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Shared fully-connected layer
        self.fc_shared = nn.Linear(64 * height * width, 128)
        # Value stream
        self.fc_value = nn.Linear(128, 1)
        # Advantage stream
        self.fc_adv = nn.Linear(128, max_actions)

        # 가중치 초기화 (Xavier)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc_shared.weight)
        nn.init.xavier_uniform_(self.fc_value.weight)
        nn.init.xavier_uniform_(self.fc_adv.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc_shared.bias)
        nn.init.zeros_(self.fc_value.bias)
        nn.init.zeros_(self.fc_adv.bias)

    def forward(self, state, num_actions):
        """
        state: Tensor of shape (batch_size, height, width), 값은 [0,1] 범위로 정규화됨
        num_actions: 유효한 행동 개수 (정수) - 배치 내 모든 샘플에 대해 동일하다고 가정
        """
        # state 차원 확장: (batch_size, 1, height, width)
        x = state.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_shared(x))
        # Value and Advantage streams
        value = self.fc_value(x)                      # shape: (batch_size, 1)
        adv = self.fc_adv(x)                          # shape: (batch_size, max_actions)
        # Q(s,a) = V(s) + [A(s,a) - mean_a' A(s,a')]
        adv_mean = adv.mean(dim=1, keepdim=True)      # shape: (batch_size, 1)
        q_values = value + (adv - adv_mean)           # shape: (batch_size, max_actions)

        # 유효한 행동 개수(num_actions) 이후의 인덱스는 -inf 처리
        device = q_values.device
        mask = torch.arange(q_values.size(-1), device=device).unsqueeze(0) < num_actions
        neg_inf_tensor = torch.full_like(q_values, -float('inf'))
        q_values = torch.where(mask, q_values, neg_inf_tensor)
        return q_values  # (batch_size, max_actions)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_num_actions):
        """
        하나의 transition을 버퍼에 저장
        state, next_state: numpy 배열 (height, width) / 정규화된 값
        action: int (행동 인덱스)
        reward: float
        done: bool
        next_num_actions: int (다음 상태에서 유효한 행동 개수)
        """
        self.buffer.append((state, action, reward, next_state, done, next_num_actions))

    def sample(self, batch_size):
        """
        무작위로 batch_size 개의 transition 샘플링
        반환: 리스트 of tuples
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    def __init__(
        self,
        env: AppleGameEnv,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_size=100000,
        min_buffer_size=1000,
        update_freq=4,
        target_update_freq=10,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        max_actions=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_actions = max_actions

        # 디바이스 설정 (CPU/GPU 자동 선택)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 듀얼링 DQN: 행동망(policy network)과 타깃망(target network) 초기화
        self.policy_net = DuelingDQNNetwork(env.height, env.width, max_actions).to(self.device)
        self.target_net = DuelingDQNNetwork(env.height, env.width, max_actions).to(self.device)
        # 타깃망을 행동망과 동일하게 초기화
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 옵티마이저
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(buffer_size)

        # TensorBoard 로그 디렉터리: runs/DQN_AppleGame/<timestamp>
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("runs", "DuelingDQN_AppleGame", timestamp)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # 체크포인트 저장 폴더
        self.checkpoint_dir = log_dir

        # 전체 학습 step 카운터
        self.total_steps = 0

    def select_action(self, state, num_actions):
        """
        ε-greedy 정책으로 행동 선택
        state: numpy array or Tensor of shape (height, width)
        num_actions: 현재 유효한 행동 개수
        반환: action_idx (int)
        """
        if num_actions == 0:
            return None
        if random.random() < self.epsilon:
            # 탐험: 무작위 행동 선택 (0 ~ num_actions-1)
            return random.randrange(num_actions)
        else:
            # 활용: 정책망으로부터 Q값 예측 후 argmax (유효한 행동 내에서)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, H, W)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor, num_actions)  # (1, max_actions)
            action_idx = q_values.argmax(dim=1).item()
            return action_idx

    def update_model(self):
        """
        replay_buffer에서 미니배치를 샘플링하여 policy_net 갱신
        """
        if len(self.replay_buffer) < self.min_buffer_size:
            return None  # 충분한 경험이 모이지 않으면 학습하지 않음

        # 미니배치 샘플링
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_num_actions_list = zip(*batch)

        # 상태/다음 상태 텐서로 변환 후 정규화 (원래 값 1~9 → 0~1로 정규화)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device) / 9.0  # (batch, H, W)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device) / 9.0

        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)      # (batch,)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # (batch,)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)      # (batch,)
        next_num_actions_tensor = torch.tensor(next_num_actions_list, dtype=torch.long, device=self.device)  # (batch,)

        # 1) 타깃 Q값 계산 (타깃망 사용)
        with torch.no_grad():
            next_q_all = self.target_net(next_states_tensor, self.max_actions)  # (batch, max_actions)
            next_max_q_list = []
            for i in range(self.batch_size):
                n_actions = next_num_actions_tensor[i].item()
                if n_actions > 0:
                    next_max_q = next_q_all[i, :n_actions].max()
                else:
                    next_max_q = torch.tensor(0.0, device=self.device)
                next_max_q_list.append(next_max_q)
            next_max_q_tensor = torch.stack(next_max_q_list)  # (batch,)
            target_q = rewards_tensor + self.gamma * next_max_q_tensor * (1 - dones_tensor)

        # 2) 현재 Q값 예측 (정책망 사용)
        q_all = self.policy_net(states_tensor, self.max_actions)  # (batch, max_actions)
        current_q = q_all.gather(dim=1, index=actions_tensor.unsqueeze(1)).squeeze(1)  # (batch,)

        # 3) 손실 함수 계산 (MSE)
        loss = F.mse_loss(current_q, target_q)

        # 4) 역전파 및 최적화
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=1000, verbose=True):
        """
        Dueling DQN 학습 루프
        """
        print("=== Dueling DQN Training Start ===")
        pbar = trange(1, num_episodes + 1, desc="Episodes", ncols=80)
        for episode in pbar:
            # 1) 환경 리셋
            state, info = self.env.reset()
            state = state.astype(np.float32) / 9.0
            state = state.copy()  # (H, W)
            feasible_actions = info['feasible_actions']
            num_actions = len(feasible_actions)

            episode_reward = 0.0
            done = False

            # 만약 초기 상태부터 가능한 행동이 없다면, 에피소드 스킵
            if num_actions == 0:
                if verbose:
                    pbar.set_postfix_str(f"Ep {episode}: No initial actions, skip")
                # ε 감소
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
                self.writer.add_scalar('Score/Episode', episode_reward, episode)
                self.writer.add_scalar('Epsilon/Episode', self.epsilon, episode)
                continue

            # main loop: 가능한 행동이 있는 동안
            while not done:
                if num_actions == 0:
                    done = True
                    break

                # 2) ε-greedy로 행동 선택
                action_idx = self.select_action(state, num_actions)
                if action_idx is None:
                    done = True
                    break

                # 3) 행동 실행
                next_state, reward, done, _, info = self.env.step(action_idx)
                next_state = next_state.astype(np.float32) / 9.0
                next_state = next_state.copy()
                next_feasible_actions = info['feasible_actions']
                next_num_actions = len(next_feasible_actions)

                episode_reward = info['score']

                # 4) 경험 저장
                self.replay_buffer.push(
                    state, action_idx, reward, next_state, done, next_num_actions
                )

                state = next_state
                num_actions = next_num_actions

                # 5) 모델 업데이트 주기마다 학습 수행
                self.total_steps += 1
                if self.total_steps % self.update_freq == 0:
                    loss = self.update_model()
                    if loss is not None:
                        self.writer.add_scalar('Loss/Step', loss, self.total_steps)

                # 환경 렌더링
                self.env.render()
                # pygame.event.pump()

            # 에피소드 종료 후: 타깃망 갱신 주기마다 weight 복사
            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # ε 감소
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # TensorBoard logging (에피소드 단위)
            self.writer.add_scalar('Score/Episode', episode_reward, episode)
            self.writer.add_scalar('Epsilon/Episode', self.epsilon, episode)

            # tqdm의 postfix에 간략한 정보 업데이트
            if verbose:
                pbar.set_postfix({
                    "Score": f"{episode_reward:.2f}",
                    "Epsilon": f"{self.epsilon:.3f}"
                })

        print("=== Dueling DQN Training Finished ===")
        self.writer.close()

        # 학습 완료 후 체크포인트 저장
        checkpoint_path = os.path.join(self.checkpoint_dir, "dueling_dqn_apple_game.pth")
        torch.save(self.policy_net.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    def close(self):
        self.env.close()


if __name__ == "__main__":
    import pygame

    # 환경 생성 (render_mode='human'으로 설정해야 시각화됨)
    env = AppleGameEnv(width=17, height=10, render_mode='human', max_actions=1000)

    # DuelingDQNAgent 생성
    agent = DuelingDQNAgent(
        env,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_size=100000,
        min_buffer_size=1000,
        update_freq=4,
        target_update_freq=10,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        max_actions=1000,
    )

    # 학습 실행
    agent.train(num_episodes=1000, verbose=True)
    agent.close()
