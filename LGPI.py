import os
import json
import openai
from agents.dqn_agent import DQNAgent  # 기존 import 유지해도 무방
from .AppleGame import *  # AppleGameEnv 정의된 위치로 수정

# OpenAI API 키 설정 (환경변수 사용)
openai.api_key = os.getenv("OPENAI_API_KEY")

class LanguageGPIAgent:
    def __init__(self, env, model="gpt-4.1-nano", num_candidates=5, lookahead_steps=1, temp=0.3):
        self.env = env
        self.model = model
        self.num_candidates = num_candidates
        self.lookahead_steps = lookahead_steps
        self.temperature = temp

    def describe_state(self, obs):
        # 2D np.array → 간단한 자연어 묘사
        rows = obs.tolist()
        desc = "The grid has rows:\n"
        for i, row in enumerate(rows):
            desc += f"Row {i}: {row}\n"
        return desc

    def get_candidate_actions(self):
        # MultiDiscrete 샘플링 후 tuple로 변환, 중복 제거
        candidates = set()
        while len(candidates) < self.num_candidates:
            sample = self.env.action_space.sample()
            action = tuple(int(x) for x in sample)
            candidates.add(action)
        return list(candidates)

    def language_td(self, state_desc, actions):
        evaluations = {}
        for act in actions:
            prompt = (
                f"You are a language value function approximator.\n\n"
                f"State:\n{state_desc}\n"
                f"Candidate action: {list(act)}\n\n"
                f"Perform a {self.lookahead_steps}-step lookahead, describe your chain-of-thought, "
                f"then provide your evaluation. Respond in JSON with keys "
                f"'Thought' and 'Evaluation'."
            )
            resp = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            # 응답에서 JSON 파싱
            txt = resp.choices[0].message.content
            data = json.loads(txt)
            evaluations[act] = data["Evaluation"]
        return evaluations

    def policy_improvement(self, state_desc, evaluations):
        eval_text = ""
        for act, ev in evaluations.items():
            eval_text += f"Action: {list(act)}\nEval: {ev}\n\n"
        prompt = (
            f"You are a language policy improvement operator.\n\n"
            f"State:\n{state_desc}\n\n"
            f"Evaluations:\n{eval_text}"
            "Select the best action and respond in JSON with keys 'Thought' and "
            "'FinalAction' where FinalAction is a list of four integers."
        )
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        return tuple(data["FinalAction"])

    def get_action(self, observation):
        # 1) 상태 묘사
        state_desc = self.describe_state(observation)
        # 2) 후보 액션 생성
        candidates = self.get_candidate_actions()
        # 3) 언어 TD 평가
        evaluations = self.language_td(state_desc, candidates)
        # 4) 정책 개선 → 최종 액션 결정
        action = self.policy_improvement(state_desc, evaluations)
        return action


# main() 수정 예시
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Apple Game')
    parser.add_argument('--mode', choices=['play', 'rl'], default='rl')
    parser.add_argument('--agent',  choices=['random', 'dqn', 'lgpi'], default='lgpi')
    parser.add_argument('--render_mode', choices=['human', 'agent'], default='human')
    parser.add_argument('--width', type=int, default=17)
    parser.add_argument('--height', type=int, default=10)
    parser.add_argument('--time_limit', type=int, default=120)
    args = parser.parse_args()

    if args.mode == 'play':
        play_game(width=args.width, height=args.height, time_limit=args.time_limit)
    else:
        env = AppleGameEnv(width=args.width, height=args.height,
                           render_mode=args.render_mode,
                           time_limit=args.time_limit)

        if args.agent == 'dqn':
            agent = DQNAgent(env)
        elif args.agent == 'lgpi':
            agent = LanguageGPIAgent(env,
                                     model="gpt-4o",
                                     num_candidates=5,
                                     lookahead_steps=1,
                                     temp=0.3)
        else:
            agent = None  # random 등 다른 에이전트

        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, _, info = env.step(action)
            if hasattr(agent, 'store_transition'):
                agent.store_transition(obs, action, reward, obs, done)
            if hasattr(agent, 'update'):
                agent.update()
            env.render()
            print(f"Action: {action}, Reward: {reward}, Info: {info}")

        env.close()

if __name__ == "__main__":
    main()
