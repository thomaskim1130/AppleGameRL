import os
import time
import json
import random
import numpy as np
import openai
import pygame
import logging  # ★ 로깅 모듈 추가

from typing import List, Tuple
from pprint import pprint
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from AppleGame import AppleGameEnv  # AppleGameEnv 코드가 정의된 모듈을 import 가정


# 누적 비용 (달러)
cum_cost = 0.0

class CostLimitExceeded(Exception):
    """누적 비용이 한도를 초과했을 때 발생시킬 예외"""
    pass

# -------------------------------------------------------------
# 1. OpenAI API 키 설정 (환경 변수 또는 직접 입력)
# -------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"

# -------------------------------------------------------------
# 2. Prompt templates (all in English)
# -------------------------------------------------------------

# 2.1. Language TD Operator (G₂) prompt template
G2_PROMPT_TEMPLATE = """
You are an expert evaluator for the AppleGame environment.
Given the current state (as a grid of numbers), an action, and the resulting next state,
you need to produce a concise natural language evaluation (Language Value) explaining
why taking this action in this state is good or bad. Use the provided information of
the next state's evaluation to help.

=== Input ===
Current state (2D grid, {height}x{width}):

{grid_current}

Action: select rectangle with top-left corner ({x1},{y1}) and bottom-right corner ({x2},{y2}).

Immediate reward from this action: {immediate_reward}

Next state (2D grid):

{grid_next}

(Optional) If provided, evaluation of the next state (Value of next state):
"{next_state_eval}"

=== Task ===
Generate a natural language evaluation_text explaining:
1. Why this action is good or bad in the current state (pros and cons).
2. A brief outlook on the next state (implicit reasoning).
Output format (strict JSON with keys):
{{
  "Thought": "YOUR chain-of-thought reasoning here",
  "LVF_Estimate": "Final concise evaluation of (state,action) pair here"
}}
"""

# 2.2. Language Aggregator (G₁) prompt template
G1_PROMPT_TEMPLATE = """
You are an expert aggregator for the AppleGame environment.
You have {K} different evaluations (Language Value Function estimates) for
the same state-action pair, each based on a different simulated rollout variation.
Your job is to merge them into one final, coherent natural language evaluation,
providing a summary that captures the consensus or highlights any disagreements.

=== Input ===
List of {K} intermediate evaluations (each includes Thought and LVF_Estimate):
{list_of_evals_json}

=== Task ===
Generate a final language evaluation (LVF) that:
1. Summarizes the key reasons (pros and cons).
2. Mentions any differences across the variations if they exist.
Output format (strict JSON with keys):
{{
  "Thought": "YOUR chain-of-thought reasoning about differences here",
  "Final_LVF": "Final concise evaluation merging all variations here"
}}
"""

# 2.x. Batch evaluation prompt template
BATCH_EVAL_PROMPT_TEMPLATE = """
You are an expert evaluator for the AppleGame environment.
Given the current state (as a grid of numbers) and a list of feasible actions,
you need to produce, for each action, a concise natural language evaluation (Language Value) 
explaining why taking that action is good or bad in context of the other actions.

Each feasible action is defined by its top-left corner (x1, y1) and dimensions (width, height).
For each action, compare and contrast pros/cons not only in absolute terms but also 
relative to the other available actions in the list.

=== Input ===
Current state (2D grid, {height}x{width}):

{grid_current}

List of feasible actions (as JSON array; each element has keys "action_index", "x1", "y1", "width", "height"):
{actions_list_json}

=== Task ===
Generate a JSON array where each element corresponds to one action. 
Each element should have the following keys:
  - "action_index": <integer index of the action in the input list>
  - "Thought": "chain-of-thought reasoning for this action here"
  - "LVF_Estimate": "final concise natural-language evaluation for this (state,action) pair"

Output format (strict JSON, array of objects):
[
  {{
    "action_index": 0,
    "Thought": "...",
    "LVF_Estimate": "..."
  }},
  {{
    "action_index": 1,
    "Thought": "...",
    "LVF_Estimate": "..."
  }},
  ...
]
"""

# 2.3. Language Policy Improvement (I) prompt template
I_PROMPT_TEMPLATE = """
You are a decision-making agent for the AppleGame environment.
Given the current state (2D grid) and a list of candidate actions, each with its
Language Value Function evaluation (Final_LVF), select the best action to maximize
score. Provide chain-of-thought reasoning (why this action is best among candidates),
and then output the best action index (0-based) from the provided list.

=== Input ===
Current state (2D grid, {height}x{width}):

{grid_current}

Candidate actions with their evaluations:
{actions_and_lvf_list_json}

List of actions (index corresponds to each entry above):
{actions_list_json}

=== Task ===
1. Compare each candidate’s evaluation, highlighting pros and cons.
2. Decide which action is best.
3. Output in strict JSON:
{{
  "Thought": "YOUR reasoning comparing actions here",
  "Best_Action_Index": <integer index of best action>
}}
"""

# -------------------------------------------------------------
# 3. Helper 함수들 정의
# -------------------------------------------------------------

def serialize_grid(grid: np.ndarray) -> str:
    """
    2D NumPy 배열을 사람이 읽기 좋은 문자열로 직렬화.
    예: [ [1,2,3], [4,5,6] ] → "1 2 3\n4 5 6"
    """
    return "\n".join(" ".join(str(int(v)) for v in row) for row in grid.tolist())

def call_openai(
    client: openai.OpenAI,
    prompt: str,
    model: str = "gpt-4o-mini-2024-07-18",
    temperature: float = 0.7,
    max_tokens: int = 512
) -> dict:
    """
    OpenAI ChatCompletion을 호출하여, JSON 형태의 응답을 파싱하여 반환.
    비용 계산: prompt_tokens, completion_tokens를 이용해 누적 비용 업데이트.
    """
    global cum_cost

    # 실제 API 호출
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a helpful assistant for evaluating AppleGame states."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    # Usage 정보 읽어오기 (속성 접근으로 수정)
    usage = response.usage
    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens

    # 해당 호출의 비용 계산
    cost_this_call = (input_tokens / 1_000_000) * INPUT_COST_PER_M + (output_tokens / 1_000_000) * OUTPUT_COST_PER_M
    cum_cost += cost_this_call
    logger.info(f"API call cost: ${cost_this_call:.4f} USD (Input: {input_tokens}, Output: {output_tokens}) | Cumulative: ${cum_cost:.4f} USD)")

    # 누적 비용이 한도를 초과하면 예외 발생
    if cum_cost > COST_LIMIT:
        raise CostLimitExceeded(f"Cost limit exceeded: cum_cost={cum_cost:.4f} USD, limit={COST_LIMIT} USD")

    # 응답 텍스트를 JSON 파싱 시도
    content = response.output[0].content[0].text.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"Thought": "Parsing Error: " + str(e), "RawResponse": content}

# -------------------------------------------------------------
# 4. Language GPI by Prompting 클래스 정의
# -------------------------------------------------------------
class LanguageGPIAgent:
    def __init__(
        self,
        env: AppleGameEnv,
        lookahead_steps: int = 1,
        num_variations: int = 2,
        model_name: str = "gpt-4.1-nano",
        batch_size: int = 25     # ★ 한 번에 요청할 action 개수
    ):
        """
        env: AppleGameEnv 인스턴스
        lookahead_steps, num_variations: 기존 용도
        model_name: 사용할 OpenAI 모델
        batch_size: 한 번에 LLM에 보낼 최대 action 개수
        """
        self.env = env
        self.N = lookahead_steps
        self.K = num_variations
        self.model = model_name
        self.client = openai.OpenAI()
        self.batch_size = batch_size

    def estimate_action_value(
        self,
        state: np.ndarray,
        feasible_actions: List[Tuple[int,int,int,int]]
    ) -> List[dict]:
        """
        현재 state에서 feasible_actions 전체를 “batch_size 개씩” 잘라서
        여러 번 LLM 호출 → 각각 (state, action)에 대한 평가(Thought, LVF)를 받아서 합쳐 반환.
        반환: [{"action_index": idx, "LVF": str, "Thought": str}, ...]
        """
        height, width = self.env.height, self.env.width
        serialized_state = serialize_grid(state)

        logger.info(f"[estimate_action_value] Evaluating state:\n{serialized_state}")
        logger.info(f"[estimate_action_value] Number of feasible actions: {len(feasible_actions)}")

        # 내부 헬퍼: actions_subset에 대해 LLM 호출 후 결과 파싱
        def _call_batch_eval(actions_subset: List[Tuple[int,int,int,int]], idx_offset: int) -> List[dict]:
            actions_list = []
            for local_idx, (x1, y1, rect_w, rect_h) in enumerate(actions_subset):
                actions_list.append({
                    "action_index": idx_offset + local_idx,
                    "x1": x1,
                    "y1": y1,
                    "width": rect_w,
                    "height": rect_h
                })
            actions_list_json = json.dumps(actions_list, ensure_ascii=False, indent=2)

            prompt = BATCH_EVAL_PROMPT_TEMPLATE.format(
                height=height,
                width=width,
                grid_current=serialized_state,
                actions_list_json=actions_list_json
            )

            logger.info(f"[estimate_action_value] Sending batch evaluation for actions {idx_offset} ~ {idx_offset + len(actions_subset)-1}")
            logger.info(f"[estimate_action_value] #Actions in batch: {len(actions_subset)}")

            batch_response = call_openai(
                self.client,
                prompt=prompt,
                model=self.model,
                temperature=0.7,
                max_tokens=1024
            )

            # 파싱
            evals = []
            try:
                parsed = batch_response
                if isinstance(parsed, list):
                    for elem in parsed:
                        idx_act = elem.get("action_index")
                        thought = elem.get("Thought", "")
                        lvf = elem.get("LVF_Estimate", "")
                        if idx_act is not None:
                            evals.append({
                                "action_index": idx_act,
                                "Thought": thought,
                                "LVF": lvf
                            })
                            logger.info(f"[evaluate] action_index={idx_act} | Thought: {thought} | LVF: {lvf}")
                else:
                    logger.warning(f"[estimate_action_value] Expected list, but got: {parsed}")
                return evals
            except Exception as e:
                logger.error(f"[Parsing Error in batch_response] {e}, raw response: {batch_response}")
                return []

        total_evaluations = []
        n_actions = len(feasible_actions)
        # batch_size만큼씩 슬라이싱해서 반복 호출
        for start in range(0, n_actions, self.batch_size):
            end = min(start + self.batch_size, n_actions)
            chunk = feasible_actions[start:end]
            evals_chunk = _call_batch_eval(chunk, idx_offset=start)
            total_evaluations.extend(evals_chunk)

        # action_index 순으로 정렬
        total_evaluations.sort(key=lambda x: x["action_index"])
        return total_evaluations

    def select_best_action(
        self,
        state: np.ndarray,
        feasible_actions: List[Tuple[int,int,int,int]],
        action_value_list: List[dict]
    ) -> int:
        """
        Language Policy Improvement (I): 후보 행동 + 그 평가를 LLM에 보내
        chain-of-thought reasoning으로 최적 행동 index를 결정.
        """
        height, width = self.env.height, self.env.width
        serialized_state = serialize_grid(state)

        # 행동 목록 JSON 준비
        actions_list = []
        for (x1, y1, rect_w, rect_h) in feasible_actions:
            actions_list.append({
                "x1": x1, "y1": y1, "width": rect_w, "height": rect_h
            })
        actions_list_json = json.dumps(actions_list, ensure_ascii=False, indent=2)

        # 각 action별 LVF 리스트 준비
        actions_and_lvf = []
        for ev in action_value_list:
            idx = ev["action_index"]
            actions_and_lvf.append({
                "action_index": idx,
                "action": actions_list[idx],
                "LVF": ev["LVF"]
            })
        actions_and_lvf_list_json = json.dumps(
            actions_and_lvf, ensure_ascii=False, indent=2
        )

        prompt_I = I_PROMPT_TEMPLATE.format(
            height=height,
            width=width,
            grid_current=serialized_state,
            actions_and_lvf_list_json=actions_and_lvf_list_json,
            actions_list_json=actions_list_json
        )

        logger.info(f"[select_best_action] Sending policy improvement prompt for state:\n{serialized_state}")
        # logger.info(f"[select_best_action] Actions & LVF JSON:\n{actions_and_lvf_list_json}")

        i_resp = call_openai(
            self.client,
            prompt=prompt_I,
            model=self.model,
            temperature=0.7,
            max_tokens=256
        )

        # LLM이 반환한 chain‐of‐thought와 최종 선택을 로깅
        reasoning = i_resp.get("Thought", "")
        chosen_idx = i_resp.get("Best_Action_Index", 0)
        logger.info(f"[select_best_action] Chain-of-thought: {reasoning}")
        logger.info(f"[select_best_action] Best_Action_Index (0-based): {chosen_idx}")

        # 유효 범위 벗어나면 0으로 초기화
        if not isinstance(chosen_idx, int) or chosen_idx < 0 or chosen_idx >= len(feasible_actions):
            logger.warning(f"[select_best_action] Received invalid index ({chosen_idx}), default to 0.")
            chosen_idx = 0

        return chosen_idx


    def run_episode(self, max_steps: int = 1000, render: bool = True):
        """
        에피소드 단위로 LLM-based 의사결정 → 환경 실행을 반복.
        이제 스텝 단위로 tqdm 진행바를 내부에서 보여줌.
        """
        obs, info = self.env.reset()
        state = obs.copy()
        feasible_actions = info["feasible_actions"]
        done = False
        total_reward = 0
        total_score = 0

        # 스텝 카운트는 진행바의 total로 사용
        pbar = tqdm(range(max_steps), desc="Episode Steps", leave=False)
        for step_count in pbar:
            feasible_actions = self.env.game.get_feasible_actions()
            num_actions = len(feasible_actions)
            if not feasible_actions:
                break

            # 1) 행동 값 추정
            # print('Estimating action values...')
            action_value_list = self.estimate_action_value(state, feasible_actions)

            pbar.set_postfix(
                {
                    "Score": total_score,
                    "NumFeasibleActions": num_actions,
                    "CostUSD": f"{cum_cost:.4f}",
                }
            )

            # 2) 최적 행동 선택
            # print('Selecting best action...')
            best_action_idx = self.select_best_action(state, feasible_actions, action_value_list)

            pbar.set_postfix(
                {
                    "Score": total_score,
                    "NumFeasibleActions": num_actions,
                    "CostUSD": f"{cum_cost:.4f}",
                }
            )

            # 3) 환경에 행동 전달
            obs, reward, done, _, info = self.env.step(best_action_idx)
            next_state = obs.copy()
            total_reward += reward
            total_score += info.get("score", 0)

            pbar.set_postfix(
                {
                    "Score": total_score,
                    "NumFeasibleActions": num_actions,
                    "CostUSD": f"{cum_cost:.4f}",
                }
            )

            # 4) 렌더링 및 이벤트 처리
            if render:
                self.env.render()
                pygame.event.pump()

            state = next_state.copy()

            # 5) 에피소드 종료 조건
            if done:
                break

        return step_count + 1, total_reward, total_score  # step_count는 0-based이므로 +1

# -------------------------------------------------------------
# 5. 실제 실행 예시 (메인 스크립트에 TensorBoard 적용 및 비용 검사)
# -------------------------------------------------------------
if __name__ == "__main__":
    # AppleGameEnv 생성 (예: 17x10 그리드)
    env = AppleGameEnv(width=17, height=10, render_mode="human")
    agent = LanguageGPIAgent(env, lookahead_steps=1, num_variations=1, model_name="gpt-4.1-nano", batch_size=10)

    # -------------------------------------------------------------
    # 0. 비용 관련 상수 및 전역 누적 변수 정의
    # -------------------------------------------------------------
    # 예시 단가 (USD 기준)
    INPUT_COST_PER_M = 0.1    # 입력 토큰 1M당 0.1달러
    OUTPUT_COST_PER_M = 0.025 # 출력 토큰 1M당 0.025달러

    # 학습 전체에 허용할 비용 한도 (USD)
    COST_LIMIT = 1  # 예: 총 1달러를 넘으면 중단

    ### Hyperparameters ###
    num_episodes = 5              # 에피소드 수
    max_steps_per_episode = 200   # 각 에피소드 최대 스텝 수
    render_mode = 'human'           # 렌더링 여부
    random.seed(42)
    np.random.seed(42)
    #######################

    # -------------------------------------------------------------
    # TensorBoard SummaryWriter 생성 (타임스탬프 기반 서브디렉토리)
    # -------------------------------------------------------------
    base_log_dir = os.path.join("runs", "LanguageGPI_AppleGame")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    #  로그 설정 (Logger Configuration)
    # -------------------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 1) 파일 핸들러 (language_gpi.log)에 기록
    file_handler = logging.FileHandler(f"{log_dir}/language_gpi.log", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2) 콘솔에도 기록
    # console_handler = logging.StreamHandler()
    # console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # console_handler.setFormatter(console_formatter)
    # logger.addHandler(console_handler)

    logger.info(f"Starting {num_episodes} episodes of Language GPI Agent with cost limit {COST_LIMIT} USD...")
    err_cnt = 0

    for episode in range(1, num_episodes + 1):
        try:
            logger.info(f"\n=== Episode {episode} ===")
            # run_episode 내부에서 스텝 단위 tqdm을 보여줌
            step_cnt, tot_reward, tot_score = agent.run_episode(
                max_steps=max_steps_per_episode,
                render=(True if render_mode == 'human' else False)
            )

            logger.info(f"Episode {episode} finished: Steps={step_cnt}, Score={tot_score}, CumulativeCost=${cum_cost:.4f}")

            # TensorBoard에 기록
            writer.add_scalar("Episode/Steps", step_cnt, episode)
            writer.add_scalar("Episode/Reward", tot_reward, episode)
            writer.add_scalar("Episode/Score", tot_score, episode)
            writer.add_scalar("Episode/CumCost_USD", cum_cost, episode)

        except CostLimitExceeded as cle:
            logger.error(f"Cost limit exceeded at episode {episode}. Message: {str(cle)}")
            break

    logger.info(f"\nTraining completed. Final cumulative cost: ${cum_cost:.4f} USD.")
    writer.close()
    env.close()
