import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from logger import Logger


# ================== Prioritized Experience Replay ================== #
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        # 🚨 (수정 1) pos 변수 추가 (버퍼가 꽉 찼을 때 덮어쓰기 위함)
        self.pos = 0

    # 🚨 (수정 2) add 함수의 시그니처 변경 (td_error 인수 제거)
    def add(self, transition):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        # 새 경험은 항상 최대 우선순위로 설정
        self.priorities[self.pos] = max_priority
        # 위치 포인터 업데이트
        self.pos = (self.pos + 1) % self.capacity

    # 🚨 (수정 3) sample 함수 수정 (beta 인수 추가 및 weights 반환)
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        prob_sum = probabilities.sum()

        # probabilities가 0인 경우 방지
        if prob_sum == 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities /= prob_sum

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]

        # 중요도 샘플링(Importance Sampling) 가중치 계산
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 안정화를 위해 정규화
        weights = np.array(weights, dtype=np.float32)  # numpy 배열로 변환

        return samples, indices, weights

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            # 🚨 (수정 4) error가 1D 배열이 되도록 .squeeze() 사용 가정 (DQN.update에서 처리)
            self.priorities[i] = (abs(error) + 1e-5) ** self.alpha

    # (신규) main.py에서 max_priority를 직접 가져갈 필요는 없지만,
    # add 함수가 내부적으로 잘 처리하고 있습니다.
    def get_max_priority(self):
        return self.priorities.max() if self.buffer else 1.0


# ================== Noisy Linear ================== #
# ... (NoisyLinear 클래스 동일) ...
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                                        self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)


# ================== Dueling Network with Noisy Nets ================== #
# ... (DuelingDQN 클래스 동일) ...
class DuelingDQN(nn.Module):
    def __init__(self, num_state, num_action):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(num_state, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.advantage_layer = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, num_action)
        )
        self.value_layer = nn.Sequential(
            NoisyLinear(256, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )

    def forward(self, x):
        x = self.feature_layer(x)
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        for layer in self.advantage_layer:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.value_layer:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


# ================== Rainbow DQN ================== #
class RainbowDQN:
    def __init__(self, num_state, num_action, num_episodes, iteration_log):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_state = num_state
        self.num_action = num_action
        self.target_net, self.act_net = DuelingDQN(num_state, num_action).to(self.device), DuelingDQN(num_state,
                                                                                                      num_action).to(
            self.device)
        self.optimizer = optim.Adam(self.act_net.parameters(), lr=3e-4)

        # 🚨 (수정 5) Loss 함수의 reduction을 'none'으로 변경
        self.loss_func = nn.SmoothL1Loss(reduction='none')

        self.buffer = PrioritizedReplayBuffer(10000)
        self.gamma = 0.98
        self.epsilon_start = 0.5
        self.epsilon = self.epsilon_start
        self.epsilon_min = 0.01
        self.num_episodes = num_episodes
        self.n_step = 3  # Multi-step learning
        self.logger = Logger().get_logger()
        self.iteration_log = iteration_log

        # 🚨 (신규) Beta 스케줄링 변수
        self.beta_start = 0.4
        self.beta_frames = 10000  # (예시) 총 스텝 수에 맞춰 조절 필요
        self.beta_by_frame = lambda frame_idx: min(1.0, self.beta_start + frame_idx * (
                    1.0 - self.beta_start) / self.beta_frames)
        self.frame_idx = 0  # 총 스텝 카운터 (update 함수에서 증가)

    def select_action(self, state, current_episode, mask=None, greedy=False, return_qvals=False):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)

        # Eval 모드: NoisyNet 노이즈 비활성화 → 결정론적 Q값
        if greedy:
            self.act_net.eval()
            with torch.no_grad():
                value = self.act_net(state_tensor)
            self.act_net.train()
        else:
            self.act_net.reset_noise()
            value = self.act_net(state_tensor)

        # XAI: 마스킹 전 DRL 원본 Q값/선호 액션 저장
        raw_qvals = value.detach().cpu().numpy().flatten()
        drl_preferred_action = int(np.argmax(raw_qvals))
        drl_preferred_qval   = float(raw_qvals[drl_preferred_action])

        # L1 Action Masking
        masked_value = value.clone()
        if mask is not None:
            # 룰 충돌로 전체 action이 금지된 경우 → mask 무시하고 DRL Q값으로 선택
            if not mask.any():
                self.logger.warning(
                    "[L1 Safety] All actions masked (rule conflict)! "
                    "Ignoring mask and selecting best Q-value action."
                )
            else:
                mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
                masked_value.masked_fill_(~mask_tensor, -float('inf'))

        action = torch.argmax(masked_value).item()
        action_type = 'rl'

        # Greedy eval: epsilon=0, 탐색 없음
        if greedy:
            if self.iteration_log:
                self.logger.info(f'action : {action} type : greedy epsilon : 0.0000')
            if return_qvals:
                return action, 0.0, drl_preferred_action, drl_preferred_qval, raw_qvals
            return action, 0.0

        # Epsilon 계산 (훈련 에피소드 기준)
        transition_episode = int(self.num_episodes * (1 / 3))
        decay_rate = 5.0

        if current_episode < transition_episode:
            self.epsilon = self.epsilon_start
        else:
            progress = (current_episode - transition_episode) / (self.num_episodes - transition_episode)
            self.epsilon = max(self.epsilon_min, self.epsilon_start * np.exp(-decay_rate * progress))

        # Epsilon-greedy 탐색 (허용된 액션 내에서만)
        if np.random.rand() < self.epsilon:
            action_type = 'random'
            if mask is not None:
                allowed_actions = np.where(mask)[0]
                if len(allowed_actions) == 0:
                    # 룰 충돌 → mask 무시, 전체 action 중 랜덤 탐색
                    self.logger.warning(
                        "[L1 Safety] All actions masked (rule conflict)! "
                        "Ignoring mask and selecting random action."
                    )
                    action = np.random.choice(range(self.num_action))
                else:
                    action = np.random.choice(allowed_actions)
            else:
                action = np.random.choice(range(self.num_action))

        if self.iteration_log:
            self.logger.info(f'action : {action} type : {action_type} epsilon : {self.epsilon:.4f}')

        if return_qvals:
            return action, self.epsilon, drl_preferred_action, drl_preferred_qval, raw_qvals
        return action, self.epsilon

    def update(self, batch_size):
        # 🚨 (신규) Beta 값 계산
        beta = self.beta_by_frame(self.frame_idx)
        self.frame_idx += 1  # 스텝 카운터 증가

        # 🚨 (수정 6) sample 함수가 'weights'도 반환
        batch, indices, weights = self.buffer.sample(batch_size, beta)

        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)

        # (신규) weights 텐서 변환
        weights = torch.tensor(np.array(weights), dtype=torch.float).unsqueeze(1).to(self.device)

        # 🚨 (수정 7) Double DQN 로직으로 수정 (선택 사항이지만 권장)
        q_values = self.act_net(states).gather(1, actions)

        # 1. '현재' 네트워크(act_net)로 다음 행동(next_actions)을 선택
        next_actions = self.act_net(next_states).argmax(1).unsqueeze(1)
        # 2. '타겟' 네트워크(target_net)로 그 행동의 Q-value를 계산
        next_q_values = self.target_net(next_states).gather(1, next_actions).detach()

        target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values

        # 🚨 (수정 8) TD-Error 계산 및 우선순위 업데이트
        # (기존) errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        # (수정) 1D numpy 배열로 변환
        td_errors = q_values - target_q_values
        abs_errors = torch.abs(td_errors).detach().cpu().numpy().squeeze(1)  # [B, 1] -> [B]
        self.buffer.update_priorities(indices, abs_errors)

        # 🚨 (수정 9) Loss 계산에 'weights' 적용
        # (기존) loss = self.loss_func(q_values, target_q_values)
        # (수정)
        loss_per_sample = self.loss_func(q_values, target_q_values)  # [B, 1]
        loss = (weights * loss_per_sample).mean()  # 가중 평균

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.act_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.act_net.reset_noise()
        self.target_net.reset_noise()

        return loss.item()

    def save_checkpoint(self, path: str):
        """DQN 가중치 + 옵티마이저 상태 저장 (중간 크래시 복구용)."""
        torch.save({
            'act_net':    self.act_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'frame_idx':  self.frame_idx,
        }, path)

    def load_checkpoint(self, path: str):
        """저장된 체크포인트 복원."""
        ckpt = torch.load(path, map_location=self.device)
        self.act_net.load_state_dict(ckpt['act_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.frame_idx = ckpt.get('frame_idx', 0)
        self.logger.info(f"[DQN] Checkpoint loaded from {path}")

# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from logger import Logger
#
# # ================== Prioritized Experience Replay ================== #
# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.buffer = []
#         self.priorities = np.zeros((capacity,), dtype=np.float32)
#         self.alpha = alpha
#
#
#     def add(self, transition, td_error):
#         max_priority = max(self.priorities.max(), 1.0)
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(transition)
#         else:
#             self.buffer.pop(0)
#         self.priorities[len(self.buffer) - 1] = max_priority
#
#     def sample(self, batch_size, beta=0.4):
#         priorities = self.priorities[:len(self.buffer)]
#         probabilities = priorities ** self.alpha / np.sum(priorities ** self.alpha)
#         indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
#         samples = [self.buffer[i] for i in indices]
#         return samples, indices
#
#     def update_priorities(self, indices, errors):
#         for i, error in zip(indices, errors):
#             self.priorities[i] = (abs(error) + 1e-5) ** self.alpha
# # ================== Noisy Linear ================== #
# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, std_init=0.5):
#         super(NoisyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.std_init = std_init
#
#         self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
#         self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
#
#         self.bias_mu = nn.Parameter(torch.empty(out_features))
#         self.bias_sigma = nn.Parameter(torch.empty(out_features))
#         self.register_buffer("bias_epsilon", torch.empty(out_features))
#
#         self.reset_parameters()
#         self.reset_noise()
#
#     def reset_parameters(self):
#         mu_range = 1 / np.sqrt(self.in_features)
#         self.weight_mu.data.uniform_(-mu_range, mu_range)
#         self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
#         self.bias_mu.data.uniform_(-mu_range, mu_range)
#         self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
#
#     def reset_noise(self):
#         self.weight_epsilon.normal_()
#         self.bias_epsilon.normal_()
#
#     def forward(self, x):
#         if self.training:
#             return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
#                                         self.bias_mu + self.bias_sigma * self.bias_epsilon)
#         else:
#             return nn.functional.linear(x, self.weight_mu, self.bias_mu)
#
#
# # ================== Dueling Network with Noisy Nets ================== #
# class DuelingDQN(nn.Module):
#     def __init__(self, num_state, num_action):
#         super(DuelingDQN, self).__init__()
#         self.feature_layer = nn.Sequential(
#             nn.Linear(num_state, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU()
#         )
#         self.advantage_layer = nn.Sequential(
#             NoisyLinear(256, 128),
#             nn.ReLU(),
#             NoisyLinear(128, num_action)
#         )
#         self.value_layer = nn.Sequential(
#             NoisyLinear(256, 128),
#             nn.ReLU(),
#             NoisyLinear(128, 1)
#         )
#
#     def forward(self, x):
#         x = self.feature_layer(x)
#         value = self.value_layer(x)
#         advantage = self.advantage_layer(x)
#         q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
#         return q_values
#
#     def reset_noise(self):
#         for layer in self.advantage_layer:
#             if isinstance(layer, NoisyLinear):
#                 layer.reset_noise()
#         for layer in self.value_layer:
#             if isinstance(layer, NoisyLinear):
#                 layer.reset_noise()
#
#
# # ================== Rainbow DQN ================== #
# class RainbowDQN:
#     def __init__(self, num_state, num_action, num_episodes, iteration_log):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.num_state = num_state
#         self.num_action = num_action
#         self.target_net, self.act_net = DuelingDQN(num_state, num_action).to(self.device), DuelingDQN(num_state, num_action).to(self.device)
#         self.optimizer = optim.Adam(self.act_net.parameters(), lr=3e-4)
#         self.loss_func = nn.SmoothL1Loss()
#         self.buffer = PrioritizedReplayBuffer(10000)
#         self.gamma = 0.98
#         self.epsilon_start = 1.0
#         self.epsilon = self.epsilon_start
#         self.epsilon_min = 0.01
#         self.num_episodes = num_episodes
#         self.n_step = 3  # Multi-step learning
#         self.logger = Logger().get_logger()
#         self.iteration_log = iteration_log
#
#     def select_action(self, state, current_episode):
#         state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
#         self.act_net.reset_noise()
#         value = self.act_net(state)
#         action = torch.argmax(value).item()
#         action_type = 'rl'
#
#         transition_episode = int(self.num_episodes * (1/3))
#         decay_rate = 5.0
#
#         if current_episode < transition_episode:
#             self.epsilon = self.epsilon_start
#         else:
#             progress = (current_episode - transition_episode) / (self.num_episodes - transition_episode)
#             self.epsilon = max(self.epsilon_min, self.epsilon_start * np.exp(-decay_rate * progress))
#
#         if np.random.rand() < self.epsilon:
#             action = np.random.choice(range(self.num_action))
#             action_type = 'random'
#
#         # 탐색 여부 결정
#         if self.iteration_log:
#             self.logger.info(f'action : {action} type : {action_type} epsilon : {self.epsilon:.4f}')
#
#         return action, self.epsilon
#
#     def update(self, batch_size):
#         batch, indices = self.buffer.sample(batch_size)
#         states, actions, rewards, next_states = zip(*batch)
#         states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
#         actions = torch.tensor(np.array(actions), dtype=torch.long).unsqueeze(1).to(self.device)
#         rewards = torch.tensor(np.array(rewards), dtype=torch.float).unsqueeze(1).to(self.device)
#         next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
#
#         q_values = self.act_net(states).gather(1, actions)
#         next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
#         target_q_values = rewards + (self.gamma ** self.n_step) * next_q_values
#
#         errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
#         self.buffer.update_priorities(indices, errors)
#
#         loss = self.loss_func(q_values, target_q_values)
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.act_net.parameters(), max_norm=1.0)
#         self.optimizer.step()
#         self.act_net.reset_noise()
#         self.target_net.reset_noise()
#
#         return loss.item()
