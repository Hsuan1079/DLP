# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import resource
import sys

gym.register_envs(ale_py)

def memory_limit(fraction=0.5):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    limit = int(total_memory * fraction)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, input,num_actions,num_layers=2,hidden_dim=64):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########
        if len(input) == 1: # # For CartPole
            print("Using MLP")
            layers = []
            layers.append(nn.Linear(input[0], hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, num_actions))
            self.network = nn.Sequential(*layers)
        elif len(input) == 3: # For Pong
            print("Using CNN")
            self.network = nn.Sequential(
                nn.Conv2d(input[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        assert x.ndim in [2, 4], f"Unexpected input shape: {x.shape}"
        if x.ndim == 4:
            return self.network(x/255.0)
        else:
            return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self,data,p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6  # 使用與原始版本相同的 alpha 值
    beta = 0.4  # 使用與原始版本相同的 beta 初始值
    beta_increment_per_sampling = 0.0005

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = self.e

    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (abs(error) + self.e) ** self.a

    def add(self, sample,error):
        p = self._get_priority(error)
        self.tree.add(sample,p)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None): # CartPole-v1,"ALE/Pong-v5"
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        print("Number of actions:", self.num_actions)
        # self.preprocessor = AtariPreprocessor()
        if self.env_name.startswith("ALE/"):
            self.use_preprocessor = True
            self.preprocessor = AtariPreprocessor()
            input_shape = (4, 84, 84)
        else:
            self.use_preprocessor = False
            self.preprocessor = None
            input_shape = self.env.observation_space.shape

        #########
        self.memory = PrioritizedReplayBuffer(args.memory_size)
        #########

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        
        self.q_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net = DQN(input_shape, self.num_actions).to(self.device)

        # self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        # self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        self.n_step = 3  # 可改成 args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        # state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
        state_tensor = torch.tensor(np.array(state, dtype=np.float32), device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=10000):
        milestones = [200_000, 400_000, 600_000, 800_000, 1_000_000]
        saved = set()  # 記錄已經存過的里程碑
        milestone_best_reward = {m: -float("inf") for m in milestones}
        milestone_best_model = {}
        for ep in range(episodes):
            obs, _ = self.env.reset()
            
            if self.use_preprocessor:
                state = self.preprocessor.reset(obs)
            else:
                state = obs

            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # if done:
                #     break
                
                if self.use_preprocessor:
                    next_state = self.preprocessor.step(next_obs)
                else:
                    next_state = next_obs
                self.n_step_buffer.append((state, action, reward, next_state, done))

                if len(self.n_step_buffer) == self.n_step:
                    # 1. 累積 n-step reward
                    R = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(self.n_step)])
                    s0, a0, _, _, _ = self.n_step_buffer[0]
                    _, _, _, s_n, done_n = self.n_step_buffer[-1]

                    # 2. 若使用 PER，計算 TD error 作為 priority
                    with torch.no_grad():
                        state_tensor = torch.tensor(np.array(s0), dtype=torch.uint8, device=self.device).unsqueeze(0)
                        next_state_tensor = torch.tensor(np.array(s_n), dtype=torch.uint8, device=self.device).unsqueeze(0)
                        q_val = self.q_net(state_tensor)[0, a0].item()
                        next_q_val = self.target_net(next_state_tensor).max(1)[0].item()

                    # 3. 計算 n-step TD error
                    td_error = abs(R + (1 - done_n) * (self.gamma ** self.n_step) * next_q_val - q_val)

                    # 4. 加入 PER replay buffer
                    self.memory.add((s0, a0, R, s_n, done_n), td_error)
                    
                for _ in range(self.train_per_step):
                    self.train()
                if done:
                    break

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                for m in milestones:
                    if self.env_count >= m and m not in saved:
                        saved.add(m)
                        # 存當前權重
                        path_cur = os.path.join(self.save_dir, f"model_{m//1000}k.pt")
                        torch.save(self.q_net.state_dict(), path_cur)
                        print(f"[AutoSave] step {self.env_count} ≥ {m}: saved current model → {path_cur}")
                        # 存里程碑之前的最佳模型
                        if m in milestone_best_model:
                            path_best = os.path.join(self.save_dir, f"best_before_{m//1000}k.pt")
                            torch.save(milestone_best_model[m], path_best)
                            print(f"✅ best-before-{m} saved → {path_best}")


                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Epsiode Reward": total_reward,
                        "Episode Length": step_count
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
            
                    ########## END OF YOUR CODE ##########   
                if self.env_count % 200000 == 0:
                    model_path = os.path.join(self.save_dir, f"model_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

                if eval_reward >= 19:
                    # save the model
                    model_path = os.path.join(self.save_dir, f"solved_model_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved solved model to {model_path} with reward {eval_reward}")
                    
            self.n_step_buffer.clear()

    def evaluate(self):
        obs, _ = self.test_env.reset()
        if self.use_preprocessor:
            state = self.preprocessor.reset(obs)
        else:
            state = obs

        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.tensor(np.array(state, dtype=np.float32), device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            if self.use_preprocessor:
                state = self.preprocessor.step(next_obs)
            else:
                state = next_obs

        return total_reward


    def train(self):
       
        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        # states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        # next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        state_tensors = [torch.as_tensor(s, device=self.device, dtype=torch.uint8) for s in states]              
        states = torch.stack(state_tensors, dim=0).float() 

        next_tensors  = [torch.as_tensor(ns, device=self.device, dtype=torch.uint8) for ns in next_states]
        next_states = torch.stack(next_tensors,  dim=0).float() 

        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # target Q-values for the next states
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            # print(f"Target Q avg: {next_q_values.mean().item():.3f}")
        
        # Compute the loss
        # loss = nn.MSELoss()(q_values, target_q_values)
        loss_fn = nn.SmoothL1Loss(reduction='none')  # 逐一元素
        losses = loss_fn(q_values, target_q_values)

        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        loss = (losses * weights.to(self.device)).mean()

        # 在 train() 最後加上更新 TD error 的程式
        td_errors = torch.abs(target_q_values - q_values).detach().cpu().numpy()
        for i, idx in enumerate(indices):
            self.memory.update(idx, td_errors[i])
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Train Loss": loss.item(),
                "Q Mean": q_values.mean().item(),
                "Q Std": q_values.std().item()
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name (e.g., CartPole-v1 or ALE/Pong-v5)")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=100000)
    # parser.add_argument("--lr", type=float, default=0.0001)
    # parser.add_argument("--discount-factor", type=float, default=0.99)
    # parser.add_argument("--epsilon-start", type=float, default=1.0)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    # parser.add_argument("--epsilon-min", type=float, default=0.05)
    # parser.add_argument("--target-update-frequency", type=int, default=1000)
    # parser.add_argument("--replay-start-size", type=int, default=50000)
    # parser.add_argument("--max-episode-steps", type=int, default=10000)
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--n-step", type=int, default=3)
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=50000)
    # parser.add_argument("--lr", type=float, default=0.0001)
    # parser.add_argument("--discount-factor", type=float, default=0.95)
    # parser.add_argument("--epsilon-start", type=float, default=0.5)
    # parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    # parser.add_argument("--epsilon-min", type=float, default=0.05)
    # parser.add_argument("--target-update-frequency", type=int, default=500)
    # parser.add_argument("--replay-start-size", type=int, default=20000)
    # parser.add_argument("--max-episode-steps", type=int, default=10000)
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)     
    parser.add_argument("--memory-size", type=int, default=400000)     
    parser.add_argument("--lr", type=float, default=0.0001)     
    parser.add_argument("--discount-factor", type=float, default=0.99)     
    parser.add_argument("--epsilon-start", type=float, default=0.5)     
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)     
    parser.add_argument("--epsilon-min", type=float, default=0.05)     
    parser.add_argument("--target-update-frequency", type=int, default=1000)     
    parser.add_argument("--replay-start-size", type=int, default=20000)     
    parser.add_argument("--max-episode-steps", type=int, default=10000)     
    parser.add_argument("--train-per-step", type=int, default=1)     
    parser.add_argument("--n-step", type=int, default=5)
    # parser.add_argument("--batch-size", type=int, default=32)     
    # parser.add_argument("--memory-size", type=int, default=100000)     
    # parser.add_argument("--lr", type=float, default=0.0001)     
    # parser.add_argument("--discount-factor", type=float, default=0.99)     
    # parser.add_argument("--epsilon-start", type=float, default=0.5)     
    # parser.add_argument("--epsilon-decay", type=float, default=0.9995)     
    # parser.add_argument("--epsilon-min", type=float, default=0.05)     
    # parser.add_argument("--target-update-frequency", type=int, default=500)     
    # parser.add_argument("--replay-start-size", type=int, default=20000)     
    # parser.add_argument("--max-episode-steps", type=int, default=10000)     
    # parser.add_argument("--train-per-step", type=int, default=1)     
    # parser.add_argument("--n-step", type=int, default=5)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    memory_limit(0.8)  # ⚠️ 加入這行限制記憶體使用為總記憶體的 50%
    try:
        agent = DQNAgent(args=args)
        agent.run()
    except MemoryError:
        sys.stderr.write('❌ MAXIMUM MEMORY EXCEEDED\n')
        sys.exit(-1)



# task3_add_all
# parser.add_argument("--batch-size", type=int, default=32)
# parser.add_argument("--memory-size", type=int, default=50000)
# parser.add_argument("--lr", type=float, default=0.0001)
# parser.add_argument("--discount-factor", type=float, default=0.95)
# parser.add_argument("--epsilon-start", type=float, default=0.5)
# parser.add_argument("--epsilon-decay", type=float, default=0.9995)
# parser.add_argument("--epsilon-min", type=float, default=0.05)
# parser.add_argument("--target-update-frequency", type=int, default=1000)
# parser.add_argument("--replay-start-size", type=int, default=20000)
# parser.add_argument("--max-episode-steps", type=int, default=10000)
# parser.add_argument("--train-per-step", type=int, default=1)
# parser.add_argument("--n-step", type=int, default=3)

# task3_add_all_100K
#  parser.add_argument("--batch-size", type=int, default=32)
#     parser.add_argument("--memory-size", type=int, default=100000)
#     parser.add_argument("--lr", type=float, default=0.0001)
#     parser.add_argument("--discount-factor", type=float, default=0.95)
#     parser.add_argument("--epsilon-start", type=float, default=0.5)
#     parser.add_argument("--epsilon-decay", type=float, default=0.9995)
#     parser.add_argument("--epsilon-min", type=float, default=0.05)
#     parser.add_argument("--target-update-frequency", type=int, default=1000)
#     parser.add_argument("--replay-start-size", type=int, default=20000)
#     parser.add_argument("--max-episode-steps", type=int, default=10000)
#     parser.add_argument("--train-per-step", type=int, default=1)
#     parser.add_argument("--n-step", type=int, default=3)

# task3_add_all_100K
    # parser.add_argument("--memory-size", type=int, default=100000)
    # parser.add_argument("--lr", type=float, default=0.0001)
    # parser.add_argument("--discount-factor", type=float, default=0.99)
    # parser.add_argument("--epsilon-start", type=float, default=0.5)
    # parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    # parser.add_argument("--epsilon-min", type=float, default=0.05)
    # parser.add_argument("--target-update-frequency", type=int, default=1000)
    # parser.add_argument("--replay-start-size", type=int, default=20000)
    # parser.add_argument("--max-episode-steps", type=int, default=10000)
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--n-step", type=int, default=3)

# task3_add_all_100K_2
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, default="CartPole-v1", help="Environment name (e.g., CartPole-v1 or ALE/Pong-v5)")
    # parser.add_argument("--save-dir", type=str, default="./results")
    # parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--memory-size", type=int, default=100000)
    # parser.add_argument("--lr", type=float, default=0.0001)
    # parser.add_argument("--discount-factor", type=float, default=0.99)
    # parser.add_argument("--epsilon-start", type=float, default=0.5)
    # parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    # parser.add_argument("--epsilon-min", type=float, default=0.05)
    # parser.add_argument("--target-update-frequency", type=int, default=1000)
    # parser.add_argument("--replay-start-size", type=int, default=20000)
    # parser.add_argument("--max-episode-steps", type=int, default=10000)
    # parser.add_argument("--train-per-step", type=int, default=1)
    # parser.add_argument("--n-step", type=int, default=3)

        # parser.add_argument("--batch-size", type=int, default=32)     
        # parser.add_argument("--memory-size", type=int, default=100000)     
        # parser.add_argument("--lr", type=float, default=0.0001)     
        # parser.add_argument("--discount-factor", type=float, default=0.99)     
        # parser.add_argument("--epsilon-start", type=float, default=0.5)     
        # parser.add_argument("--epsilon-decay", type=float, default=0.9995)     
        # parser.add_argument("--epsilon-min", type=float, default=0.05)     
        # parser.add_argument("--target-update-frequency", type=int, default=1000)     
        # parser.add_argument("--replay-start-size", type=int, default=20000)     
        # parser.add_argument("--max-episode-steps", type=int, default=10000)     
        # parser.add_argument("--train-per-step", type=int, default=1)     
        # parser.add_argument("--n-step", type=int, default=5)
    
# V11
#    parser.add_argument("--batch-size", type=int, default=32)     
    # parser.add_argument("--memory-size", type=int, default=100000)     
    # parser.add_argument("--lr", type=float, default=0.0001)     
    # parser.add_argument("--discount-factor", type=float, default=0.99)     
    # parser.add_argument("--epsilon-start", type=float, default=0.5)     
    # parser.add_argument("--epsilon-decay", type=float, default=0.9995)     
    # parser.add_argument("--epsilon-min", type=float, default=0.05)     
    # parser.add_argument("--target-update-frequency", type=int, default=1000)     
    # parser.add_argument("--replay-start-size", type=int, default=20000)     
    # parser.add_argument("--max-episode-steps", type=int, default=10000)     
    # parser.add_argument("--train-per-step", type=int, default=1)     
    # parser.add_argument("--n-step", type=int, default=5)