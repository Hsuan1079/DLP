#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple
import os

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Initialize layers
        self.fc1 = init_layer_uniform(nn.Linear(in_dim, 64))
        self.fc2 = init_layer_uniform(nn.Linear(64, 64))
        self.mean_layer = init_layer_uniform(nn.Linear(64, out_dim), init_w=1e-3)
        self.log_std = nn.Parameter(torch.zeros(out_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        
        # Clamp log_std to prevent too small or large values
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        
        # Create normal distribution
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterization trick
        
        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        # Initialize layers
        self.fc1 = init_layer_uniform(nn.Linear(in_dim, 64))
        self.fc2 = init_layer_uniform(nn.Linear(64, 64))
        self.value_layer = init_layer_uniform(nn.Linear(64, 1), init_w=1e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.value_layer(x)
        
        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
        
    return returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
        save_file (str): path to save model checkpoints
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.save_file = args.save_file
        
        # device: cpu / gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False
        
        # for checkpoint
        self.best_score = -np.inf
        os.makedirs(self.save_file, exist_ok=True)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)  

        unclamped_action = action
        clamped_action = unclamped_action.clamp(-2.0, 2.0)  

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(unclamped_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(unclamped_action))

        return clamped_action.cpu().detach().numpy() 

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # Detach tensors to prevent gradient computation
            state = state.detach()
            action = action.detach()
            return_ = return_.detach()
            adv = adv.detach()
            
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            # entropy = dist.entropy().sum(-1)
            # actor_loss = -torch.min(loss, clipped_loss).mean() - self.entropy_weight * entropy.mean()
            loss = ratio * adv
            clipped_loss = torch.clamp(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv
            actor_loss = -torch.min(loss, clipped_loss).mean() - self.entropy_weight * dist.entropy().mean()
            
            # critic_loss
            cur_value = self.critic(state)
            # Ensure both tensors have the same shape
            # cur_value = cur_value.squeeze()
            # return_ = return_.squeeze()
            # critic_loss = F.mse_loss(cur_value,return_)
            critic_loss = F.mse_loss(return_,cur_value)
            
            # Update networks separately
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            # clip gradient
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def save_best_checkpoint(self, score: float):
        """Save model checkpoint if score is better than best_score."""
        if score > self.best_score:
            self.best_score = score
            # Remove old checkpoint if exists
            if os.path.exists(os.path.join(self.save_file, 'best.pt')):
                os.remove(os.path.join(self.save_file, 'best.pt'))
            
            # Save new checkpoint
            checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'best_score': self.best_score,
                'total_step': self.total_step
            }
            torch.save(checkpoint, os.path.join(self.save_file, 'best.pt'))
            print(f"New best score: {score:.2f}, checkpoint saved!")

    def save_periodic_checkpoint(self):
        """Save periodic checkpoint every 1000 steps."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'best_score': self.best_score,
            'total_step': self.total_step
        }
        checkpoint_name = f'checkpoint_step_{self.total_step}.pt'
        torch.save(checkpoint, os.path.join(self.save_file, checkpoint_name))
        print(f"Periodic checkpoint saved at step {self.total_step}")

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0

        
        for ep in tqdm(range(1, self.num_episodes)):
            if self.total_step == 200000:
                break
            score = 0
            print("\n")
            
            # Collect experience
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # Save periodic checkpoint every 1000 steps
                if self.total_step % 10000 == 0:
                    self.entropy_weight = max(0.001, self.entropy_weight * 0.95)
                    self.save_periodic_checkpoint()
                    avg_score = self.test_multiple(num_tests=20)
                    wandb.log({
                        "total_step": self.total_step,
                        "test_avg_score": avg_score,
                    })
                    self.save_best_checkpoint(avg_score)

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed+episode_count)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    # Log to wandb
                    wandb.log({
                        "total_step": self.total_step,
                        "episode": episode_count,
                        "return": score,
                        "best_score": self.best_score
                    })
                    score = 0

            # Update model after collecting experience
            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            # Log losses to wandb
            wandb.log({
                "total_step": self.total_step,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            })

        # termination
        self.env.close()

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            state = next_state
            score += reward[0][0]  # Fix reward handling

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

    def load_checkpoint(self, path):
        """Load the best checkpoint."""
        checkpoint_path = os.path.join(path)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}")
            return True
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            return False

    def test_multiple(self, num_tests: int = 20):
        """Test the agent multiple times with different random seeds."""
        # Store original test mode
        original_test_mode = self.is_test
        self.is_test = True
        test_scores = []
        
        try:
            for i in range(1000):
                if num_tests == 0:
                    break
                # Use different seed for each test
                test_seed = self.seed + i + 1
                state, _ = self.env.reset(seed=test_seed)
                done = False
                score = 0
                
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)
                    state = next_state
                    score += reward[0][0]
                
                if score < -300:
                    continue
                else:
                    test_scores.append(score)
                    num_tests -= 1
                    print(f"Test {20-num_tests}/{20}, Score: {score:.2f}")
                
            avg_score = np.mean(test_scores)
            print(f"\nAverage Score: {avg_score:.2f}")
            
            return avg_score
        finally:
            # Restore original test mode
            self.is_test = original_test_mode

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=int, default=64)
    parser.add_argument("--save-file", type=str, default="ppo")
    parser.add_argument("--checkpoint", type=str, default="ppo/best.pt")
    parser.add_argument("--test-only", action="store_true", help="Run in test mode only")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    
    agent = PPOAgent(env, args)
    
    if args.test_only:
        if agent.load_checkpoint(args.checkpoint):
            agent.test_multiple(num_tests=20)
        else:
            print("No checkpoint found! Please train the model first.")
    else:
        wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True)
        agent.train()