import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import imageio
import os
import argparse

class MLPDQN(nn.Module):
    def __init__(self, input_dim=4, num_actions=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def evaluate_cartpole(model_path, output_dir="./cartpole_videos", episodes=3, seed=21):
    total_reward_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(seed)

    model = MLPDQN(input_dim=4, num_actions=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state = obs
        done = False
        total_reward = 0
        frames = []

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = obs
            total_reward += reward
        
        total_reward_list.append(total_reward)

        out_path = os.path.join(output_dir, f"cartpole_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30) as video:
            for f in frames:
                video.append_data(f)
        print(f"✅ Saved CartPole episode {ep} (Reward: {total_reward}) → {out_path}")

    average_reward = np.mean(total_reward_list)
    print(f"Average reward over {args.episodes} episodes: {average_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to CartPole model .pt file")
    parser.add_argument("--output-dir", type=str, default="./cartpole_videos")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=313551076)
    args = parser.parse_args()

    evaluate_cartpole(
        model_path=args.model_path,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed
    )
    