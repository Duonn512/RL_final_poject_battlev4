from magent2.environments import battle_v4
import torch
import torch.nn as nn
import os
import cv2

from biggerDQN import DQN
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 24

    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)

    final_q_network = FinalQNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    final_q_network.load_state_dict(
        torch.load("red_final.pt", weights_only=True, map_location="cpu")
    )
    final_q_network.to(device)

    q_network_blue = DQN(
        env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    q_network_blue.load_state_dict(
        torch.load("blue.pt", weights_only=True, map_location="cpu")
    )
    q_network_blue.to(device)

    def blue_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network_blue(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]
    
    def pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]

    def final_pretrain_policy(env, agent, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = final_q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]



    def write_video(red_policy, blue_policy, video_name):
        frames = []
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            

            if termination or truncation:
                action = None  # this agent has died
            else:
                agent_team = agent.split("_")[0]
                if agent_team == "red":
                    action = red_policy(env, agent, observation)
                else:
                    action = blue_policy(env, agent, observation)

            env.step(action)

            if agent == "red_0":
                frames.append(env.render())

        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            os.path.join(vid_dir, f"{video_name}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"Done recording blue agents vs {video_name} agents")

    write_video(random_policy, blue_policy, "random")
    write_video(pretrain_policy, blue_policy, "pretrain")
    write_video(final_pretrain_policy, blue_policy, "final_pretrain")
    env.close()
