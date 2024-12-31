import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cribbage_env.core.environment import CribbageEnv
from .dqn_agent import DQNAgent
from ..base.random_agent import RandomAgent


def train_episode(env: CribbageEnv, agent: DQNAgent,
                 opponent: RandomAgent) -> Tuple[float, float]:
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    agent_score = 0
    
    while not done:
        if env.current_player == agent.player_id:
            # Agent's turn
            state = obs
            
            if env.phase == "discard":
                discard = agent.choose_discard(
                    env.hands[agent.player_id],
                    env.dealer == agent.player_id
                )
                for card in discard:
                    action = env.hands[agent.player_id].index(card)
                    next_obs, reward, done, _, _ = env.step(action)
                    episode_reward += reward
                    agent.update(state, action, reward, next_obs, done)
                    state = next_obs
                    
            elif env.phase == "play":
                card = agent.choose_play(
                    env.hands[agent.player_id],
                    env.played_cards,
                    sum(c.value for c in env.played_cards)
                )
                if card:
                    action = env.hands[agent.player_id].index(card)
                    next_obs, reward, done, _, _ = env.step(action)
                    episode_reward += reward
                    agent.update(state, action, reward, next_obs, done)
                    state = next_obs
                else:
                    next_obs, reward, done, _, _ = env.step(-1)
                    episode_reward += reward
                    state = next_obs
            else:
                next_obs, reward, done, _, _ = env.step(0)
                episode_reward += reward
                state = next_obs
                
        else:
            # Opponent's turn
            if env.phase == "discard":
                discard = opponent.choose_discard(
                    env.hands[opponent.player_id],
                    env.dealer == opponent.player_id
                )
                for card in discard:
                    action = env.hands[opponent.player_id].index(card)
                    obs, _, done, _, _ = env.step(action)
                    
            elif env.phase == "play":
                card = opponent.choose_play(
                    env.hands[opponent.player_id],
                    env.played_cards,
                    sum(c.value for c in env.played_cards)
                )
                if card:
                    action = env.hands[opponent.player_id].index(card)
                    obs, _, done, _, _ = env.step(action)
                else:
                    obs, _, done, _, _ = env.step(-1)
            else:
                obs, _, done, _, _ = env.step(0)
        
        if done:
            agent_score = env.scores[agent.player_id]
            
    return episode_reward, agent_score


def train(episodes: int = 10000, eval_interval: int = 100,
          save_dir: str = "models", log_dir: str = "logs"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    env = CribbageEnv()
    agent = DQNAgent(player_id=0)
    opponent = RandomAgent(player_id=1)
    
    best_score = float('-inf')
    eval_scores: List[float] = []
    
    for episode in range(episodes):
        episode_reward, agent_score = train_episode(env, agent, opponent)
        
        writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
        writer.add_scalar('Training/Agent_Score', agent_score, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        
        if (episode + 1) % eval_interval == 0:
            # Evaluate agent
            eval_rewards = []
            eval_scores = []
            eval_episodes = 100
            
            for _ in range(eval_episodes):
                old_epsilon = agent.epsilon
                agent.epsilon = 0  # No exploration during evaluation
                eval_reward, eval_score = train_episode(env, agent, opponent)
                agent.epsilon = old_epsilon
                
                eval_rewards.append(eval_reward)
                eval_scores.append(eval_score)
            
            avg_reward = np.mean(eval_rewards)
            avg_score = np.mean(eval_scores)
            
            writer.add_scalar('Evaluation/Average_Reward', avg_reward, episode)
            writer.add_scalar('Evaluation/Average_Score', avg_score, episode)
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Average Evaluation Reward: {avg_reward:.2f}")
            print(f"Average Evaluation Score: {avg_score:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print("-" * 50)
            
            if avg_score > best_score:
                best_score = avg_score
                agent.save(os.path.join(save_dir, "best_model.pth"))
        
        if (episode + 1) % 1000 == 0:
            agent.save(os.path.join(save_dir, f"model_episode_{episode+1}.pth"))
    
    writer.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Cribbage")
    parser.add_argument("--episodes", type=int, default=10000,
                      help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=100,
                      help="Evaluation interval")
    parser.add_argument("--save-dir", type=str, default="models",
                      help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs",
                      help="Directory to save tensorboard logs")
    
    args = parser.parse_args()
    
    trained_agent = train(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )