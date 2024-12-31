import argparse
import os
import subprocess
import sys
import time

from cribbage_rl.agents.base.random_agent import RandomAgent
from cribbage_rl.agents.rl.dqn_agent import DQNAgent
from cribbage_rl.cribbage_env.core.environment import CribbageEnv, GamePhase


def train_agent(episodes: int = 1000, eval_interval: int = 100):
    """Train a DQN agent against a random opponent."""
    print("Training DQN agent...")
    
    env = CribbageEnv()
    dqn_agent = DQNAgent(0)  # DQN agent is player 0
    random_agent = RandomAgent(1)  # Random agent is player 1
    
    # Create directories for models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Training loop
    best_score = float('-inf')
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if env.current_player == 0:  # DQN agent's turn
                if env.phase == GamePhase.DISCARD:
                    discard = dqn_agent.choose_discard(
                        env.hands[0],
                        env.dealer == 0
                    )
                    for card in discard:
                        action = env.hands[0].index(card)
                        next_obs, reward, done, _, _ = env.step(action)
                        episode_reward += reward
                        dqn_agent.update(obs, action, reward, next_obs, done)
                        obs = next_obs
                        
                elif env.phase == GamePhase.PLAY:
                    card = dqn_agent.choose_play(
                        env.hands[0],
                        env.played_cards,
                        sum(c.value for c in env.played_cards)
                    )
                    if card:
                        action = env.hands[0].index(card)
                        next_obs, reward, done, _, _ = env.step(action)
                        episode_reward += reward
                        dqn_agent.update(obs, action, reward, next_obs, done)
                        obs = next_obs
                    else:
                        next_obs, reward, done, _, _ = env.step(-1)
                        episode_reward += reward
                        obs = next_obs
                        
                else:
                    next_obs, reward, done, _, _ = env.step(0)
                    episode_reward += reward
                    obs = next_obs
                    
            else:  # Random agent's turn
                if env.phase == GamePhase.DISCARD:
                    discard = random_agent.choose_discard(
                        env.hands[1],
                        env.dealer == 1
                    )
                    for card in discard:
                        action = env.hands[1].index(card)
                        obs, _, done, _, _ = env.step(action)
                        
                elif env.phase == GamePhase.PLAY:
                    card = random_agent.choose_play(
                        env.hands[1],
                        env.played_cards,
                        sum(c.value for c in env.played_cards)
                    )
                    if card:
                        action = env.hands[1].index(card)
                        obs, _, done, _, _ = env.step(action)
                    else:
                        obs, _, done, _, _ = env.step(-1)
                        
                else:
                    obs, _, done, _, _ = env.step(0)
        
        # Print progress
        if (episode + 1) % eval_interval == 0:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Episode reward: {episode_reward}")
            print(f"DQN agent score: {env.scores[0]}")
            print(f"Random agent score: {env.scores[1]}")
            print(f"Epsilon: {dqn_agent.epsilon:.3f}")
            print("-" * 50)
            
            # Save best model
            if env.scores[0] > best_score:
                best_score = env.scores[0]
                dqn_agent.save("models/best_model.pth")
    
    print("Training completed!")
    return dqn_agent


def play_game(agent_type: str = "random", model_path: str = None):
    """Play a game against a computer opponent."""
    env = CribbageEnv(render_mode="human")
    
    # Create computer opponent
    if agent_type == "random":
        computer = RandomAgent(1)
    else:  # DQN
        computer = DQNAgent(1)
        if model_path and os.path.exists(model_path):
            computer.load(model_path)
            print(f"Loaded model from {model_path}")
    
    # Game loop
    obs, _ = env.reset()
    done = False
    
    while not done:
        env.render()
        
        if env.current_player == 0:  # Human's turn
            print("\nYour turn!")
            if env.phase == GamePhase.DISCARD:
                print("\nYour hand:", ' '.join(str(card) for card in env.hands[0]))
                while True:
                    try:
                        indices = input("Choose two cards to discard (e.g., '0 3'): ").split()
                        if len(indices) != 2:
                            print("Please select exactly two cards.")
                            continue
                        indices = [int(i) for i in indices]
                        if not all(0 <= i < len(env.hands[0]) for i in indices):
                            print("Invalid card indices.")
                            continue
                        for idx in indices:
                            obs, reward, done, _, _ = env.step(idx)
                        break
                    except ValueError:
                        print("Invalid input. Please enter two numbers.")
                        
            elif env.phase == GamePhase.PLAY:
                print("\nYour hand:", ' '.join(str(card) for card in env.hands[0]))
                print("Played cards:", ' '.join(str(card) for card in env.played_cards))
                print(f"Play count: {sum(c.value for c in env.played_cards)}")
                
                # Check if any legal plays
                play_sum = sum(c.value for c in env.played_cards)
                legal_plays = [
                    i for i, card in enumerate(env.hands[0])
                    if play_sum + card.value <= 31
                ]
                
                if legal_plays:
                    while True:
                        try:
                            idx = int(input("Choose a card to play (index): "))
                            if idx in legal_plays:
                                obs, reward, done, _, _ = env.step(idx)
                                break
                            print("Invalid choice. Please select a legal play.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                else:
                    print("No legal plays available. Saying 'Go'.")
                    obs, reward, done, _, _ = env.step(-1)
                    
            else:
                input("Press Enter to continue...")
                obs, reward, done, _, _ = env.step(0)
                
        else:  # Computer's turn
            print("\nComputer's turn...")
            time.sleep(1)  # Add a small delay for better UX
            
            if env.phase == GamePhase.DISCARD:
                discard = computer.choose_discard(
                    env.hands[1],
                    env.dealer == 1
                )
                for card in discard:
                    action = env.hands[1].index(card)
                    obs, reward, done, _, _ = env.step(action)
                    
            elif env.phase == GamePhase.PLAY:
                card = computer.choose_play(
                    env.hands[1],
                    env.played_cards,
                    sum(c.value for c in env.played_cards)
                )
                if card:
                    action = env.hands[1].index(card)
                    obs, reward, done, _, _ = env.step(action)
                else:
                    obs, reward, done, _, _ = env.step(-1)
                    
            else:
                obs, reward, done, _, _ = env.step(0)
    
    # Game over
    env.render()
    print("\nGame Over!")
    print(f"Final scores - You: {env.scores[0]}, Computer: {env.scores[1]}")
    winner = "You" if env.scores[0] >= 121 else "Computer"
    print(f"{winner} won!")


def main():
    parser = argparse.ArgumentParser(description="Cribbage RL Demo")
    parser.add_argument("mode", choices=["train", "play"],
                      help="Mode to run: train or play")
    parser.add_argument("--episodes", type=int, default=1000,
                      help="Number of training episodes")
    parser.add_argument("--eval-interval", type=int, default=100,
                      help="Evaluation interval")
    parser.add_argument("--agent", choices=["random", "dqn"], default="random",
                      help="Type of computer opponent")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                      help="Path to model file for DQN agent")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_agent(args.episodes, args.eval_interval)
    else:  # play
        play_game(args.agent, args.model)


if __name__ == "__main__":
    main()