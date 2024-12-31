import argparse
import os
import subprocess
import sys


def train_agent(args):
    from agents.rl.train_dqn import train
    
    print("Starting DQN agent training...")
    trained_agent = train(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    print("Training completed!")


def run_streamlit():
    print("Starting Streamlit interface...")
    subprocess.run([
        "streamlit", "run",
        os.path.join("ui", "streamlit", "app.py")
    ])


def main():
    parser = argparse.ArgumentParser(description="Cribbage RL Project Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training arguments
    train_parser = subparsers.add_parser("train", help="Train DQN agent")
    train_parser.add_argument("--episodes", type=int, default=10000,
                           help="Number of training episodes")
    train_parser.add_argument("--eval-interval", type=int, default=100,
                           help="Evaluation interval")
    train_parser.add_argument("--save-dir", type=str, default="models",
                           help="Directory to save models")
    train_parser.add_argument("--log-dir", type=str, default="logs",
                           help="Directory to save tensorboard logs")
    
    # Play arguments
    play_parser = subparsers.add_parser("play", help="Start Streamlit interface")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_agent(args)
    elif args.command == "play":
        run_streamlit()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()