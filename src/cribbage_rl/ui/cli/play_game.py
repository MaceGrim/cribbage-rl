import argparse
from typing import List, Optional

from cribbage_env.core.cards import Card
from cribbage_env.core.environment import CribbageEnv, GamePhase
from agents.base.random_agent import RandomAgent


class HumanAgent:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand: List[Card] = []

    def choose_discard(self, hand: List[Card], is_dealer: bool) -> List[Card]:
        self.hand = hand.copy()
        print("\nYour hand:", ' '.join(f"{i}:{card}" for i, card in enumerate(hand)))
        while True:
            try:
                indices = input("Choose two cards to discard (e.g., '0 3'): ").split()
                if len(indices) != 2:
                    print("Please select exactly two cards.")
                    continue
                indices = [int(i) for i in indices]
                if not all(0 <= i < len(hand) for i in indices):
                    print("Invalid card indices.")
                    continue
                discard = [hand[i] for i in indices]
                for card in discard:
                    self.hand.remove(card)
                return discard
            except ValueError:
                print("Invalid input. Please enter two numbers.")

    def choose_play(self, hand: List[Card], played_cards: List[Card], play_count: int) -> Optional[Card]:
        legal_plays = [
            card for card in hand
            if play_count + card.value <= 31
        ]
        if not legal_plays:
            return None
            
        print("\nPlayed cards:", ' '.join(str(card) for card in played_cards))
        print(f"Play count: {play_count}")
        print("\nYour hand:", ' '.join(f"{i}:{card}" for i, card in enumerate(hand)))
        print("Legal plays:", ' '.join(str(card) for card in legal_plays))
        
        while True:
            try:
                idx = int(input("Choose a card to play (index): "))
                if 0 <= idx < len(hand) and hand[idx] in legal_plays:
                    return hand[idx]
                print("Invalid choice. Please select a legal play.")
            except ValueError:
                print("Invalid input. Please enter a number.")


def play_game(human_player_id: int = 0):
    env = CribbageEnv(render_mode="human")
    human = HumanAgent(human_player_id)
    computer = RandomAgent(1 - human_player_id)
    agents = {human_player_id: human, 1 - human_player_id: computer}
    
    obs, info = env.reset()
    done = False
    
    while not done:
        env.render()
        
        if env.phase == GamePhase.DISCARD:
            current_agent = agents[env.current_player]
            discard = current_agent.choose_discard(
                env.hands[env.current_player],
                env.dealer == env.current_player
            )
            for card in discard:
                action = env.hands[env.current_player].index(card)
                obs, reward, done, truncated, info = env.step(action)
                
        elif env.phase == GamePhase.PLAY:
            current_agent = agents[env.current_player]
            card = current_agent.choose_play(
                env.hands[env.current_player],
                env.played_cards,
                sum(c.value for c in env.played_cards)
            )
            if card:
                action = env.hands[env.current_player].index(card)
                obs, reward, done, truncated, info = env.step(action)
            else:
                # No legal play, pass
                obs, reward, done, truncated, info = env.step(-1)
                
        else:
            # Handle other phases automatically
            obs, reward, done, truncated, info = env.step(0)
    
    env.render()
    winner = 0 if env.scores[0] >= 121 else 1
    print(f"\nGame Over! {'You' if winner == human_player_id else 'Computer'} won!")
    print(f"Final scores - You: {env.scores[human_player_id]}, Computer: {env.scores[1-human_player_id]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Cribbage against a computer")
    parser.add_argument("--player", type=int, choices=[0, 1], default=0,
                      help="Choose player ID (0 or 1)")
    args = parser.parse_args()
    play_game(args.player)