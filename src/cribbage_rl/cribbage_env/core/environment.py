from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .cards import Card, Deck, Rank, Suit
from .scoring import CribbageScoring


class GamePhase(Enum):
    DEAL = "deal"
    DISCARD = "discard"
    CUT = "cut"
    PLAY = "play"
    SHOW = "show"
    GAME_OVER = "game_over"


class CribbageEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.deck = Deck()
        self.hands = {0: [], 1: []}  # Player hands
        self.crib = []  # Crib cards
        self.starter: Optional[Card] = None
        self.played_cards: List[Card] = []
        self.scores = {0: 0, 1: 0}
        self.current_player = 0
        self.dealer = 0
        self.phase = GamePhase.DEAL
        self.render_mode = render_mode
        
        # Define action spaces for different phases
        self.action_space = spaces.Discrete(52)  # Maximum possible actions
        
        # Observation space includes:
        # - Player's hand (52 binary values)
        # - Opponent's known played cards (52 binary values)
        # - Crib ownership (1 value)
        # - Current scores (2 values)
        # - Phase (6 values)
        # - Current play count (1 value)
        # Total: 114 values
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(114,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        self.deck = Deck()
        self.hands = {0: [], 1: []}
        self.crib = []
        self.starter = None
        self.played_cards = []
        self.scores = {0: 0, 1: 0}
        self.current_player = 0
        self.dealer = 0
        
        # Deal initial hands
        for player in [0, 1]:
            self.hands[player] = self.deck.draw(6)
            
        # Start with discard phase, non-dealer discards first
        self.phase = GamePhase.DISCARD
        self.current_player = 1 - self.dealer
            
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(114, dtype=np.float32)
        
        # Encode player's hand
        if self.hands[self.current_player]:
            for card in self.hands[self.current_player]:
                card_idx = self._card_to_index(card)
                obs[card_idx] = 1
                
        # Encode opponent's known played cards
        opponent = 1 - self.current_player
        start_idx = 52
        for card in self.played_cards:
            if card in self.hands[opponent]:
                card_idx = start_idx + self._card_to_index(card)
                obs[card_idx] = 1
                
        # Encode crib ownership
        obs[104] = float(self.dealer == self.current_player)
        
        # Encode scores
        obs[105] = self.scores[self.current_player] / 121
        obs[106] = self.scores[1 - self.current_player] / 121
        
        # Encode phase
        phase_idx = 107 + list(GamePhase).index(self.phase)
        obs[phase_idx] = 1
        
        # Encode current play count
        play_count = sum(card.value for card in self.played_cards)
        obs[113] = play_count / 31
        
        return obs

    def _card_to_index(self, card: Card) -> int:
        # Convert card to index in range [0, 51]
        # Suits: Hearts (0), Diamonds (1), Clubs (2), Spades (3)
        # Ranks: Ace (0) through King (12)
        suit_map = {
            Suit.HEARTS: 0,
            Suit.DIAMONDS: 1,
            Suit.CLUBS: 2,
            Suit.SPADES: 3
        }
        rank_map = {
            Rank.ACE: 0,
            Rank.TWO: 1,
            Rank.THREE: 2,
            Rank.FOUR: 3,
            Rank.FIVE: 4,
            Rank.SIX: 5,
            Rank.SEVEN: 6,
            Rank.EIGHT: 7,
            Rank.NINE: 8,
            Rank.TEN: 9,
            Rank.JACK: 10,
            Rank.QUEEN: 11,
            Rank.KING: 12
        }
        return suit_map[card.suit] * 13 + rank_map[card.rank]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0
        done = False
        truncated = False
        info = {}
        
        if self.phase == GamePhase.DISCARD:
            reward = self._handle_discard(action)
        elif self.phase == GamePhase.CUT:
            reward = self._handle_cut(action)
        elif self.phase == GamePhase.PLAY:
            reward = self._handle_play(action)
        elif self.phase == GamePhase.SHOW:
            reward = self._handle_show()
            
        # Check if game is over
        if max(self.scores.values()) >= 121:
            done = True
            winner = 0 if self.scores[0] >= 121 else 1
            info["winner"] = winner
            reward = 1 if winner == self.current_player else -1
            
        return self._get_observation(), reward, done, truncated, info

    def _handle_discard(self, action: int) -> float:
        if len(self.crib) >= 4:
            self.phase = GamePhase.CUT
            self.current_player = self.dealer  # Dealer cuts
            return 0
            
        hand = self.hands[self.current_player]
        if 0 <= action < len(hand):
            card = hand.pop(action)
            self.crib.append(card)
            
            # Each player discards 2 cards
            if len(self.crib) < 4:
                if len(self.hands[self.current_player]) > 4:
                    # Same player discards again
                    return 0
                else:
                    # Switch to other player
                    self.current_player = 1 - self.current_player
                    return 0
            else:
                # Both players have discarded, move to cut phase
                self.phase = GamePhase.CUT
                self.current_player = self.dealer  # Dealer cuts
                return 0
                
        return -1  # Invalid action

    def _handle_cut(self, action: int) -> float:
        if not self.starter:
            self.starter = self.deck.cut()
            self.phase = GamePhase.PLAY
            self.current_player = 1 - self.dealer  # Non-dealer plays first
            return 0
        return -1  # Invalid action

    def _handle_play(self, action: int) -> float:
        if action == -1:  # Player says "go"
            # Check if other player can play
            other_player = 1 - self.current_player
            play_sum = sum(c.value for c in self.played_cards)
            other_can_play = any(
                card.value + play_sum <= 31
                for card in self.hands[other_player]
            )
            
            if other_can_play:
                self.current_player = other_player
                return 0
            else:
                # Neither player can play, start new count
                self.played_cards = []
                # Check if any player has cards left
                if any(len(hand) > 0 for hand in self.hands.values()):
                    # Find first player with cards
                    for player in [0, 1]:
                        if len(self.hands[player]) > 0:
                            self.current_player = player
                            break
                    return 0
                else:
                    # No cards left, move to show phase
                    self.phase = GamePhase.SHOW
                    self.current_player = 1 - self.dealer  # Non-dealer shows first
                    return 0
        
        hand = self.hands[self.current_player]
        if 0 <= action < len(hand):
            card = hand[action]
            play_sum = sum(c.value for c in self.played_cards) + card.value
            
            if play_sum <= 31:
                hand.pop(action)
                self.played_cards.append(card)
                score, reason = CribbageScoring.score_play(self.played_cards)
                self.scores[self.current_player] += score
                
                if play_sum == 31:
                    # Start new count
                    self.played_cards = []
                    # Find first player with cards
                    if any(len(hand) > 0 for hand in self.hands.values()):
                        for player in [0, 1]:
                            if len(self.hands[player]) > 0:
                                self.current_player = player
                                break
                    else:
                        # No cards left, move to show phase
                        self.phase = GamePhase.SHOW
                        self.current_player = 1 - self.dealer  # Non-dealer shows first
                else:
                    # Check if next player can play
                    other_player = 1 - self.current_player
                    other_can_play = any(
                        card.value + play_sum <= 31
                        for card in self.hands[other_player]
                    )
                    if other_can_play:
                        self.current_player = other_player
                    else:
                        # Other player must say "go"
                        score += 1  # Point for go
                        self.scores[self.current_player] += 1
                        # Check if current player can continue
                        current_can_play = any(
                            card.value + play_sum <= 31
                            for card in self.hands[self.current_player]
                        )
                        if not current_can_play:
                            # Neither player can play, start new count
                            self.played_cards = []
                            # Find first player with cards
                            if any(len(hand) > 0 for hand in self.hands.values()):
                                for player in [0, 1]:
                                    if len(self.hands[player]) > 0:
                                        self.current_player = player
                                        break
                            else:
                                # No cards left, move to show phase
                                self.phase = GamePhase.SHOW
                                self.current_player = 1 - self.dealer  # Non-dealer shows first
                
                return score
                
        return -1  # Invalid action

    def _handle_show(self) -> float:
        if not self.starter:
            self.starter = self.deck.cut()
            self.phase = GamePhase.PLAY
            self.current_player = 1 - self.dealer  # Non-dealer plays first
            return 0
            
        score = 0
        for player in [1 - self.dealer, self.dealer]:  # Non-dealer shows first
            hand_score = CribbageScoring.score_hand(
                self.hands[player],
                self.starter,
                is_crib=False
            )
            self.scores[player] += hand_score
            if player == self.current_player:
                score = hand_score
                
        # Score crib (dealer's)
        if self.crib:
            crib_score = CribbageScoring.score_hand(
                self.crib,
                self.starter,
                is_crib=True
            )
            self.scores[self.dealer] += crib_score
            if self.dealer == self.current_player:
                score += crib_score
                
        # Start new hand
        self.phase = GamePhase.DEAL
        self.dealer = 1 - self.dealer
        self.current_player = self.dealer  # Dealer deals
        self.deck = Deck()  # New deck
        self.hands = {0: [], 1: []}  # Clear hands
        self.crib = []  # Clear crib
        self.starter = None  # Clear starter
        self.played_cards = []  # Clear played cards
        
        # Deal new hands
        for player in [0, 1]:
            self.hands[player] = self.deck.draw(6)
            
        self.phase = GamePhase.DISCARD
        self.current_player = 1 - self.dealer  # Non-dealer discards first
        
        return score

    def render(self, mode: str = None):
        if self.render_mode == "human":
            print("\n=== Cribbage Game State ===")
            print(f"Phase: {self.phase.value}")
            print(f"Scores - Player 0: {self.scores[0]}, Player 1: {self.scores[1]}")
            print(f"Current player: {self.current_player}")
            print(f"Dealer: {self.dealer}")
            
            if self.starter:
                print(f"Starter card: {self.starter}")
                
            print("\nHands:")
            for player, hand in self.hands.items():
                print(f"Player {player}: {' '.join(str(card) for card in hand)}")
                
            if self.played_cards:
                print("\nPlayed cards:", ' '.join(str(card) for card in self.played_cards))
                print(f"Play count: {sum(card.value for card in self.played_cards)}")
                
            if self.crib:
                print("\nCrib:", ' '.join(str(card) for card in self.crib))
                
        return None