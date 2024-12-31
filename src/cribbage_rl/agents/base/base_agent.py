from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from cribbage_rl.cribbage_env.core.cards import Card


class BaseAgent(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand: List[Card] = []
        self.opponent_known_cards: List[Card] = []
        self.is_dealer = False

    @abstractmethod
    def choose_discard(self, hand: List[Card], is_dealer: bool) -> List[Card]:
        """Choose which cards to discard to the crib."""
        pass

    @abstractmethod
    def choose_play(self, hand: List[Card], played_cards: List[Card], play_count: int) -> Optional[Card]:
        """Choose which card to play during the play phase."""
        pass

    def observe_opponent_play(self, card: Card) -> None:
        """Record a card played by the opponent."""
        self.opponent_known_cards.append(card)

    def reset(self) -> None:
        """Reset the agent's state for a new game."""
        self.hand = []
        self.opponent_known_cards = []
        self.is_dealer = False