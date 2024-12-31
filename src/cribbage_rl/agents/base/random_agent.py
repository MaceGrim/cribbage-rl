import random
from typing import List, Optional

from cribbage_rl.cribbage_env.core.cards import Card
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def choose_discard(self, hand: List[Card], is_dealer: bool) -> List[Card]:
        """Randomly choose two cards to discard."""
        self.hand = hand.copy()
        self.is_dealer = is_dealer
        discard = random.sample(hand, 2)
        for card in discard:
            self.hand.remove(card)
        return discard

    def choose_play(self, hand: List[Card], played_cards: List[Card], play_count: int) -> Optional[Card]:
        """Randomly choose a legal card to play."""
        legal_plays = [
            card for card in hand
            if play_count + card.value <= 31
        ]
        return random.choice(legal_plays) if legal_plays else None