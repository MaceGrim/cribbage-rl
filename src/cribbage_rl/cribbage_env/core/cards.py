from dataclasses import dataclass
from enum import Enum
import random
from typing import List, Optional, Set, Tuple


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Rank(Enum):
    ACE = (1, "A")
    TWO = (2, "2")
    THREE = (3, "3")
    FOUR = (4, "4")
    FIVE = (5, "5")
    SIX = (6, "6")
    SEVEN = (7, "7")
    EIGHT = (8, "8")
    NINE = (9, "9")
    TEN = (10, "10")
    JACK = (10, "J")
    QUEEN = (10, "Q")
    KING = (10, "K")

    def __init__(self, value: int, symbol: str):
        self._value_ = value
        self.symbol = symbol

    @property
    def value(self) -> int:
        return self._value_


@dataclass(frozen=True)
class Card:
    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.symbol}{self.suit.value}"

    @property
    def value(self) -> int:
        return self.rank.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


class Deck:
    def __init__(self):
        self.cards: List[Card] = []
        self.reset()

    def reset(self) -> None:
        self.cards = [
            Card(rank, suit)
            for rank in Rank
            for suit in Suit
        ]
        random.shuffle(self.cards)

    def draw(self, n: int = 1) -> List[Card]:
        if n > len(self.cards):
            raise ValueError("Not enough cards in deck")
        drawn = self.cards[:n]
        self.cards = self.cards[n:]
        return drawn

    def cut(self) -> Card:
        if not self.cards:
            raise ValueError("No cards left to cut")
        cut_index = random.randint(0, len(self.cards) - 1)
        cut_card = self.cards[cut_index]
        self.cards.pop(cut_index)
        return cut_card