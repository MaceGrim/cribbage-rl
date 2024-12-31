from collections import Counter
from itertools import combinations
from typing import List, Set, Tuple

from .cards import Card, Rank, Suit


class CribbageScoring:
    @staticmethod
    def score_fifteen(cards: List[Card]) -> int:
        score = 0
        for r in range(2, len(cards) + 1):
            for combo in combinations(cards, r):
                if sum(card.value for card in combo) == 15:
                    score += 2
        return score

    @staticmethod
    def score_pairs(cards: List[Card]) -> int:
        score = 0
        for r in range(2, min(5, len(cards) + 1)):
            for combo in combinations(cards, r):
                if all(card.rank == combo[0].rank for card in combo):
                    score += 2 * (r - 1)
        return score

    @staticmethod
    def score_runs(cards: List[Card]) -> int:
        score = 0
        values = sorted([card.rank.value for card in cards])
        max_run = 0
        current_run = 1
        
        for i in range(1, len(values)):
            if values[i] == values[i-1] + 1:
                current_run += 1
            else:
                if current_run >= 3:
                    max_run = max(max_run, current_run)
                current_run = 1
        
        if current_run >= 3:
            max_run = max(max_run, current_run)
        
        if max_run >= 3:
            score = max_run
            
        return score

    @staticmethod
    def score_flush(cards: List[Card], is_crib: bool = False) -> int:
        suits = [card.suit for card in cards]
        if len(set(suits)) == 1:
            return 4 if is_crib else 5
        return 0

    @staticmethod
    def score_nobs(hand: List[Card], starter: Card) -> int:
        return 1 if any(
            card.rank == Rank.JACK and card.suit == starter.suit
            for card in hand
        ) else 0

    @staticmethod
    def score_play(played_cards: List[Card]) -> Tuple[int, str]:
        total = sum(card.value for card in played_cards)
        if total > 31:
            return 0, "Invalid play - sum exceeds 31"
        
        score = 0
        reason = []
        
        # Check for 15 or 31
        if total == 15:
            score += 2
            reason.append("fifteen for 2")
        elif total == 31:
            score += 2
            reason.append("thirty-one for 2")
            
        # Check for pairs/trips/quads
        if len(played_cards) >= 2:
            last_cards = played_cards[-2:]
            if last_cards[0].rank == last_cards[1].rank:
                score += 2
                reason.append("pair for 2")
                if len(played_cards) >= 3 and played_cards[-3].rank == last_cards[0].rank:
                    score += 4
                    reason[-1] = "three of a kind for 6"
                    if len(played_cards) >= 4 and played_cards[-4].rank == last_cards[0].rank:
                        score += 6
                        reason[-1] = "four of a kind for 12"
        
        # Check for runs
        for run_length in range(min(len(played_cards), 7), 2, -1):
            subset = played_cards[-run_length:]
            values = sorted([card.rank.value for card in subset])
            is_run = all(values[i] == values[i-1] + 1 for i in range(1, len(values)))
            if is_run:
                score += run_length
                reason.append(f"run of {run_length} for {run_length}")
                break
                
        return score, " and ".join(reason) if reason else "no score"

    @staticmethod
    def score_hand(hand: List[Card], starter: Card, is_crib: bool = False) -> int:
        all_cards = hand + [starter]
        score = (
            CribbageScoring.score_fifteen(all_cards) +
            CribbageScoring.score_pairs(all_cards) +
            CribbageScoring.score_runs(all_cards) +
            CribbageScoring.score_flush(hand, is_crib) +
            CribbageScoring.score_nobs(hand, starter)
        )
        return score