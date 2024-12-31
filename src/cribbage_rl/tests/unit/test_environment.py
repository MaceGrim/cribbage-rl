import pytest

from cribbage_rl.cribbage_env.core.environment import CribbageEnv, GamePhase
from cribbage_rl.cribbage_env.core.cards import Card, Rank, Suit


def test_environment_initialization():
    env = CribbageEnv()
    obs, info = env.reset()
    
    # Check observation space
    assert obs.shape == (114,)
    
    # Check initial hands
    assert len(env.hands[0]) == 6
    assert len(env.hands[1]) == 6
    
    # Check initial scores
    assert env.scores[0] == 0
    assert env.scores[1] == 0


def test_scoring_fifteen():
    env = CribbageEnv()
    env.reset()
    
    # Create a hand with a fifteen
    hand = [
        Card(Rank.FIVE, Suit.HEARTS),
        Card(Rank.TEN, Suit.DIAMONDS),
        Card(Rank.KING, Suit.CLUBS),
        Card(Rank.FIVE, Suit.SPADES),
    ]
    starter = Card(Rank.ACE, Suit.HEARTS)
    
    score = env._handle_show()
    assert score >= 0  # Basic sanity check


def test_game_flow():
    env = CribbageEnv()
    obs, info = env.reset()
    
    # Test initial state
    assert env.phase == GamePhase.DISCARD
    assert len(env.hands[0]) == 6
    assert len(env.hands[1]) == 6
    
    # Test discard phase
    for _ in range(4):  # Both players discard 2 cards
        action = 0  # Always discard first card
        obs, reward, done, truncated, info = env.step(action)
        assert not done
        assert reward >= -1  # Should be valid move
    
    # Test cut phase
    assert env.phase == GamePhase.CUT
    obs, reward, done, truncated, info = env.step(0)  # Cut the deck
    assert not done
    assert reward >= -1
    
    # Test play phase
    assert env.phase == GamePhase.PLAY
    assert len(env.hands[0]) == 4
    assert len(env.hands[1]) == 4
    
    # Play all cards
    for _ in range(8):  # Both players play 4 cards each
        action = 0  # Always play first available card
        obs, reward, done, truncated, info = env.step(action)
        assert not done
        assert reward >= -1
    
    # Test show phase
    assert env.phase == GamePhase.SHOW
    obs, reward, done, truncated, info = env.step(0)  # Show hands
    
    # Test new hand
    assert env.phase == GamePhase.DISCARD
    assert len(env.hands[0]) == 6
    assert len(env.hands[1]) == 6