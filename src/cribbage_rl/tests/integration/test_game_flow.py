import pytest

from cribbage_rl.cribbage_env.core.environment import CribbageEnv, GamePhase
from cribbage_rl.agents.base.random_agent import RandomAgent
from cribbage_rl.agents.rl.dqn_agent import DQNAgent


def test_full_game_random_vs_random():
    env = CribbageEnv()
    agent1 = RandomAgent(0)
    agent2 = RandomAgent(1)
    
    obs, _ = env.reset()
    done = False
    max_steps = 1000
    step = 0
    
    while not done and step < max_steps:
        current_agent = agent1 if env.current_player == 0 else agent2
        
        if env.phase == GamePhase.DISCARD:
            discard = current_agent.choose_discard(
                env.hands[current_agent.player_id],
                env.dealer == current_agent.player_id
            )
            for card in discard:
                action = env.hands[current_agent.player_id].index(card)
                obs, reward, done, _, _ = env.step(action)
                
        elif env.phase == GamePhase.PLAY:
            card = current_agent.choose_play(
                env.hands[current_agent.player_id],
                env.played_cards,
                sum(c.value for c in env.played_cards)
            )
            if card:
                action = env.hands[current_agent.player_id].index(card)
                obs, reward, done, _, _ = env.step(action)
            else:
                obs, reward, done, _, _ = env.step(-1)
                
        else:
            obs, reward, done, _, _ = env.step(0)
            
        step += 1
    
    assert step < max_steps, "Game did not finish within maximum steps"
    assert max(env.scores.values()) >= 121, "Game ended without a winner"


def test_dqn_vs_random():
    env = CribbageEnv()
    dqn_agent = DQNAgent(0)
    random_agent = RandomAgent(1)
    
    obs, _ = env.reset()
    done = False
    max_steps = 1000
    step = 0
    
    while not done and step < max_steps:
        current_agent = dqn_agent if env.current_player == 0 else random_agent
        
        if env.phase == GamePhase.DISCARD:
            discard = current_agent.choose_discard(
                env.hands[current_agent.player_id],
                env.dealer == current_agent.player_id
            )
            for card in discard:
                action = env.hands[current_agent.player_id].index(card)
                obs, reward, done, _, _ = env.step(action)
                
        elif env.phase == GamePhase.PLAY:
            card = current_agent.choose_play(
                env.hands[current_agent.player_id],
                env.played_cards,
                sum(c.value for c in env.played_cards)
            )
            if card:
                action = env.hands[current_agent.player_id].index(card)
                obs, reward, done, _, _ = env.step(action)
            else:
                obs, reward, done, _, _ = env.step(-1)
                
        else:
            obs, reward, done, _, _ = env.step(0)
            
        step += 1
    
    assert step < max_steps, "Game did not finish within maximum steps"
    assert max(env.scores.values()) >= 121, "Game ended without a winner"


def test_game_state_transitions():
    env = CribbageEnv()
    obs, _ = env.reset()
    
    # Test initial state
    assert env.phase == GamePhase.DISCARD
    assert len(env.hands[0]) == 6
    assert len(env.hands[1]) == 6
    
    # Test discard phase
    for _ in range(4):  # Both players discard 2 cards
        action = 0  # Always discard first card
        obs, _, done, _, _ = env.step(action)
        
    assert len(env.hands[0]) == 4
    assert len(env.hands[1]) == 4
    assert len(env.crib) == 4
    
    # Continue game until done
    agents = {0: RandomAgent(0), 1: RandomAgent(1)}
    max_steps = 1000
    step = 0
    done = False
    
    while not done and step < max_steps:
        current_agent = agents[env.current_player]
        
        if env.phase == GamePhase.PLAY:
            card = current_agent.choose_play(
                env.hands[current_agent.player_id],
                env.played_cards,
                sum(c.value for c in env.played_cards)
            )
            if card:
                action = env.hands[current_agent.player_id].index(card)
                obs, _, done, _, _ = env.step(action)
            else:
                obs, _, done, _, _ = env.step(-1)
        else:
            obs, _, done, _, _ = env.step(0)
            
        step += 1
    
    assert step < max_steps, "Game did not finish within maximum steps"
    assert max(env.scores.values()) >= 121, "Game ended without a winner"