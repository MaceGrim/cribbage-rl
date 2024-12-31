import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from cribbage_env.core.cards import Card, Rank, Suit
from cribbage_env.core.environment import CribbageEnv, GamePhase
from agents.base.random_agent import RandomAgent
from agents.rl.dqn_agent import DQNAgent


def get_card_emoji(card: Card) -> str:
    suit_map = {
        Suit.HEARTS: "♥️",
        Suit.DIAMONDS: "♦️",
        Suit.CLUBS: "♣️",
        Suit.SPADES: "♠️"
    }
    return f"{card.rank.symbol}{suit_map[card.suit]}"


def display_game_state(env: CribbageEnv, human_id: int):
    st.markdown("## Game State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Scores")
        st.markdown(f"You: {env.scores[human_id]}")
        st.markdown(f"Computer: {env.scores[1-human_id]}")
        
    with col2:
        st.markdown("### Current Phase")
        st.markdown(f"{env.phase.value}")
        if env.current_player == human_id:
            st.markdown("**Your turn**")
        else:
            st.markdown("*Computer's turn*")
    
    st.markdown("### Your Hand")
    if env.hands[human_id]:
        hand_str = " ".join(get_card_emoji(card) for card in env.hands[human_id])
        st.markdown(f"<h2>{hand_str}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("*No cards*")
    
    if env.played_cards:
        st.markdown("### Played Cards")
        played_str = " ".join(get_card_emoji(card) for card in env.played_cards)
        st.markdown(f"<h3>{played_str}</h3>", unsafe_allow_html=True)
        st.markdown(f"Play count: {sum(card.value for card in env.played_cards)}")
    
    if env.starter:
        st.markdown("### Starter Card")
        st.markdown(f"<h3>{get_card_emoji(env.starter)}</h3>",
                   unsafe_allow_html=True)


def human_select_discard(hand: List[Card]) -> List[Card]:
    st.markdown("### Select Two Cards to Discard")
    
    cards_to_discard = []
    cols = st.columns(len(hand))
    
    for i, (card, col) in enumerate(zip(hand, cols)):
        with col:
            if st.button(get_card_emoji(card), key=f"discard_{i}"):
                cards_to_discard.append(card)
                if len(cards_to_discard) == 2:
                    return cards_to_discard
    
    return cards_to_discard


def human_select_play(hand: List[Card], played_cards: List[Card]) -> Optional[Card]:
    play_count = sum(card.value for card in played_cards)
    legal_plays = [
        card for card in hand
        if play_count + card.value <= 31
    ]
    
    if not legal_plays:
        st.markdown("*No legal plays available*")
        return None
    
    st.markdown("### Select a Card to Play")
    cols = st.columns(len(hand))
    
    for i, (card, col) in enumerate(zip(hand, cols)):
        with col:
            if card in legal_plays:
                if st.button(get_card_emoji(card), key=f"play_{i}"):
                    return card
            else:
                st.markdown(f"~~{get_card_emoji(card)}~~")
    
    return None


def play_game(agent_type: str = "random", model_path: Optional[str] = None):
    if "env" not in st.session_state:
        st.session_state.env = CribbageEnv()
        st.session_state.human_id = 0
        
        if agent_type == "random":
            st.session_state.computer = RandomAgent(1 - st.session_state.human_id)
        else:  # DQN
            st.session_state.computer = DQNAgent(1 - st.session_state.human_id)
            if model_path and os.path.exists(model_path):
                st.session_state.computer.load(model_path)
        
        st.session_state.game_history = []
        obs, _ = st.session_state.env.reset()
    
    env = st.session_state.env
    human_id = st.session_state.human_id
    computer = st.session_state.computer
    
    display_game_state(env, human_id)
    
    done = False
    
    if env.current_player == human_id:
        if env.phase == GamePhase.DISCARD:
            discard = human_select_discard(env.hands[human_id])
            if len(discard) == 2:
                for card in discard:
                    action = env.hands[human_id].index(card)
                    obs, reward, done, _, _ = env.step(action)
                st.experimental_rerun()
                
        elif env.phase == GamePhase.PLAY:
            card = human_select_play(env.hands[human_id], env.played_cards)
            if card is not None:
                action = env.hands[human_id].index(card)
                obs, reward, done, _, _ = env.step(action)
                st.experimental_rerun()
            elif st.button("Pass (No Legal Plays)"):
                obs, reward, done, _, _ = env.step(-1)
                st.experimental_rerun()
                
        else:
            if st.button("Continue"):
                obs, reward, done, _, _ = env.step(0)
                st.experimental_rerun()
                
    else:  # Computer's turn
        if st.button("Let Computer Play"):
            if env.phase == GamePhase.DISCARD:
                discard = computer.choose_discard(
                    env.hands[computer.player_id],
                    env.dealer == computer.player_id
                )
                for card in discard:
                    action = env.hands[computer.player_id].index(card)
                    obs, reward, done, _, _ = env.step(action)
                    
            elif env.phase == GamePhase.PLAY:
                card = computer.choose_play(
                    env.hands[computer.player_id],
                    env.played_cards,
                    sum(c.value for c in env.played_cards)
                )
                if card:
                    action = env.hands[computer.player_id].index(card)
                    obs, reward, done, _, _ = env.step(action)
                else:
                    obs, reward, done, _, _ = env.step(-1)
                    
            else:
                obs, reward, done, _, _ = env.step(0)
                
            st.experimental_rerun()
    
    if done:
        winner = human_id if env.scores[human_id] >= 121 else 1-human_id
        st.markdown(f"## Game Over! {'You' if winner == human_id else 'Computer'} won!")
        st.markdown(f"Final scores - You: {env.scores[human_id]}, "
                   f"Computer: {env.scores[1-human_id]}")
        
        if st.button("Play Again"):
            del st.session_state.env
            st.experimental_rerun()


def main():
    st.title("Cribbage Game")
    
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    
    if not st.session_state.game_started:
        st.markdown("## Welcome to Cribbage!")
        agent_type = st.selectbox(
            "Select opponent type:",
            ["random", "dqn"],
            index=0
        )
        
        model_path = None
        if agent_type == "dqn":
            model_path = st.text_input(
                "Enter path to model file (optional):",
                value="models/best_model.pth"
            )
        
        if st.button("Start Game"):
            st.session_state.game_started = True
            st.session_state.agent_type = agent_type
            st.session_state.model_path = model_path
            st.experimental_rerun()
    else:
        play_game(st.session_state.agent_type, st.session_state.model_path)
        
        if st.button("Quit Game"):
            st.session_state.game_started = False
            if "env" in st.session_state:
                del st.session_state.env
            st.experimental_rerun()


if __name__ == "__main__":
    main()