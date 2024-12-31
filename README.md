# Cribbage Reinforcement Learning Project

This project implements a Cribbage environment for training reinforcement learning agents and playing against them. The implementation follows the standard Bicycle Cards rules for Cribbage.

## Features

- Complete Cribbage game environment with Gymnasium interface
- DQN (Deep Q-Network) agent implementation
- Random agent for baseline comparison
- Command-line interface for human vs computer play
- Streamlit web interface with interactive gameplay
- Comprehensive test suite
- TensorBoard integration for training visualization

## Project Structure

```
cribbage/
├── cribbage_env/        # Core environment implementation
│   ├── core/           # Core game logic
│   └── utils/          # Utility functions
├── agents/             # Agent implementations
│   ├── base/          # Base agent classes
│   └── rl/            # RL agent implementations
├── ui/                 # User interfaces
│   ├── cli/           # Command-line interface
│   └── streamlit/     # Streamlit web interface
├── tests/             # Test suite
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
└── docs/              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cribbage-rl.git
cd cribbage-rl
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

### Quick Start

The easiest way to get started is to use the demo script:

1. Train a DQN agent:
```bash
poetry run python demo.py train --episodes 1000 --eval-interval 100
```

2. Play against a random agent:
```bash
poetry run python demo.py play --agent random
```

3. Play against a trained DQN agent:
```bash
poetry run python demo.py play --agent dqn --model models/best_model.pth
```

### Advanced Usage

For more control and features:

1. Train DQN agent with TensorBoard monitoring:
```bash
poetry run python run.py train --episodes 10000 --eval-interval 100
tensorboard --logdir logs
```

2. Start the Streamlit interface:
```bash
poetry run python run.py play
```

3. Use the command-line interface:
```bash
poetry run python ui/cli/play_game.py --player 0
```

### Running Tests

Run the test suite:
```bash
poetry run pytest
```

## Game Rules

The implementation follows standard Cribbage rules:

1. Each player is dealt 6 cards
2. Players discard 2 cards each to form the crib
3. A starter card is cut
4. Players take turns playing cards, trying to make:
   - Fifteens (2 points)
   - Pairs (2 points)
   - Runs (1 point per card)
   - Thirty-one (2 points)
5. After the play, hands are scored including:
   - Fifteens
   - Pairs
   - Runs
   - Flushes
   - Nobs (Jack of same suit as starter)
6. First player to 121 points wins

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Bicycle Cards for the official Cribbage rules
- OpenAI Gymnasium for the environment interface
- PyTorch for the deep learning framework
- Streamlit for the web interface