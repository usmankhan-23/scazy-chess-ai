# ♟️ Scazy Chess AI: Deep Reinforcement Learning Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Pygame](https://img.shields.io/badge/GUI-Pygame-green.svg)](https://www.pygame.org/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**Scazy Chess AI** is a self-play Deep Reinforcement Learning (RL) chess engine built from scratch in Python and PyTorch. Combining Convolutional Neural Networks (CNNs) with Minimax decision trees, Scazy learns positional evaluation directly from board states—moving beyond traditional hand-crafted heuristic tables.

---

## 📸 Interface & Training Overview

*(Tip: Upload screenshots of your GUI and terminal output to a `docs/` folder in your repo and link them here!)*

| Interactive PyGame GUI | Self-Play RL Training |
| :---: | :---: |
| ![GUI Preview](docs/gui_preview.png) | ![Terminal Training](docs/training_preview.png) |
| *Play live against the neural network engine* | *Real-time self-play training loop* |

---

## 🔑 Key Features

* **14-Channel Spatial Representation:** Encodes piece locations, active player turn, and castling rights into a tensor grid of size $8 \times 8 \times 14$ for spatial feature extraction.
* **Deep Value Network:** Built with multi-layer 2D Convolutional layers (`nn.Conv2d`) to extract complex spatial board dynamics like pin tactics, pawn structures, and king safety.
* **Self-Play RL Pipeline:** Trains autonomously via Deep Q-Learning principles using experience replay buffers and dynamic $\epsilon$-greedy exploration decay.
* **Minimax Search Engine:** Blends deep neural network evaluations with tactical depth-first lookahead search.
* **Custom PyGame GUI:** Clean, interactive user interface allowing human vs. AI matches with real-time move validation and status tracking.

---

## 🧠 Brain Architecture & Board Representation

Rather than feeding raw FEN strings or basic piece counts to a fully connected network, **Scazy** transforms the standard $8 \times 8$ board into a **14-channel binary tensor matrix**:

$$\text{State Tensor Shape} = (14, 8, 8)$$

### Channel Breakdown
* **Channels 0–5:** White pieces ($\text{Pawn}, \text{Knight}, \text{Bishop}, \text{Rook}, \text{Queen}, \text{King}$)
* **Channels 6–11:** Black pieces ($\text{Pawn}, \text{Knight}, \text{Bishop}, \text{Rook}, \text{Queen}, \text{King}$)
* **Channel 12:** Turn indicator ($1.0$ for White, $0.0$ for Black)
* **Channel 13:** Castling rights encoding

```text
[Board State] ---> [14x8x8 Tensor] ---> Conv2D (64) ---> Conv2D (128) ---> Conv2D (128) ---> FC (512) ---> FC (1) [Position Evaluation V(s)]