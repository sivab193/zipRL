# Zippy RL — Full Build Plan

A Python web app that trains a reinforcement-learning agent **from scratch** to play Zip (the LinkedIn / 8zippy path puzzle), with a live dashboard that shows the puzzle, the agent's moves, its learning curves, and a 3D visualization of the neural network's weights updating as it learns.

**How to use this document:** paste it into Google AI Studio (or any code-generating AI) and ask it to build the complete app. Everything below is the spec — do not invent different rules, rewards, or architecture without flagging the change.

---

## 1. The game — Zip rules (exact, do not change)

- Square grid. Start at **5×5**; later scale to 6×6, 7×7, 8×8, up to 10×10.
- Numbered waypoint dots must be visited in ascending order: 1, 2, 3, … N.
- Draw **one continuous path** starting at 1.
- Moves are **orthogonal only** (up / down / left / right). **NO diagonals.**
- The path must fill **every cell exactly once** (a Hamiltonian path).
- **Walls** sit on edges between adjacent cells and block movement across that edge.
- Never revisit a cell, never cross the path.
- **Win:** reach the last number with all cells filled. **Lose:** dead end (no legal moves) before finishing.

## 2. Board generator

- Sample a random Hamiltonian path covering the whole grid (randomized DFS self-avoiding walk from a random start cell).
- Place waypoints along that path: `1` at the path start, the last number at the path end, 1–3 intermediate waypoints spaced along the path.
- Optionally add walls: pick edges that are **NOT** on the generated path and mark them as walls, so the generated path stays valid.
- Every generated board is solvable **by construction** (the generated path is a valid solution).
- Difficulty knobs: grid size, number of waypoints, wall density.
- **Fresh board every episode** — never train on a fixed set of boards.

## 3. Solver — the oracle / baseline (not the product)

- Plain DFS backtracking solver: solves any board.
- Used to (a) verify generated boards are solvable, (b) provide the **100% baseline** the RL agent is measured against, (c) optionally generate expert demonstrations later.
- The solver is the measuring stick. The RL agent is the product.

## 4. The RL environment

- Gymnasium-style environment (`reset()` / `step()`), or a minimal equivalent with the same shape.
- **State (observation):** grid channels — walls, waypoint numbers, visited path, current head position, next target waypoint.
- **Actions:** 4 directions (up / down / left / right), **masked to legal moves only** — illegal moves are never offered to the agent.
- **Rewards:** +0.1 per valid step, **+10 on solve**, **−1 and episode end on dead end**.
- Episode ends on solve, dead end, or a step cap (e.g. 2× the number of cells).
- **Curriculum:** start on 5×5; move up a size when the eval win rate passes a threshold (e.g. >80% over 100 eval episodes).

## 5. The agent — the brain

- **Double DQN** (Deep Q-Network).
- Network: **small MLP** — flattened board input → hidden layer(s) (e.g. 128 units) → 4 Q-value outputs (one per direction).
- **Small MLP deliberately, not a CNN** — so the 3D weight visualization stays readable neuron-by-neuron.
- Epsilon-greedy exploration with decay (e.g. 1.0 → 0.05).
- Experience replay buffer; target network updated periodically.
- Framework: **PyTorch**.
- **Device auto-detect:** CUDA if available, else MPS (Apple Silicon), else CPU. Must run on CPU — no GPU required. Target machine: Mac mini M4, 16 GB. (Colab GPU is an optional scaling path later, not a requirement.)

## 6. Training loop + checkpointing

- Initial training block: **~1,000 generated games** (episodes).
- After each training block, save a **checkpoint**: model weights, optimizer state, epsilon, episode count, training history.
- On start, **resume from the latest checkpoint if one exists** — every run continues learning from where the last run stopped, never restarts from scratch.
- Periodic greedy evaluation on a fixed held-out set of boards (e.g. every 100 episodes, 50 boards) → eval win rate.
- Log per episode to JSONL: reward, solved?, steps, dead-end?, epsilon.

## 7. The web app — what the user runs

- Python web app. Suggested stack: **FastAPI (or Flask)** backend running the training loop + a REST/WebSocket API, frontend as a single-page app (HTML / CSS / JS, Three.js for the 3D view).
- Run: `python app.py` → starts the server and **opens the dashboard in the browser automatically** (or prints the localhost URL).
- Training runs in a background thread/process so the UI stays live; the UI receives live state via WebSocket (or polling).
- Controls: Start / Pause / Resume training, new puzzle, step speed, board-size selector, reset model, load checkpoint.

## 8. Dashboard views — the whole point (he wants to SEE it learn)

1. **Live board:** the exact current puzzle — grid, walls, waypoints, path drawn so far, current head highlighted. Updates move by move as the agent plays.
2. **Move feed (the "thinking" view):** each move with its Q-values for **all** legal moves — which move it picked and what the alternatives scored.
3. **Learning curves:** rolling win rate, average reward, average steps, epsilon over episodes. Live-updating charts.
4. **3D network visualization:** Three.js 3D view of the MLP (input → hidden → output neurons), connections colored/lit by **weight-update magnitude after each training step**. After a solved episode, highlight the update burst. Rotatable / zoomable.
5. **Stats panel:** episodes played, current win rate, best win rate, boards solved, dead-end rate, current board size, checkpoint info.
6. **Replay:** pick any past episode and replay it step by step with Q-values.

## 9. Honest mechanics note (do not fake this)

- Weights update on **every training step** (each replay-buffer batch), not only when the agent solves a puzzle. The dashboard shows updates live after each step and highlights the larger update burst following a solved episode. Do not present a fake "solve → update" causality.

## 10. Suggested project structure

```
zippy-rl/
  app.py              # entry point: starts server + opens browser
  requirements.txt
  README.md           # how to run
  game/
    generator.py      # board generator (Section 2)
    solver.py         # DFS oracle solver (Section 3)
    env.py            # RL environment (Section 4)
  agent/
    model.py          # DQN network — small MLP (Section 5)
    dqn.py            # Double DQN agent: replay, target net, epsilon
    checkpoint.py     # save / load checkpoints (Section 6)
  training/
    train.py          # training loop + logging + eval (Section 6)
  dashboard/
    server.py         # FastAPI/Flask app + WebSocket live state
    static/           # HTML / CSS / JS frontend, Three.js 3D view
  checkpoints/        # saved model states
  logs/               # per-episode JSONL logs
```

## 11. Milestones (build in this order)

- **M1 — Game core:** generator + solver + env, verified (solver solves 100/100 generated boards).
- **M2 — Learning:** DQN agent + training loop + checkpointing; smoke run shows the win rate climbing on 5×5.
- **M3 — App + live views:** web app running, live board + move feed + learning curves.
- **M4 — 3D + replay:** 3D weight visualization, episode replay, full controls.
- **M5 — Scale:** curriculum to larger boards (6×6 → 10×10).

## 12. What success looks like

- On 5×5, the agent goes from ~0% to a high win rate (target **>80% eval**) within the first few thousand episodes.
- The dashboard makes the learning visible: you can watch a single game, see why it picked each move, and see the network's weights change as it trains.

---

*Plan written Nov 21, 2026. Game rules verified against 8zippy.com (Zip: drag one continuous path, orthogonal moves only, walls block, numbered dots in order, fill every cell). RL approach grounded in standard DQN practice for grid pathfinding (Double DQN, replay buffer, target network, reward shaping).*
