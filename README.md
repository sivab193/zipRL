# Zippy RL

Zippy RL is an interactive reinforcement-learning lab for **Zip Hamiltonian path puzzles**. It trains a Double Deep Q-Network (Double DQN) from scratch in the browser while exposing the board state, action values, learning metrics, neural-network activity, and episode replays.

The entire training loop runs client-side: puzzle generation, environment stepping, replay-buffer updates, evaluation, checkpointing, and visualization. No backend is required.

## What it includes

- **Live Lab** for watching an agent train on generated 4×4–10×10 boards
- Double DQN with a compact multilayer perceptron, target network, epsilon-greedy exploration, and experience replay
- Generated boards built from guaranteed Hamiltonian paths, randomized walls, and numbered waypoints
- Optional **Solve Till Win** backtracking mode and clean-solve/extra-step tracking
- Held-out evaluation, rolling learning curves, reward/loss metrics, and curriculum progression
- DFS oracle solver for comparing learned behavior with an exact baseline
- 3D neural-network view with activations, Q-values, and weight-update telemetry
- Episode replay with move-level action evaluations
- Browser persistence through IndexedDB plus portable `.ziprl` checkpoint export/import
- Automatic WebGPU detection with a CPU fallback and an in-app compute benchmark

## Getting started

Requirements:

- Node.js 18+ (or Bun)
- A modern browser; WebGPU is optional

```bash
git clone https://github.com/sivab193/zipRL.git
cd zipRL
npm install
npm run dev
```

Open the local URL printed by Vite. With Bun, use `bun install` and `bun run dev`.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Vite development server on port 3000 |
| `npm run build` | Create a production build in `dist/` |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run the TypeScript no-emit check |
| `npm run clean` | Remove generated `dist/` and `server.js` files |

## How training works

Each generated board contains a legal Hamiltonian solution path, but the agent only receives the environment observation and legal-action mask. The Double DQN learns from `(state, action, reward, next state)` transitions sampled from a 20,000-transition replay buffer. A separate target network stabilizes temporal-difference updates.

The default configuration starts at 5×5 with varied waypoint counts, randomized walls, backtracking enabled, and automatic curriculum support. Training settings can be changed from the UI without resetting learned weights. The exact DFS oracle is available for baseline solving and timing comparisons.

## Checkpoints and privacy

Checkpoints are stored in the browser's IndexedDB database and can be exported as `.ziprl` JSON files for portability. Training data stays local to the browser unless you explicitly export a checkpoint. The optional `.env.example` contains configuration placeholders for environments that provide Gemini or hosting integrations; the core training lab does not require an API key or backend.

## Project structure

```text
src/
├── components/   Dashboard, charts, replay, board, and network views
└── rl/           Board generator, environment, solver, DQN, storage, and tests
```

## License

No license has been specified yet.
