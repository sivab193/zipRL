export type Direction = 0 | 1 | 2 | 3; // 0: Up, 1: Right, 2: Down, 3: Left

export const DIRECTION_NAMES: Record<Direction, string> = {
  0: 'Up',
  1: 'Right',
  2: 'Down',
  3: 'Left',
};

export const DIRECTION_DELTAS: Record<Direction, readonly [number, number]> = {
  0: [-1, 0],
  1: [0, 1],
  2: [1, 0],
  3: [0, -1],
};

export interface CellCoord {
  row: number;
  col: number;
}

/**
 * Wall edge between two orthogonally adjacent cells.
 * Stored canonically with (r1 < r2) or (r1 === r2 && c1 < c2).
 */
export interface WallEdge {
  r1: number;
  c1: number;
  r2: number;
  c2: number;
}

export interface Waypoint {
  order: number; // 1-indexed ascending waypoint number
  row: number;
  col: number;
}

export interface ZipBoard {
  id: string;
  size: number;
  waypoints: Waypoint[];
  walls: WallEdge[];
  /** Guaranteed Hamiltonian solution generated during board construction. */
  solutionPath: CellCoord[];
  wallDensity: number;
}

export interface ActionEvaluation {
  direction: Direction;
  label: string;
  targetRow: number;
  targetCol: number;
  qValue: number;
  legal: boolean;
  chosen: boolean;
  isBacktrack?: boolean;
}

export interface MoveRecord {
  step: number;
  from: CellCoord;
  to: CellCoord;
  direction: Direction;
  reward: number;
  cumulativeReward: number;
  exploratory: boolean;
  isBacktrack?: boolean;
  epsilon: number;
  evaluations: ActionEvaluation[];
  nextWaypointOrder: number;
  outcome: 'continue' | 'solved' | 'dead_end' | 'step_cap';
}

export interface EpisodeReplay {
  episode: number;
  timestamp: string;
  board: ZipBoard;
  moves: MoveRecord[];
  solved: boolean;
  deadEnd: boolean;
  cleanSolve?: boolean;
  steps: number;
  optimalSteps?: number;
  extraSteps?: number;
  backtracks?: number;
  totalReward: number;
  epsilon: number;
  curriculumSize: number;
}

export interface EpisodeMetric {
  episode: number;
  reward: number;
  solved: boolean;
  deadEnd: boolean;
  cleanSolve?: boolean;
  steps: number;
  extraSteps?: number;
  backtracks?: number;
  epsilon: number;
  boardSize: number;
  rollingWinRate: number;
  rollingReward: number;
  rollingSteps: number;
  rollingExtraSteps?: number;
  rollingBacktracks?: number;
  loss: number;
}

export interface EvalPoint {
  episode: number;
  winRate: number;
  avgReward: number;
  avgSteps: number;
  avgExtraSteps?: number;
  avgBacktracks?: number;
  boardSize: number;
  evalCount: number;
}

export interface NetworkVisualSnapshot {
  stepIndex: number;
  inputSize: number;
  hidden1Size: number;
  hidden2Size: number;
  outputSize: number;
  /** Downsampled / representative input activations in [0, 1] for 3D display */
  inputActivations: number[];
  /** Hidden layer 1 post-ReLU normalized activations */
  hidden1Activations: number[];
  /** Hidden layer 2 post-ReLU normalized activations */
  hidden2Activations: number[];
  /** Output Q-values for [Up, Right, Down, Left] */
  outputQValues: number[];
  /** Legal move mask for [Up, Right, Down, Left] */
  legalMask: boolean[];
  /** Representative connection weight update magnitudes [0..1] for layer 1->2 */
  w1DeltaStrengths: number[];
  /** Representative connection weight update magnitudes [0..1] for layer 2->3 */
  w2DeltaStrengths: number[];
  /** All hidden2 -> output weight update magnitudes [0..1] */
  w3DeltaStrengths: number[];
  /** Raw mean absolute gradient update magnitude on this training step */
  meanWeightDelta: number;
  /** Max weight delta in this batch */
  maxWeightDelta: number;
  /** TD loss on the most recent replay-buffer update step */
  lastLoss: number;
  /** True when the latest replay update follows a solved episode reward burst */
  solveBurstActive: boolean;
  /** Counter of total gradient update steps executed */
  totalGradientSteps: number;
}

export interface CheckpointInfo {
  id: string;
  filename: string;
  createdAt: string;
  episode: number;
  boardSize: number;
  evalWinRate: number;
  rollingWinRate: number;
  epsilon: number;
  totalSolved: number;
}

export type TargetType = 'none' | 'episodes' | 'clean_rate' | 'grid_level';

export interface TrainingTarget {
  type: TargetType;
  value: number;
}

export interface WebGPUStatus {
  supported: boolean;
  active: boolean;
  adapterName: string;
  reason: string;
}

export interface TrainingConfig {
  boardSize: number;
  /** 0 = Varied Zip range per board; >= 2 = fixed waypoint count */
  waypointCount: number;
  wallDensity: number;
  stepDelayMs: number;
  allowBacktrack: boolean;
  autoCurriculum: boolean;
  curriculumThreshold: number;
  /** Auto-save checkpoint every N episodes (0 = disabled). Supports 250, 1k, 5k, 10k, or custom. */
  checkpointInterval: number;
  target: TrainingTarget;
  preferredCompute: 'auto' | 'cpu';
  batchSize: number;
  learningRate: number;
  gamma: number;
  epsilonDecay: number;
  epsilonMin: number;
  targetUpdateInterval: number;
}

export interface LiveBoardState {
  board: ZipBoard;
  path: CellCoord[];
  nextWaypointOrder: number;
  step: number;
  optimalSteps: number;
  extraSteps: number;
  backtracks: number;
  done: boolean;
  solved: boolean;
  deadEnd: boolean;
  cumulativeReward: number;
  oracleSolution: CellCoord[] | null;
  oracleSolveMs: number | null;
}

export interface DashboardSnapshot {
  running: boolean;
  episode: number;
  totalSolved: number;
  totalDeadEnds: number;
  totalCleanSolves: number;
  rollingWinRate: number;
  rollingExtraSteps: number;
  rollingBacktracks: number;
  bestWinRate: number;
  latestEvalWinRate: number;
  epsilon: number;
  replayBufferSize: number;
  replayBufferCapacity: number;
  deviceLabel: string;
  webgpu: WebGPUStatus;
  targetReachedMessage: string | null;
  activeCheckpoint: CheckpointInfo | null;
  config: TrainingConfig;
  liveBoard: LiveBoardState;
  recentMoves: MoveRecord[];
  metrics: EpisodeMetric[];
  evalHistory: EvalPoint[];
  network: NetworkVisualSnapshot;
  recentEpisodes: Array<{
    episode: number;
    boardSize: number;
    solved: boolean;
    deadEnd: boolean;
    cleanSolve: boolean;
    steps: number;
    optimalSteps: number;
    extraSteps: number;
    backtracks: number;
    totalReward: number;
    epsilon: number;
    timestamp: string;
  }>;
  checkpoints: CheckpointInfo[];
}
