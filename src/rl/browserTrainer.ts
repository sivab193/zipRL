import {
  ActionEvaluation,
  CheckpointInfo,
  DashboardSnapshot,
  DIRECTION_DELTAS,
  DIRECTION_NAMES,
  Direction,
  EpisodeMetric,
  EpisodeReplay,
  EvalPoint,
  MoveRecord,
  TrainingConfig,
  TrainingTarget,
  WebGPUStatus,
  ZipBoard,
} from './types';
import { generateZipBoard, getMaxWaypointCount } from './generator';
import { solveZipBoard } from './solver';
import { ZipEnv } from './env';
import { DoubleDQNAgent } from './dqn';
import { WebGPUMlpAccelerator } from './webgpuMlp';
import {
  ZIPRL_FORMAT_ID,
  ZIPRL_FORMAT_VERSION,
  ZiprlCheckpointPayload,
  listBrowserCheckpoints,
  loadCheckpointFromBrowserDb,
  parseZiprlCheckpoint,
  saveCheckpointToBrowserDb,
  triggerZiprlDownload,
} from './storage';

export function checkTargetReached(
  target: TrainingTarget,
  state: { episode: number; rollingWinRate: number; latestEvalWinRate: number; boardSize: number; allowBacktrack: boolean },
): string | null {
  if (!target || target.type === 'none' || target.value <= 0) return null;

  if (target.type === 'episodes' && state.episode >= target.value) {
    const unit = state.allowBacktrack ? 'boards solved' : 'episodes';
    return `Target reached: ${state.episode.toLocaleString()} ${unit} · Training paused & checkpoint saved`;
  }
  if (target.type === 'clean_rate' && state.episode >= 25) {
    const bestPct = Math.max(state.rollingWinRate, state.latestEvalWinRate) * 100;
    if (bestPct >= target.value) {
      const label = state.allowBacktrack ? 'clean solve rate' : 'win rate';
      return `Target reached: ${bestPct.toFixed(1)}% ${label} (goal ${target.value}%) · Training paused & checkpoint saved`;
    }
  }
  if (target.type === 'grid_level' && state.boardSize >= target.value) {
    return `Target reached: graduated to ${state.boardSize}×${state.boardSize} grid · Training paused & checkpoint saved`;
  }
  return null;
}

const DEFAULT_CONFIG: TrainingConfig = {
  boardSize: 5,
  waypointCount: 0, // 0 = Varied Zip range (e.g., 4–11 numbers on 5×5)
  wallDensity: 0.28,
  stepDelayMs: 40,
  allowBacktrack: true,
  autoCurriculum: true,
  curriculumThreshold: 0.8,
  checkpointInterval: 1000,
  target: { type: 'none', value: 500 },
  preferredCompute: 'auto',
  batchSize: 32,
  learningRate: 0.0012,
  gamma: 0.96,
  epsilonDecay: 0.996,
  epsilonMin: 0.05,
  targetUpdateInterval: 120,
};

/**
 * 100% Client-Side Browser Training Engine for Zippy RL.
 * Runs the entire Double DQN training loop, held-out evaluation, curriculum,
 * episode replay buffer, IndexedDB checkpoint persistence, and portable .ziprl
 * file export/import directly inside the browser with zero backend required.
 */
export class BrowserZipTrainer {
  private env: ZipEnv;
  private agent: DoubleDQNAgent;
  private config: TrainingConfig;

  private running = false;
  private loopTimer: number | null = null;
  private episode = 0;
  private totalSolved = 0;
  private totalDeadEnds = 0;
  private totalCleanSolves = 0;
  private bestWinRate = 0;
  private latestEvalWinRate = 0;

  private currentObs!: Float32Array;
  private currentEpisodeMoves: MoveRecord[] = [];
  private oracleSolution: Array<{ row: number; col: number }> | null = null;
  private oracleSolveMs: number | null = null;

  private metrics: EpisodeMetric[] = [];
  private evalHistory: EvalPoint[] = [];
  private recentReplays: EpisodeReplay[] = [];
  private evalBoardsBySize = new Map<number, ZipBoard[]>();

  private activeCheckpoint: CheckpointInfo | null = null;
  private cachedCheckpoints: CheckpointInfo[] = [];
  private listeners = new Set<(snapshot: DashboardSnapshot) => void>();
  private lastEmitTime = 0;
  private totalEnvSteps = 0;
  private webgpuAccel: WebGPUMlpAccelerator = WebGPUMlpAccelerator.createCpuFallback(
    'Detecting browser WebGPU capability…',
  );
  private targetReachedMessage: string | null = null;

  constructor() {
    this.config = { ...DEFAULT_CONFIG, target: { ...DEFAULT_CONFIG.target } };
    this.env = new ZipEnv({
      size: this.config.boardSize,
      waypointCount: this.config.waypointCount,
      wallDensity: this.config.wallDensity,
      allowBacktrack: this.config.allowBacktrack,
    });
    this.agent = this.createAgentForEnv(this.env);
    this.resetLiveEpisode();
  }

  /**
   * Initializes WebGPU compute shaders (when supported in this browser) and
   * auto-resumes the latest compatible IndexedDB checkpoint if one exists locally.
   */
  public async init(): Promise<void> {
    const [accel, checkpoints] = await Promise.all([
      WebGPUMlpAccelerator.detectAndInit(),
      listBrowserCheckpoints(),
    ]);
    this.webgpuAccel = accel;
    this.cachedCheckpoints = checkpoints;
    if (this.cachedCheckpoints.length > 0) {
      const latest = this.cachedCheckpoints[0];
      try {
        await this.loadCheckpointById(latest.id);
      } catch {
        // Ignore incompatible legacy checkpoint
      }
    }
    this.emit(true);
  }

  public getWebGPUStatus(): WebGPUStatus {
    const raw = this.webgpuAccel.status;
    if (this.config.preferredCompute === 'cpu' && raw.supported) {
      return {
        supported: true,
        active: false,
        adapterName: raw.adapterName,
        reason: 'WebGPU adapter detected, currently forced to Float32 CPU by user preference.',
      };
    }
    return raw;
  }

  public async runWebGPUComputeBenchmark(batchSize = 128): Promise<{
    batchSize: number;
    cpuMs: number;
    gpuMs: number | null;
    maxDiff: number | null;
    usedWebGPU: boolean;
  }> {
    const states: Float32Array[] = [];
    for (let i = 0; i < batchSize; i++) {
      states.push(this.env.getObservation());
    }
    const tCpu0 = performance.now();
    const cpuOuts = states.map((s) => this.agent.onlineNet.predict(s));
    const cpuMs = Math.round((performance.now() - tCpu0) * 100) / 100;

    if (!this.webgpuAccel.isReady() || this.config.preferredCompute === 'cpu') {
      return { batchSize, cpuMs, gpuMs: null, maxDiff: null, usedWebGPU: false };
    }

    const tGpu0 = performance.now();
    const gpuOuts = await this.webgpuAccel.predictBatchMLP(this.agent.onlineNet, states);
    const gpuMs = Math.round((performance.now() - tGpu0) * 100) / 100;

    let maxDiff = 0;
    for (let b = 0; b < batchSize; b++) {
      for (let a = 0; a < 4; a++) {
        const diff = Math.abs(cpuOuts[b][a] - gpuOuts[b][a]);
        if (diff > maxDiff) maxDiff = diff;
      }
    }
    return {
      batchSize,
      cpuMs,
      gpuMs,
      maxDiff: Math.round(maxDiff * 1e6) / 1e6,
      usedWebGPU: true,
    };
  }

  private createAgentForEnv(env: ZipEnv): DoubleDQNAgent {
    return new DoubleDQNAgent({
      inputSize: env.obsSize,
      hidden1Size: 128,
      hidden2Size: 64,
      learningRate: this.config.learningRate,
      gamma: this.config.gamma,
      epsilonStart: 1.0,
      epsilonMin: this.config.epsilonMin,
      epsilonDecay: this.config.epsilonDecay,
      batchSize: this.config.batchSize,
      replayCapacity: 20000,
      targetUpdateInterval: this.config.targetUpdateInterval,
    });
  }

  private getEvalBoards(size: number): ZipBoard[] {
    const existing = this.evalBoardsBySize.get(size);
    if (existing) return existing;

    const boards: ZipBoard[] = [];
    for (let i = 0; i < 50; i++) {
      boards.push(
        generateZipBoard({
          size,
          waypointCount: this.config.waypointCount,
          wallDensity: this.config.wallDensity,
          seed: size * 100000 + i * 7919 + 42,
        }),
      );
    }
    this.evalBoardsBySize.set(size, boards);
    return boards;
  }

  private resetLiveEpisode(customBoard?: ZipBoard): void {
    const { observation } = this.env.reset(customBoard);
    this.currentObs = observation;
    this.currentEpisodeMoves = [];
    this.oracleSolution = null;
    this.oracleSolveMs = null;
    const legal = this.env.getLegalActions();
    if (legal.length > 0) {
      this.agent.selectAction(this.currentObs, legal, true);
    }
  }

  public subscribe(listener: (snapshot: DashboardSnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private emit(force = false): void {
    const now = performance.now();
    if (!force && now - this.lastEmitTime < 65) return;
    this.lastEmitTime = now;
    const snap = this.getSnapshot();
    for (const listener of this.listeners) {
      listener(snap);
    }
  }

  public start(): void {
    if (this.running) return;
    this.targetReachedMessage = null;
    this.running = true;
    this.emit(true);
    this.scheduleLoop();
  }

  public clearTargetNotice(): void {
    this.targetReachedMessage = null;
    this.emit(true);
  }

  public pause(): void {
    this.running = false;
    if (this.loopTimer !== null) {
      window.clearTimeout(this.loopTimer);
      this.loopTimer = null;
    }
    this.emit(true);
  }

  private scheduleLoop(): void {
    if (!this.running) return;
    const delay = Math.max(0, this.config.stepDelayMs);

    this.loopTimer = window.setTimeout(() => {
      if (!this.running) return;
      if (delay === 0) {
        // Burst mode: run a fast slice of steps while keeping UI frames smooth
        const sliceStart = performance.now();
        for (let i = 0; i < 60 && this.running; i++) {
          this.executeSingleStep(true);
          if (performance.now() - sliceStart > 16) break;
        }
        this.emit(false);
      } else {
        this.executeSingleStep(false);
        this.emit(true);
      }
      this.scheduleLoop();
    }, delay === 0 ? 10 : delay);
  }

  /**
   * Runs a chunked non-blocking block of N episodes in the browser, then evaluates
   * on held-out boards and saves a checkpoint to IndexedDB.
   */
  public async runEpisodeBlockAsync(
    count: number,
  ): Promise<{ episodesCompleted: number; evalWinRate: number }> {
    const wasRunning = this.running;
    if (wasRunning) this.pause();

    const startEp = this.episode;
    const targetEp = startEp + count;
    // Generous safety cap so even an untrained large board won't stall a batch button indefinitely,
    // while 5x5–7x7 Solve-Till-Win episodes run cleanly to completion.
    const maxBlockSteps = this.config.allowBacktrack
      ? count * Math.max(600, this.env.totalCells * 40)
      : count * (this.env.totalCells * 2) + 200;
    let stepsExecuted = 0;
    let lastProgressEmit = performance.now();

    this.targetReachedMessage = null;
    while (this.episode < targetEp && stepsExecuted < maxBlockSteps && !this.targetReachedMessage) {
      const sliceStart = performance.now();
      while (
        this.episode < targetEp &&
        stepsExecuted < maxBlockSteps &&
        !this.targetReachedMessage &&
        performance.now() - sliceStart < 30
      ) {
        this.executeSingleStep(true);
        stepsExecuted++;
      }
      const now = performance.now();
      if (now - lastProgressEmit > 250) {
        lastProgressEmit = now;
        this.emit(true);
      }
      // Yield to browser paint & event queue
      await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
    }

    const evalPt = this.runGreedyEvaluation();
    await this.saveCheckpoint();
    this.emit(true);

    if (wasRunning && !this.targetReachedMessage) this.start();

    return {
      episodesCompleted: this.episode - startEp,
      evalWinRate: evalPt.winRate,
    };
  }

  public executeSingleStep(fastBatchMode = false): void {
    if (this.env.done) {
      this.resetLiveEpisode();
    }

    const legal = this.env.getLegalActions();
    if (legal.length === 0) {
      this.resetLiveEpisode();
      return;
    }

    const head = this.env.getHead();
    const backtrackDir = this.env.getBacktrackDirection();
    const { action, exploratory, qValues } = this.agent.selectAction(this.currentObs, legal, false);

    const evaluations: ActionEvaluation[] = ([0, 1, 2, 3] as Direction[]).map((dir) => {
      const [dr, dc] = DIRECTION_DELTAS[dir];
      return {
        direction: dir,
        label: DIRECTION_NAMES[dir],
        targetRow: head.row + dr,
        targetCol: head.col + dc,
        qValue: Math.round(qValues[dir] * 1000) / 1000,
        legal: legal.includes(dir),
        chosen: dir === action,
        isBacktrack: backtrackDir !== null && dir === backtrackDir,
      };
    });

    const stepRes = this.env.step(action);
    const nextLegalMask = this.env.getLegalMask();

    const moveRecord: MoveRecord = {
      step: stepRes.step,
      from: { row: head.row, col: head.col },
      to: { row: stepRes.head.row, col: stepRes.head.col },
      direction: action,
      reward: stepRes.reward,
      cumulativeReward: this.env.cumulativeReward,
      exploratory,
      isBacktrack: stepRes.isBacktrack,
      epsilon: Math.round(this.agent.epsilon * 10000) / 10000,
      evaluations,
      nextWaypointOrder: stepRes.nextWaypointOrder,
      outcome: stepRes.outcome,
    };
    this.currentEpisodeMoves.push(moveRecord);

    this.agent.storeTransition({
      state: this.currentObs,
      action,
      reward: stepRes.reward,
      nextState: stepRes.observation,
      nextLegalMask,
      done: stepRes.done,
      solvedTransition: stepRes.solved,
    });

    this.currentObs = stepRes.observation;
    this.totalEnvSteps += 1;

    // Honest RL mechanics: train on replay batch every step in live inspection mode,
    // or every 40 env steps during fast burst/block training — always updating immediately on solve.
    if (!fastBatchMode || stepRes.solved || this.totalEnvSteps % 40 === 0) {
      this.agent.trainStep(stepRes.solved);
    }

    if (stepRes.done) {
      this.finalizeEpisode();
    }
  }

  public stepOnceAndEmit(): void {
    this.executeSingleStep();
    this.emit(true);
  }

  private finalizeEpisode(): void {
    this.episode += 1;
    const cleanSolve = this.env.solved && this.env.backtrackCount === 0;
    if (this.env.solved) this.totalSolved += 1;
    if (this.env.deadEnd) this.totalDeadEnds += 1;
    if (cleanSolve) this.totalCleanSolves += 1;

    this.agent.decayEpsilon();

    const windowSize = 50;
    const recentSlice = this.metrics.slice(-(windowSize - 1));
    const currentWinFlag = this.config.allowBacktrack ? cleanSolve : this.env.solved;
    const solvedCount =
      recentSlice.filter((m) => (this.config.allowBacktrack ? (m.cleanSolve ?? m.solved) : m.solved)).length +
      (currentWinFlag ? 1 : 0);
    const denom = recentSlice.length + 1;
    const rollingWinRate = solvedCount / denom;
    const rollingReward =
      (recentSlice.reduce((s, m) => s + m.reward, 0) + this.env.cumulativeReward) / denom;
    const rollingSteps =
      (recentSlice.reduce((s, m) => s + m.steps, 0) + this.env.stepCount) / denom;
    const rollingExtraSteps =
      (recentSlice.reduce((s, m) => s + (m.extraSteps ?? 0), 0) + this.env.extraSteps) / denom;
    const rollingBacktracks =
      (recentSlice.reduce((s, m) => s + (m.backtracks ?? 0), 0) + this.env.backtrackCount) / denom;

    if (rollingWinRate > this.bestWinRate && this.episode >= 10) {
      this.bestWinRate = rollingWinRate;
    }

    const metric: EpisodeMetric = {
      episode: this.episode,
      reward: this.env.cumulativeReward,
      solved: this.env.solved,
      deadEnd: this.env.deadEnd,
      cleanSolve,
      steps: this.env.stepCount,
      extraSteps: this.env.extraSteps,
      backtracks: this.env.backtrackCount,
      epsilon: Math.round(this.agent.epsilon * 10000) / 10000,
      boardSize: this.env.size,
      rollingWinRate: Math.round(rollingWinRate * 10000) / 10000,
      rollingReward: Math.round(rollingReward * 1000) / 1000,
      rollingSteps: Math.round(rollingSteps * 100) / 100,
      rollingExtraSteps: Math.round(rollingExtraSteps * 100) / 100,
      rollingBacktracks: Math.round(rollingBacktracks * 100) / 100,
      loss: Math.round(this.agent.lastLoss * 10000) / 10000,
    };

    this.metrics.push(metric);
    if (this.metrics.length > 2000) {
      this.metrics = this.metrics.slice(-2000);
    }

    const replay: EpisodeReplay = {
      episode: this.episode,
      timestamp: new Date().toISOString(),
      board: this.env.board,
      moves: this.currentEpisodeMoves.length <= 2000
        ? [...this.currentEpisodeMoves]
        : this.currentEpisodeMoves.slice(0, 2000),
      solved: this.env.solved,
      deadEnd: this.env.deadEnd,
      cleanSolve,
      steps: this.env.stepCount,
      optimalSteps: this.env.optimalSteps,
      extraSteps: this.env.extraSteps,
      backtracks: this.env.backtrackCount,
      totalReward: this.env.cumulativeReward,
      epsilon: metric.epsilon,
      curriculumSize: this.env.size,
    };

    this.recentReplays.unshift(replay);
    if (this.recentReplays.length > 100) {
      this.recentReplays.pop();
    }

    // Periodic greedy eval every 100 episodes
    if (this.episode % 100 === 0) {
      const evalPoint = this.runGreedyEvaluation();
      if (evalPoint.winRate > this.bestWinRate) {
        this.bestWinRate = evalPoint.winRate;
      }
      if (
        this.config.autoCurriculum &&
        evalPoint.winRate >= this.config.curriculumThreshold &&
        this.config.boardSize < 10
      ) {
        void this.saveCheckpoint();
        this.setBoardSize(this.config.boardSize + 1);
        this.evaluateTargetStop();
        return;
      }
    }

    // Customizable auto-checkpoint interval into IndexedDB (e.g., every 1k or 10k episodes)
    if (this.config.checkpointInterval > 0 && this.episode % this.config.checkpointInterval === 0) {
      void this.saveCheckpoint();
    }

    this.evaluateTargetStop();
  }

  private evaluateTargetStop(): void {
    const latestMetric = this.metrics[this.metrics.length - 1];
    const msg = checkTargetReached(this.config.target, {
      episode: this.episode,
      rollingWinRate: latestMetric ? latestMetric.rollingWinRate : 0,
      latestEvalWinRate: this.latestEvalWinRate,
      boardSize: this.config.boardSize,
      allowBacktrack: this.config.allowBacktrack,
    });
    if (msg) {
      this.targetReachedMessage = msg;
      void this.saveCheckpoint();
      if (this.running) {
        this.pause();
      } else {
        this.emit(true);
      }
    }
  }

  public runGreedyEvaluation(): EvalPoint {
    const evalBoards = this.getEvalBoards(this.env.size);
    const evalEnv = new ZipEnv({
      size: this.env.size,
      waypointCount: this.config.waypointCount,
      wallDensity: this.config.wallDensity,
      allowBacktrack: this.config.allowBacktrack,
    });

    let winCount = 0;
    let totalReward = 0;
    let totalSteps = 0;
    let totalExtraSteps = 0;
    let totalBacktracks = 0;
    const perBoardEvalGuard = Math.max(400, evalEnv.totalCells * 24);

    for (let i = 0; i < evalBoards.length; i++) {
      let { observation } = evalEnv.reset(evalBoards[i]);
      let guardSteps = 0;
      while (!evalEnv.done && guardSteps < perBoardEvalGuard) {
        const legal = evalEnv.getLegalActions();
        if (legal.length === 0) break;
        const { action } = this.agent.selectAction(observation, legal, true);
        const step = evalEnv.step(action);
        observation = step.observation;
        guardSteps++;
      }
      const isWin = this.config.allowBacktrack
        ? evalEnv.solved && evalEnv.backtrackCount === 0
        : evalEnv.solved;
      if (isWin) winCount += 1;
      totalReward += evalEnv.cumulativeReward;
      totalSteps += evalEnv.stepCount;
      totalExtraSteps += evalEnv.extraSteps;
      totalBacktracks += evalEnv.backtrackCount;
    }

    const winRate = winCount / evalBoards.length;
    this.latestEvalWinRate = Math.round(winRate * 10000) / 10000;
    if (this.latestEvalWinRate > this.bestWinRate) {
      this.bestWinRate = this.latestEvalWinRate;
    }

    const pt: EvalPoint = {
      episode: this.episode,
      winRate: this.latestEvalWinRate,
      avgReward: Math.round((totalReward / evalBoards.length) * 1000) / 1000,
      avgSteps: Math.round((totalSteps / evalBoards.length) * 100) / 100,
      avgExtraSteps: Math.round((totalExtraSteps / evalBoards.length) * 100) / 100,
      avgBacktracks: Math.round((totalBacktracks / evalBoards.length) * 100) / 100,
      boardSize: this.env.size,
      evalCount: evalBoards.length,
    };

    this.evalHistory.push(pt);
    if (this.evalHistory.length > 200) {
      this.evalHistory = this.evalHistory.slice(-200);
    }

    return pt;
  }

  public newPuzzle(): void {
    this.resetLiveEpisode();
    this.emit(true);
  }

  public runOracleOnCurrentBoard(): { solved: boolean; path: Array<{ row: number; col: number }>; elapsedMs: number } {
    const res = solveZipBoard(this.env.board);
    this.oracleSolution = res.solved ? res.path : null;
    this.oracleSolveMs = res.elapsedMs;
    this.emit(true);
    return { solved: res.solved, path: res.path, elapsedMs: res.elapsedMs };
  }

  public setBoardSize(newSize: number): void {
    const clamped = Math.max(4, Math.min(10, Math.round(newSize)));
    if (clamped === this.config.boardSize && this.env.size === clamped) return;

    this.config.boardSize = clamped;
    if (this.config.waypointCount > 0) {
      this.config.waypointCount = Math.max(2, Math.min(getMaxWaypointCount(clamped), this.config.waypointCount));
    }
    this.env = new ZipEnv({
      size: clamped,
      waypointCount: this.config.waypointCount,
      wallDensity: this.config.wallDensity,
      allowBacktrack: this.config.allowBacktrack,
    });
    this.agent = this.createAgentForEnv(this.env);
    this.latestEvalWinRate = 0;
    this.evalBoardsBySize.delete(clamped);
    this.resetLiveEpisode();
    this.emit(true);
  }

  public updateSettings(partial: Partial<TrainingConfig>): void {
    if (partial.boardSize !== undefined && partial.boardSize !== this.config.boardSize) {
      this.setBoardSize(partial.boardSize);
    }
    if (partial.waypointCount !== undefined) {
      const raw = Math.round(Number(partial.waypointCount));
      const maxWp = getMaxWaypointCount(this.config.boardSize);
      const nextWp = raw <= 0 ? 0 : Math.max(2, Math.min(maxWp, raw));
      if (nextWp !== this.config.waypointCount) {
        this.config.waypointCount = nextWp;
        this.evalBoardsBySize.delete(this.config.boardSize);
        this.env = new ZipEnv({
          size: this.config.boardSize,
          waypointCount: this.config.waypointCount,
          wallDensity: this.config.wallDensity,
          allowBacktrack: this.config.allowBacktrack,
        });
        this.resetLiveEpisode();
      }
    }
    if (partial.stepDelayMs !== undefined) {
      this.config.stepDelayMs = Math.max(0, Math.min(1000, Math.round(partial.stepDelayMs)));
    }
    if (partial.allowBacktrack !== undefined) {
      this.config.allowBacktrack = Boolean(partial.allowBacktrack);
      this.env.allowBacktrack = this.config.allowBacktrack;
      if (this.config.allowBacktrack && this.env.deadEnd && !this.env.solved) {
        this.env.reviveFromDeadEnd();
      }
    }
    if (partial.wallDensity !== undefined) {
      this.config.wallDensity = Math.max(0, Math.min(0.6, Number(partial.wallDensity)));
      this.evalBoardsBySize.delete(this.config.boardSize);
      this.env = new ZipEnv({
        size: this.config.boardSize,
        waypointCount: this.config.waypointCount,
        wallDensity: this.config.wallDensity,
        allowBacktrack: this.config.allowBacktrack,
      });
    }
    if (partial.autoCurriculum !== undefined) {
      this.config.autoCurriculum = Boolean(partial.autoCurriculum);
    }
    if (partial.checkpointInterval !== undefined) {
      const interval = Math.max(0, Math.min(100000, Math.round(Number(partial.checkpointInterval))));
      this.config.checkpointInterval = interval;
    }
    if (partial.target !== undefined) {
      this.config.target = {
        type: partial.target.type,
        value: Math.max(1, Math.round(Number(partial.target.value) || 1)),
      };
      this.targetReachedMessage = null;
    }
    if (partial.preferredCompute !== undefined) {
      this.config.preferredCompute = partial.preferredCompute === 'cpu' ? 'cpu' : 'auto';
    }
    if (partial.learningRate !== undefined) {
      this.config.learningRate = Math.max(0.0001, Math.min(0.05, Number(partial.learningRate)));
      this.agent.learningRate = this.config.learningRate;
    }
    this.emit(true);
  }

  public resetModel(): void {
    this.pause();
    this.episode = 0;
    this.totalSolved = 0;
    this.totalDeadEnds = 0;
    this.totalCleanSolves = 0;
    this.bestWinRate = 0;
    this.latestEvalWinRate = 0;
    this.targetReachedMessage = null;
    this.metrics = [];
    this.evalHistory = [];
    this.recentReplays = [];
    this.activeCheckpoint = null;
    this.env = new ZipEnv({
      size: this.config.boardSize,
      waypointCount: this.config.waypointCount,
      wallDensity: this.config.wallDensity,
      allowBacktrack: this.config.allowBacktrack,
    });
    this.agent = this.createAgentForEnv(this.env);
    this.resetLiveEpisode();
    this.emit(true);
  }

  public buildCheckpointPayload(): ZiprlCheckpointPayload {
    const timestamp = new Date().toISOString();
    const id = `zippy-${this.env.size}x${this.env.size}-ep${this.episode}-${Date.now()}`;
    const filename = `zippy-${this.env.size}x${this.env.size}-ep${this.episode}.ziprl`;
    const rollingWinRate =
      this.metrics.length > 0 ? this.metrics[this.metrics.length - 1].rollingWinRate : 0;

    const info: CheckpointInfo = {
      id,
      filename,
      createdAt: timestamp,
      episode: this.episode,
      boardSize: this.env.size,
      evalWinRate: this.latestEvalWinRate,
      rollingWinRate,
      epsilon: Math.round(this.agent.epsilon * 10000) / 10000,
      totalSolved: this.totalSolved,
    };

    return {
      format: ZIPRL_FORMAT_ID,
      version: ZIPRL_FORMAT_VERSION,
      info,
      config: { ...this.config },
      agent: this.agent.serialize(),
      episode: this.episode,
      totalSolved: this.totalSolved,
      totalDeadEnds: this.totalDeadEnds,
      bestWinRate: this.bestWinRate,
      latestEvalWinRate: this.latestEvalWinRate,
      metrics: this.metrics.slice(-1000),
      evalHistory: this.evalHistory.slice(-100),
    };
  }

  public async saveCheckpoint(): Promise<CheckpointInfo> {
    const payload = this.buildCheckpointPayload();
    await saveCheckpointToBrowserDb(payload);
    this.activeCheckpoint = payload.info;
    this.cachedCheckpoints = await listBrowserCheckpoints();
    this.emit(true);
    return payload.info;
  }

  public async downloadCheckpoint(id?: string): Promise<CheckpointInfo> {
    let payload: ZiprlCheckpointPayload | null = null;
    if (id) {
      payload = await loadCheckpointFromBrowserDb(id);
    }
    if (!payload) {
      payload = this.buildCheckpointPayload();
      await saveCheckpointToBrowserDb(payload);
      this.activeCheckpoint = payload.info;
      this.cachedCheckpoints = await listBrowserCheckpoints();
      this.emit(true);
    }
    triggerZiprlDownload(payload);
    return payload.info;
  }

  public applyCheckpointPayload(payload: ZiprlCheckpointPayload): CheckpointInfo {
    this.pause();
    const mergedConfig: TrainingConfig = {
      ...DEFAULT_CONFIG,
      ...payload.config,
      target: payload.config?.target ? { ...payload.config.target } : { ...DEFAULT_CONFIG.target },
    };
    const nextEnv = new ZipEnv({
      size: mergedConfig.boardSize,
      waypointCount: mergedConfig.waypointCount,
      wallDensity: mergedConfig.wallDensity,
      allowBacktrack: mergedConfig.allowBacktrack,
    });
    if (payload.agent.inputSize !== nextEnv.obsSize) {
      throw new Error(
        `Checkpoint observation dimension (${payload.agent.inputSize}) is incompatible with ${mergedConfig.boardSize}×${mergedConfig.boardSize} (${nextEnv.obsSize}).`,
      );
    }

    this.config = mergedConfig;
    this.env = nextEnv;
    this.agent = this.createAgentForEnv(this.env);
    this.agent.loadSerialized(payload.agent);
    this.episode = payload.episode ?? 0;
    this.totalSolved = payload.totalSolved ?? 0;
    this.totalDeadEnds = payload.totalDeadEnds ?? 0;
    this.bestWinRate = payload.bestWinRate ?? 0;
    this.latestEvalWinRate = payload.latestEvalWinRate ?? 0;
    this.metrics = payload.metrics ?? [];
    this.totalCleanSolves = this.metrics.filter((m) => m.cleanSolve ?? (m.solved && (m.backtracks ?? 0) === 0)).length;
    this.evalHistory = payload.evalHistory ?? [];
    this.activeCheckpoint = payload.info;
    this.targetReachedMessage = null;
    this.resetLiveEpisode();
    this.emit(true);
    return payload.info;
  }

  public async loadCheckpointById(id: string): Promise<CheckpointInfo> {
    const payload = await loadCheckpointFromBrowserDb(id);
    if (!payload) {
      throw new Error('Checkpoint not found in browser storage.');
    }
    return this.applyCheckpointPayload(payload);
  }

  public async importZiprlText(rawText: string): Promise<CheckpointInfo> {
    const payload = parseZiprlCheckpoint(rawText);
    const info = this.applyCheckpointPayload(payload);
    await saveCheckpointToBrowserDb(payload);
    this.cachedCheckpoints = await listBrowserCheckpoints();
    this.emit(true);
    return info;
  }

  public getEpisodeReplay(episodeNum: number): EpisodeReplay | null {
    return this.recentReplays.find((r) => r.episode === episodeNum) ?? null;
  }

  public getSnapshot(): DashboardSnapshot {
    const latestMetric = this.metrics[this.metrics.length - 1];
    const webgpu = this.getWebGPUStatus();
    const deviceLabel = webgpu.active
      ? `WebGPU Compute (${webgpu.adapterName}) + Float32`
      : 'In-Browser CPU (Float32 Double DQN)';
    return {
      running: this.running,
      episode: this.episode,
      totalSolved: this.totalSolved,
      totalDeadEnds: this.totalDeadEnds,
      totalCleanSolves: this.totalCleanSolves,
      rollingWinRate: latestMetric ? latestMetric.rollingWinRate : 0,
      rollingExtraSteps: latestMetric?.rollingExtraSteps ?? 0,
      rollingBacktracks: latestMetric?.rollingBacktracks ?? 0,
      bestWinRate: this.bestWinRate,
      latestEvalWinRate: this.latestEvalWinRate,
      epsilon: Math.round(this.agent.epsilon * 10000) / 10000,
      replayBufferSize: this.agent.replay.size,
      replayBufferCapacity: this.agent.replay.capacity,
      deviceLabel,
      webgpu,
      targetReachedMessage: this.targetReachedMessage,
      activeCheckpoint: this.activeCheckpoint,
      config: { ...this.config, target: { ...this.config.target } },
      liveBoard: {
        board: this.env.board,
        path: [...this.env.path],
        nextWaypointOrder: this.env.nextWaypointOrder,
        step: this.env.stepCount,
        optimalSteps: this.env.optimalSteps,
        extraSteps: this.env.extraSteps,
        backtracks: this.env.backtrackCount,
        done: this.env.done,
        solved: this.env.solved,
        deadEnd: this.env.deadEnd,
        cumulativeReward: this.env.cumulativeReward,
        oracleSolution: this.oracleSolution,
        oracleSolveMs: this.oracleSolveMs,
      },
      recentMoves: this.currentEpisodeMoves.slice(-24),
      metrics: this.metrics.slice(-400),
      evalHistory: this.evalHistory,
      network: this.agent.getVisualSnapshot(),
      recentEpisodes: this.recentReplays.slice(0, 60).map((r) => {
        const opt = r.optimalSteps ?? r.board.size * r.board.size - 1;
        return {
          episode: r.episode,
          boardSize: r.board.size,
          solved: r.solved,
          deadEnd: r.deadEnd,
          cleanSolve: r.cleanSolve ?? (r.solved && (r.backtracks ?? 0) === 0),
          steps: r.steps,
          optimalSteps: opt,
          extraSteps: r.extraSteps ?? Math.max(0, r.steps - opt),
          backtracks: r.backtracks ?? 0,
          totalReward: r.totalReward,
          epsilon: r.epsilon,
          timestamp: r.timestamp,
        };
      }),
      checkpoints: this.cachedCheckpoints,
    };
  }
}
