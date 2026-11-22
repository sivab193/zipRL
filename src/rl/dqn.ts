import { Direction, NetworkVisualSnapshot } from './types';
import { BatchSample, SerializedMLP, SmallMLP } from './mlp';

export interface Transition {
  state: Float32Array;
  action: Direction;
  reward: number;
  nextState: Float32Array;
  nextLegalMask: [boolean, boolean, boolean, boolean];
  done: boolean;
  solvedTransition: boolean;
}

export interface DoubleDQNConfig {
  inputSize: number;
  hidden1Size?: number;
  hidden2Size?: number;
  learningRate?: number;
  gamma?: number;
  epsilonStart?: number;
  epsilonMin?: number;
  epsilonDecay?: number;
  batchSize?: number;
  replayCapacity?: number;
  targetUpdateInterval?: number;
}

export interface SerializedAgent {
  inputSize: number;
  epsilon: number;
  totalGradientSteps: number;
  onlineNet: SerializedMLP;
  targetNet: SerializedMLP;
}

export class ReplayBuffer {
  private buffer: Transition[] = [];
  private cursor = 0;
  public readonly capacity: number;

  constructor(capacity = 20000) {
    this.capacity = capacity;
  }

  public get size(): number {
    return this.buffer.length;
  }

  public push(t: Transition): void {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(t);
    } else {
      this.buffer[this.cursor] = t;
    }
    this.cursor = (this.cursor + 1) % this.capacity;
  }

  public sample(batchSize: number, includeLatest = false): Transition[] {
    const n = this.buffer.length;
    if (n === 0) return [];
    const count = Math.min(batchSize, n);
    const out: Transition[] = [];

    if (includeLatest && n > 0) {
      const latestIdx = (this.cursor - 1 + n) % n;
      out.push(this.buffer[latestIdx]);
    }

    while (out.length < count) {
      const idx = Math.floor(Math.random() * n);
      out.push(this.buffer[idx]);
    }
    return out;
  }

  public clear(): void {
    this.buffer = [];
    this.cursor = 0;
  }
}

/**
 * Double DQN agent with legal-action masking, experience replay buffer,
 * periodic target network synchronization, epsilon-greedy decay, and live
 * 3D network snapshot generation.
 */
export class DoubleDQNAgent {
  public onlineNet: SmallMLP;
  public targetNet: SmallMLP;
  public replay: ReplayBuffer;

  public learningRate: number;
  public gamma: number;
  public epsilon: number;
  public epsilonMin: number;
  public epsilonDecay: number;
  public batchSize: number;
  public targetUpdateInterval: number;
  public totalGradientSteps = 0;

  public lastLoss = 0;
  public lastMeanDelta = 0;
  public lastMaxDelta = 0;
  public solveBurstActive = false;
  private lastForwardInput: Float32Array;
  private lastHidden1: Float32Array;
  private lastHidden2: Float32Array;
  private lastQValues: [number, number, number, number] = [0, 0, 0, 0];
  private lastLegalMask: [boolean, boolean, boolean, boolean] = [true, true, true, true];

  constructor(config: DoubleDQNConfig) {
    const h1 = config.hidden1Size ?? 128;
    const h2 = config.hidden2Size ?? 64;
    this.onlineNet = new SmallMLP(config.inputSize, h1, h2, 4);
    this.targetNet = new SmallMLP(config.inputSize, h1, h2, 4);
    this.targetNet.copyWeightsFrom(this.onlineNet);

    this.replay = new ReplayBuffer(config.replayCapacity ?? 20000);
    this.learningRate = config.learningRate ?? 0.0012;
    this.gamma = config.gamma ?? 0.96;
    this.epsilon = config.epsilonStart ?? 1.0;
    this.epsilonMin = config.epsilonMin ?? 0.05;
    this.epsilonDecay = config.epsilonDecay ?? 0.996;
    this.batchSize = config.batchSize ?? 32;
    this.targetUpdateInterval = config.targetUpdateInterval ?? 120;

    this.lastForwardInput = new Float32Array(config.inputSize);
    this.lastHidden1 = new Float32Array(h1);
    this.lastHidden2 = new Float32Array(h2);
  }

  /**
   * Computes Q-values for all 4 directions and picks an action strictly from legalActions.
   */
  public selectAction(
    observation: Float32Array,
    legalActions: Direction[],
    greedy = false,
  ): {
    action: Direction;
    exploratory: boolean;
    qValues: [number, number, number, number];
  } {
    if (legalActions.length === 0) {
      throw new Error('selectAction called with no legal actions');
    }

    const cache = this.onlineNet.forward(observation);
    this.lastForwardInput = observation;
    this.lastHidden1 = cache.a1;
    this.lastHidden2 = cache.a2;

    const qValues: [number, number, number, number] = [
      cache.out[0],
      cache.out[1],
      cache.out[2],
      cache.out[3],
    ];
    this.lastQValues = qValues;
    this.lastLegalMask = [
      legalActions.includes(0),
      legalActions.includes(1),
      legalActions.includes(2),
      legalActions.includes(3),
    ];

    if (!greedy && Math.random() < this.epsilon) {
      const action = legalActions[Math.floor(Math.random() * legalActions.length)];
      return { action, exploratory: true, qValues };
    }

    let bestAction = legalActions[0];
    let bestQ = qValues[bestAction];
    for (let i = 1; i < legalActions.length; i++) {
      const dir = legalActions[i];
      if (qValues[dir] > bestQ) {
        bestQ = qValues[dir];
        bestAction = dir;
      }
    }

    return { action: bestAction, exploratory: false, qValues };
  }

  public storeTransition(t: Transition): void {
    this.replay.push(t);
  }

  /**
   * Runs one Double DQN optimization step from the experience replay buffer.
   * Weights update on every training step once the buffer has at least batchSize samples.
   */
  public trainStep(triggeredBySolve = false): { trained: boolean; loss: number; meanDelta: number; maxDelta: number } {
    if (this.replay.size < Math.min(this.batchSize, 16)) {
      return { trained: false, loss: 0, meanDelta: 0, maxDelta: 0 };
    }

    const batch = this.replay.sample(this.batchSize, triggeredBySolve);
    const samples: BatchSample[] = new Array(batch.length);
    let containsSolve = triggeredBySolve;

    for (let i = 0; i < batch.length; i++) {
      const tr = batch[i];
      if (tr.solvedTransition) containsSolve = true;

      let target = tr.reward;
      if (!tr.done) {
        // Double DQN: choose best legal next action with onlineNet, evaluate with targetNet
        const nextOnlineQ = this.onlineNet.predict(tr.nextState);
        let bestNextAction = -1;
        let bestNextVal = -Infinity;
        for (let a = 0; a < 4; a++) {
          if (tr.nextLegalMask[a] && nextOnlineQ[a] > bestNextVal) {
            bestNextVal = nextOnlineQ[a];
            bestNextAction = a;
          }
        }
        if (bestNextAction !== -1) {
          const nextTargetQ = this.targetNet.predict(tr.nextState);
          target = tr.reward + this.gamma * nextTargetQ[bestNextAction];
        }
      }

      samples[i] = {
        state: tr.state,
        action: tr.action,
        target,
      };
    }

    const stats = this.onlineNet.trainBatch(samples, this.learningRate);
    this.totalGradientSteps += 1;
    this.lastLoss = stats.loss;
    this.lastMeanDelta = stats.meanWeightDelta;
    this.lastMaxDelta = stats.maxWeightDelta;
    this.solveBurstActive = triggeredBySolve || (containsSolve && stats.meanWeightDelta > 0.0005);

    if (this.totalGradientSteps % this.targetUpdateInterval === 0) {
      this.targetNet.copyWeightsFrom(this.onlineNet);
    }

    return {
      trained: true,
      loss: stats.loss,
      meanDelta: stats.meanWeightDelta,
      maxDelta: stats.maxWeightDelta,
    };
  }

  public decayEpsilon(): void {
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
  }

  /**
   * Produces a compact, faithful visualization snapshot of the MLP's current
   * activations and real per-step weight-update magnitudes for the Three.js 3D view.
   */
  public getVisualSnapshot(): NetworkVisualSnapshot {
    const VISUAL_IN = 25;
    const VISUAL_H1 = 24;
    const VISUAL_H2 = 16;
    const VISUAL_OUT = 4;

    const inputActivations = new Array<number>(VISUAL_IN).fill(0);
    const inChunk = Math.max(1, Math.floor(this.lastForwardInput.length / VISUAL_IN));
    for (let i = 0; i < VISUAL_IN; i++) {
      let acc = 0;
      const start = i * inChunk;
      const end = Math.min(this.lastForwardInput.length, start + inChunk);
      for (let k = start; k < end; k++) {
        acc += Math.abs(this.lastForwardInput[k]);
      }
      inputActivations[i] = Math.min(1, acc / Math.max(1, end - start));
    }

    const sampleLayerAct = (src: Float32Array, count: number): number[] => {
      const res = new Array<number>(count).fill(0);
      let maxVal = 0.001;
      for (let i = 0; i < src.length; i++) {
        if (src[i] > maxVal) maxVal = src[i];
      }
      const stride = src.length / count;
      for (let i = 0; i < count; i++) {
        const idx = Math.min(src.length - 1, Math.floor(i * stride));
        res[i] = Math.min(1, src[idx] / maxVal);
      }
      return res;
    };

    const sampleDeltaGrid = (
      deltaArr: Float32Array,
      inSize: number,
      outSize: number,
      visIn: number,
      visOut: number,
    ): number[] => {
      const out = new Array<number>(visIn * visOut).fill(0);
      const inStride = inSize / visIn;
      const outStride = outSize / visOut;
      const scale = Math.max(this.lastMaxDelta, 1e-5);
      for (let o = 0; o < visOut; o++) {
        const realO = Math.min(outSize - 1, Math.floor(o * outStride));
        for (let i = 0; i < visIn; i++) {
          const realI = Math.min(inSize - 1, Math.floor(i * inStride));
          const raw = deltaArr[realO * inSize + realI] ?? 0;
          out[o * visIn + i] = Math.min(1, Math.pow(raw / scale, 0.55));
        }
      }
      return out;
    };

    const [l0, l1, l2] = this.onlineNet.layers;

    return {
      stepIndex: this.totalGradientSteps,
      inputSize: this.onlineNet.inputSize,
      hidden1Size: this.onlineNet.hidden1Size,
      hidden2Size: this.onlineNet.hidden2Size,
      outputSize: VISUAL_OUT,
      inputActivations,
      hidden1Activations: sampleLayerAct(this.lastHidden1, VISUAL_H1),
      hidden2Activations: sampleLayerAct(this.lastHidden2, VISUAL_H2),
      outputQValues: [...this.lastQValues],
      legalMask: [...this.lastLegalMask],
      w1DeltaStrengths: sampleDeltaGrid(l0.lastDeltaW, l0.inSize, l0.outSize, VISUAL_IN, VISUAL_H1),
      w2DeltaStrengths: sampleDeltaGrid(l1.lastDeltaW, l1.inSize, l1.outSize, VISUAL_H1, VISUAL_H2),
      w3DeltaStrengths: sampleDeltaGrid(l2.lastDeltaW, l2.inSize, l2.outSize, VISUAL_H2, VISUAL_OUT),
      meanWeightDelta: this.lastMeanDelta,
      maxWeightDelta: this.lastMaxDelta,
      lastLoss: this.lastLoss,
      solveBurstActive: this.solveBurstActive,
      totalGradientSteps: this.totalGradientSteps,
    };
  }

  public serialize(): SerializedAgent {
    return {
      inputSize: this.onlineNet.inputSize,
      epsilon: this.epsilon,
      totalGradientSteps: this.totalGradientSteps,
      onlineNet: this.onlineNet.serialize(),
      targetNet: this.targetNet.serialize(),
    };
  }

  public loadSerialized(data: SerializedAgent): void {
    this.onlineNet = SmallMLP.deserialize(data.onlineNet);
    this.targetNet = SmallMLP.deserialize(data.targetNet);
    this.epsilon = data.epsilon;
    this.totalGradientSteps = data.totalGradientSteps ?? 0;
  }
}
