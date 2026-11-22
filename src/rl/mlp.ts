/**
 * Pure TypeScript Multi-Layer Perceptron (MLP) with He initialization,
 * ReLU activations, Huber loss backpropagation, Adam optimizer, and
 * real per-step weight-delta instrumentation for 3D visualization.
 */

export interface LayerParams {
  inSize: number;
  outSize: number;
  weights: Float32Array; // [outSize * inSize], row-major: w[o * inSize + i]
  biases: Float32Array;  // [outSize]
  // Adam moments
  mW: Float32Array;
  vW: Float32Array;
  mB: Float32Array;
  vB: Float32Array;
  // Last parameter step deltas (|w_new - w_old|) for live 3D visualization
  lastDeltaW: Float32Array;
}

export interface SerializedLayer {
  inSize: number;
  outSize: number;
  weights: number[];
  biases: number[];
  mW: number[];
  vW: number[];
  mB: number[];
  vB: number[];
}

export interface SerializedMLP {
  inputSize: number;
  hidden1Size: number;
  hidden2Size: number;
  outputSize: number;
  adamStep: number;
  layers: SerializedLayer[];
}

export interface ForwardCache {
  input: Float32Array;
  z1: Float32Array;
  a1: Float32Array;
  z2: Float32Array;
  a2: Float32Array;
  out: Float32Array;
}

export interface BatchSample {
  state: Float32Array;
  action: number;
  target: number;
}

export interface TrainStepStats {
  loss: number;
  meanWeightDelta: number;
  maxWeightDelta: number;
}

function createLayer(inSize: number, outSize: number, seedFn: () => number): LayerParams {
  const weights = new Float32Array(outSize * inSize);
  const biases = new Float32Array(outSize);
  // He uniform initialization for ReLU layers
  const limit = Math.sqrt(6 / Math.max(1, inSize));
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (seedFn() * 2 - 1) * limit;
  }
  return {
    inSize,
    outSize,
    weights,
    biases,
    mW: new Float32Array(weights.length),
    vW: new Float32Array(weights.length),
    mB: new Float32Array(outSize),
    vB: new Float32Array(outSize),
    lastDeltaW: new Float32Array(weights.length),
  };
}

export class SmallMLP {
  public readonly inputSize: number;
  public readonly hidden1Size: number;
  public readonly hidden2Size: number;
  public readonly outputSize: number;
  public layers: [LayerParams, LayerParams, LayerParams];
  public adamStep = 0;

  // Reusable gradient accumulators to avoid GC churn during fast training
  private gW0: Float32Array;
  private gB0: Float32Array;
  private gW1: Float32Array;
  private gB1: Float32Array;
  private gW2: Float32Array;
  private gB2: Float32Array;

  // Reusable scratch buffers for single-sample forward/backward
  private dOut = new Float32Array(4);
  private dA2: Float32Array;
  private dA1: Float32Array;

  constructor(
    inputSize: number,
    hidden1Size = 128,
    hidden2Size = 64,
    outputSize = 4,
    seedFn: () => number = Math.random,
  ) {
    this.inputSize = inputSize;
    this.hidden1Size = hidden1Size;
    this.hidden2Size = hidden2Size;
    this.outputSize = outputSize;

    const l0 = createLayer(inputSize, hidden1Size, seedFn);
    const l1 = createLayer(hidden1Size, hidden2Size, seedFn);
    const l2 = createLayer(hidden2Size, outputSize, seedFn);
    // Smaller init on final Q-head for calm initial Q-values
    for (let i = 0; i < l2.weights.length; i++) {
      l2.weights[i] *= 0.25;
    }

    this.layers = [l0, l1, l2];
    this.gW0 = new Float32Array(l0.weights.length);
    this.gB0 = new Float32Array(l0.biases.length);
    this.gW1 = new Float32Array(l1.weights.length);
    this.gB1 = new Float32Array(l1.biases.length);
    this.gW2 = new Float32Array(l2.weights.length);
    this.gB2 = new Float32Array(l2.biases.length);
    this.dA2 = new Float32Array(hidden2Size);
    this.dA1 = new Float32Array(hidden1Size);
  }

  public forward(input: Float32Array): ForwardCache {
    const [l0, l1, l2] = this.layers;
    const z1 = new Float32Array(this.hidden1Size);
    const a1 = new Float32Array(this.hidden1Size);
    const z2 = new Float32Array(this.hidden2Size);
    const a2 = new Float32Array(this.hidden2Size);
    const out = new Float32Array(this.outputSize);

    // Layer 0: input -> hidden1 (ReLU)
    const w0 = l0.weights;
    const b0 = l0.biases;
    const in0 = l0.inSize;
    for (let o = 0; o < this.hidden1Size; o++) {
      let sum = b0[o];
      const rowOff = o * in0;
      for (let i = 0; i < in0; i++) {
        sum += w0[rowOff + i] * input[i];
      }
      z1[o] = sum;
      a1[o] = sum > 0 ? sum : 0;
    }

    // Layer 1: hidden1 -> hidden2 (ReLU)
    const w1 = l1.weights;
    const b1 = l1.biases;
    const in1 = l1.inSize;
    for (let o = 0; o < this.hidden2Size; o++) {
      let sum = b1[o];
      const rowOff = o * in1;
      for (let i = 0; i < in1; i++) {
        sum += w1[rowOff + i] * a1[i];
      }
      z2[o] = sum;
      a2[o] = sum > 0 ? sum : 0;
    }

    // Layer 2: hidden2 -> output Q-values (linear)
    const w2 = l2.weights;
    const b2 = l2.biases;
    const in2 = l2.inSize;
    for (let o = 0; o < this.outputSize; o++) {
      let sum = b2[o];
      const rowOff = o * in2;
      for (let i = 0; i < in2; i++) {
        sum += w2[rowOff + i] * a2[i];
      }
      out[o] = sum;
    }

    return { input, z1, a1, z2, a2, out };
  }

  public predict(input: Float32Array): Float32Array {
    return this.forward(input).out;
  }

  /**
   * Executes one real mini-batch gradient descent update with Huber loss and Adam optimizer.
   * Captures exact per-weight update magnitudes |w_new - w_old| for live visualization.
   */
  public trainBatch(samples: BatchSample[], learningRate: number): TrainStepStats {
    const batchSize = samples.length;
    if (batchSize === 0) {
      return { loss: 0, meanWeightDelta: 0, maxWeightDelta: 0 };
    }

    const [l0, l1, l2] = this.layers;
    this.gW0.fill(0);
    this.gB0.fill(0);
    this.gW1.fill(0);
    this.gB1.fill(0);
    this.gW2.fill(0);
    this.gB2.fill(0);

    let totalLoss = 0;
    const invBatch = 1 / batchSize;

    for (let s = 0; s < batchSize; s++) {
      const sample = samples[s];
      const cache = this.forward(sample.state);
      const qPred = cache.out[sample.action];
      const diff = qPred - sample.target;
      const absDiff = Math.abs(diff);

      // Huber loss (delta = 1.0)
      if (absDiff <= 1.0) {
        totalLoss += 0.5 * diff * diff;
      } else {
        totalLoss += absDiff - 0.5;
      }

      // dLoss / dOut
      this.dOut.fill(0);
      const gradScalar = Math.max(-1.0, Math.min(1.0, diff)) * invBatch;
      this.dOut[sample.action] = gradScalar;

      // Backprop Layer 2 (hidden2 -> output)
      const a2 = cache.a2;
      const w2 = l2.weights;
      const in2 = l2.inSize;
      const act = sample.action;
      this.gB2[act] += gradScalar;
      const rowOff2 = act * in2;
      for (let i = 0; i < in2; i++) {
        this.gW2[rowOff2 + i] += gradScalar * a2[i];
        this.dA2[i] = w2[rowOff2 + i] * gradScalar;
      }

      // Backprop Layer 1 (hidden1 -> hidden2) through ReLU
      const z2 = cache.z2;
      const a1 = cache.a1;
      const w1 = l1.weights;
      const in1 = l1.inSize;
      this.dA1.fill(0);
      for (let o = 0; o < this.hidden2Size; o++) {
        if (z2[o] <= 0) continue;
        const g = this.dA2[o];
        if (g === 0) continue;
        this.gB1[o] += g;
        const rowOff1 = o * in1;
        for (let i = 0; i < in1; i++) {
          this.gW1[rowOff1 + i] += g * a1[i];
          this.dA1[i] += w1[rowOff1 + i] * g;
        }
      }

      // Backprop Layer 0 (input -> hidden1) through ReLU
      const z1 = cache.z1;
      const inp = cache.input;
      const in0 = l0.inSize;
      for (let o = 0; o < this.hidden1Size; o++) {
        if (z1[o] <= 0) continue;
        const g = this.dA1[o];
        if (g === 0) continue;
        this.gB0[o] += g;
        const rowOff0 = o * in0;
        for (let i = 0; i < in0; i++) {
          const x = inp[i];
          if (x !== 0) {
            this.gW0[rowOff0 + i] += g * x;
          }
        }
      }
    }

    // Apply Adam update across all layers and measure exact weight deltas
    this.adamStep += 1;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const bc1 = 1 - Math.pow(beta1, this.adamStep);
    const bc2 = 1 - Math.pow(beta2, this.adamStep);
    const stepSize = learningRate * (Math.sqrt(bc2) / bc1);

    let deltaSum = 0;
    let deltaCount = 0;
    let maxDelta = 0;

    const applyAdam = (layer: LayerParams, gW: Float32Array, gB: Float32Array) => {
      const { weights, biases, mW, vW, mB, vB, lastDeltaW } = layer;
      for (let i = 0; i < weights.length; i++) {
        const g = Math.max(-5, Math.min(5, gW[i]));
        const m = beta1 * mW[i] + (1 - beta1) * g;
        const v = beta2 * vW[i] + (1 - beta2) * g * g;
        mW[i] = m;
        vW[i] = v;
        const delta = (stepSize * m) / (Math.sqrt(v) + eps);
        weights[i] -= delta;
        const absD = Math.abs(delta);
        lastDeltaW[i] = absD;
        deltaSum += absD;
        if (absD > maxDelta) maxDelta = absD;
      }
      deltaCount += weights.length;

      for (let i = 0; i < biases.length; i++) {
        const g = Math.max(-5, Math.min(5, gB[i]));
        const m = beta1 * mB[i] + (1 - beta1) * g;
        const v = beta2 * vB[i] + (1 - beta2) * g * g;
        mB[i] = m;
        vB[i] = v;
        biases[i] -= (stepSize * m) / (Math.sqrt(v) + eps);
      }
    };

    applyAdam(l0, this.gW0, this.gB0);
    applyAdam(l1, this.gW1, this.gB1);
    applyAdam(l2, this.gW2, this.gB2);

    return {
      loss: totalLoss * invBatch,
      meanWeightDelta: deltaCount > 0 ? deltaSum / deltaCount : 0,
      maxWeightDelta: maxDelta,
    };
  }

  public copyWeightsFrom(other: SmallMLP): void {
    for (let l = 0; l < 3; l++) {
      this.layers[l].weights.set(other.layers[l].weights);
      this.layers[l].biases.set(other.layers[l].biases);
    }
  }

  public serialize(): SerializedMLP {
    return {
      inputSize: this.inputSize,
      hidden1Size: this.hidden1Size,
      hidden2Size: this.hidden2Size,
      outputSize: this.outputSize,
      adamStep: this.adamStep,
      layers: this.layers.map((l) => ({
        inSize: l.inSize,
        outSize: l.outSize,
        weights: Array.from(l.weights),
        biases: Array.from(l.biases),
        mW: Array.from(l.mW),
        vW: Array.from(l.vW),
        mB: Array.from(l.mB),
        vB: Array.from(l.vB),
      })),
    };
  }

  public static deserialize(data: SerializedMLP): SmallMLP {
    const net = new SmallMLP(data.inputSize, data.hidden1Size, data.hidden2Size, data.outputSize);
    net.adamStep = data.adamStep ?? 0;
    for (let l = 0; l < 3; l++) {
      const src = data.layers[l];
      const dst = net.layers[l];
      dst.weights.set(src.weights);
      dst.biases.set(src.biases);
      if (src.mW) dst.mW.set(src.mW);
      if (src.vW) dst.vW.set(src.vW);
      if (src.mB) dst.mB.set(src.mB);
      if (src.vB) dst.vB.set(src.vB);
    }
    return net;
  }
}
