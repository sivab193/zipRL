import { WebGPUStatus } from './types';
import { SmallMLP } from './mlp';

// Minimal structural typings for WebGPU APIs so TypeScript compiles cleanly
// across browsers and Node without requiring extra @webgpu/types packages.
interface GPUAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
}

interface GPUDeviceLike {
  createShaderModule(desc: { code: string }): unknown;
  createComputePipeline(desc: unknown): unknown;
  createBindGroup(desc: unknown): unknown;
  createBuffer(desc: { size: number; usage: number }): GPUBufferLike;
  createCommandEncoder(): GPUCommandEncoderLike;
  queue: {
    writeBuffer(buffer: GPUBufferLike, offset: number, data: ArrayBufferView): void;
    submit(commands: unknown[]): void;
  };
  destroy?: () => void;
}

interface GPUBufferLike {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GPUCommandEncoderLike {
  beginComputePass(): {
    setPipeline(pipeline: unknown): void;
    setBindGroup(index: number, bindGroup: unknown): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  };
  copyBufferToBuffer(
    src: GPUBufferLike,
    srcOffset: number,
    dst: GPUBufferLike,
    dstOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

interface GPUAdapterLike {
  info?: GPUAdapterInfoLike;
  requestDevice(): Promise<GPUDeviceLike>;
}

interface NavigatorGPU {
  requestAdapter(opts?: { powerPreference?: 'high-performance' | 'low-power' }): Promise<GPUAdapterLike | null>;
}

const GPU_BUF_MAP_READ = 0x0001;
const GPU_BUF_COPY_SRC = 0x0004;
const GPU_BUF_COPY_DST = 0x0008;
const GPU_BUF_UNIFORM = 0x0040;
const GPU_BUF_STORAGE = 0x0080;

/**
 * WGSL Compute Shader for batched dense layer forward pass:
 *   Y[b, o] = activation( sum_i( W[o, i] * X[b, i] ) + B[o] )
 */
const DENSE_BATCH_WGSL = /* wgsl */ `
struct Dims {
  batchSize : u32,
  inSize    : u32,
  outSize   : u32,
  applyRelu : u32,
};

@group(0) @binding(0) var<uniform> dims : Dims;
@group(0) @binding(1) var<storage, read> inputX : array<f32>;
@group(0) @binding(2) var<storage, read> weights : array<f32>;
@group(0) @binding(3) var<storage, read> biases : array<f32>;
@group(0) @binding(4) var<storage, read_write> outputY : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  let total = dims.batchSize * dims.outSize;
  if (idx >= total) {
    return;
  }
  let b = idx / dims.outSize;
  let o = idx % dims.outSize;
  let xBase = b * dims.inSize;
  let wBase = o * dims.inSize;

  var sum : f32 = biases[o];
  for (var i : u32 = 0u; i < dims.inSize; i = i + 1u) {
    sum = sum + weights[wBase + i] * inputX[xBase + i];
  }
  if (dims.applyRelu == 1u && sum < 0.0) {
    sum = 0.0;
  }
  outputY[idx] = sum;
}
`;

/**
 * Optional WebGPU Compute Shader backend for batched MLP inference.
 * Automatically detects `navigator.gpu` and falls back cleanly to Float32 CPU
 * on environments without WebGPU support (Safari older versions, headless CI, etc.).
 */
export class WebGPUMlpAccelerator {
  public readonly status: WebGPUStatus;
  private device: GPUDeviceLike | null = null;
  private pipeline: unknown = null;

  private constructor(status: WebGPUStatus, device: GPUDeviceLike | null, pipeline: unknown) {
    this.status = status;
    this.device = device;
    this.pipeline = pipeline;
  }

  public static createCpuFallback(reason: string): WebGPUMlpAccelerator {
    return new WebGPUMlpAccelerator(
      {
        supported: false,
        active: false,
        adapterName: 'CPU Float32',
        reason,
      },
      null,
      null,
    );
  }

  public static async detectAndInit(): Promise<WebGPUMlpAccelerator> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return WebGPUMlpAccelerator.createCpuFallback(
        'WebGPU API (navigator.gpu) is not exposed in this browser; running optimized Float32 CPU.',
      );
    }

    try {
      const gpu = (navigator as unknown as { gpu: NavigatorGPU }).gpu;
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return WebGPUMlpAccelerator.createCpuFallback(
          'No WebGPU GPU adapter found on this device; running optimized Float32 CPU.',
        );
      }

      const device = await adapter.requestDevice();
      const module = device.createShaderModule({ code: DENSE_BATCH_WGSL });
      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module,
          entryPoint: 'main',
        },
      });

      const info = adapter.info;
      const adapterName =
        [info?.description, info?.device, info?.vendor, info?.architecture]
          .filter((s): s is string => Boolean(s && s.trim().length > 0))
          .slice(0, 2)
          .join(' · ') || 'WebGPU Hardware Adapter';

      return new WebGPUMlpAccelerator(
        {
          supported: true,
          active: true,
          adapterName,
          reason: 'WGSL batched dense compute shaders compiled and ready.',
        },
        device,
        pipeline,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return WebGPUMlpAccelerator.createCpuFallback(
        `WebGPU initialization failed (${msg}); fell back to Float32 CPU.`,
      );
    }
  }

  public isReady(): boolean {
    return this.device !== null && this.pipeline !== null && this.status.active;
  }

  /**
   * Executes a single dense layer forward pass over a batch of B vectors on the GPU.
   */
  public async denseForwardBatch(
    xBatch: Float32Array,
    weights: Float32Array,
    biases: Float32Array,
    batchSize: number,
    inSize: number,
    outSize: number,
    applyRelu: boolean,
  ): Promise<Float32Array> {
    if (!this.device || !this.pipeline) {
      throw new Error('WebGPU device not initialized');
    }
    const device = this.device;
    const outCount = batchSize * outSize;
    const outBytes = outCount * 4;

    const dimsData = new Uint32Array([batchSize, inSize, outSize, applyRelu ? 1 : 0]);
    const dimsBuf = device.createBuffer({ size: 16, usage: GPU_BUF_UNIFORM | GPU_BUF_COPY_DST });
    const xBuf = device.createBuffer({ size: Math.max(4, xBatch.byteLength), usage: GPU_BUF_STORAGE | GPU_BUF_COPY_DST });
    const wBuf = device.createBuffer({ size: Math.max(4, weights.byteLength), usage: GPU_BUF_STORAGE | GPU_BUF_COPY_DST });
    const bBuf = device.createBuffer({ size: Math.max(4, biases.byteLength), usage: GPU_BUF_STORAGE | GPU_BUF_COPY_DST });
    const yBuf = device.createBuffer({ size: Math.max(4, outBytes), usage: GPU_BUF_STORAGE | GPU_BUF_COPY_SRC });
    const readBuf = device.createBuffer({ size: Math.max(4, outBytes), usage: GPU_BUF_MAP_READ | GPU_BUF_COPY_DST });

    device.queue.writeBuffer(dimsBuf, 0, dimsData);
    device.queue.writeBuffer(xBuf, 0, xBatch);
    device.queue.writeBuffer(wBuf, 0, weights);
    device.queue.writeBuffer(bBuf, 0, biases);

    const bindGroup = device.createBindGroup({
      layout: (this.pipeline as { getBindGroupLayout(i: number): unknown }).getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: dimsBuf } },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: wBuf } },
        { binding: 3, resource: { buffer: bBuf } },
        { binding: 4, resource: { buffer: yBuf } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outCount / 64));
    pass.end();
    encoder.copyBufferToBuffer(yBuf, 0, readBuf, 0, outBytes);
    device.queue.submit([encoder.finish()]);

    await readBuf.mapAsync(GPU_BUF_MAP_READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    dimsBuf.destroy();
    xBuf.destroy();
    wBuf.destroy();
    bBuf.destroy();
    yBuf.destroy();
    readBuf.destroy();

    return result;
  }

  /**
   * Runs a full 3-layer batched MLP forward pass on WebGPU (used for batched Q-value
   * evaluation and live GPU vs CPU compute benchmark).
   */
  public async predictBatchMLP(mlp: SmallMLP, states: Float32Array[]): Promise<Float32Array[]> {
    const batchSize = states.length;
    if (batchSize === 0) return [];
    const inSize = mlp.inputSize;
    const flatIn = new Float32Array(batchSize * inSize);
    for (let b = 0; b < batchSize; b++) {
      flatIn.set(states[b], b * inSize);
    }

    const [l0, l1, l2] = mlp.layers;
    const h1 = await this.denseForwardBatch(flatIn, l0.weights, l0.biases, batchSize, l0.inSize, l0.outSize, true);
    const h2 = await this.denseForwardBatch(h1, l1.weights, l1.biases, batchSize, l1.inSize, l1.outSize, true);
    const out = await this.denseForwardBatch(h2, l2.weights, l2.biases, batchSize, l2.inSize, l2.outSize, false);

    const results: Float32Array[] = new Array(batchSize);
    for (let b = 0; b < batchSize; b++) {
      results[b] = out.slice(b * l2.outSize, (b + 1) * l2.outSize);
    }
    return results;
  }

  public dispose(): void {
    try {
      this.device?.destroy?.();
    } catch {
      // Ignore cleanup issues
    }
    this.device = null;
    this.pipeline = null;
  }
}
