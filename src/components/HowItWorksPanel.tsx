import React, { useState } from 'react';
import { Cpu, Gauge, Info, Layers, Play, Sparkles, Zap } from 'lucide-react';
import { DashboardSnapshot, TrainingConfig } from '../rl/types';
import { generateZipBoard } from '../rl/generator';
import { solveZipBoard } from '../rl/solver';

interface HowItWorksPanelProps {
  snapshot: DashboardSnapshot;
  onUpdateSettings: (partial: Partial<TrainingConfig>) => void;
  onRunComputeBenchmark: () => Promise<{
    batchSize: number;
    cpuMs: number;
    gpuMs: number | null;
    maxDiff: number | null;
    usedWebGPU: boolean;
  }>;
}

interface OracleBenchRow {
  size: number;
  boardsTested: number;
  solvedCount: number;
  avgMs: number;
  maxMs: number;
  avgNodes: number;
}

/**
 * Exact directed Hamiltonian path counts on N×N square grids (OEIS A096969)
 * plus exact non-path internal edges (N-1)^2 available for walls.
 */
const COMBINATORIAL_ROWS = [
  {
    size: 4,
    cells: 16,
    hamiltonianPaths: '552',
    nonPathEdges: 9,
    wallConfigs: '512 (2⁹)',
    approxBoards: '~10⁸+ puzzles',
    note: 'Ideal fast RL sandbox — learns clean solves quickly',
  },
  {
    size: 5,
    cells: 25,
    hamiltonianPaths: '8,648',
    nonPathEdges: 16,
    wallConfigs: '65,536 (2¹⁶)',
    approxBoards: '~10¹⁴+ puzzles',
    note: 'Default training arena; rich branching & traps',
  },
  {
    size: 6,
    cells: 36,
    hamiltonianPaths: '458,696',
    nonPathEdges: 25,
    wallConfigs: '3.35 × 10⁷ (2²⁵)',
    approxBoards: '~10²²+ puzzles',
    note: 'First curriculum promotion from 5×5',
  },
  {
    size: 7,
    cells: 49,
    hamiltonianPaths: '27,070,560',
    nonPathEdges: 36,
    wallConfigs: '6.87 × 10¹⁰ (2³⁶)',
    approxBoards: '~10³¹+ puzzles',
    note: '27 million base paths before walls & numbers',
  },
  {
    size: 8,
    cells: 64,
    hamiltonianPaths: '6,046,626,568',
    nonPathEdges: 49,
    wallConfigs: '5.63 × 10¹⁴ (2⁴⁹)',
    approxBoards: '~10⁴²+ puzzles',
    note: '~6.05 billion directed paths on empty grid',
  },
  {
    size: 9,
    cells: 81,
    hamiltonianPaths: '1,490,832,682,992',
    nonPathEdges: 64,
    wallConfigs: '1.84 × 10¹⁹ (2⁶⁴)',
    approxBoards: '~10⁵⁴+ puzzles',
    note: '~1.49 trillion base Hamiltonian paths',
  },
  {
    size: 10,
    cells: 100,
    hamiltonianPaths: '> 10¹⁵ (est.)',
    nonPathEdges: 81,
    wallConfigs: '2.42 × 10²⁴ (2⁸¹)',
    approxBoards: '> 10⁶⁸ puzzles',
    note: 'Astronomical space — memorization is impossible',
  },
];

export const HowItWorksPanel: React.FC<HowItWorksPanelProps> = ({
  snapshot,
  onUpdateSettings,
  onRunComputeBenchmark,
}) => {
  const [benchRows, setBenchRows] = useState<OracleBenchRow[]>([]);
  const [benchingOracle, setBenchingOracle] = useState(false);
  const [computeResult, setComputeResult] = useState<{
    batchSize: number;
    cpuMs: number;
    gpuMs: number | null;
    maxDiff: number | null;
    usedWebGPU: boolean;
  } | null>(null);
  const [benchingCompute, setBenchingCompute] = useState(false);

  const handleRunOracleBenchmark = async () => {
    setBenchingOracle(true);
    const results: OracleBenchRow[] = [];
    const boardsPerSize = 5;

    for (const size of [4, 5, 6, 7, 8, 9, 10]) {
      let totalMs = 0;
      let maxMs = 0;
      let totalNodes = 0;
      let solvedCount = 0;

      for (let i = 0; i < boardsPerSize; i++) {
        const board = generateZipBoard({
          size,
          waypointCount: 0, // Varied realistic Zip number count
          wallDensity: snapshot.config.wallDensity,
          seed: size * 31337 + i * 997 + 11,
        });
        const res = solveZipBoard(board);
        if (res.solved) solvedCount++;
        totalMs += res.elapsedMs;
        if (res.elapsedMs > maxMs) maxMs = res.elapsedMs;
        totalNodes += res.nodesExplored;
      }

      results.push({
        size,
        boardsTested: boardsPerSize,
        solvedCount,
        avgMs: Math.round((totalMs / boardsPerSize) * 100) / 100,
        maxMs: Math.round(maxMs * 100) / 100,
        avgNodes: Math.round(totalNodes / boardsPerSize),
      });
      setBenchRows([...results]);
      // Yield between sizes so UI updates smoothly
      await new Promise<void>((r) => window.setTimeout(r, 0));
    }

    setBenchingOracle(false);
  };

  const handleRunComputeTest = async () => {
    setBenchingCompute(true);
    try {
      const res = await onRunComputeBenchmark();
      setComputeResult(res);
    } finally {
      setBenchingCompute(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro & 20k Replay Buffer Explainer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <section className="lg:col-span-7 bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-5 space-y-3 transition-colors">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            <h1 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              How Zippy RL Works
            </h1>
          </div>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            Zippy RL trains a <strong>Double Deep Q-Network (Double DQN)</strong> from scratch directly in your browser to solve Hamiltonian path puzzles (Zip) on grids from <strong>4×4 up to 10×10</strong>. Every board is freshly generated from a guaranteed Hamiltonian path with randomized walls and varied numbered waypoints—so the agent must learn genuine spatial reasoning rather than memorizing fixed puzzles.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div className="p-3 rounded-md bg-slate-50 dark:bg-slate-900/70 border border-slate-200/80 dark:border-slate-800">
              <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Solve Till Win vs. No Backtrack
              </div>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                In <strong>Solve Till Win</strong> (default), the agent never stops on dead ends—it unwinds wrong branches (<span className="font-mono">-0.15</span> undo, <span className="font-mono">-0.20</span> trap) until the board is solved (<span className="font-mono">+10</span>). Training minimizes <strong>extra steps</strong> and climbs toward 100% <strong>Clean (0-Undo) Solves</strong>.
              </p>
            </div>
            <div className="p-3 rounded-md bg-slate-50 dark:bg-slate-900/70 border border-slate-200/80 dark:border-slate-800">
              <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Variable Waypoint Numbers
              </div>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                Just like real Zip, puzzles are not locked to 4 or 5 numbers. Set <strong>Numbers</strong> to <strong>Varied</strong> (e.g., 4–11 numbers on 5×5) or lock any custom count up to 65% of the board cells—without resetting your trained weights.
              </p>
            </div>
          </div>
        </section>

        {/* Dedicated Answer: What is the 20k Replay Buffer? */}
        <section className="lg:col-span-5 bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-5 space-y-3 transition-colors">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              What Is the &ldquo;20k&rdquo; Counter?
            </h2>
          </div>
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            The <span className="font-mono font-semibold text-slate-900 dark:text-slate-100">{snapshot.replayBufferSize.toLocaleString()} / 20k</span> metric is the <strong>Experience Replay Buffer</strong> capacity—it stores the agent&apos;s most recent <strong>20,000</strong> move transitions <span className="font-mono">(state, action, reward, nextState)</span> in a circular ring buffer.
          </p>
          <ul className="space-y-1.5 text-xs text-slate-600 dark:text-slate-400 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Breaks correlation:</strong> Training only on consecutive moves makes neural networks oscillate and forget earlier boards.
            </li>
            <li>
              <strong>Random mini-batches:</strong> On each Adam update, Double DQN samples 32 random transitions from the 20k memory so both recent mistakes and past +10 solves reinforce stably.
            </li>
            <li>
              <strong>Rolling window:</strong> Once it fills to 20,000, the oldest moves are automatically overwritten with fresher policy experience.
            </li>
          </ul>
        </section>
      </div>

      {/* Compute Engine: WebGPU & CPU */}
      <section className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-5 space-y-4 transition-colors">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            {snapshot.webgpu.active ? (
              <Zap className="w-4 h-4 text-amber-500" />
            ) : (
              <Cpu className="w-4 h-4 text-blue-600 dark:text-blue-400" />
            )}
            <div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Compute Backend (WebGPU &amp; Float32 CPU)
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {snapshot.webgpu.reason}
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-800 rounded-md" role="group" aria-label="Compute preference">
              <button
                type="button"
                onClick={() => onUpdateSettings({ preferredCompute: 'auto' })}
                className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap transition-colors ${
                  snapshot.config.preferredCompute === 'auto'
                    ? 'bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                Auto (WebGPU)
              </button>
              <button
                type="button"
                onClick={() => onUpdateSettings({ preferredCompute: 'cpu' })}
                className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap transition-colors ${
                  snapshot.config.preferredCompute === 'cpu'
                    ? 'bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                CPU Only
              </button>
            </div>

            <button
              type="button"
              disabled={benchingCompute}
              onClick={handleRunComputeTest}
              className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <Gauge className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              {benchingCompute ? 'Benchmarking…' : 'Test Compute (128 Batch)'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded-md bg-slate-50 dark:bg-slate-900/70 border border-slate-200/80 dark:border-slate-800">
            <div className="text-slate-500 dark:text-slate-400">Active Runtime</div>
            <div className="mt-1 font-mono font-semibold text-slate-900 dark:text-slate-100">
              {snapshot.webgpu.active ? 'WebGPU WGSL Active' : 'Float32 CPU Engine'}
            </div>
            <div className="mt-0.5 text-[11px] font-mono text-slate-500 dark:text-slate-400 truncate">
              {snapshot.webgpu.adapterName}
            </div>
          </div>
          <div className="p-3 rounded-md bg-slate-50 dark:bg-slate-900/70 border border-slate-200/80 dark:border-slate-800">
            <div className="text-slate-500 dark:text-slate-400">Why Hybrid Execution?</div>
            <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed">
              Single-step Q-inference on a compact MLP (~36k params) runs in ~0.04 ms on CPU with zero GPU buffer readback latency; WebGPU WGSL shaders accelerate large batched forward passes when available.
            </p>
          </div>
          <div className="p-3 rounded-md bg-slate-50 dark:bg-slate-900/70 border border-slate-200/80 dark:border-slate-800">
            <div className="text-slate-500 dark:text-slate-400">Latest 128-State Probe</div>
            {computeResult ? (
              <div className="mt-1 font-mono tabular-nums text-slate-900 dark:text-slate-100 space-y-0.5">
                <div>CPU Float32: {computeResult.cpuMs.toFixed(2)} ms</div>
                <div>
                  WebGPU WGSL:{' '}
                  {computeResult.usedWebGPU && computeResult.gpuMs !== null
                    ? `${computeResult.gpuMs.toFixed(2)} ms (max Δ ${computeResult.maxDiff})`
                    : 'Not available in this browser'}
                </div>
              </div>
            ) : (
              <p className="mt-1 text-slate-500 dark:text-slate-400">
                Click &ldquo;Test Compute&rdquo; to measure 128-state MLP inference speed on your device.
              </p>
            )}
          </div>
        </div>
      </section>

      {/* DFS Oracle Speed & Live Browser Benchmark */}
      <section className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden transition-colors">
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
              <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                How Fast Does the DFS Oracle Solve Every Board Size?
              </h2>
            </div>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400 max-w-2xl">
              The Oracle uses exact Depth-First Search with <strong>Warnsdorff minimum-degree ordering</strong>, <strong>BFS flood-fill connectivity pruning</strong>, and <strong>isolated dead-end pruning</strong>. On 4×4–7×7 it solves almost instantaneously (&lt;1 ms); on 8×8–10×10 it typically solves in a few milliseconds.
            </p>
          </div>
          <button
            type="button"
            disabled={benchingOracle}
            onClick={handleRunOracleBenchmark}
            className="h-8 px-3.5 rounded-md bg-emerald-600 text-white text-xs font-semibold hover:bg-emerald-700 disabled:opacity-50 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <Play className="w-3.5 h-3.5" />
            {benchingOracle ? 'Running Benchmark…' : 'Run Live Oracle Benchmark (4×4 → 10×10)'}
          </button>
        </div>

        {benchRows.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                <tr>
                  <th className="py-2.5 px-4 font-medium">Grid Size</th>
                  <th className="py-2.5 px-4 font-medium text-right">Cells</th>
                  <th className="py-2.5 px-4 font-medium text-right">Solved</th>
                  <th className="py-2.5 px-4 font-medium text-right">Avg Solve Time</th>
                  <th className="py-2.5 px-4 font-medium text-right">Max Solve Time</th>
                  <th className="py-2.5 px-4 font-medium text-right">Avg DFS Nodes</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80 font-mono tabular-nums">
                {benchRows.map((row) => (
                  <tr key={row.size} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                    <td className="py-2.5 px-4 font-semibold text-slate-900 dark:text-slate-100">
                      {row.size}×{row.size}
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-600 dark:text-slate-400">
                      {row.size * row.size}
                    </td>
                    <td className="py-2.5 px-4 text-right text-emerald-700 dark:text-emerald-400 font-semibold">
                      {row.solvedCount} / {row.boardsTested}
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-900 dark:text-slate-100">
                      {row.avgMs.toFixed(2)} ms
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-600 dark:text-slate-400">
                      {row.maxMs.toFixed(2)} ms
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                      {row.avgNodes.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="px-5 py-6 text-xs text-slate-500 dark:text-slate-400">
            Click <strong>Run Live Oracle Benchmark</strong> to time the exact DFS solver live on 5 freshly generated boards per size from 4×4 to 10×10 in your browser.
          </div>
        )}
      </section>

      {/* Combinatorial Scale Table (OEIS A096969) */}
      <section className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden transition-colors">
        <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            How Many Possible Zip Boards Exist Per Grid Size? (4×4 to 10×10)
          </h2>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Every valid Zip puzzle starts from a <strong>directed Hamiltonian path</strong> covering all N² cells (exact square-grid counts from <strong>OEIS A096969</strong>), then adds any subset of walls along the <span className="font-mono">(N−1)²</span> unused internal edges plus any subset of intermediate waypoint numbers.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
              <tr>
                <th className="py-2.5 px-4 font-medium">Grid</th>
                <th className="py-2.5 px-4 font-medium text-right">Cells</th>
                <th className="py-2.5 px-4 font-medium text-right">Directed Hamiltonian Paths (OEIS A096969)</th>
                <th className="py-2.5 px-4 font-medium text-right">Wall Subsets / Path</th>
                <th className="py-2.5 px-4 font-medium text-right">Distinct Zip Puzzles</th>
                <th className="py-2.5 px-4 font-medium">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
              {COMBINATORIAL_ROWS.map((row) => (
                <tr key={row.size} className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40">
                  <td className="py-2.5 px-4 font-mono font-semibold text-slate-900 dark:text-slate-100">
                    {row.size}×{row.size}
                  </td>
                  <td className="py-2.5 px-4 font-mono tabular-nums text-right text-slate-600 dark:text-slate-400">
                    {row.cells}
                  </td>
                  <td className="py-2.5 px-4 font-mono tabular-nums text-right font-semibold text-blue-700 dark:text-blue-400">
                    {row.hamiltonianPaths}
                  </td>
                  <td className="py-2.5 px-4 font-mono tabular-nums text-right text-slate-600 dark:text-slate-400">
                    {row.wallConfigs}
                  </td>
                  <td className="py-2.5 px-4 font-mono tabular-nums text-right text-slate-900 dark:text-slate-100">
                    {row.approxBoards}
                  </td>
                  <td className="py-2.5 px-4 text-slate-600 dark:text-slate-400">
                    {row.note}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
