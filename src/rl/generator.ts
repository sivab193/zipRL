import { CellCoord, DIRECTION_DELTAS, Direction, WallEdge, Waypoint, ZipBoard } from './types';

export interface GeneratorOptions {
  size: number;
  waypointCount?: number;
  wallDensity?: number;
  seed?: number;
}

/** Fast deterministic PRNG (Mulberry32) for reproducible tests and seeded boards. */
export function createRng(seed?: number): () => number {
  let state = (seed ?? Math.floor(Math.random() * 0xffffffff)) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function edgeKey(r1: number, c1: number, r2: number, c2: number): string {
  if (r1 < r2 || (r1 === r2 && c1 < c2)) {
    return `${r1},${c1}-${r2},${c2}`;
  }
  return `${r2},${c2}-${r1},${c1}`;
}

export function normalizeWallEdge(r1: number, c1: number, r2: number, c2: number): WallEdge {
  if (r1 < r2 || (r1 === r2 && c1 < c2)) {
    return { r1, c1, r2, c2 };
  }
  return { r1: r2, c1: c2, r2: r1, c2: c1 };
}

function getNeighbors(row: number, col: number, size: number): CellCoord[] {
  const result: CellCoord[] = [];
  for (let d = 0; d < 4; d++) {
    const [dr, dc] = DIRECTION_DELTAS[d as Direction];
    const nr = row + dr;
    const nc = col + dc;
    if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
      result.push({ row: nr, col: nc });
    }
  }
  return result;
}

function shuffleInPlace<T>(items: T[], rng: () => number): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = items[i];
    items[i] = items[j];
    items[j] = tmp;
  }
  return items;
}

/**
 * Samples a randomized Hamiltonian path covering every cell of an N x N grid.
 * Uses randomized DFS with Warnsdorff tie-breaking (preferring neighbors with
 * fewest unvisited exits) so even 10x10 grids sample reliably in milliseconds,
 * while preserving genuine randomness across generated puzzles.
 */
export function sampleHamiltonianPath(size: number, rng: () => number): CellCoord[] {
  const totalCells = size * size;
  const maxAttempts = 20;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    // Parity check: on odd grids (e.g., 5x5, 7x7, 9x9), a Hamiltonian path
    // must start on a cell with even (row + col) parity!
    const validStarts: CellCoord[] = [];
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (size % 2 === 0 || (r + c) % 2 === 0) {
          validStarts.push({ row: r, col: c });
        }
      }
    }
    const start = validStarts[Math.floor(rng() * validStarts.length)];

    const visited = new Uint8Array(totalCells);
    const path: CellCoord[] = [start];
    visited[start.row * size + start.col] = 1;

    let backtracks = 0;
    const backtrackBudget = size <= 6 ? 1200 : 4000;

    const dfs = (row: number, col: number): boolean => {
      if (path.length === totalCells) {
        return true;
      }
      if (backtracks > backtrackBudget) {
        return false;
      }

      const candidates = getNeighbors(row, col, size).filter(
        (n) => visited[n.row * size + n.col] === 0,
      );
      if (candidates.length === 0) {
        return false;
      }

      // Compute Warnsdorff degree with small random jitter for diversity
      const scored = candidates.map((cand) => {
        let deg = 0;
        for (let d = 0; d < 4; d++) {
          const [dr, dc] = DIRECTION_DELTAS[d as Direction];
          const nr = cand.row + dr;
          const nc = cand.col + dc;
          if (nr >= 0 && nr < size && nc >= 0 && nc < size && visited[nr * size + nc] === 0) {
            deg++;
          }
        }
        return { cand, deg, tie: rng() };
      });

      // On smaller grids, allow occasional non-Warnsdorff exploration for maximum path variety
      const usePureRandom = size <= 5 && rng() < 0.35;
      if (usePureRandom) {
        shuffleInPlace(scored, rng);
      } else {
        scored.sort((a, b) => a.deg - b.deg || a.tie - b.tie);
      }

      // Dead-end pruning: if >1 unvisited neighbor has degree 1 (besides the very last step), prune
      const remaining = totalCells - path.length;
      if (remaining > 2) {
        let forcedEnds = 0;
        for (let i = 0; i < scored.length; i++) {
          if (scored[i].deg === 0) forcedEnds++;
        }
        if (forcedEnds > 1) {
          return false;
        }
      }

      for (const { cand } of scored) {
        const idx = cand.row * size + cand.col;
        visited[idx] = 1;
        path.push(cand);
        if (dfs(cand.row, cand.col)) {
          return true;
        }
        path.pop();
        visited[idx] = 0;
        backtracks++;
        if (backtracks > backtrackBudget) {
          return false;
        }
      }

      return false;
    };

    if (dfs(start.row, start.col)) {
      return path;
    }
  }

  // Deterministic snake fallback with random rotation/flip (guaranteed Hamiltonian)
  const snake: CellCoord[] = [];
  for (let r = 0; r < size; r++) {
    if (r % 2 === 0) {
      for (let c = 0; c < size; c++) snake.push({ row: r, col: c });
    } else {
      for (let c = size - 1; c >= 0; c--) snake.push({ row: r, col: c });
    }
  }
  const flipH = rng() < 0.5;
  const flipV = rng() < 0.5;
  const transpose = rng() < 0.5;
  return snake.map(({ row, col }) => {
    let r = flipV ? size - 1 - row : row;
    let c = flipH ? size - 1 - col : col;
    if (transpose) {
      const t = r;
      r = c;
      c = t;
    }
    return { row: r, col: c };
  });
}

let boardCounter = 0;

/**
 * Returns the recommended varied waypoint range [min, max] for a given grid size N,
 * matching real Zip puzzles where a 5×5 can have anywhere from 4 to 11 numbers and
 * larger grids can have many more numbered waypoints.
 */
export function getVariedWaypointRange(size: number): readonly [number, number] {
  const clamped = Math.max(4, Math.min(10, Math.round(size)));
  const minWp = Math.max(3, clamped - 1);
  const maxWp = Math.min(Math.floor(clamped * clamped * 0.48), Math.max(minWp + 3, Math.round(clamped * 2.2)));
  return [minWp, maxWp] as const;
}

export function getMaxWaypointCount(size: number): number {
  const clamped = Math.max(4, Math.min(10, Math.round(size)));
  return Math.max(4, Math.floor(clamped * clamped * 0.65));
}

/**
 * Generates a fresh, guaranteed-solvable Zip puzzle board.
 * - Supports grids from 4×4 up to 10×10.
 * - Samples a random Hamiltonian path covering all N*N cells.
 * - Places waypoint 1 at path start, final waypoint at path end, and intermediate waypoints
 *   along the path. If waypointCount is 0 or omitted, picks a varied Zip number count.
 * - Adds walls only on edges NOT traversed by the sampled solution path.
 */
export function generateZipBoard(options: GeneratorOptions): ZipBoard {
  const size = Math.max(4, Math.min(10, Math.round(options.size)));
  const rng = createRng(options.seed);
  const path = sampleHamiltonianPath(size, rng);
  const totalCells = size * size;

  const [minAuto, maxAuto] = getVariedWaypointRange(size);
  const defaultWaypoints = minAuto + Math.floor(rng() * (maxAuto - minAuto + 1));
  const requestedWp =
    options.waypointCount !== undefined && options.waypointCount > 0
      ? options.waypointCount
      : defaultWaypoints;
  const waypointCount = Math.max(2, Math.min(totalCells, Math.round(requestedWp)));
  const wallDensity = Math.max(0, Math.min(0.65, options.wallDensity ?? 0.25));

  // Space waypoints along path with gentle jitter so they never clump
  const indices = new Set<number>([0, totalCells - 1]);
  const intermediateCount = waypointCount - 2;
  if (intermediateCount > 0) {
    const stride = (totalCells - 1) / (intermediateCount + 1);
    for (let i = 1; i <= intermediateCount; i++) {
      const ideal = Math.round(i * stride);
      const jitter = Math.floor((rng() - 0.5) * Math.max(1, stride * 0.4));
      let idx = Math.max(1, Math.min(totalCells - 2, ideal + jitter));
      while (indices.has(idx) && idx < totalCells - 2) idx++;
      while (indices.has(idx) && idx > 1) idx--;
      indices.add(idx);
    }
  }

  const sortedIndices = Array.from(indices).sort((a, b) => a - b);
  const waypoints: Waypoint[] = sortedIndices.map((pathIdx, i) => ({
    order: i + 1,
    row: path[pathIdx].row,
    col: path[pathIdx].col,
  }));

  // Collect edges traversed by the Hamiltonian path so we NEVER place a wall on them
  const pathEdges = new Set<string>();
  for (let i = 0; i < path.length - 1; i++) {
    pathEdges.add(edgeKey(path[i].row, path[i].col, path[i + 1].row, path[i + 1].col));
  }

  // Enumerate all internal non-path edges and sample walls by wallDensity
  const nonPathEdges: WallEdge[] = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (c + 1 < size) {
        const key = edgeKey(r, c, r, c + 1);
        if (!pathEdges.has(key)) {
          nonPathEdges.push(normalizeWallEdge(r, c, r, c + 1));
        }
      }
      if (r + 1 < size) {
        const key = edgeKey(r, c, r + 1, c);
        if (!pathEdges.has(key)) {
          nonPathEdges.push(normalizeWallEdge(r, c, r + 1, c));
        }
      }
    }
  }

  const walls: WallEdge[] = [];
  for (const edge of nonPathEdges) {
    if (rng() < wallDensity) {
      walls.push(edge);
    }
  }

  boardCounter += 1;
  return {
    id: `zip-${size}x${size}-${boardCounter}-${Math.floor(rng() * 100000)}`,
    size,
    waypoints,
    walls,
    solutionPath: path,
    wallDensity,
  };
}
