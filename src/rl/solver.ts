import { CellCoord, DIRECTION_DELTAS, Direction, ZipBoard } from './types';
import { edgeKey } from './generator';

export interface SolverResult {
  solved: boolean;
  path: CellCoord[];
  nodesExplored: number;
  elapsedMs: number;
}

/**
 * Verifies whether a candidate path obeys every Zip game rule on the given board:
 * 1. Starts at waypoint 1 and ends at the last waypoint.
 * 2. Visits every cell in the N x N grid exactly once (Hamiltonian path).
 * 3. Every consecutive step is orthogonal (distance 1) and does not cross a wall.
 * 4. All waypoints are visited in strictly ascending order 1, 2, ..., K.
 */
export function verifySolutionPath(board: ZipBoard, path: CellCoord[]): { valid: boolean; reason?: string } {
  const { size, waypoints, walls } = board;
  const totalCells = size * size;

  if (path.length !== totalCells) {
    return { valid: false, reason: `Path visits ${path.length} cells, expected ${totalCells}` };
  }

  const wallSet = new Set<string>(walls.map((w) => edgeKey(w.r1, w.c1, w.r2, w.c2)));
  const waypointMap = new Map<number, number>();
  for (const wp of waypoints) {
    waypointMap.set(wp.row * size + wp.col, wp.order);
  }

  const visited = new Uint8Array(totalCells);
  let expectedWaypoint = 1;

  for (let i = 0; i < path.length; i++) {
    const { row, col } = path[i];
    if (row < 0 || row >= size || col < 0 || col >= size) {
      return { valid: false, reason: `Cell (${row},${col}) out of bounds` };
    }
    const idx = row * size + col;
    if (visited[idx] === 1) {
      return { valid: false, reason: `Cell (${row},${col}) revisited at step ${i}` };
    }
    visited[idx] = 1;

    if (i > 0) {
      const prev = path[i - 1];
      const dist = Math.abs(row - prev.row) + Math.abs(col - prev.col);
      if (dist !== 1) {
        return { valid: false, reason: `Non-orthogonal move from (${prev.row},${prev.col}) to (${row},${col})` };
      }
      if (wallSet.has(edgeKey(prev.row, prev.col, row, col))) {
        return { valid: false, reason: `Wall crossed between (${prev.row},${prev.col}) and (${row},${col})` };
      }
    }

    const wpOrder = waypointMap.get(idx);
    if (wpOrder !== undefined) {
      if (wpOrder !== expectedWaypoint) {
        return {
          valid: false,
          reason: `Visited waypoint ${wpOrder} when expecting waypoint ${expectedWaypoint}`,
        };
      }
      expectedWaypoint++;
    }
  }

  if (expectedWaypoint !== waypoints.length + 1) {
    return { valid: false, reason: `Did not reach all waypoints (stopped before ${expectedWaypoint})` };
  }

  const lastWp = waypoints[waypoints.length - 1];
  const lastCell = path[path.length - 1];
  if (lastCell.row !== lastWp.row || lastCell.col !== lastWp.col) {
    return { valid: false, reason: 'Path did not terminate on the final waypoint' };
  }

  return { valid: true };
}

/**
 * Plain DFS backtracking oracle solver with wall checks, waypoint ordering,
 * dead-end degree pruning, and flood-fill connectivity pruning.
 * Serves as the 100% baseline and solvability verifier (independent of board.solutionPath).
 */
export function solveZipBoard(board: ZipBoard, maxNodes = 250000): SolverResult {
  const startTime = performance.now();
  const { size, waypoints, walls } = board;
  const totalCells = size * size;
  const wallSet = new Set<string>(walls.map((w) => edgeKey(w.r1, w.c1, w.r2, w.c2)));

  const sortedWaypoints = [...waypoints].sort((a, b) => a.order - b.order);
  const startWp = sortedWaypoints[0];
  const endWp = sortedWaypoints[sortedWaypoints.length - 1];
  const endIdx = endWp.row * size + endWp.col;

  // Map cell index -> waypoint order (0 if none)
  const cellWaypoint = new Int16Array(totalCells);
  for (const wp of sortedWaypoints) {
    cellWaypoint[wp.row * size + wp.col] = wp.order;
  }

  // Precompute legal adjacency list per cell (respecting walls)
  const adj: number[][] = Array.from({ length: totalCells }, () => []);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const idx = r * size + c;
      for (let d = 0; d < 4; d++) {
        const [dr, dc] = DIRECTION_DELTAS[d as Direction];
        const nr = r + dr;
        const nc = c + dc;
        if (nr >= 0 && nr < size && nc >= 0 && nc < size) {
          if (!wallSet.has(edgeKey(r, c, nr, nc))) {
            adj[idx].push(nr * size + nc);
          }
        }
      }
    }
  }

  const visited = new Uint8Array(totalCells);
  const bfsVisited = new Uint8Array(totalCells);
  const bfsQueue = new Int32Array(totalCells);
  const pathIdx = new Int32Array(totalCells);

  const startIdx = startWp.row * size + startWp.col;
  visited[startIdx] = 1;
  pathIdx[0] = startIdx;

  let nodesExplored = 0;
  let solved = false;

  const isRemainingConnected = (currIdx: number, visitedCount: number): boolean => {
    const targetCount = totalCells - visitedCount;
    if (targetCount <= 1) return true;

    bfsVisited.fill(0);
    let head = 0;
    let tail = 0;
    let seen = 0;

    // Seed BFS from currIdx's unvisited neighbors
    const neighbors = adj[currIdx];
    for (let i = 0; i < neighbors.length; i++) {
      const n = neighbors[i];
      if (visited[n] === 0 && bfsVisited[n] === 0) {
        bfsVisited[n] = 1;
        bfsQueue[tail++] = n;
        seen++;
      }
    }
    if (seen === 0) return false;

    while (head < tail) {
      const u = bfsQueue[head++];
      const uAdj = adj[u];
      for (let i = 0; i < uAdj.length; i++) {
        const v = uAdj[i];
        if (visited[v] === 0 && bfsVisited[v] === 0) {
          bfsVisited[v] = 1;
          bfsQueue[tail++] = v;
          seen++;
        }
      }
    }

    return seen === targetCount;
  };

  const dfs = (currIdx: number, depth: number, nextWpOrder: number): boolean => {
    nodesExplored++;
    if (nodesExplored > maxNodes) return false;

    if (depth === totalCells) {
      return currIdx === endIdx && nextWpOrder > sortedWaypoints.length;
    }

    // Connectivity check (fast and eliminates disconnected pockets immediately)
    if (!isRemainingConnected(currIdx, depth)) {
      return false;
    }

    // Check unvisited cells for dead ends (degree 0, or >1 cell with degree 1 besides endIdx)
    const remaining = totalCells - depth;
    let forcedMove = -1;
    if (remaining > 1) {
      for (let i = 0; i < totalCells; i++) {
        if (visited[i] === 1 || i === endIdx) continue;
        let deg = 0;
        let adjToCurr = false;
        const iAdj = adj[i];
        for (let j = 0; j < iAdj.length; j++) {
          const n = iAdj[j];
          if (visited[n] === 0) deg++;
          else if (n === currIdx) adjToCurr = true;
        }
        if (deg === 0 && !adjToCurr) {
          return false; // Isolated unreachable cell
        }
        if (deg === 1 && !adjToCurr) {
          // A non-end cell with only 1 unvisited exit and not adjacent to head is a dead end
          return false;
        }
        if (deg === 0 && adjToCurr) {
          if (forcedMove !== -1) return false; // Two dead ends adjacent to head
          forcedMove = i;
        }
      }
    }

    const neighbors = adj[currIdx];
    const candidates: Array<{ idx: number; deg: number }> = [];

    for (let i = 0; i < neighbors.length; i++) {
      const nextIdx = neighbors[i];
      if (visited[nextIdx] === 1) continue;
      if (forcedMove !== -1 && nextIdx !== forcedMove) continue;

      const wp = cellWaypoint[nextIdx];
      // Cannot step on a future waypoint out of order
      if (wp !== 0 && wp !== nextWpOrder) continue;
      // Final waypoint can only be entered on the very last move
      if (nextIdx === endIdx && depth + 1 !== totalCells) continue;

      let deg = 0;
      const nAdj = adj[nextIdx];
      for (let j = 0; j < nAdj.length; j++) {
        if (visited[nAdj[j]] === 0) deg++;
      }
      candidates.push({ idx: nextIdx, deg });
    }

    // Order candidates by Warnsdorff degree ascending
    candidates.sort((a, b) => a.deg - b.deg);

    for (let i = 0; i < candidates.length; i++) {
      const nextIdx = candidates[i].idx;
      const wp = cellWaypoint[nextIdx];
      const nextOrder = wp === nextWpOrder ? nextWpOrder + 1 : nextWpOrder;

      visited[nextIdx] = 1;
      pathIdx[depth] = nextIdx;
      if (dfs(nextIdx, depth + 1, nextOrder)) {
        return true;
      }
      visited[nextIdx] = 0;
    }

    return false;
  };

  solved = dfs(startIdx, 1, 2);

  const path: CellCoord[] = [];
  if (solved) {
    for (let i = 0; i < totalCells; i++) {
      const idx = pathIdx[i];
      path.push({ row: Math.floor(idx / size), col: idx % size });
    }
  }

  return {
    solved,
    path,
    nodesExplored,
    elapsedMs: Math.round((performance.now() - startTime) * 100) / 100,
  };
}
