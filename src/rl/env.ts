import { CellCoord, DIRECTION_DELTAS, Direction, ZipBoard } from './types';
import { edgeKey, generateZipBoard } from './generator';

export interface StepResult {
  observation: Float32Array;
  reward: number;
  done: boolean;
  solved: boolean;
  deadEnd: boolean;
  isBacktrack: boolean;
  outcome: 'continue' | 'solved' | 'dead_end' | 'step_cap';
  legalActions: Direction[];
  head: CellCoord;
  nextWaypointOrder: number;
  step: number;
  backtrackCount: number;
  extraSteps: number;
}

export interface ZipEnvConfig {
  size: number;
  waypointCount?: number;
  wallDensity?: number;
  allowBacktrack?: boolean;
}

const OPPOSITE_DIR: Record<Direction, Direction> = {
  0: 2,
  1: 3,
  2: 0,
  3: 1,
};

/**
 * Exact Zip RL environment (Gymnasium-style reset/step interface).
 *
 * Modes:
 * - Solve Till Win (allowBacktrack = true, real Zip rules):
 *     Episode NEVER ends on dead ends or step caps — it only completes when the board
 *     is solved (+10.0). When the agent hits a dead end (no untried forward moves),
 *     moving backward onto the immediate previous cell (path[len-2]) unwinds the path
 *     and marks the exhausted branch so it tries another direction instead of looping
 *     endlessly. Each wrong move / dead-end backtrack incurs a penalty (-0.15 undo,
 *     -0.2 dead-end entry) and increments backtrackCount / extraSteps, while +0.1
 *     forward reward is gated by high-water path length to prevent reward farming.
 * - No Backtrack (allowBacktrack = false, strict RL mode):
 *     Never revisit a cell; dead end gives -1.0 and ends the episode immediately.
 */
export class ZipEnv {
  public readonly size: number;
  public readonly totalCells: number;
  public readonly optimalSteps: number;
  public readonly obsSize: number;
  public allowBacktrack: boolean;

  public board!: ZipBoard;
  public path: CellCoord[] = [];
  public visited!: Uint8Array;
  public nextWaypointOrder = 2;
  public stepCount = 0;
  public backtrackCount = 0;
  public done = false;
  public solved = false;
  public deadEnd = false;
  public cumulativeReward = 0;

  private wallSet = new Set<string>();
  private cellWaypoint!: Int16Array;
  private waypointByOrder = new Map<number, CellCoord>();
  private highWaterPathLen = 1;
  /** Parallel to `path`: bitmask of forward directions already proven dead from each prefix. */
  private exhaustedDirs: number[] = [0];
  private defaultWaypointCount?: number;
  private defaultWallDensity: number;

  constructor(config: ZipEnvConfig) {
    this.size = Math.max(4, Math.min(10, Math.round(config.size)));
    this.totalCells = this.size * this.size;
    this.optimalSteps = this.totalCells - 1;
    // 8 grid channels + 20 local head-centric features (5 per dir) + 1 fill-progress scalar
    this.obsSize = this.totalCells * 8 + 21;
    this.allowBacktrack = Boolean(config.allowBacktrack);
    this.defaultWaypointCount = config.waypointCount;
    this.defaultWallDensity = config.wallDensity ?? 0.25;
  }

  public get maxSteps(): number {
    return this.allowBacktrack ? Number.POSITIVE_INFINITY : this.totalCells * 2;
  }

  public get extraSteps(): number {
    return Math.max(0, this.stepCount - this.optimalSteps);
  }

  public reset(customBoard?: ZipBoard): { observation: Float32Array; legalActions: Direction[]; board: ZipBoard } {
    this.board =
      customBoard && customBoard.size === this.size
        ? customBoard
        : generateZipBoard({
            size: this.size,
            waypointCount: this.defaultWaypointCount,
            wallDensity: this.defaultWallDensity,
          });

    this.wallSet = new Set<string>(
      this.board.walls.map((w) => edgeKey(w.r1, w.c1, w.r2, w.c2)),
    );
    this.cellWaypoint = new Int16Array(this.totalCells);
    this.waypointByOrder.clear();
    for (const wp of this.board.waypoints) {
      this.cellWaypoint[wp.row * this.size + wp.col] = wp.order;
      this.waypointByOrder.set(wp.order, { row: wp.row, col: wp.col });
    }

    const startWp = this.waypointByOrder.get(1) ?? this.board.waypoints[0];
    this.visited = new Uint8Array(this.totalCells);
    this.visited[startWp.row * this.size + startWp.col] = 1;
    this.path = [{ row: startWp.row, col: startWp.col }];
    this.exhaustedDirs = [0];
    this.nextWaypointOrder = 2;
    this.stepCount = 0;
    this.backtrackCount = 0;
    this.highWaterPathLen = 1;
    this.done = false;
    this.solved = false;
    this.deadEnd = false;
    this.cumulativeReward = 0;

    return {
      observation: this.getObservation(),
      legalActions: this.getLegalActions(),
      board: this.board,
    };
  }

  /**
   * When switching from No Backtrack to Solve Till Win mid-episode while sitting on
   * a dead end, revive the episode so the agent can immediately backtrack out.
   */
  public reviveFromDeadEnd(): void {
    if (this.allowBacktrack && this.deadEnd && !this.solved) {
      this.done = false;
      this.deadEnd = false;
    }
  }

  public getHead(): CellCoord {
    return this.path[this.path.length - 1];
  }

  public getPreviousCell(): CellCoord | null {
    return this.path.length >= 2 ? this.path[this.path.length - 2] : null;
  }

  public getBacktrackDirection(): Direction | null {
    if (!this.allowBacktrack || this.path.length < 2) return null;
    const head = this.getHead();
    const prev = this.path[this.path.length - 2];
    const dr = prev.row - head.row;
    const dc = prev.col - head.col;
    for (let d = 0; d < 4; d++) {
      const [ddr, ddc] = DIRECTION_DELTAS[d as Direction];
      if (ddr === dr && ddc === dc) return d as Direction;
    }
    return null;
  }

  public hasWallBetween(r1: number, c1: number, r2: number, c2: number): boolean {
    return this.wallSet.has(edgeKey(r1, c1, r2, c2));
  }

  private getForwardLegalActions(ignoreExhausted = false): Direction[] {
    const head = this.getHead();
    const legal: Direction[] = [];
    const lastOrder = this.board.waypoints.length;
    const isLastStep = this.path.length + 1 === this.totalCells;
    const exhaustedMask = this.allowBacktrack && !ignoreExhausted
      ? (this.exhaustedDirs[this.path.length - 1] ?? 0)
      : 0;

    for (let d = 0; d < 4; d++) {
      const dir = d as Direction;
      if ((exhaustedMask & (1 << dir)) !== 0) continue;

      const [dr, dc] = DIRECTION_DELTAS[dir];
      const nr = head.row + dr;
      const nc = head.col + dc;

      if (nr < 0 || nr >= this.size || nc < 0 || nc >= this.size) continue;
      if (this.hasWallBetween(head.row, head.col, nr, nc)) continue;

      const nIdx = nr * this.size + nc;
      if (this.visited[nIdx] === 1) continue;

      const wpOrder = this.cellWaypoint[nIdx];
      if (wpOrder !== 0) {
        if (wpOrder !== this.nextWaypointOrder) continue;
        // Final waypoint can only be entered when filling the very last cell
        if (wpOrder === lastOrder && !isLastStep) continue;
      }

      legal.push(dir);
    }

    return legal;
  }

  /**
   * Returns legal orthogonal directions from the current head.
   * In Solve Till Win (allowBacktrack) mode:
   * - If the current path prefix has already disconnected the remaining unvisited cells
   *   (a trapped pocket that cannot be completed), immediately offers backtracking so the
   *   agent unwinds out of the mistake instead of thrashing inside an unsolvable pocket.
   * - Otherwise offers all untried forward moves not yet proven dead from this prefix.
   * - Whenever 0 untried forward moves remain, offers backtracking onto the previous cell.
   */
  public getLegalActions(): Direction[] {
    if (this.done) return [];
    const forward = this.getForwardLegalActions(false);

    if (!this.allowBacktrack) {
      return forward;
    }

    const backDir = this.getBacktrackDirection();
    if (
      this.path.length > 1 &&
      this.path.length < this.totalCells &&
      !this.isRemainingConnectedFromHead() &&
      backDir !== null
    ) {
      return [backDir];
    }

    if (forward.length > 0) {
      return forward;
    }

    if (backDir !== null) {
      return [backDir];
    }

    // Safety fallback at start cell if exhausted mask was ever over-constrained: reset mask
    if (this.exhaustedDirs[0] !== 0) {
      this.exhaustedDirs[0] = 0;
      return this.getForwardLegalActions(true);
    }

    return [];
  }

  private isRemainingConnectedFromHead(): boolean {
    const remaining = this.totalCells - this.path.length;
    if (remaining <= 1) return true;
    const head = this.getHead();
    const N = this.size;
    const seen = new Uint8Array(this.totalCells);
    const queue = new Int32Array(this.totalCells);
    let qHead = 0;
    let qTail = 0;

    for (let d = 0; d < 4; d++) {
      const [dr, dc] = DIRECTION_DELTAS[d as Direction];
      const rr = head.row + dr;
      const cc = head.col + dc;
      if (rr < 0 || rr >= N || cc < 0 || cc >= N) continue;
      const idx = rr * N + cc;
      if (this.visited[idx] === 0 && !this.hasWallBetween(head.row, head.col, rr, cc)) {
        seen[idx] = 1;
        queue[qTail++] = idx;
      }
    }
    if (qTail === 0) return false;

    let count = qTail;
    while (qHead < qTail) {
      const u = queue[qHead++];
      const ur = Math.floor(u / N);
      const uc = u % N;
      for (let d = 0; d < 4; d++) {
        const [dr, dc] = DIRECTION_DELTAS[d as Direction];
        const vr = ur + dr;
        const vc = uc + dc;
        if (vr < 0 || vr >= N || vc < 0 || vc >= N) continue;
        const v = vr * N + vc;
        if (this.visited[v] === 0 && seen[v] === 0 && !this.hasWallBetween(ur, uc, vr, vc)) {
          seen[v] = 1;
          queue[qTail++] = v;
          count++;
        }
      }
    }
    return count === remaining;
  }

  public getLegalMask(): [boolean, boolean, boolean, boolean] {
    const mask: [boolean, boolean, boolean, boolean] = [false, false, false, false];
    const legal = this.getLegalActions();
    for (let i = 0; i < legal.length; i++) {
      mask[legal[i]] = true;
    }
    return mask;
  }

  private countOpenNeighbors(row: number, col: number, ignoreIdx: number): number {
    let deg = 0;
    for (let d = 0; d < 4; d++) {
      const [dr, dc] = DIRECTION_DELTAS[d as Direction];
      const nr = row + dr;
      const nc = col + dc;
      if (nr < 0 || nr >= this.size || nc < 0 || nc >= this.size) continue;
      const nIdx = nr * this.size + nc;
      if (nIdx === ignoreIdx || this.visited[nIdx] === 1) continue;
      if (this.hasWallBetween(row, col, nr, nc)) continue;
      deg++;
    }
    return deg;
  }

  public getObservation(): Float32Array {
    const obs = new Float32Array(this.obsSize);
    const N = this.size;
    const C = this.totalCells;
    const head = this.getHead();
    const prev = this.getPreviousCell();
    const lastOrder = this.board.waypoints.length;
    const nextWp = this.waypointByOrder.get(this.nextWaypointOrder);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx = r * N + c;
        obs[idx] = this.visited[idx];
        obs[C + idx] = r === head.row && c === head.col ? 1 : 0;
        obs[C * 2 + idx] = nextWp && r === nextWp.row && c === nextWp.col ? 1 : 0;
        const wpOrder = this.cellWaypoint[idx];
        obs[C * 3 + idx] = wpOrder >= this.nextWaypointOrder ? wpOrder / lastOrder : 0;
        obs[C * 4 + idx] = r === 0 || this.hasWallBetween(r, c, r - 1, c) ? 1 : 0;
        obs[C * 5 + idx] = c === N - 1 || this.hasWallBetween(r, c, r, c + 1) ? 1 : 0;
        obs[C * 6 + idx] = r === N - 1 || this.hasWallBetween(r, c, r + 1, c) ? 1 : 0;
        obs[C * 7 + idx] = c === 0 || this.hasWallBetween(r, c, r, c - 1) ? 1 : 0;
      }
    }

    const base = C * 8;
    const legalMask = this.getLegalMask();
    const headDistToTarget = nextWp
      ? Math.abs(head.row - nextWp.row) + Math.abs(head.col - nextWp.col)
      : 0;
    const remainingAfterMove = C - this.path.length - 1;

    for (let d = 0; d < 4; d++) {
      const offset = base + d * 5;
      const isLegal = legalMask[d];
      obs[offset] = isLegal ? 1 : 0;
      if (isLegal) {
        const [dr, dc] = DIRECTION_DELTAS[d as Direction];
        const nr = head.row + dr;
        const nc = head.col + dc;
        const isBacktrackMove = this.allowBacktrack && prev !== null && nr === prev.row && nc === prev.col;
        if (isBacktrackMove) {
          obs[offset + 1] = -0.5;
          obs[offset + 2] = 0;
          obs[offset + 3] = -0.5;
          obs[offset + 4] = 0;
        } else {
          const nIdx = nr * N + nc;
          const deg = this.countOpenNeighbors(nr, nc, nIdx);
          obs[offset + 1] = deg / 3;
          obs[offset + 2] = nextWp && nr === nextWp.row && nc === nextWp.col ? 1 : 0;
          const nextDist = nextWp ? Math.abs(nr - nextWp.row) + Math.abs(nc - nextWp.col) : 0;
          obs[offset + 3] = headDistToTarget > 0 ? headDistToTarget - nextDist : 0;
          obs[offset + 4] = this.keepsRemainingConnected(nr, nc, remainingAfterMove) ? 1 : -1;
        }
      }
    }

    obs[base + 20] = this.path.length / C;
    return obs;
  }

  private keepsRemainingConnected(nextRow: number, nextCol: number, remainingAfterMove: number): boolean {
    if (remainingAfterMove <= 1) return true;
    const N = this.size;
    const nextIdx = nextRow * N + nextCol;
    const seen = new Uint8Array(this.totalCells);
    seen[nextIdx] = 1;
    const queue = new Int32Array(this.totalCells);
    let qHead = 0;
    let qTail = 0;

    for (let d = 0; d < 4; d++) {
      const [dr, dc] = DIRECTION_DELTAS[d as Direction];
      const rr = nextRow + dr;
      const cc = nextCol + dc;
      if (rr < 0 || rr >= N || cc < 0 || cc >= N) continue;
      const idx = rr * N + cc;
      if (this.visited[idx] === 0 && !this.hasWallBetween(nextRow, nextCol, rr, cc)) {
        seen[idx] = 1;
        queue[qTail++] = idx;
      }
    }
    if (qTail === 0) return false;

    let count = qTail;
    while (qHead < qTail) {
      const u = queue[qHead++];
      const ur = Math.floor(u / N);
      const uc = u % N;
      for (let d = 0; d < 4; d++) {
        const [dr, dc] = DIRECTION_DELTAS[d as Direction];
        const vr = ur + dr;
        const vc = uc + dc;
        if (vr < 0 || vr >= N || vc < 0 || vc >= N) continue;
        const v = vr * N + vc;
        if (this.visited[v] === 0 && seen[v] === 0 && !this.hasWallBetween(ur, uc, vr, vc)) {
          seen[v] = 1;
          queue[qTail++] = v;
          count++;
        }
      }
    }

    return count === remainingAfterMove;
  }

  public step(action: Direction): StepResult {
    if (this.done) {
      return {
        observation: this.getObservation(),
        reward: 0,
        done: true,
        solved: this.solved,
        deadEnd: this.deadEnd,
        isBacktrack: false,
        outcome: this.solved ? 'solved' : this.deadEnd ? 'dead_end' : 'step_cap',
        legalActions: [],
        head: this.getHead(),
        nextWaypointOrder: this.nextWaypointOrder,
        step: this.stepCount,
        backtrackCount: this.backtrackCount,
        extraSteps: this.extraSteps,
      };
    }

    const legal = this.getLegalActions();
    if (!legal.includes(action)) {
      throw new Error(`Illegal action ${action} attempted; legal actions are [${legal.join(', ')}]`);
    }

    const head = this.getHead();
    const prev = this.getPreviousCell();
    const [dr, dc] = DIRECTION_DELTAS[action];
    const nr = head.row + dr;
    const nc = head.col + dc;

    const isBacktrack = this.allowBacktrack && prev !== null && nr === prev.row && nc === prev.col;
    this.stepCount += 1;

    let reward = 0;
    let outcome: 'continue' | 'solved' | 'dead_end' | 'step_cap' = 'continue';

    if (isBacktrack) {
      // Pop current head back one cell, restore waypoint order if leaving a waypoint,
      // and mark the popped direction as exhausted on the parent prefix.
      const popped = this.path.pop()!;
      this.exhaustedDirs.pop();
      const poppedIdx = popped.row * this.size + popped.col;
      this.visited[poppedIdx] = 0;
      const poppedWp = this.cellWaypoint[poppedIdx];
      if (poppedWp !== 0 && poppedWp === this.nextWaypointOrder - 1) {
        this.nextWaypointOrder -= 1;
      }
      const failedForwardDir = OPPOSITE_DIR[action];
      const parentDepth = this.path.length - 1;
      if (parentDepth >= 0) {
        this.exhaustedDirs[parentDepth] = (this.exhaustedDirs[parentDepth] ?? 0) | (1 << failedForwardDir);
      }

      this.backtrackCount += 1;
      reward = -0.15;
    } else {
      const nIdx = nr * this.size + nc;
      this.visited[nIdx] = 1;
      this.path.push({ row: nr, col: nc });
      this.exhaustedDirs.push(0);

      const wpOrder = this.cellWaypoint[nIdx];
      if (wpOrder === this.nextWaypointOrder) {
        this.nextWaypointOrder += 1;
      }

      const allWaypointsVisited = this.nextWaypointOrder > this.board.waypoints.length;
      const allCellsFilled = this.path.length === this.totalCells;

      // Grant +0.1 on forward progress (only past highWaterPathLen in backtrack mode to prevent reward farming)
      if (!this.allowBacktrack || this.path.length > this.highWaterPathLen) {
        reward = 0.1;
        if (this.path.length > this.highWaterPathLen) {
          this.highWaterPathLen = this.path.length;
        }
      } else {
        reward = 0.0;
      }

      if (allCellsFilled && allWaypointsVisited) {
        reward = 10.0; // +10 on solve
        this.done = true;
        this.solved = true;
        outcome = 'solved';
      } else if (!this.allowBacktrack) {
        // Strict No-Backtrack mode: end episode on step cap or dead end
        if (this.stepCount >= this.maxSteps) {
          reward = -1.0;
          this.done = true;
          this.deadEnd = true;
          outcome = 'step_cap';
        } else {
          const nextForward = this.getForwardLegalActions(true);
          if (nextForward.length === 0) {
            reward = -1.0;
            this.done = true;
            this.deadEnd = true;
            outcome = 'dead_end';
          }
        }
      } else {
        // Solve Till Win mode: never terminate on dead end!
        // If this move entered a dead end or disconnected the remaining grid, give -0.2 feedback
        // so Double DQN learns directly to avoid wrong turns, then let it backtrack out.
        const nextForward = this.getForwardLegalActions(false);
        if (nextForward.length === 0 || !this.isRemainingConnectedFromHead()) {
          reward = -0.2;
        }
      }
    }

    this.cumulativeReward = Math.round((this.cumulativeReward + reward) * 1000) / 1000;

    return {
      observation: this.getObservation(),
      reward,
      done: this.done,
      solved: this.solved,
      deadEnd: this.deadEnd,
      isBacktrack,
      outcome,
      legalActions: this.done ? [] : this.getLegalActions(),
      head: this.getHead(),
      nextWaypointOrder: this.nextWaypointOrder,
      step: this.stepCount,
      backtrackCount: this.backtrackCount,
      extraSteps: this.extraSteps,
    };
  }
}
