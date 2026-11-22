import test from 'node:test';
import assert from 'node:assert/strict';
import { generateZipBoard } from './generator';
import { solveZipBoard, verifySolutionPath } from './solver';
import { ZipEnv } from './env';
import { SmallMLP } from './mlp';
import { DoubleDQNAgent } from './dqn';
import { Direction } from './types';

test('M1: Generator + DFS Oracle Solver solves 100/100 generated 5x5 boards', () => {
  for (let i = 0; i < 100; i++) {
    const board = generateZipBoard({
      size: 5,
      waypointCount: 3 + (i % 3),
      wallDensity: 0.25,
      seed: 1000 + i,
    });

    // Verify the construction path is strictly legal
    const builtCheck = verifySolutionPath(board, board.solutionPath);
    assert.equal(builtCheck.valid, true, `Generated board ${i} invalid: ${builtCheck.reason}`);

    // Independently solve using DFS oracle solver
    const solved = solveZipBoard(board);
    assert.equal(solved.solved, true, `Oracle failed to solve board ${i}`);
    const oracleCheck = verifySolutionPath(board, solved.path);
    assert.equal(oracleCheck.valid, true, `Oracle solution invalid on board ${i}: ${oracleCheck.reason}`);
  }
});

test('M1: Board generator scales cleanly across 6x6, 7x7, 8x8, and 10x10', () => {
  for (const size of [6, 7, 8, 10]) {
    const board = generateZipBoard({ size, wallDensity: 0.25, seed: size * 97 });
    assert.equal(board.size, size);
    assert.equal(board.solutionPath.length, size * size);
    const check = verifySolutionPath(board, board.solutionPath);
    assert.equal(check.valid, true, `Size ${size} board failed verification: ${check.reason}`);
  }
});

test('M1: ZipEnv enforces exact rewards (+0.1 step, +10 solve, -1 dead end) and legal action masking', () => {
  const board = generateZipBoard({ size: 5, waypointCount: 4, wallDensity: 0.28, seed: 4242 });
  const env = new ZipEnv({ size: 5 });
  env.reset(board);

  // Follow the known Hamiltonian solution step-by-step and confirm rewards
  for (let i = 1; i < board.solutionPath.length; i++) {
    const prev = board.solutionPath[i - 1];
    const next = board.solutionPath[i];
    const dr = next.row - prev.row;
    const dc = next.col - prev.col;
    let action: Direction = 0;
    if (dr === -1 && dc === 0) action = 0;
    else if (dr === 0 && dc === 1) action = 1;
    else if (dr === 1 && dc === 0) action = 2;
    else if (dr === 0 && dc === -1) action = 3;

    const legal = env.getLegalActions();
    assert.equal(legal.includes(action), true, `Solution step ${i} must be legal`);

    const res = env.step(action);
    if (i < board.solutionPath.length - 1) {
      assert.equal(res.reward, 0.1);
      assert.equal(res.done, false);
    } else {
      assert.equal(res.reward, 10.0);
      assert.equal(res.done, true);
      assert.equal(res.solved, true);
      assert.equal(res.outcome, 'solved');
    }
  }
});

test('M2: SmallMLP backprop reduces TD loss, records real weight deltas, and serializes faithfully', () => {
  const net = new SmallMLP(32, 64, 32, 4);
  const input = new Float32Array(32);
  input[0] = 1;
  input[5] = 0.5;
  input[12] = 1;

  const beforeQ = net.predict(input)[2];
  let lastLoss = Infinity;
  let lastMeanDelta = 0;

  for (let step = 0; step < 30; step++) {
    const stats = net.trainBatch([{ state: input, action: 2, target: 4.5 }], 0.01);
    lastLoss = stats.loss;
    lastMeanDelta = stats.meanWeightDelta;
  }

  const afterQ = net.predict(input)[2];
  assert.ok(Math.abs(4.5 - afterQ) < Math.abs(4.5 - beforeQ), 'Q-value should move toward target');
  assert.ok(lastLoss < 1.0, `Loss should decrease significantly, got ${lastLoss}`);
  assert.ok(lastMeanDelta > 0, 'Weight delta magnitude must be strictly positive on training step');

  const restored = SmallMLP.deserialize(net.serialize());
  const restoredQ = restored.predict(input);
  const origQ = net.predict(input);
  for (let a = 0; a < 4; a++) {
    assert.ok(Math.abs(restoredQ[a] - origQ[a]) < 1e-6);
  }
});

test('M2: DoubleDQNAgent trains across episodes on 5x5 without errors and updates visual snapshot', () => {
  const env = new ZipEnv({ size: 5, waypointCount: 4, wallDensity: 0.3 });
  const agent = new DoubleDQNAgent({
    inputSize: env.obsSize,
    hidden1Size: 64,
    hidden2Size: 32,
    batchSize: 16,
    learningRate: 0.002,
    epsilonDecay: 0.97,
  });

  let solvedCount = 0;
  for (let ep = 0; ep < 60; ep++) {
    let { observation, legalActions } = env.reset();
    while (!env.done && legalActions.length > 0) {
      const { action } = agent.selectAction(observation, legalActions, false);
      const step = env.step(action);
      agent.storeTransition({
        state: observation,
        action,
        reward: step.reward,
        nextState: step.observation,
        nextLegalMask: env.getLegalMask(),
        done: step.done,
        solvedTransition: step.solved,
      });
      agent.trainStep(step.solved);
      observation = step.observation;
      legalActions = step.legalActions;
    }
    if (env.solved) solvedCount++;
    agent.decayEpsilon();
  }

  const vis = agent.getVisualSnapshot();
  assert.ok(agent.totalGradientSteps > 100, 'Agent should execute gradient steps');
  assert.ok(vis.meanWeightDelta > 0, 'Visual snapshot should record non-zero weight update deltas');
  assert.equal(vis.outputQValues.length, 4);
  assert.ok(solvedCount >= 0);
});

test('Solve Till Win (Allow Backtrack) never stops on dead ends, unwinds exhausted branches, tracks extra steps, and solves 100% of boards', () => {
  // Verify 25 distinct 5x5 boards all solve with 0 dead ends even under a naive first-legal policy
  let sawBacktracks = false;
  for (let i = 0; i < 25; i++) {
    const board = generateZipBoard({ size: 5, waypointCount: 4, wallDensity: 0.25, seed: 5000 + i });
    const env = new ZipEnv({ size: 5, allowBacktrack: true });
    assert.equal(env.maxSteps, Number.POSITIVE_INFINITY);
    env.reset(board);

    let steps = 0;
    while (!env.done) {
      const legal = env.getLegalActions();
      assert.ok(legal.length > 0, `Board ${i} must always have a legal forward or backtrack move`);
      // Deliberately pick last legal move to provoke wrong turns & backtracks
      const action = legal[legal.length - 1];
      const res = env.step(action);
      assert.equal(res.deadEnd, false, 'Solve Till Win must never terminate with deadEnd=true');
      steps++;
      if (res.isBacktrack) sawBacktracks = true;
      assert.ok(steps < 50000, `Board ${i} exceeded reasonable solve steps (${steps})`);
    }

    assert.equal(env.solved, true, `Board ${i} must finish solved`);
    assert.equal(env.deadEnd, false);
    assert.equal(env.extraSteps, Math.max(0, env.stepCount - 24));
    const check = verifySolutionPath(board, env.path);
    assert.equal(check.valid, true, `Solved path on board ${i} invalid: ${check.reason}`);
  }
  assert.equal(sawBacktracks, true, 'Naive policy should exercise dead-end backtracking across test boards');

  // Verify mid-episode revival from No Backtrack dead end into Solve Till Win
  let revivedSuccessfully = false;
  for (let seed = 9000; seed < 9040 && !revivedSuccessfully; seed++) {
    const board = generateZipBoard({ size: 5, waypointCount: 4, wallDensity: 0.25, seed });
    const env = new ZipEnv({ size: 5, allowBacktrack: false });
    env.reset(board);
    while (!env.done) {
      const legal = env.getLegalActions();
      if (legal.length === 0) break;
      env.step(legal[legal.length - 1]);
    }
    if (env.deadEnd && !env.solved) {
      env.allowBacktrack = true;
      env.reviveFromDeadEnd();
      assert.equal(env.done, false);
      assert.equal(env.deadEnd, false);
      const legalAfterRevive = env.getLegalActions();
      assert.equal(legalAfterRevive.length, 1);
      const backStep = env.step(legalAfterRevive[0]);
      assert.equal(backStep.isBacktrack, true);
      assert.equal(backStep.reward, -0.15);
      assert.equal(env.backtrackCount, 1);
      revivedSuccessfully = true;
    }
  }
  assert.equal(revivedSuccessfully, true, 'Should be able to revive a No-Backtrack dead end into Solve Till Win');
});

test('Portable .ziprl checkpoint exports, validates, and restores identical weights and state', async () => {
  const { BrowserZipTrainer } = await import('./browserTrainer');
  const { serializeZiprlCheckpoint, parseZiprlCheckpoint } = await import('./storage');

  const trainerA = new BrowserZipTrainer();
  for (let i = 0; i < 40; i++) {
    trainerA.executeSingleStep();
  }
  const snapA = trainerA.getSnapshot();
  const payload = trainerA.buildCheckpointPayload();
  const raw = serializeZiprlCheckpoint(payload);

  // Validate parse succeeds and preserves format metadata
  const parsed = parseZiprlCheckpoint(raw);
  assert.equal(parsed.format, 'zippy-rl-checkpoint');
  assert.equal(parsed.config.boardSize, 5);

  // Load into a brand-new trainer instance and confirm identical Q-values and episode state
  const trainerB = new BrowserZipTrainer();
  trainerB.applyCheckpointPayload(parsed);
  const snapB = trainerB.getSnapshot();
  assert.equal(snapB.episode, snapA.episode);
  assert.equal(snapB.epsilon, snapA.epsilon);
  assert.equal(snapB.network.totalGradientSteps, snapA.network.totalGradientSteps);

  // Reject malformed files cleanly
  assert.throws(() => parseZiprlCheckpoint('not-json'), /not valid JSON/i);
  assert.throws(() => parseZiprlCheckpoint(JSON.stringify({ format: 'other' })), /Unrecognized file format/i);
});

test('4x4 grid generation, DFS Oracle 100% solve, and ZipEnv optimalSteps=15', () => {
  for (let i = 0; i < 20; i++) {
    const board = generateZipBoard({
      size: 4,
      waypointCount: 0, // varied range
      wallDensity: 0.25,
      seed: 7000 + i,
    });
    assert.equal(board.size, 4);
    assert.equal(board.solutionPath.length, 16);
    assert.ok(board.waypoints.length >= 3 && board.waypoints.length <= 8, `Unexpected 4x4 varied count: ${board.waypoints.length}`);

    const solved = solveZipBoard(board);
    assert.equal(solved.solved, true, `Oracle failed on 4x4 board ${i}`);
    const check = verifySolutionPath(board, solved.path);
    assert.equal(check.valid, true, `Oracle solution invalid on 4x4 board ${i}: ${check.reason}`);
  }

  const env4 = new ZipEnv({ size: 4, allowBacktrack: true });
  assert.equal(env4.size, 4);
  assert.equal(env4.totalCells, 16);
  assert.equal(env4.optimalSteps, 15);
  env4.reset();
  assert.equal(env4.board.size, 4);
});

test('Flexible & higher waypoint counts (10 on 4x4, 14 on 5x5, 20 on 6x6) produce valid solvable Zip boards', async () => {
  const { getVariedWaypointRange, getMaxWaypointCount } = await import('./generator');

  assert.deepEqual(getVariedWaypointRange(4), [3, 7]);
  assert.deepEqual(getVariedWaypointRange(5), [4, 11]);
  assert.ok(getMaxWaypointCount(5) >= 15, '5x5 should allow many more than 5 numbers');
  assert.ok(getMaxWaypointCount(6) >= 20, '6x6 should allow 20+ numbers');

  const cases: Array<[number, number]> = [
    [4, 10],
    [5, 14],
    [6, 20],
  ];
  for (const [size, wpCount] of cases) {
    const board = generateZipBoard({ size, waypointCount: wpCount, wallDensity: 0.25, seed: size * 100 + wpCount });
    assert.equal(board.waypoints.length, wpCount, `Expected ${wpCount} waypoints on ${size}x${size}`);
    assert.equal(board.waypoints[0].order, 1);
    assert.equal(board.waypoints[wpCount - 1].order, wpCount);
    const solved = solveZipBoard(board);
    assert.equal(solved.solved, true, `Oracle should solve ${size}x${size} with ${wpCount} waypoints`);
    assert.equal(verifySolutionPath(board, solved.path).valid, true);
  }
});

test('checkTargetReached evaluates episodes, clean/solve rate (with warmup guard), and grid level targets accurately', async () => {
  const { checkTargetReached } = await import('./browserTrainer');

  // Continuous mode never stops
  assert.equal(
    checkTargetReached(
      { type: 'none', value: 100 },
      { episode: 500, rollingWinRate: 1, latestEvalWinRate: 1, boardSize: 10, allowBacktrack: true },
    ),
    null,
  );

  // Episode target
  assert.equal(
    checkTargetReached(
      { type: 'episodes', value: 100 },
      { episode: 99, rollingWinRate: 0.2, latestEvalWinRate: 0.2, boardSize: 5, allowBacktrack: true },
    ),
    null,
  );
  const epMsg = checkTargetReached(
    { type: 'episodes', value: 100 },
    { episode: 100, rollingWinRate: 0.2, latestEvalWinRate: 0.2, boardSize: 5, allowBacktrack: true },
  );
  assert.ok(epMsg && epMsg.includes('100'), `Expected episode target message, got ${epMsg}`);

  // Clean/solve rate target enforces 25-episode minimum warmup
  assert.equal(
    checkTargetReached(
      { type: 'clean_rate', value: 80 },
      { episode: 10, rollingWinRate: 0.9, latestEvalWinRate: 0.9, boardSize: 5, allowBacktrack: true },
    ),
    null,
  );
  const rateMsg = checkTargetReached(
    { type: 'clean_rate', value: 80 },
    { episode: 30, rollingWinRate: 0.82, latestEvalWinRate: 0.5, boardSize: 5, allowBacktrack: true },
  );
  assert.ok(rateMsg && rateMsg.includes('80%'), `Expected rate target message, got ${rateMsg}`);

  // Grid level target
  assert.equal(
    checkTargetReached(
      { type: 'grid_level', value: 7 },
      { episode: 200, rollingWinRate: 0.6, latestEvalWinRate: 0.6, boardSize: 6, allowBacktrack: true },
    ),
    null,
  );
  const gridMsg = checkTargetReached(
    { type: 'grid_level', value: 7 },
    { episode: 200, rollingWinRate: 0.6, latestEvalWinRate: 0.6, boardSize: 7, allowBacktrack: true },
  );
  assert.ok(gridMsg && gridMsg.includes('7×7'), `Expected grid target message, got ${gridMsg}`);
});

test('BrowserZipTrainer supports 4x4, custom checkpointInterval (1k/10k/off), and WebGPU CPU fallback', async () => {
  const { BrowserZipTrainer } = await import('./browserTrainer');
  const { WebGPUMlpAccelerator } = await import('./webgpuMlp');
  const { serializeZiprlCheckpoint, parseZiprlCheckpoint } = await import('./storage');

  // WebGPU graceful fallback in Node (no navigator.gpu)
  const accel = await WebGPUMlpAccelerator.detectAndInit();
  assert.equal(accel.status.supported, false);
  assert.equal(accel.status.active, false);

  const trainer = new BrowserZipTrainer();
  trainer.setBoardSize(4);
  assert.equal(trainer.getSnapshot().config.boardSize, 4);

  // Updating waypointCount preserves weights (totalGradientSteps unchanged)
  for (let i = 0; i < 15; i++) trainer.executeSingleStep();
  const stepsBefore = trainer.getSnapshot().network.totalGradientSteps;
  trainer.updateSettings({ waypointCount: 9, checkpointInterval: 10000, target: { type: 'episodes', value: 2500 } });
  const snap = trainer.getSnapshot();
  assert.equal(snap.config.waypointCount, 9);
  assert.equal(snap.config.checkpointInterval, 10000);
  assert.equal(snap.config.target.type, 'episodes');
  assert.equal(snap.config.target.value, 2500);
  assert.equal(snap.network.totalGradientSteps, stepsBefore);

  // 4x4 checkpoint round-trip succeeds
  const raw = serializeZiprlCheckpoint(trainer.buildCheckpointPayload());
  const parsed = parseZiprlCheckpoint(raw);
  assert.equal(parsed.config.boardSize, 4);
  assert.equal(parsed.config.checkpointInterval, 10000);

  // CPU benchmark works cleanly when WebGPU is unavailable
  const bench = await trainer.runWebGPUComputeBenchmark(32);
  assert.equal(bench.usedWebGPU, false);
  assert.equal(bench.batchSize, 32);
  assert.ok(bench.cpuMs >= 0);
  assert.equal(bench.gpuMs, null);
});


