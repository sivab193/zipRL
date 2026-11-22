import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Pause, Play } from 'lucide-react';
import { DashboardSnapshot, EpisodeReplay, TrainingConfig } from './rl/types';
import { BrowserZipTrainer } from './rl/browserTrainer';
import { ZipBoardView } from './components/ZipBoardView';
import { MoveFeed } from './components/MoveFeed';
import { LearningCurves } from './components/LearningCurves';
import { Network3DView } from './components/Network3DView';
import { ReplayPanel } from './components/ReplayPanel';
import { ControlBar } from './components/ControlBar';
import { StatsStrip } from './components/StatsStrip';
import { HowItWorksPanel } from './components/HowItWorksPanel';

type ActiveTab = 'lab' | 'curves' | 'network' | 'replay' | 'about';

const NAV_TABS: Array<{ id: ActiveTab; label: string }> = [
  { id: 'lab', label: 'Live Lab' },
  { id: 'curves', label: 'Learning Curves' },
  { id: 'network', label: '3D Network' },
  { id: 'replay', label: 'Episode Replay' },
  { id: 'about', label: 'How It Works' },
];

export default function App() {
  const trainerRef = useRef<BrowserZipTrainer | null>(null);
  if (!trainerRef.current) {
    trainerRef.current = new BrowserZipTrainer();
  }

  const [snapshot, setSnapshot] = useState<DashboardSnapshot>(() =>
    trainerRef.current!.getSnapshot(),
  );
  const [activeTab, setActiveTab] = useState<ActiveTab>('lab');
  const [busyBlock, setBusyBlock] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof document !== 'undefined') {
      return document.documentElement.classList.contains('dark');
    }
    return false;
  });

  const showNotice = useCallback((msg: string) => {
    setNotice(msg);
  }, []);

  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(t);
  }, [notice]);

  // Sync light/dark class on <html> + persist preference
  const handleToggleTheme = useCallback(() => {
    setIsDark((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle('dark', next);
      try {
        localStorage.setItem('zippy-rl-theme', next ? 'dark' : 'light');
      } catch {
        // Ignore storage restrictions
      }
      return next;
    });
  }, []);

  // Subscribe to browser-native trainer & auto-resume latest IndexedDB checkpoint
  useEffect(() => {
    const trainer = trainerRef.current!;
    const unsubscribe = trainer.subscribe((snap) => {
      setSnapshot(snap);
    });
    void trainer.init();
    return () => {
      trainer.pause();
      unsubscribe();
    };
  }, []);

  const handleToggleTraining = useCallback(() => {
    const trainer = trainerRef.current!;
    if (snapshot.running) {
      trainer.pause();
    } else {
      trainer.start();
    }
  }, [snapshot.running]);

  const handleStepOnce = useCallback(() => {
    trainerRef.current!.stepOnceAndEmit();
  }, []);

  const handleTrainBlock = useCallback(
    async (episodes: number) => {
      setBusyBlock(true);
      try {
        const summary = await trainerRef.current!.runEpisodeBlockAsync(episodes);
        showNotice(
          `Completed ${summary.episodesCompleted} episodes · Held-out eval win rate: ${(summary.evalWinRate * 100).toFixed(1)}% · Saved to browser storage`,
        );
      } catch (err) {
        showNotice(err instanceof Error ? err.message : 'Training block failed');
      } finally {
        setBusyBlock(false);
      }
    },
    [showNotice],
  );

  const handleNewPuzzle = useCallback(() => {
    trainerRef.current!.newPuzzle();
  }, []);

  const handleRunOracle = useCallback(() => {
    const oracle = trainerRef.current!.runOracleOnCurrentBoard();
    if (oracle.solved) {
      showNotice(
        `DFS Oracle solved ${snapshot.config.boardSize}×${snapshot.config.boardSize} board in ${oracle.elapsedMs.toFixed(2)} ms (100% baseline)`,
      );
    }
  }, [snapshot.config.boardSize, showNotice]);

  const handleUpdateSettings = useCallback((partial: Partial<TrainingConfig>) => {
    trainerRef.current!.updateSettings(partial);
  }, []);

  const handleSaveCheckpoint = useCallback(async () => {
    const info = await trainerRef.current!.saveCheckpoint();
    showNotice(`Checkpoint saved locally at Episode ${info.episode}`);
  }, [showNotice]);

  const handleLoadCheckpoint = useCallback(
    async (id: string) => {
      try {
        const info = await trainerRef.current!.loadCheckpointById(id);
        showNotice(
          `Resumed checkpoint from Episode ${info.episode} (${info.boardSize}×${info.boardSize})`,
        );
      } catch (err) {
        showNotice(err instanceof Error ? err.message : 'Failed to load checkpoint');
      }
    },
    [showNotice],
  );

  const handleDownloadCheckpoint = useCallback(
    async (id?: string) => {
      try {
        const info = await trainerRef.current!.downloadCheckpoint(id);
        showNotice(`Exported ${info.filename} (Episode ${info.episode}, ${info.boardSize}×${info.boardSize})`);
      } catch (err) {
        showNotice(err instanceof Error ? err.message : 'Failed to export .ziprl checkpoint');
      }
    },
    [showNotice],
  );

  const handleUploadZiprlFile = useCallback(
    async (file: File) => {
      try {
        const rawText = await file.text();
        const info = await trainerRef.current!.importZiprlText(rawText);
        showNotice(
          `Loaded ${file.name} · Resuming at Episode ${info.episode} (${info.boardSize}×${info.boardSize}, ε=${info.epsilon.toFixed(3)})`,
        );
      } catch (err) {
        showNotice(err instanceof Error ? err.message : 'Invalid .ziprl checkpoint file');
      }
    },
    [showNotice],
  );

  const handleResetModel = useCallback(() => {
    trainerRef.current!.resetModel();
    showNotice('Model weights, replay buffer, and episode counters reset');
  }, [showNotice]);

  const getReplay = useCallback((episodeNum: number): EpisodeReplay | null => {
    return trainerRef.current!.getEpisodeReplay(episodeNum);
  }, []);

  const handleRunComputeBenchmark = useCallback(async () => {
    return trainerRef.current!.runWebGPUComputeBenchmark(128);
  }, []);

  const { liveBoard } = snapshot;
  const totalCells = liveBoard.board.size * liveBoard.board.size;
  const latestMove =
    snapshot.recentMoves.length > 0
      ? snapshot.recentMoves[snapshot.recentMoves.length - 1]
      : null;

  // Compute active target progress bar metrics
  const targetProgress = (() => {
    const { target, boardSize, allowBacktrack } = snapshot.config;
    if (target.type === 'none' || target.value <= 0) return null;
    if (target.type === 'episodes') {
      const pct = Math.min(100, (snapshot.episode / target.value) * 100);
      return {
        label: `Target: ${target.value.toLocaleString()} ${allowBacktrack ? 'boards solved' : 'episodes'}`,
        detail: `${snapshot.episode.toLocaleString()} / ${target.value.toLocaleString()} (${pct.toFixed(1)}%)`,
        pct,
      };
    }
    if (target.type === 'clean_rate') {
      const curr = Math.max(snapshot.rollingWinRate, snapshot.latestEvalWinRate) * 100;
      const pct = Math.min(100, (curr / target.value) * 100);
      return {
        label: `Target: ${target.value}% ${allowBacktrack ? 'clean solve rate' : 'win rate'}`,
        detail: `${curr.toFixed(1)}% / ${target.value}%${snapshot.episode < 25 ? ' (warming up 25 ep)' : ''}`,
        pct,
      };
    }
    const pct = Math.min(100, Math.max(0, ((boardSize - 4) / Math.max(1, target.value - 4)) * 100));
    return {
      label: `Target: Graduate to ${target.value}×${target.value} grid`,
      detail: `Current ${boardSize}×${boardSize} → Goal ${target.value}×${target.value}`,
      pct,
    };
  })();

  return (
    <div className="min-h-screen flex flex-col bg-[#F6F8FB] dark:bg-[#090D16] text-[#0F172A] dark:text-slate-100 transition-colors">
      {/* Top Navigation Bar — Strict 3-Zone Contract on Desktop */}
      <header className="sticky top-0 z-20 bg-white dark:bg-[#0F1523] border-b border-slate-200 dark:border-slate-800 transition-colors">
        <div className="max-w-[1440px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4 md:gap-8">
          {/* Zone 1: Single-line Brand Wordmark */}
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('lab');
            }}
            className="text-lg font-bold tracking-tight text-slate-900 dark:text-white whitespace-nowrap shrink-0"
          >
            Zippy RL
          </a>

          {/* Zone 2: 4 Clean Single-Line Nav Tabs (Desktop) */}
          <nav aria-label="Dashboard views" className="hidden md:flex items-center gap-7">
            {NAV_TABS.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`relative h-14 text-sm font-medium whitespace-nowrap shrink-0 transition-colors ${
                    isActive
                      ? 'text-slate-900 dark:text-white'
                      : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  {tab.label}
                  {isActive && (
                    <span
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-500"
                      aria-hidden="true"
                    />
                  )}
                </button>
              );
            })}
          </nav>

          {/* Zone 3: 1 Primary Action */}
          <div className="shrink-0">
            <button
              type="button"
              onClick={handleToggleTraining}
              className={`px-4 py-2 rounded-md text-xs font-semibold inline-flex items-center gap-1.5 whitespace-nowrap shrink-0 transition-colors ${
                snapshot.running
                  ? 'bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-white'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {snapshot.running ? (
                <>
                  <Pause className="w-3.5 h-3.5" />
                  Pause Training
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5" />
                  Start Training
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Mobile Navigation Strip (non-sticky so mobile sticky height stays < 7% of viewport) */}
      <nav
        aria-label="Mobile dashboard views"
        className="md:hidden bg-white dark:bg-[#0F1523] border-b border-slate-200 dark:border-slate-800 px-4 flex items-center gap-5 overflow-x-auto transition-colors"
      >
        {NAV_TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`relative py-2.5 text-xs font-medium whitespace-nowrap shrink-0 transition-colors ${
                isActive
                  ? 'text-slate-900 dark:text-white'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              {tab.label}
              {isActive && (
                <span
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600 dark:bg-blue-500"
                  aria-hidden="true"
                />
              )}
            </button>
          );
        })}
      </nav>

      {/* Secondary Workspace Control Bar */}
      <ControlBar
        running={snapshot.running}
        isDark={isDark}
        config={snapshot.config}
        checkpoints={snapshot.checkpoints}
        activeCheckpoint={snapshot.activeCheckpoint}
        busyBlock={busyBlock}
        onToggleTheme={handleToggleTheme}
        onStepOnce={handleStepOnce}
        onTrainBlock={handleTrainBlock}
        onNewPuzzle={handleNewPuzzle}
        onRunOracle={handleRunOracle}
        onUpdateSettings={handleUpdateSettings}
        onSaveCheckpoint={handleSaveCheckpoint}
        onLoadCheckpoint={handleLoadCheckpoint}
        onDownloadCheckpoint={handleDownloadCheckpoint}
        onUploadZiprlFile={handleUploadZiprlFile}
        onResetModel={handleResetModel}
      />

      {/* Main Content Container */}
      <main id="top" className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 py-5 space-y-5">
        {(notice || snapshot.targetReachedMessage) && (
          <div
            role="status"
            className="px-4 py-2.5 rounded-md bg-slate-900 dark:bg-slate-800 border border-slate-800 dark:border-slate-700 text-white text-xs font-medium flex items-center justify-between gap-4"
          >
            <span>{snapshot.targetReachedMessage ?? notice}</span>
            <button
              type="button"
              onClick={() => {
                setNotice(null);
                if (snapshot.targetReachedMessage) {
                  trainerRef.current!.clearTargetNotice();
                }
              }}
              className="text-slate-300 hover:text-white text-xs shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Active Training Target Progress Strip */}
        {targetProgress && (
          <div className="bg-white dark:bg-[#0F1523] border border-blue-200 dark:border-blue-900/60 rounded-lg px-4 py-2.5 space-y-1.5 transition-colors">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="font-semibold text-slate-900 dark:text-slate-100">
                {targetProgress.label}
              </span>
              <div className="flex items-center gap-3">
                <span className="font-mono tabular-nums text-blue-700 dark:text-blue-400 font-medium">
                  {targetProgress.detail}
                </span>
                <button
                  type="button"
                  onClick={() => handleUpdateSettings({ target: { type: 'none', value: snapshot.config.target.value } })}
                  className="text-[11px] text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 underline"
                >
                  Clear
                </button>
              </div>
            </div>
            <div className="h-1.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <div
                className="h-full rounded-full bg-blue-600 dark:bg-blue-500 transition-all duration-300"
                style={{ width: `${Math.max(2, targetProgress.pct)}%` }}
              />
            </div>
          </div>
        )}

        {/* Persistent Quantitative Stats Strip */}
        <StatsStrip snapshot={snapshot} />

        {/* TAB 1: LIVE LAB */}
        {activeTab === 'lab' && (
          <div className="space-y-5">
            {/* Primary Row: Live Puzzle Board + Move Feed / Q-Values */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
              {/* Live Zip Board */}
              <section
                aria-label="Live puzzle board"
                className="lg:col-span-5 bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-5 flex flex-col justify-between transition-colors"
              >
                <div className="flex items-baseline justify-between gap-3 mb-4">
                  <div>
                    <h1 className="text-base font-semibold text-slate-900 dark:text-slate-100">
                      Live Zip Board
                    </h1>
                    <p className="text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums mt-0.5">
                      {liveBoard.board.size}×{liveBoard.board.size} · {liveBoard.path.length}/{totalCells} cells ·{' '}
                      {liveBoard.solved
                        ? liveBoard.backtracks === 0
                          ? 'Clean Solve (+10.0)'
                          : `Solved (+10.0 · +${liveBoard.extraSteps} extra)`
                        : liveBoard.deadEnd
                          ? 'Dead end (-1.0)'
                          : snapshot.config.allowBacktrack
                            ? `Step ${liveBoard.step} (${liveBoard.backtracks} undos) · Wp ${liveBoard.nextWaypointOrder}/${liveBoard.board.waypoints.length}`
                            : `Target waypoint ${liveBoard.nextWaypointOrder}/${liveBoard.board.waypoints.length}`}
                    </p>
                  </div>
                  <div className="text-right font-mono tabular-nums">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Return</div>
                    <div
                      className={`text-sm font-semibold ${
                        liveBoard.cumulativeReward >= 5
                          ? 'text-emerald-700 dark:text-emerald-400'
                          : liveBoard.cumulativeReward < 0
                            ? 'text-rose-700 dark:text-rose-400'
                            : 'text-slate-900 dark:text-slate-100'
                      }`}
                    >
                      {liveBoard.cumulativeReward >= 0
                        ? `+${liveBoard.cumulativeReward.toFixed(1)}`
                        : liveBoard.cumulativeReward.toFixed(1)}
                    </div>
                  </div>
                </div>

                <ZipBoardView
                  board={liveBoard.board}
                  path={liveBoard.path}
                  nextWaypointOrder={liveBoard.nextWaypointOrder}
                  solved={liveBoard.solved}
                  deadEnd={liveBoard.deadEnd}
                  oracleSolution={liveBoard.oracleSolution}
                  latestEvaluations={latestMove?.evaluations ?? []}
                  isDark={isDark}
                />

                <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
                  <span>
                    {liveBoard.board.walls.length} walls · {liveBoard.board.waypoints.length} waypoints ·{' '}
                    {snapshot.config.allowBacktrack
                      ? `Solve Till Win (optimal ${liveBoard.optimalSteps} steps)`
                      : 'No Backtrack (dead ends stop)'}
                  </span>
                  {liveBoard.oracleSolveMs !== null && (
                    <span className="font-mono tabular-nums text-emerald-700 dark:text-emerald-400 font-medium">
                      Oracle DFS: {liveBoard.oracleSolveMs.toFixed(2)} ms
                    </span>
                  )}
                </div>
              </section>

              {/* Move Feed (Agent Thinking View) */}
              <section aria-label="Move feed and Q-values" className="lg:col-span-7 flex flex-col">
                <MoveFeed moves={snapshot.recentMoves} />
              </section>
            </div>

            {/* 3D Neural Network Visualization */}
            <section aria-label="3D neural network weight updates" className="space-y-2">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    3D MLP Weight-Update Visualization
                  </h2>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Connections illuminate by exact Adam |Δw| magnitude on every replay-buffer step, with a gold burst when a solved episode&apos;s +10 reward propagates.
                  </p>
                </div>
                <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                  {snapshot.deviceLabel}
                </span>
              </div>
              <Network3DView network={snapshot.network} heightClass="h-[380px]" />
            </section>

            {/* Learning Curves */}
            <section aria-label="Learning curves" className="space-y-2">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Learning Curves
                </h2>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  Hover any chart to inspect episode values
                </span>
              </div>
              <LearningCurves
                metrics={snapshot.metrics}
                evalHistory={snapshot.evalHistory}
                curriculumThreshold={snapshot.config.curriculumThreshold}
                allowBacktrack={snapshot.config.allowBacktrack}
                boardSize={snapshot.config.boardSize}
                isDark={isDark}
                onTrainBlock={handleTrainBlock}
                busyBlock={busyBlock}
              />
            </section>
          </div>
        )}

        {/* TAB 2: LEARNING CURVES & EVAL TABLE */}
        {activeTab === 'curves' && (
          <div className="space-y-5">
            <LearningCurves
              metrics={snapshot.metrics}
              evalHistory={snapshot.evalHistory}
              curriculumThreshold={snapshot.config.curriculumThreshold}
              allowBacktrack={snapshot.config.allowBacktrack}
              boardSize={snapshot.config.boardSize}
              isDark={isDark}
              onTrainBlock={handleTrainBlock}
              busyBlock={busyBlock}
            />

            <section className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden transition-colors">
              <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  Held-Out Greedy Evaluation Checkpoints (50 Fixed Boards vs. 100% Oracle Baseline)
                </h2>
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                  {snapshot.evalHistory.length} evaluations
                </span>
              </div>
              {snapshot.evalHistory.length === 0 ? (
                <div className="p-8 text-center text-xs text-slate-500 dark:text-slate-400">
                  Greedy evaluations run automatically every 100 episodes or after clicking Train +100 Ep.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead className="bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                      <tr>
                        <th className="py-2.5 px-4 font-medium">Episode</th>
                        <th className="py-2.5 px-4 font-medium">Grid Size</th>
                        <th className="py-2.5 px-4 font-medium text-right">
                          {snapshot.config.allowBacktrack ? 'Clean (0-Undo) Eval' : 'Agent Eval Win Rate'}
                        </th>
                        <th className="py-2.5 px-4 font-medium text-right">Oracle Baseline</th>
                        <th className="py-2.5 px-4 font-medium text-right">Mean Reward</th>
                        <th className="py-2.5 px-4 font-medium text-right">Mean Steps</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80 font-mono tabular-nums">
                      {[...snapshot.evalHistory].reverse().map((ev, i) => (
                        <tr
                          key={`${ev.episode}-${i}`}
                          className="hover:bg-slate-50/70 dark:hover:bg-slate-800/40"
                        >
                          <td className="py-2.5 px-4 text-slate-900 dark:text-slate-100 font-medium">
                            #{ev.episode}
                          </td>
                          <td className="py-2.5 px-4 text-slate-600 dark:text-slate-400">
                            {ev.boardSize}×{ev.boardSize}
                          </td>
                          <td className="py-2.5 px-4 text-right font-semibold text-blue-700 dark:text-blue-400">
                            {(ev.winRate * 100).toFixed(1)}%
                          </td>
                          <td className="py-2.5 px-4 text-right text-emerald-700 dark:text-emerald-400">
                            100.0%
                          </td>
                          <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                            {ev.avgReward >= 0 ? `+${ev.avgReward.toFixed(2)}` : ev.avgReward.toFixed(2)}
                          </td>
                          <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                            {ev.avgSteps.toFixed(1)}
                            {ev.avgExtraSteps !== undefined && ev.avgExtraSteps > 0
                              ? ` (+${ev.avgExtraSteps.toFixed(1)})`
                              : ''}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </div>
        )}

        {/* TAB 3: EXPANDED 3D NETWORK VIEW */}
        {activeTab === 'network' && (
          <div className="space-y-5">
            <Network3DView network={snapshot.network} heightClass="h-[540px]" />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-4">
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  100% Browser-Native MLP
                </h3>
                <p className="mt-1.5 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  Compact MLP ({snapshot.network.inputSize} inputs → {snapshot.network.hidden1Size} ReLU → {snapshot.network.hidden2Size} ReLU → 4 directional Q-outputs) runs directly in your browser with zero backend required. Export or upload portable .ziprl checkpoints anytime.
                </p>
              </div>
              <div className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-4">
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  Honest Update Mechanics
                </h3>
                <p className="mt-1.5 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  Parameters update via Adam + Huber loss on every training step once the replay buffer warms up — never only on solved puzzles. Solved episodes inject a +10 terminal reward that creates a visibly larger gradient burst.
                </p>
              </div>
              <div className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-4">
                <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                  Double DQN Stability
                </h3>
                <p className="mt-1.5 text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  The online network selects the best legal next action while a periodically synchronized target network (every {snapshot.config.targetUpdateInterval} steps) evaluates it, preventing Q-value overestimation.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: EPISODE REPLAY */}
        {activeTab === 'replay' && (
          <ReplayPanel
            recentEpisodes={snapshot.recentEpisodes}
            getReplay={getReplay}
            isDark={isDark}
            onTrainBlock={handleTrainBlock}
            busyBlock={busyBlock}
          />
        )}

        {/* TAB 5: HOW IT WORKS / ABOUT / ORACLE BENCHMARK */}
        {activeTab === 'about' && (
          <HowItWorksPanel
            snapshot={snapshot}
            onUpdateSettings={handleUpdateSettings}
            onRunComputeBenchmark={handleRunComputeBenchmark}
          />
        )}
      </main>
    </div>
  );
}
