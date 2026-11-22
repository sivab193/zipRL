import React, { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { CellCoord, DashboardSnapshot, EpisodeReplay } from '../rl/types';
import { ZipBoardView } from './ZipBoardView';
import { QValueBreakdown } from './MoveFeed';

interface ReplayPanelProps {
  recentEpisodes: DashboardSnapshot['recentEpisodes'];
  getReplay: (episodeNum: number) => EpisodeReplay | null;
  isDark?: boolean;
  onTrainBlock?: (episodes: number) => void;
  busyBlock?: boolean;
}

export const ReplayPanel: React.FC<ReplayPanelProps> = ({
  recentEpisodes,
  getReplay,
  isDark = false,
  onTrainBlock,
  busyBlock = false,
}) => {
  const [filter, setFilter] = useState<'all' | 'clean' | 'with_errors'>('all');
  const [selectedEpisodeNum, setSelectedEpisodeNum] = useState<number | null>(null);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const filteredEpisodes = useMemo(() => {
    return recentEpisodes.filter((ep) => {
      if (filter === 'clean') return ep.cleanSolve;
      if (filter === 'with_errors') return !ep.cleanSolve;
      return true;
    });
  }, [recentEpisodes, filter]);

  useEffect(() => {
    if (selectedEpisodeNum === null && filteredEpisodes.length > 0) {
      setSelectedEpisodeNum(filteredEpisodes[0].episode);
    }
  }, [filteredEpisodes, selectedEpisodeNum]);

  const replay = useMemo(
    () => (selectedEpisodeNum !== null ? getReplay(selectedEpisodeNum) : null),
    [selectedEpisodeNum, getReplay, recentEpisodes],
  );

  useEffect(() => {
    if (replay) {
      setPlaying(false);
      setStepIdx(replay.moves.length);
    }
  }, [selectedEpisodeNum]);

  useEffect(() => {
    if (!playing || !replay) return;
    if (stepIdx >= replay.moves.length) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setStepIdx((prev) => Math.min(replay.moves.length, prev + 1));
    }, 320);
    return () => window.clearTimeout(timer);
  }, [playing, stepIdx, replay]);

  // Reconstruct active path up to stepIdx (faithfully handling backtrack pops!)
  const replayPath: CellCoord[] = useMemo(() => {
    if (!replay) return [];
    const startWp = replay.board.waypoints.find((w) => w.order === 1) ?? replay.board.waypoints[0];
    const pts: CellCoord[] = [{ row: startWp.row, col: startWp.col }];
    const count = Math.min(stepIdx, replay.moves.length);
    for (let i = 0; i < count; i++) {
      const m = replay.moves[i];
      if (m.isBacktrack && pts.length > 1) {
        pts.pop();
      } else {
        pts.push(m.to);
      }
    }
    return pts;
  }, [replay, stepIdx]);

  const currentMove = replay && stepIdx > 0 ? replay.moves[stepIdx - 1] : null;
  const nextWaypointOrder = currentMove ? currentMove.nextWaypointOrder : 2;
  const atEnd = replay !== null && stepIdx === replay.moves.length;

  if (recentEpisodes.length === 0) {
    return (
      <div className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-8 text-center transition-colors">
        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          No completed episodes recorded yet
        </h3>
        <p className="mt-1.5 text-sm text-slate-600 dark:text-slate-400 max-w-lg mx-auto">
          Once the agent finishes episodes, every board, path, and step-by-step Q-value evaluation can be inspected and scrubbed here.
        </p>
        {onTrainBlock && (
          <div className="mt-4">
            <button
              type="button"
              disabled={busyBlock}
              onClick={() => onTrainBlock(100)}
              className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {busyBlock ? 'Running Episodes…' : 'Run 100 Episodes to Populate Replays'}
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
      {/* Left Episode List */}
      <div className="lg:col-span-4 bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden transition-colors">
        <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-2">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Recorded Episodes</h2>
          <div className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-800 rounded-md">
            {(
              [
                ['all', 'All'],
                ['clean', 'Clean'],
                ['with_errors', 'Undos / Fail'],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                type="button"
                onClick={() => setFilter(key)}
                className={`px-2 py-1 text-xs font-medium rounded whitespace-nowrap transition-colors ${
                  filter === key
                    ? 'bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="max-h-[480px] overflow-y-auto thin-scroll divide-y divide-slate-100 dark:divide-slate-800/80">
          {filteredEpisodes.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-500 dark:text-slate-400">
              No episodes match this filter yet.
            </div>
          ) : (
            filteredEpisodes.map((ep) => {
              const isSelected = ep.episode === selectedEpisodeNum;
              return (
                <button
                  key={ep.episode}
                  type="button"
                  onClick={() => setSelectedEpisodeNum(ep.episode)}
                  className={`w-full text-left px-4 py-2.5 flex items-center justify-between gap-3 transition-colors ${
                    isSelected
                      ? 'bg-blue-50/80 dark:bg-blue-950/40'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-800/50'
                  }`}
                >
                  <div>
                    <div className="flex items-center gap-2 text-xs font-medium text-slate-900 dark:text-slate-100">
                      <span className="font-mono tabular-nums font-semibold">Episode #{ep.episode}</span>
                      <span aria-hidden="true" className="text-slate-300 dark:text-slate-600">·</span>
                      <span className="font-mono text-slate-500 dark:text-slate-400">
                        {ep.boardSize}×{ep.boardSize}
                      </span>
                    </div>
                    <div className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400 font-mono tabular-nums">
                      {ep.steps} steps{ep.extraSteps > 0 ? ` (+${ep.extraSteps} extra)` : ''} · {ep.backtracks} undos
                    </div>
                  </div>
                  <div className="text-right">
                    <div
                      className={`text-xs font-semibold ${
                        ep.cleanSolve
                          ? 'text-emerald-700 dark:text-emerald-400'
                          : ep.solved
                            ? 'text-blue-700 dark:text-blue-400'
                            : 'text-rose-700 dark:text-rose-400'
                      }`}
                    >
                      {ep.cleanSolve ? 'Clean Solve' : ep.solved ? 'Solved' : 'Dead end'}
                    </div>
                    <div className="text-[11px] font-mono tabular-nums text-slate-500 dark:text-slate-400">
                      {ep.totalReward >= 0 ? `+${ep.totalReward.toFixed(1)}` : ep.totalReward.toFixed(1)}
                    </div>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Right Replay Stage */}
      <div className="lg:col-span-8 bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-5 transition-colors">
        {!replay ? (
          <div className="py-12 text-center text-sm text-slate-500 dark:text-slate-400">
            Select an episode from the list to inspect its moves and Q-values.
          </div>
        ) : (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 dark:border-slate-800">
              <div>
                <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
                  Episode #{replay.episode} Replay
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums mt-0.5">
                  {replay.board.size}×{replay.board.size} grid ·{' '}
                  {replay.solved
                    ? (replay.backtracks ?? 0) === 0
                      ? 'Clean Solve (0 undos)'
                      : `Solved (+${replay.extraSteps ?? 0} extra · ${replay.backtracks ?? 0} undos)`
                    : 'Dead end'}{' '}
                  · Return{' '}
                  {replay.totalReward >= 0 ? `+${replay.totalReward.toFixed(1)}` : replay.totalReward.toFixed(1)}
                </p>
              </div>

              {/* Transport Controls */}
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx(0);
                  }}
                  className="p-2 rounded-md border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
                  aria-label="Jump to start of episode"
                >
                  <SkipBack className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx((s) => Math.max(0, s - 1));
                  }}
                  className="p-2 rounded-md border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
                  aria-label="Previous step"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (stepIdx >= replay.moves.length) setStepIdx(0);
                    setPlaying((p) => !p);
                  }}
                  className="px-3 py-2 rounded-md bg-blue-600 text-white text-xs font-semibold hover:bg-blue-700 inline-flex items-center gap-1.5 whitespace-nowrap"
                >
                  {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {playing ? 'Pause' : 'Play'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx((s) => Math.min(replay.moves.length, s + 1));
                  }}
                  className="p-2 rounded-md border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
                  aria-label="Next step"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setStepIdx(replay.moves.length);
                  }}
                  className="p-2 rounded-md border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
                  aria-label="Jump to end of episode"
                >
                  <SkipForward className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Step Scrubber */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400 font-mono tabular-nums">
                <label htmlFor="replay-scrubber">
                  Step {stepIdx} / {replay.moves.length}
                </label>
                <span>
                  Cells filled: {replayPath.length} / {replay.board.size * replay.board.size}
                </span>
              </div>
              <input
                id="replay-scrubber"
                type="range"
                min={0}
                max={replay.moves.length}
                value={stepIdx}
                onChange={(e) => {
                  setPlaying(false);
                  setStepIdx(Number(e.target.value));
                }}
                className="w-full accent-blue-600 cursor-pointer"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 items-start">
              <ZipBoardView
                board={replay.board}
                path={replayPath}
                nextWaypointOrder={nextWaypointOrder}
                solved={atEnd && replay.solved}
                deadEnd={atEnd && replay.deadEnd}
                latestEvaluations={currentMove?.evaluations ?? []}
                compact
                isDark={isDark}
              />

              <div className="space-y-4">
                {currentMove ? (
                  <>
                    <div className="p-3.5 rounded-md bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs space-y-1">
                      <div className="font-semibold text-slate-900 dark:text-slate-100">
                        Move #{currentMove.step}: ({currentMove.from.row},{currentMove.from.col}) → ({currentMove.to.row},{currentMove.to.col})
                        {currentMove.isBacktrack ? ' (Backtrack)' : ''}
                      </div>
                      <div className="font-mono tabular-nums text-slate-600 dark:text-slate-400">
                        Step reward:{' '}
                        {currentMove.reward >= 0
                          ? `+${currentMove.reward.toFixed(2)}`
                          : currentMove.reward.toFixed(2)}{' '}
                        · Cumulative:{' '}
                        {currentMove.cumulativeReward >= 0
                          ? `+${currentMove.cumulativeReward.toFixed(2)}`
                          : currentMove.cumulativeReward.toFixed(2)}
                      </div>
                    </div>
                    <QValueBreakdown
                      evaluations={currentMove.evaluations}
                      exploratory={currentMove.exploratory}
                    />
                  </>
                ) : (
                  <div className="p-4 rounded-md bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400">
                    Initial board state at waypoint 1. Step forward or press Play to inspect each move and its Q-values.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
