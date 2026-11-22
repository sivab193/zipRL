import React from 'react';
import { DashboardSnapshot } from '../rl/types';

interface StatsStripProps {
  snapshot: DashboardSnapshot;
}

export const StatsStrip: React.FC<StatsStripProps> = ({ snapshot }) => {
  const deadEndRate =
    snapshot.episode > 0 ? (snapshot.totalDeadEnds / snapshot.episode) * 100 : 0;
  const optimalSteps = snapshot.config.boardSize * snapshot.config.boardSize - 1;
  const isSolveTillWin = snapshot.config.allowBacktrack;

  const autoSaveText =
    snapshot.config.checkpointInterval > 0
      ? `Auto-save ${snapshot.config.checkpointInterval >= 1000 && snapshot.config.checkpointInterval % 1000 === 0 ? `${snapshot.config.checkpointInterval / 1000}k` : snapshot.config.checkpointInterval} ep`
      : 'Manual checkpoints';

  const items = isSolveTillWin
    ? [
        {
          label: 'Boards Solved',
          value: snapshot.totalSolved.toLocaleString(),
          sub: `${snapshot.config.boardSize}×${snapshot.config.boardSize} · Solve Till Win · ε ${snapshot.epsilon.toFixed(3)}`,
        },
        {
          label: 'Clean Solve Rate',
          value: `${(snapshot.rollingWinRate * 100).toFixed(1)}%`,
          sub: `0-undo eval ${(snapshot.latestEvalWinRate * 100).toFixed(1)}% · Best ${(snapshot.bestWinRate * 100).toFixed(1)}%`,
        },
        {
          label: 'Avg Extra Steps',
          value: `+${snapshot.rollingExtraSteps.toFixed(1)}`,
          sub: `Optimal ${optimalSteps} · Avg ${snapshot.rollingBacktracks.toFixed(1)} undos`,
        },
        {
          label: 'Replay Memory (Moves)',
          value: `${snapshot.replayBufferSize.toLocaleString()} / ${(snapshot.replayBufferCapacity / 1000).toFixed(0)}k`,
          sub: snapshot.activeCheckpoint
            ? `Saved Ep ${snapshot.activeCheckpoint.episode} · ${autoSaveText}`
            : `${autoSaveText} · Export .ziprl`,
        },
      ]
    : [
        {
          label: 'Episodes Played',
          value: snapshot.episode.toLocaleString(),
          sub: `${snapshot.config.boardSize}×${snapshot.config.boardSize} · No Backtrack · ε ${snapshot.epsilon.toFixed(3)}`,
        },
        {
          label: 'Rolling Win Rate',
          value: `${(snapshot.rollingWinRate * 100).toFixed(1)}%`,
          sub: `Eval ${(snapshot.latestEvalWinRate * 100).toFixed(1)}% · Best ${(snapshot.bestWinRate * 100).toFixed(1)}%`,
        },
        {
          label: 'Boards Solved',
          value: snapshot.totalSolved.toLocaleString(),
          sub: `Dead-end rate ${deadEndRate.toFixed(1)}%`,
        },
        {
          label: 'Replay Memory (Moves)',
          value: `${snapshot.replayBufferSize.toLocaleString()} / ${(snapshot.replayBufferCapacity / 1000).toFixed(0)}k`,
          sub: snapshot.activeCheckpoint
            ? `Saved Ep ${snapshot.activeCheckpoint.episode} · ${autoSaveText}`
            : `${autoSaveText} · Export .ziprl`,
        },
      ];

  return (
    <section
      aria-label="Training statistics"
      className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg transition-colors"
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 divide-y lg:divide-y-0 lg:divide-x divide-slate-200 dark:divide-slate-800">
        {items.map((item) => (
          <div key={item.label} className="px-4 py-3.5">
            <div className="text-xs font-medium text-slate-500 dark:text-slate-400">{item.label}</div>
            <div className="mt-1 text-xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-100">
              {item.value}
            </div>
            <div className="mt-0.5 text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400 truncate">
              {item.sub}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};
