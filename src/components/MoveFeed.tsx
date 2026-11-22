import React from 'react';
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Undo2 } from 'lucide-react';
import { ActionEvaluation, DIRECTION_NAMES, Direction, MoveRecord } from '../rl/types';

interface MoveFeedProps {
  moves: MoveRecord[];
  activeMove?: MoveRecord | null;
  title?: string;
}

const DirectionIcon: React.FC<{ dir: Direction; className?: string }> = ({ dir, className = 'w-3.5 h-3.5' }) => {
  switch (dir) {
    case 0:
      return <ArrowUp className={className} />;
    case 1:
      return <ArrowRight className={className} />;
    case 2:
      return <ArrowDown className={className} />;
    case 3:
      return <ArrowLeft className={className} />;
  }
};

export const QValueBreakdown: React.FC<{ evaluations: ActionEvaluation[]; exploratory?: boolean }> = ({
  evaluations,
  exploratory = false,
}) => {
  const legalEvals = evaluations.filter((e) => e.legal);
  const maxAbs = Math.max(0.5, ...legalEvals.map((e) => Math.abs(e.qValue)));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>Action Q-Values (Legal Mask Active)</span>
        <span className="font-mono text-slate-600 dark:text-slate-300">
          {exploratory ? 'Explore (ε-random)' : 'Exploit (argmax Q)'}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {evaluations.map((ev) => {
          const barPct = ev.legal ? Math.min(100, Math.round((Math.abs(ev.qValue) / maxAbs) * 100)) : 0;
          const isPositive = ev.qValue >= 0;
          return (
            <div
              key={ev.direction}
              className={`px-2.5 py-2 rounded-md border text-xs transition-colors ${
                ev.chosen
                  ? ev.isBacktrack
                    ? 'border-amber-500 bg-amber-50/70 dark:bg-amber-950/40 text-slate-900 dark:text-slate-100'
                    : 'border-blue-600 dark:border-blue-500 bg-blue-50/70 dark:bg-blue-950/40 text-slate-900 dark:text-slate-100'
                  : ev.legal
                    ? 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200'
                    : 'border-slate-100 dark:border-slate-800/80 bg-slate-50 dark:bg-slate-900/40 text-slate-400 dark:text-slate-500'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 font-medium whitespace-nowrap">
                  {ev.isBacktrack ? (
                    <Undo2 className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                  ) : (
                    <DirectionIcon dir={ev.direction} className="w-3.5 h-3.5" />
                  )}
                  <span>{ev.label}</span>
                  {ev.isBacktrack && (
                    <span className="text-[10px] font-mono text-amber-700 dark:text-amber-400">
                      (Back)
                    </span>
                  )}
                </span>
                <span className="font-mono tabular-nums font-semibold">
                  {ev.legal ? (ev.qValue >= 0 ? `+${ev.qValue.toFixed(3)}` : ev.qValue.toFixed(3)) : 'Masked'}
                </span>
              </div>
              <div className="mt-1.5 h-1.5 w-full rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                {ev.legal && (
                  <div
                    className={`h-full rounded-full ${
                      ev.chosen
                        ? ev.isBacktrack
                          ? 'bg-amber-500'
                          : 'bg-blue-600 dark:bg-blue-500'
                        : isPositive
                          ? 'bg-slate-400 dark:bg-slate-500'
                          : 'bg-rose-400 dark:bg-rose-500'
                    }`}
                    style={{ width: `${Math.max(8, barPct)}%` }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const MoveFeed: React.FC<MoveFeedProps> = ({
  moves,
  activeMove = null,
  title = 'Move Feed & Q-Value Policy',
}) => {
  const focusMove = activeMove ?? (moves.length > 0 ? moves[moves.length - 1] : null);
  const reversed = [...moves].reverse();

  return (
    <div className="flex flex-col h-full bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg overflow-hidden transition-colors">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-4">
        <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
        <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
          {moves.length} {moves.length === 1 ? 'step' : 'steps'}
        </span>
      </div>

      {/* Latest / Focused Move Q-Value Inspector */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40">
        {focusMove ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-300">
              <span className="font-mono tabular-nums font-medium text-slate-900 dark:text-slate-100">
                Step #{focusMove.step} · ({focusMove.from.row},{focusMove.from.col}) → ({focusMove.to.row},{focusMove.to.col})
                {focusMove.isBacktrack ? ' · Backtrack' : ''}
              </span>
              <span
                className={`font-mono tabular-nums font-semibold ${
                  focusMove.reward >= 5
                    ? 'text-emerald-700 dark:text-emerald-400'
                    : focusMove.reward < 0
                      ? 'text-rose-700 dark:text-rose-400'
                      : 'text-slate-700 dark:text-slate-300'
                }`}
              >
                Reward {focusMove.reward >= 0 ? `+${focusMove.reward.toFixed(2)}` : focusMove.reward.toFixed(2)}
              </span>
            </div>
            <QValueBreakdown
              evaluations={focusMove.evaluations}
              exploratory={focusMove.exploratory}
            />
          </div>
        ) : (
          <div className="py-5 text-center text-xs text-slate-500 dark:text-slate-400">
            Start training or click Step Once to inspect the agent&apos;s Q-values across legal moves.
          </div>
        )}
      </div>

      {/* Chronological Move Ledger */}
      <div className="flex-1 overflow-y-auto thin-scroll max-h-[260px]">
        {reversed.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-slate-400 dark:text-slate-500">
            No moves recorded in the current episode yet.
          </div>
        ) : (
          <table className="w-full text-left border-collapse text-xs">
            <thead className="sticky top-0 bg-slate-50 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
              <tr>
                <th className="py-2 pl-4 pr-2 font-medium">#</th>
                <th className="py-2 px-2 font-medium">Picked</th>
                <th className="py-2 px-2 font-medium">Alternatives (Q)</th>
                <th className="py-2 pl-2 pr-4 font-medium text-right">Reward</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">
              {reversed.map((m) => {
                const chosenEval = m.evaluations.find((e) => e.chosen);
                const otherLegals = m.evaluations.filter((e) => e.legal && !e.chosen);
                const isFocused = focusMove?.step === m.step;

                return (
                  <tr
                    key={m.step}
                    className={
                      isFocused
                        ? 'bg-blue-50/50 dark:bg-blue-950/30'
                        : 'hover:bg-slate-50/80 dark:hover:bg-slate-800/40'
                    }
                  >
                    <td className="py-2 pl-4 pr-2 font-mono tabular-nums text-slate-500 dark:text-slate-400">
                      {m.step}
                    </td>
                    <td className="py-2 px-2">
                      <div className="flex items-center gap-1.5 text-slate-900 dark:text-slate-100 font-medium whitespace-nowrap">
                        {m.isBacktrack ? (
                          <Undo2 className="w-3 h-3 text-amber-600 dark:text-amber-400" />
                        ) : (
                          <DirectionIcon dir={m.direction} className="w-3 h-3 text-blue-600 dark:text-blue-400" />
                        )}
                        <span>{m.isBacktrack ? 'Backtrack' : DIRECTION_NAMES[m.direction]}</span>
                        <span className="font-mono tabular-nums text-slate-500 dark:text-slate-400">
                          {chosenEval ? chosenEval.qValue.toFixed(2) : ''}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-2 font-mono tabular-nums text-slate-500 dark:text-slate-400">
                      {otherLegals.length > 0
                        ? otherLegals
                            .map((o) => `${o.isBacktrack ? 'Undo' : o.label[0]}:${o.qValue.toFixed(2)}`)
                            .join(' · ')
                        : 'Forced'}
                    </td>
                    <td
                      className={`py-2 pl-2 pr-4 font-mono tabular-nums text-right font-medium ${
                        m.reward >= 5
                          ? 'text-emerald-700 dark:text-emerald-400'
                          : m.reward < 0
                            ? 'text-rose-700 dark:text-rose-400'
                            : 'text-slate-700 dark:text-slate-300'
                      }`}
                    >
                      {m.reward >= 0 ? `+${m.reward.toFixed(2)}` : m.reward.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
