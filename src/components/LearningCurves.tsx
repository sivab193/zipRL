import React, { useMemo, useState } from 'react';
import { EpisodeMetric, EvalPoint } from '../rl/types';

interface LearningCurvesProps {
  metrics: EpisodeMetric[];
  evalHistory: EvalPoint[];
  curriculumThreshold: number;
  allowBacktrack?: boolean;
  boardSize?: number;
  isDark?: boolean;
  onTrainBlock?: (episodes: number) => void;
  busyBlock?: boolean;
}

interface LineSeriesProps {
  title: string;
  subtitle: string;
  data: Array<{ x: number; y: number }>;
  secondaryData?: Array<{ x: number; y: number }>;
  secondaryLabel?: string;
  yMin: number;
  yMax: number;
  color: string;
  secondaryColor?: string;
  formatY: (v: number) => string;
  referenceLineY?: number;
  referenceLabel?: string;
  unitLabel: string;
  isDark: boolean;
}

const ScientificLineChart: React.FC<LineSeriesProps> = ({
  title,
  subtitle,
  data,
  secondaryData = [],
  secondaryLabel,
  yMin,
  yMax,
  color,
  secondaryColor = '#10B981',
  formatY,
  referenceLineY,
  referenceLabel,
  unitLabel,
  isDark,
}) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const width = 560;
  const height = 188;
  const margin = { top: 14, right: 18, bottom: 26, left: 48 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const { minX, maxX } = useMemo(() => {
    if (data.length === 0) return { minX: 1, maxX: 100 };
    const xs = data.map((d) => d.x);
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    return { minX: lo, maxX: Math.max(lo + 1, hi) };
  }, [data]);

  const safeYMax = yMax === yMin ? yMin + 1 : yMax;
  const mapX = (x: number) => margin.left + ((x - minX) / (maxX - minX)) * plotW;
  const mapY = (y: number) =>
    margin.top + plotH - ((Math.max(yMin, Math.min(safeYMax, y)) - yMin) / (safeYMax - yMin)) * plotH;

  const primaryPath = useMemo(() => {
    if (data.length === 0) return '';
    return data.map((pt, i) => `${i === 0 ? 'M' : 'L'}${mapX(pt.x).toFixed(1)},${mapY(pt.y).toFixed(1)}`).join(' ');
  }, [data, minX, maxX, yMin, safeYMax]);

  const secondaryPath = useMemo(() => {
    if (secondaryData.length < 2) return '';
    return secondaryData
      .map((pt, i) => `${i === 0 ? 'M' : 'L'}${mapX(pt.x).toFixed(1)},${mapY(pt.y).toFixed(1)}`)
      .join(' ');
  }, [secondaryData, minX, maxX, yMin, safeYMax]);

  const yTicks = [yMin, yMin + (safeYMax - yMin) * 0.5, safeYMax];
  const latest = data.length > 0 ? data[data.length - 1] : null;
  const hovered = hoverIdx !== null && data[hoverIdx] ? data[hoverIdx] : null;
  const displayPoint = hovered ?? latest;

  const gridStroke = isDark ? '#1E293B' : '#E2E8F0';
  const axisFill = isDark ? '#94A3B8' : '#64748B';
  const dotStroke = isDark ? '#0F1523' : '#FFFFFF';

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (data.length === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width;
    const clampedX = Math.max(margin.left, Math.min(margin.left + plotW, relX));
    const ratio = (clampedX - margin.left) / plotW;
    const targetEp = minX + ratio * (maxX - minX);
    let nearest = 0;
    let bestDist = Infinity;
    for (let i = 0; i < data.length; i++) {
      const dist = Math.abs(data[i].x - targetEp);
      if (dist < bestDist) {
        bestDist = dist;
        nearest = i;
      }
    }
    setHoverIdx(nearest);
  };

  return (
    <div className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-4 transition-colors">
      <div className="flex items-baseline justify-between gap-4 mb-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>
        </div>
        <div className="text-right font-mono tabular-nums">
          {displayPoint ? (
            <>
              <span className="text-base font-semibold text-slate-900 dark:text-slate-100">
                {formatY(displayPoint.y)}
              </span>
              <span className="text-xs text-slate-400 ml-1.5">
                {unitLabel} · Ep {displayPoint.x}
              </span>
            </>
          ) : (
            <span className="text-xs text-slate-400">No episodes yet</span>
          )}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto block cursor-crosshair"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
        role="img"
        aria-label={`${title} learning curve`}
      >
        {/* Horizontal grid lines */}
        {yTicks.map((tick, i) => {
          const y = mapY(tick);
          return (
            <g key={i}>
              <line
                x1={margin.left}
                y1={y}
                x2={width - margin.right}
                y2={y}
                stroke={gridStroke}
                strokeWidth={1}
              />
              <text
                x={margin.left - 8}
                y={y}
                textAnchor="end"
                dominantBaseline="central"
                fill={axisFill}
                fontSize={10}
                fontFamily="IBM Plex Mono, monospace"
              >
                {formatY(tick)}
              </text>
            </g>
          );
        })}

        {/* Optional curriculum / baseline reference line */}
        {referenceLineY !== undefined && (
          <g>
            <line
              x1={margin.left}
              y1={mapY(referenceLineY)}
              x2={width - margin.right}
              y2={mapY(referenceLineY)}
              stroke="#10B981"
              strokeWidth={1}
              strokeDasharray="4 4"
              opacity={0.75}
            />
            {referenceLabel && (
              <text
                x={width - margin.right}
                y={mapY(referenceLineY) - 5}
                textAnchor="end"
                fill="#10B981"
                fontSize={9.5}
                fontFamily="IBM Plex Mono, monospace"
              >
                {referenceLabel}
              </text>
            )}
          </g>
        )}

        {/* Primary line */}
        {primaryPath && (
          <path d={primaryPath} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
        )}

        {/* Secondary line & markers */}
        {secondaryPath && (
          <path
            d={secondaryPath}
            fill="none"
            stroke={secondaryColor}
            strokeWidth={1.75}
            strokeDasharray="5 3"
          />
        )}
        {secondaryData.map((pt, i) => (
          <circle
            key={i}
            cx={mapX(pt.x)}
            cy={mapY(pt.y)}
            r={3.5}
            fill={secondaryColor}
            stroke={dotStroke}
            strokeWidth={1.5}
          />
        ))}

        {/* Interactive crosshair */}
        {hovered && (
          <g>
            <line
              x1={mapX(hovered.x)}
              y1={margin.top}
              x2={mapX(hovered.x)}
              y2={margin.top + plotH}
              stroke={axisFill}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <circle
              cx={mapX(hovered.x)}
              cy={mapY(hovered.y)}
              r={4}
              fill={color}
              stroke={dotStroke}
              strokeWidth={2}
            />
          </g>
        )}

        {/* X-axis labels */}
        <text
          x={margin.left}
          y={height - 6}
          fill={axisFill}
          fontSize={10}
          fontFamily="IBM Plex Mono, monospace"
        >
          Ep {minX}
        </text>
        <text
          x={width - margin.right}
          y={height - 6}
          textAnchor="end"
          fill={axisFill}
          fontSize={10}
          fontFamily="IBM Plex Mono, monospace"
        >
          Ep {maxX}
        </text>
      </svg>

      {secondaryLabel && (
        <div className="mt-2 flex items-center gap-4 text-xs text-slate-500 dark:text-slate-400">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0.5 inline-block" style={{ backgroundColor: color }} />
            Primary curve
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-3 h-0.5 inline-block" style={{ backgroundColor: secondaryColor }} />
            {secondaryLabel}
          </span>
        </div>
      )}
    </div>
  );
};

export const LearningCurves: React.FC<LearningCurvesProps> = ({
  metrics,
  evalHistory,
  curriculumThreshold,
  allowBacktrack = true,
  boardSize = 5,
  isDark = false,
  onTrainBlock,
  busyBlock = false,
}) => {
  const optimalSteps = boardSize * boardSize - 1;
  const winRateSeries = useMemo(
    () => metrics.map((m) => ({ x: m.episode, y: m.rollingWinRate * 100 })),
    [metrics],
  );
  const evalSeries = useMemo(
    () => evalHistory.map((e) => ({ x: e.episode, y: e.winRate * 100 })),
    [evalHistory],
  );
  const rewardSeries = useMemo(
    () => metrics.map((m) => ({ x: m.episode, y: m.rollingReward })),
    [metrics],
  );
  const stepsSeries = useMemo(
    () => metrics.map((m) => ({ x: m.episode, y: m.rollingSteps })),
    [metrics],
  );
  const epsilonSeries = useMemo(
    () => metrics.map((m) => ({ x: m.episode, y: m.epsilon })),
    [metrics],
  );

  const rewardBounds = useMemo(() => {
    if (rewardSeries.length === 0) return { min: -2, max: 12 };
    const vals = rewardSeries.map((d) => d.y);
    return {
      min: Math.min(-1.5, Math.floor(Math.min(...vals) - 1)),
      max: Math.max(4, Math.ceil(Math.max(...vals) + 1)),
    };
  }, [rewardSeries]);

  const maxSteps = useMemo(() => {
    if (stepsSeries.length === 0) return 30;
    return Math.max(25, Math.ceil(Math.max(...stepsSeries.map((d) => d.y)) + 4));
  }, [stepsSeries]);

  if (metrics.length === 0) {
    return (
      <div className="bg-white dark:bg-[#0F1523] border border-slate-200 dark:border-slate-800 rounded-lg p-8 text-center transition-colors">
        <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          Learning curves awaiting first episodes
        </h3>
        <p className="mt-1.5 text-sm text-slate-600 dark:text-slate-400 max-w-lg mx-auto">
          Rolling win rate, held-out greedy evaluation, average reward, average steps, and epsilon decay plot here automatically as episodes complete.
        </p>
        {onTrainBlock && (
          <div className="mt-4 flex items-center justify-center gap-3">
            <button
              type="button"
              disabled={busyBlock}
              onClick={() => onTrainBlock(100)}
              className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 transition-colors whitespace-nowrap"
            >
              {busyBlock ? 'Running 100-Episode Block…' : 'Run 100-Episode Training Block'}
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ScientificLineChart
        title={allowBacktrack ? 'Clean Solve Rate (0 Wrong Moves) & Eval' : 'Rolling Win Rate & Held-Out Eval'}
        subtitle={
          allowBacktrack
            ? '50-episode rolling 0-backtrack solve rate vs. 50-board greedy eval'
            : '50-episode rolling solve rate vs. 50-board greedy evaluation'
        }
        data={winRateSeries}
        secondaryData={evalSeries}
        secondaryLabel="Held-out greedy eval (every 100 ep)"
        yMin={0}
        yMax={100}
        color={isDark ? '#60A5FA' : '#2563EB'}
        secondaryColor="#10B981"
        formatY={(v) => `${v.toFixed(1)}%`}
        referenceLineY={curriculumThreshold * 100}
        referenceLabel={`Curriculum target (${Math.round(curriculumThreshold * 100)}%)`}
        unitLabel={allowBacktrack ? 'clean' : 'win'}
        isDark={isDark}
      />
      <ScientificLineChart
        title="Average Episode Reward"
        subtitle={
          allowBacktrack
            ? 'Rolling 50-ep mean return (+0.1 progress, -0.15 undo, -0.2 trap, +10 solve)'
            : 'Rolling 50-episode mean return (+0.1 step, +10 solve, -1 fail)'
        }
        data={rewardSeries}
        yMin={rewardBounds.min}
        yMax={rewardBounds.max}
        color={isDark ? '#E2E8F0' : '#0F172A'}
        formatY={(v) => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
        referenceLineY={0}
        referenceLabel="Zero return"
        unitLabel="pts"
        isDark={isDark}
      />
      <ScientificLineChart
        title="Average Steps to Solve"
        subtitle={
          allowBacktrack
            ? `Total steps per solve approaching optimal ${optimalSteps} steps (fewer wrong moves)`
            : 'Mean path steps executed per episode'
        }
        data={stepsSeries}
        yMin={0}
        yMax={maxSteps}
        color={isDark ? '#38BDF8' : '#0284C7'}
        formatY={(v) => v.toFixed(1)}
        referenceLineY={optimalSteps}
        referenceLabel={`Optimal (${optimalSteps} steps)`}
        unitLabel="steps"
        isDark={isDark}
      />
      <ScientificLineChart
        title="Exploration Schedule (Epsilon)"
        subtitle="ε-greedy probability decaying from 1.00 toward 0.05 floor"
        data={epsilonSeries}
        yMin={0}
        yMax={1}
        color="#F59E0B"
        formatY={(v) => v.toFixed(3)}
        referenceLineY={0.05}
        referenceLabel="Min ε (0.05)"
        unitLabel="ε"
        isDark={isDark}
      />
    </div>
  );
};
