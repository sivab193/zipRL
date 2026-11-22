import React, { useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  Download,
  FastForward,
  FolderOpen,
  Moon,
  RotateCcw,
  Save,
  Shuffle,
  Sparkles,
  StepForward,
  Sun,
  Target,
  Upload,
} from 'lucide-react';
import { CheckpointInfo, TargetType, TrainingConfig } from '../rl/types';
import { getMaxWaypointCount, getVariedWaypointRange } from '../rl/generator';

interface ControlBarProps {
  running: boolean;
  isDark: boolean;
  config: TrainingConfig;
  checkpoints: CheckpointInfo[];
  activeCheckpoint: CheckpointInfo | null;
  busyBlock: boolean;
  onToggleTheme: () => void;
  onStepOnce: () => void;
  onTrainBlock: (episodes: number) => void;
  onNewPuzzle: () => void;
  onRunOracle: () => void;
  onUpdateSettings: (partial: Partial<TrainingConfig>) => void;
  onSaveCheckpoint: () => void;
  onLoadCheckpoint: (id: string) => void;
  onDownloadCheckpoint: (id?: string) => void;
  onUploadZiprlFile: (file: File) => void;
  onResetModel: () => void;
}

const SPEED_OPTIONS = [
  { label: 'Burst', ms: 0 },
  { label: 'Fast', ms: 40 },
  { label: 'Medium', ms: 120 },
  { label: 'Inspect', ms: 340 },
];

const CHECKPOINT_INTERVAL_PRESETS = [
  { label: 'Off', value: 0 },
  { label: '250 ep', value: 250 },
  { label: '500 ep', value: 500 },
  { label: '1,000 (1k)', value: 1000 },
  { label: '2,500 ep', value: 2500 },
  { label: '5,000 (5k)', value: 5000 },
  { label: '10,000 (10k)', value: 10000 },
];

export const ControlBar: React.FC<ControlBarProps> = ({
  isDark,
  config,
  checkpoints,
  activeCheckpoint,
  busyBlock,
  onToggleTheme,
  onStepOnce,
  onTrainBlock,
  onNewPuzzle,
  onRunOracle,
  onUpdateSettings,
  onSaveCheckpoint,
  onLoadCheckpoint,
  onDownloadCheckpoint,
  onUploadZiprlFile,
  onResetModel,
}) => {
  const [ckptOpen, setCkptOpen] = useState(false);
  const [targetOpen, setTargetOpen] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [customCkptVal, setCustomCkptVal] = useState(String(config.checkpointInterval || 1000));
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [minAutoWp, maxAutoWp] = useMemo(
    () => getVariedWaypointRange(config.boardSize),
    [config.boardSize],
  );
  const maxWp = useMemo(() => getMaxWaypointCount(config.boardSize), [config.boardSize]);

  const waypointChoices = useMemo(() => {
    const base = [3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60];
    const filtered = base.filter((v) => v <= maxWp);
    if (!filtered.includes(maxWp)) filtered.push(maxWp);
    return filtered;
  }, [maxWp]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadZiprlFile(file);
      setCkptOpen(false);
    }
    e.target.value = '';
  };

  const targetButtonLabel = useMemo(() => {
    if (config.target.type === 'none') return 'Target: None';
    if (config.target.type === 'episodes') {
      const val =
        config.target.value >= 1000 && config.target.value % 1000 === 0
          ? `${config.target.value / 1000}k`
          : config.target.value.toLocaleString();
      return `Target: ${val} ep`;
    }
    if (config.target.type === 'clean_rate') return `Target: ${config.target.value}% rate`;
    return `Target: ${config.target.value}×${config.target.value}`;
  }, [config.target]);

  const handleTargetTypeChange = (nextType: TargetType) => {
    let defaultVal = config.target.value;
    if (nextType === 'episodes' && (defaultVal < 10 || config.target.type !== 'episodes')) {
      defaultVal = 500;
    } else if (nextType === 'clean_rate' && (defaultVal > 100 || config.target.type !== 'clean_rate')) {
      defaultVal = 80;
    } else if (nextType === 'grid_level') {
      defaultVal = Math.min(10, Math.max(config.boardSize + 1, 6));
    }
    onUpdateSettings({ target: { type: nextType, value: defaultVal } });
  };

  return (
    <div className="bg-white dark:bg-[#0F1523] border-b border-slate-200 dark:border-slate-800 transition-colors">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 py-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Left: Grid Size (4x4..10x10), Numbers (Waypoints), Rule, Speed, Curriculum */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5">
            <label
              htmlFor="board-size-select"
              className="text-xs font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap"
            >
              Grid
            </label>
            <select
              id="board-size-select"
              value={config.boardSize}
              onChange={(e) => onUpdateSettings({ boardSize: Number(e.target.value) })}
              className="h-8 px-2 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-mono font-medium text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/25 focus:border-blue-600"
            >
              {[4, 5, 6, 7, 8, 9, 10].map((n) => (
                <option key={n} value={n}>
                  {n}×{n}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5">
            <label
              htmlFor="waypoint-count-select"
              className="text-xs font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap"
            >
              Numbers
            </label>
            <select
              id="waypoint-count-select"
              value={config.waypointCount}
              onChange={(e) => onUpdateSettings({ waypointCount: Number(e.target.value) })}
              className="h-8 px-2 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-mono font-medium text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600/25 focus:border-blue-600"
              title="Number of waypoints per board (Varied picks a realistic Zip range each puzzle)"
            >
              <option value={0}>
                Varied ({minAutoWp}–{maxAutoWp})
              </option>
              {waypointChoices.map((n) => (
                <option key={n} value={n}>
                  {n} numbers
                </option>
              ))}
            </select>
          </div>

          {/* Episode Rule Toggle: Solve Till Win vs No Backtrack */}
          <div
            className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-800/90 rounded-md"
            role="group"
            aria-label="Episode rule"
          >
            <button
              type="button"
              onClick={() => onUpdateSettings({ allowBacktrack: true })}
              className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap transition-colors ${
                config.allowBacktrack
                  ? 'bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
              title="Never stops on dead ends — backtracks out of wrong moves and keeps learning until the board is solved"
            >
              Solve Till Win
            </button>
            <button
              type="button"
              onClick={() => onUpdateSettings({ allowBacktrack: false })}
              className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap transition-colors ${
                !config.allowBacktrack
                  ? 'bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 shadow-xs'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
              }`}
              title="One-way path only — hitting a dead end immediately terminates the episode (-1.0)"
            >
              No Backtrack
            </button>
          </div>

          <div
            className="flex items-center gap-1 p-0.5 bg-slate-100 dark:bg-slate-800/90 rounded-md"
            role="group"
            aria-label="Step speed"
          >
            {SPEED_OPTIONS.map((opt) => {
              const active = config.stepDelayMs === opt.ms;
              return (
                <button
                  key={opt.label}
                  type="button"
                  onClick={() => onUpdateSettings({ stepDelayMs: opt.ms })}
                  className={`px-2.5 py-1 text-xs font-medium rounded whitespace-nowrap transition-colors ${
                    active
                      ? 'bg-white dark:bg-slate-950 text-slate-900 dark:text-slate-100 shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100'
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>

          <label className="inline-flex items-center gap-1.5 text-xs text-slate-700 dark:text-slate-300 cursor-pointer select-none whitespace-nowrap">
            <input
              type="checkbox"
              checked={config.autoCurriculum}
              onChange={(e) => onUpdateSettings({ autoCurriculum: e.target.checked })}
              className="rounded border-slate-300 dark:border-slate-600 text-blue-600 focus:ring-blue-600"
            />
            <span>Auto-curriculum (&gt;80%)</span>
          </label>
        </div>

        {/* Right: Training Actions, Targets, Puzzle Actions, Checkpoints (.ziprl), Theme, Reset */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onStepOnce}
            className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <StepForward className="w-3.5 h-3.5" />
            Step Once
          </button>

          <button
            type="button"
            disabled={busyBlock}
            onClick={() => onTrainBlock(100)}
            className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-50 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <FastForward className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
            {busyBlock ? 'Training…' : 'Train +100 Ep'}
          </button>

          {/* Training Target / Auto-Stop Popover */}
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                setTargetOpen((v) => !v);
                setCkptOpen(false);
              }}
              aria-expanded={targetOpen}
              className={`h-8 px-3 rounded-md border text-xs font-medium inline-flex items-center gap-1.5 whitespace-nowrap transition-colors ${
                config.target.type !== 'none'
                  ? 'border-blue-500/60 bg-blue-50/70 dark:bg-blue-950/40 text-blue-800 dark:text-blue-300'
                  : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800'
              }`}
            >
              <Target className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              {targetButtonLabel}
              <ChevronDown className="w-3 h-3 text-slate-400" />
            </button>

            {targetOpen && (
              <div className="absolute right-0 mt-1.5 w-80 max-w-[92vw] bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-30 p-3.5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Training Stop Target
                  </span>
                  <button
                    type="button"
                    onClick={() => setTargetOpen(false)}
                    className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                  >
                    Done
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                  {(
                    [
                      ['none', 'Continuous (Off)'],
                      ['episodes', 'Boards / Episodes'],
                      ['clean_rate', 'Solve Rate (%)'],
                      ['grid_level', 'Grid Level'],
                    ] as const
                  ).map(([tType, label]) => (
                    <button
                      key={tType}
                      type="button"
                      onClick={() => handleTargetTypeChange(tType)}
                      className={`px-2.5 py-1.5 rounded border text-xs font-medium text-left transition-colors ${
                        config.target.type === tType
                          ? 'border-blue-600 bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300'
                          : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>

                {config.target.type === 'episodes' && (
                  <div className="space-y-2 pt-1 border-t border-slate-100 dark:border-slate-800">
                    <div className="text-[11px] text-slate-500 dark:text-slate-400">
                      Auto-pause &amp; save when episodes solved reach:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {[50, 100, 500, 1000, 5000, 10000].map((v) => (
                        <button
                          key={v}
                          type="button"
                          onClick={() => onUpdateSettings({ target: { type: 'episodes', value: v } })}
                          className={`px-2 py-1 rounded text-xs font-mono border ${
                            config.target.value === v
                              ? 'border-blue-600 bg-blue-600 text-white'
                              : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                          }`}
                        >
                          {v >= 1000 ? `${v / 1000}k` : v}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-2">
                      <label htmlFor="custom-target-ep" className="text-xs text-slate-600 dark:text-slate-400">
                        Custom:
                      </label>
                      <input
                        id="custom-target-ep"
                        type="number"
                        min={1}
                        max={1000000}
                        value={config.target.value}
                        onChange={(e) =>
                          onUpdateSettings({
                            target: { type: 'episodes', value: Math.max(1, Number(e.target.value) || 1) },
                          })
                        }
                        className="w-28 h-7 px-2 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-mono text-slate-900 dark:text-slate-100"
                      />
                    </div>
                  </div>
                )}

                {config.target.type === 'clean_rate' && (
                  <div className="space-y-2 pt-1 border-t border-slate-100 dark:border-slate-800">
                    <div className="text-[11px] text-slate-500 dark:text-slate-400">
                      Auto-pause when rolling/eval {config.allowBacktrack ? 'clean (0-undo)' : ''} solve rate reaches (after 25+ ep):
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {[50, 75, 80, 90, 95].map((pct) => (
                        <button
                          key={pct}
                          type="button"
                          onClick={() => onUpdateSettings({ target: { type: 'clean_rate', value: pct } })}
                          className={`px-2.5 py-1 rounded text-xs font-mono border ${
                            config.target.value === pct
                              ? 'border-blue-600 bg-blue-600 text-white'
                              : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                          }`}
                        >
                          {pct}%
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-2">
                      <label htmlFor="custom-target-rate" className="text-xs text-slate-600 dark:text-slate-400">
                        Custom %:
                      </label>
                      <input
                        id="custom-target-rate"
                        type="number"
                        min={5}
                        max={100}
                        value={config.target.value}
                        onChange={(e) =>
                          onUpdateSettings({
                            target: {
                              type: 'clean_rate',
                              value: Math.max(5, Math.min(100, Number(e.target.value) || 80)),
                            },
                          })
                        }
                        className="w-24 h-7 px-2 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-mono text-slate-900 dark:text-slate-100"
                      />
                    </div>
                  </div>
                )}

                {config.target.type === 'grid_level' && (
                  <div className="space-y-2 pt-1 border-t border-slate-100 dark:border-slate-800">
                    <div className="text-[11px] text-slate-500 dark:text-slate-400">
                      Auto-pause once Auto-curriculum graduates to grid size:
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {[5, 6, 7, 8, 9, 10].map((lvl) => (
                        <button
                          key={lvl}
                          type="button"
                          onClick={() => onUpdateSettings({ target: { type: 'grid_level', value: lvl } })}
                          className={`px-2.5 py-1 rounded text-xs font-mono border ${
                            config.target.value === lvl
                              ? 'border-blue-600 bg-blue-600 text-white'
                              : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                          }`}
                        >
                          {lvl}×{lvl}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={onNewPuzzle}
            className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <Shuffle className="w-3.5 h-3.5" />
            New Puzzle
          </button>

          <button
            type="button"
            onClick={onRunOracle}
            className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            <Sparkles className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            Oracle Solver
          </button>

          {/* Checkpoints & .ziprl File Import/Export Popover */}
          <div className="relative">
            <input
              ref={fileInputRef}
              type="file"
              accept=".ziprl,.json"
              onChange={handleFileChange}
              className="hidden"
              aria-label="Upload .ziprl checkpoint file"
              data-testid="ziprl-upload-input"
            />
            <button
              type="button"
              onClick={() => {
                setCkptOpen((v) => !v);
                setTargetOpen(false);
              }}
              aria-expanded={ckptOpen}
              className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              Checkpoints ({checkpoints.length})
              <ChevronDown className="w-3 h-3 text-slate-400" />
            </button>

            {ckptOpen && (
              <div className="absolute right-0 mt-1.5 w-88 max-w-[92vw] bg-white dark:bg-[#111827] border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg z-30 p-3 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Checkpoints (.ziprl)
                  </span>
                  <button
                    type="button"
                    onClick={() => onSaveCheckpoint()}
                    className="px-2.5 py-1 rounded bg-slate-900 dark:bg-blue-600 text-white text-xs font-medium hover:bg-slate-800 dark:hover:bg-blue-500 inline-flex items-center gap-1 whitespace-nowrap"
                  >
                    <Save className="w-3 h-3" />
                    Save Locally
                  </button>
                </div>

                {/* Customizable Auto-Save Interval (1k, 5k, 10k, or Custom) */}
                <div className="space-y-1.5 pb-2 border-b border-slate-100 dark:border-slate-800">
                  <div className="flex items-center justify-between gap-2">
                    <label
                      htmlFor="checkpoint-interval-select"
                      className="text-[11px] font-medium text-slate-600 dark:text-slate-400"
                    >
                      Auto-save interval
                    </label>
                    <select
                      id="checkpoint-interval-select"
                      value={
                        CHECKPOINT_INTERVAL_PRESETS.some((p) => p.value === config.checkpointInterval)
                          ? config.checkpointInterval
                          : -1
                      }
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        if (val >= 0) {
                          setCustomCkptVal(String(val));
                          onUpdateSettings({ checkpointInterval: val });
                        }
                      }}
                      className="h-7 px-2 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-mono text-slate-900 dark:text-slate-100"
                    >
                      {CHECKPOINT_INTERVAL_PRESETS.map((p) => (
                        <option key={p.value} value={p.value}>
                          {p.label}
                        </option>
                      ))}
                      {!CHECKPOINT_INTERVAL_PRESETS.some((p) => p.value === config.checkpointInterval) && (
                        <option value={-1}>Custom ({config.checkpointInterval} ep)</option>
                      )}
                    </select>
                  </div>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-slate-500 dark:text-slate-400">
                      Or set custom episodes:
                    </span>
                    <div className="flex items-center gap-1">
                      <input
                        type="number"
                        min={0}
                        max={100000}
                        aria-label="Custom checkpoint interval episodes"
                        value={customCkptVal}
                        onChange={(e) => setCustomCkptVal(e.target.value)}
                        className="w-20 h-6 px-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11px] font-mono text-slate-900 dark:text-slate-100"
                      />
                      <button
                        type="button"
                        onClick={() => {
                          const parsed = Math.max(0, Math.round(Number(customCkptVal) || 0));
                          onUpdateSettings({ checkpointInterval: parsed });
                        }}
                        className="h-6 px-2 rounded border border-slate-200 dark:border-slate-700 text-[11px] font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
                      >
                        Set
                      </button>
                    </div>
                  </div>
                </div>

                {/* Portable .ziprl File Actions */}
                <div className="grid grid-cols-2 gap-2 pb-2 border-b border-slate-100 dark:border-slate-800">
                  <button
                    type="button"
                    onClick={() => onDownloadCheckpoint()}
                    className="h-8 px-2.5 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-xs font-medium text-slate-800 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                    Download .ziprl
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="h-8 px-2.5 rounded-md border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-xs font-medium text-slate-800 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 inline-flex items-center justify-center gap-1.5 whitespace-nowrap"
                  >
                    <Upload className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                    Upload .ziprl
                  </button>
                </div>

                {checkpoints.length === 0 ? (
                  <p className="text-xs text-slate-500 dark:text-slate-400 py-1">
                    Saved checkpoints stay in browser IndexedDB and can be exported as portable .ziprl files to resume anywhere.
                  </p>
                ) : (
                  <div className="max-h-56 overflow-y-auto thin-scroll divide-y divide-slate-100 dark:divide-slate-800">
                    {checkpoints.map((ck) => {
                      const isActive = activeCheckpoint?.id === ck.id;
                      return (
                        <div
                          key={ck.id}
                          className="py-2 flex items-center justify-between gap-2 text-xs"
                        >
                          <div className="min-w-0">
                            <div className="font-mono tabular-nums font-medium text-slate-900 dark:text-slate-100 truncate">
                              {ck.boardSize}×{ck.boardSize} · Ep {ck.episode}
                              {isActive ? ' (Active)' : ''}
                            </div>
                            <div className="text-[11px] font-mono tabular-nums text-slate-500 dark:text-slate-400">
                              Eval {(ck.evalWinRate * 100).toFixed(0)}% · ε={ck.epsilon.toFixed(2)}
                            </div>
                          </div>
                          <div className="flex items-center gap-1 shrink-0">
                            <button
                              type="button"
                              onClick={() => onDownloadCheckpoint(ck.id)}
                              className="p-1.5 rounded border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800"
                              aria-label={`Download ${ck.filename}`}
                              title="Download .ziprl"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                onLoadCheckpoint(ck.id);
                                setCkptOpen(false);
                              }}
                              className="px-2.5 py-1 rounded border border-slate-200 dark:border-slate-700 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
                            >
                              Load
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Dark / Light Mode Toggle */}
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="h-8 px-2.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 inline-flex items-center gap-1.5 whitespace-nowrap"
          >
            {isDark ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-slate-600" />}
            {isDark ? 'Light' : 'Dark'}
          </button>

          {/* Safe Reset Model Action */}
          {!confirmReset ? (
            <button
              type="button"
              onClick={() => setConfirmReset(true)}
              className="h-8 px-3 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-xs font-medium text-rose-700 dark:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/50 inline-flex items-center gap-1.5 whitespace-nowrap"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          ) : (
            <div className="inline-flex items-center gap-1">
              <button
                type="button"
                onClick={() => {
                  onResetModel();
                  setConfirmReset(false);
                }}
                className="h-8 px-2.5 rounded-md bg-rose-600 text-white text-xs font-semibold hover:bg-rose-700 whitespace-nowrap"
              >
                Confirm Reset
              </button>
              <button
                type="button"
                onClick={() => setConfirmReset(false)}
                className="h-8 px-2 rounded-md border border-slate-200 dark:border-slate-700 text-xs text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 whitespace-nowrap"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
