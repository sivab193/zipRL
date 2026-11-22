import React, { useMemo } from 'react';
import { ActionEvaluation, CellCoord, ZipBoard } from '../rl/types';

interface ZipBoardViewProps {
  board: ZipBoard;
  path: CellCoord[];
  nextWaypointOrder: number;
  solved?: boolean;
  deadEnd?: boolean;
  oracleSolution?: CellCoord[] | null;
  latestEvaluations?: ActionEvaluation[];
  compact?: boolean;
  isDark?: boolean;
}

export const ZipBoardView: React.FC<ZipBoardViewProps> = ({
  board,
  path,
  nextWaypointOrder,
  solved = false,
  deadEnd = false,
  oracleSolution = null,
  latestEvaluations = [],
  compact = false,
  isDark = false,
}) => {
  const { size, waypoints, walls } = board;

  const cellSize = 64;
  const pad = 20;
  const gridPx = size * cellSize;
  const totalPx = gridPx + pad * 2;

  const visitedSet = useMemo(() => {
    const s = new Set<number>();
    for (let i = 0; i < path.length; i++) {
      s.add(path[i].row * size + path[i].col);
    }
    return s;
  }, [path, size]);

  const head = path.length > 0 ? path[path.length - 1] : null;

  const cellCenter = (row: number, col: number) => ({
    x: pad + col * cellSize + cellSize / 2,
    y: pad + row * cellSize + cellSize / 2,
  });

  const buildPolyline = (coords: CellCoord[]) =>
    coords
      .map((c) => {
        const pt = cellCenter(c.row, c.col);
        return `${pt.x},${pt.y}`;
      })
      .join(' ');

  const agentPolyline = useMemo(() => buildPolyline(path), [path, size]);
  const oraclePolyline = useMemo(
    () => (oracleSolution && oracleSolution.length > 0 ? buildPolyline(oracleSolution) : ''),
    [oracleSolution, size],
  );

  const pathStrokeColor = solved ? '#10B981' : deadEnd ? '#EF4444' : isDark ? '#3B82F6' : '#2563EB';
  const visitedFillColor = solved
    ? isDark
      ? 'rgba(16, 185, 129, 0.16)'
      : 'rgba(5, 150, 105, 0.09)'
    : deadEnd
      ? isDark
        ? 'rgba(239, 68, 68, 0.16)'
        : 'rgba(220, 38, 38, 0.08)'
      : isDark
        ? 'rgba(59, 130, 246, 0.18)'
        : 'rgba(37, 99, 235, 0.08)';

  const cellBg = isDark ? '#111827' : '#FFFFFF';
  const headBg = isDark ? 'rgba(59, 130, 246, 0.28)' : 'rgba(37, 99, 235, 0.14)';
  const cellStroke = isDark ? '#1E293B' : '#E2E8F0';
  const outerStroke = isDark ? '#64748B' : '#334155';
  const wallStroke = isDark ? '#E2E8F0' : '#0F172A';
  const waypointStroke = isDark ? '#0F172A' : '#FFFFFF';

  const waypointRadius = size >= 9 ? 14 : size >= 7 ? 16 : 18;
  const waypointFontSize = size >= 9 ? 13 : size >= 7 ? 15 : 17;

  return (
    <div
      className={`relative flex items-center justify-center w-full ${compact ? 'max-w-[340px]' : 'max-w-[460px]'} mx-auto select-none`}
      data-testid="zip-board-container"
    >
      <svg
        viewBox={`0 0 ${totalPx} ${totalPx}`}
        className="w-full h-auto block bg-transparent"
        role="img"
        aria-label={`Zip ${size} by ${size} puzzle board, ${path.length} of ${size * size} cells visited`}
      >
        {/* Outer grid background */}
        <rect
          x={pad}
          y={pad}
          width={gridPx}
          height={gridPx}
          rx={6}
          fill={isDark ? '#0B101A' : '#F8FAFC'}
          stroke={isDark ? '#334155' : '#CBD5E1'}
          strokeWidth={2}
        />

        {/* Cells */}
        {Array.from({ length: size * size }, (_, idx) => {
          const r = Math.floor(idx / size);
          const c = idx % size;
          const x = pad + c * cellSize;
          const y = pad + r * cellSize;
          const isVisited = visitedSet.has(idx);
          const isHead = head !== null && head.row === r && head.col === c;

          return (
            <g key={idx}>
              <rect
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                fill={isHead ? headBg : isVisited ? visitedFillColor : cellBg}
                stroke={cellStroke}
                strokeWidth={1}
              />
            </g>
          );
        })}

        {/* Optional DFS Oracle Solution Overlay */}
        {oraclePolyline && (
          <polyline
            points={oraclePolyline}
            fill="none"
            stroke="#10B981"
            strokeWidth={size >= 8 ? 5 : 7}
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeDasharray="10 8"
            opacity={0.78}
          />
        )}

        {/* Agent's continuous path ribbon */}
        {path.length > 1 && (
          <polyline
            points={agentPolyline}
            fill="none"
            stroke={pathStrokeColor}
            strokeWidth={size >= 8 ? 10 : 14}
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={0.9}
          />
        )}

        {/* Legal move directional cues from current head */}
        {head && !solved && !deadEnd && latestEvaluations.length > 0 && (
          <g>
            {latestEvaluations
              .filter((ev) => ev.legal)
              .map((ev) => {
                const start = cellCenter(head.row, head.col);
                const target = cellCenter(ev.targetRow, ev.targetCol);
                const midX = start.x + (target.x - start.x) * 0.56;
                const midY = start.y + (target.y - start.y) * 0.56;
                return (
                  <g key={ev.direction}>
                    <line
                      x1={start.x}
                      y1={start.y}
                      x2={midX}
                      y2={midY}
                      stroke={
                        ev.isBacktrack
                          ? '#F59E0B'
                          : ev.chosen
                            ? pathStrokeColor
                            : isDark
                              ? '#64748B'
                              : '#94A3B8'
                      }
                      strokeWidth={ev.chosen ? 3 : 2}
                      strokeDasharray={ev.chosen ? undefined : '4 3'}
                    />
                  </g>
                );
              })}
          </g>
        )}

        {/* Walls sitting strictly on edges between adjacent cells */}
        {walls.map((w, idx) => {
          const isVerticalWall = w.r1 === w.r2;
          if (isVerticalWall) {
            const wallX = pad + Math.max(w.c1, w.c2) * cellSize;
            const wallY1 = pad + w.r1 * cellSize + 3;
            const wallY2 = pad + (w.r1 + 1) * cellSize - 3;
            return (
              <line
                key={`w-${idx}`}
                x1={wallX}
                y1={wallY1}
                x2={wallX}
                y2={wallY2}
                stroke={wallStroke}
                strokeWidth={6}
                strokeLinecap="round"
              />
            );
          } else {
            const wallY = pad + Math.max(w.r1, w.r2) * cellSize;
            const wallX1 = pad + w.c1 * cellSize + 3;
            const wallX2 = pad + (w.c1 + 1) * cellSize - 3;
            return (
              <line
                key={`w-${idx}`}
                x1={wallX1}
                y1={wallY}
                x2={wallX2}
                y2={wallY}
                stroke={wallStroke}
                strokeWidth={6}
                strokeLinecap="round"
              />
            );
          }
        })}

        {/* Current head highlight ring */}
        {head && (
          <g>
            <circle
              cx={cellCenter(head.row, head.col).x}
              cy={cellCenter(head.row, head.col).y}
              r={waypointRadius + 5}
              fill="none"
              stroke={pathStrokeColor}
              strokeWidth={3}
              opacity={0.95}
            />
            <circle
              cx={cellCenter(head.row, head.col).x}
              cy={cellCenter(head.row, head.col).y}
              r={8}
              fill={pathStrokeColor}
              stroke={waypointStroke}
              strokeWidth={2.5}
            />
          </g>
        )}

        {/* Numbered waypoints */}
        {waypoints.map((wp) => {
          const pt = cellCenter(wp.row, wp.col);
          const isVisitedWp = wp.order < nextWaypointOrder;
          const isNextTarget = wp.order === nextWaypointOrder && !solved;
          const fill = isVisitedWp
            ? solved
              ? '#059669'
              : isDark
                ? '#334155'
                : '#1E293B'
            : isNextTarget
              ? '#2563EB'
              : isDark
                ? '#1E293B'
                : '#0F172A';

          return (
            <g key={`wp-${wp.order}`}>
              {isNextTarget && (
                <circle
                  cx={pt.x}
                  cy={pt.y}
                  r={waypointRadius + 4}
                  fill="none"
                  stroke="#3B82F6"
                  strokeWidth={2}
                  strokeDasharray="4 3"
                />
              )}
              <circle
                cx={pt.x}
                cy={pt.y}
                r={waypointRadius}
                fill={fill}
                stroke={waypointStroke}
                strokeWidth={2.5}
              />
              <text
                x={pt.x}
                y={pt.y}
                textAnchor="middle"
                dominantBaseline="central"
                fill="#FFFFFF"
                fontSize={wp.order >= 10 ? Math.max(10, waypointFontSize - 2.5) : waypointFontSize}
                fontWeight={700}
                fontFamily="IBM Plex Mono, monospace"
              >
                {wp.order}
              </text>
            </g>
          );
        })}

        {/* Crisp outer border redraw so edge walls meet cleanly */}
        <rect
          x={pad}
          y={pad}
          width={gridPx}
          height={gridPx}
          rx={6}
          fill="none"
          stroke={outerStroke}
          strokeWidth={2.5}
        />
      </svg>
    </div>
  );
};
