import { CheckpointInfo, EpisodeMetric, EvalPoint, TrainingConfig } from './types';
import { SerializedAgent } from './dqn';

export const ZIPRL_FORMAT_ID = 'zippy-rl-checkpoint';
export const ZIPRL_FORMAT_VERSION = 1;

export interface ZiprlCheckpointPayload {
  format: typeof ZIPRL_FORMAT_ID;
  version: typeof ZIPRL_FORMAT_VERSION;
  info: CheckpointInfo;
  config: TrainingConfig;
  agent: SerializedAgent;
  episode: number;
  totalSolved: number;
  totalDeadEnds: number;
  bestWinRate: number;
  latestEvalWinRate: number;
  metrics: EpisodeMetric[];
  evalHistory: EvalPoint[];
}

/**
 * Serializes a checkpoint payload into the portable .ziprl JSON format.
 */
export function serializeZiprlCheckpoint(payload: ZiprlCheckpointPayload): string {
  return JSON.stringify(payload);
}

/**
 * Parses and validates an uploaded .ziprl checkpoint file.
 */
export function parseZiprlCheckpoint(rawText: string): ZiprlCheckpointPayload {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    throw new Error('Invalid .ziprl file: content is not valid JSON.');
  }

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Invalid .ziprl file: expected a checkpoint object.');
  }

  const obj = parsed as Partial<ZiprlCheckpointPayload>;
  if (obj.format !== ZIPRL_FORMAT_ID) {
    throw new Error('Unrecognized file format: expected a Zippy RL (.ziprl) checkpoint.');
  }
  if (!obj.info || !obj.config || !obj.agent || !obj.agent.onlineNet || !obj.agent.targetNet) {
    throw new Error('Corrupted .ziprl checkpoint: missing required model weights or metadata.');
  }
  if (
    !Array.isArray(obj.agent.onlineNet.layers) ||
    obj.agent.onlineNet.layers.length !== 3 ||
    !Array.isArray(obj.agent.targetNet.layers) ||
    obj.agent.targetNet.layers.length !== 3
  ) {
    throw new Error('Corrupted .ziprl checkpoint: invalid MLP layer structure.');
  }
  const boardSize = Number(obj.config.boardSize);
  if (!Number.isInteger(boardSize) || boardSize < 4 || boardSize > 10) {
    throw new Error(`Invalid board size (${obj.config.boardSize}) in .ziprl checkpoint.`);
  }

  return obj as ZiprlCheckpointPayload;
}

/**
 * Triggers a browser file download of a .ziprl checkpoint file.
 */
export function triggerZiprlDownload(payload: ZiprlCheckpointPayload): void {
  const json = serializeZiprlCheckpoint(payload);
  const blob = new Blob([json], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  const safeName = payload.info.filename.endsWith('.ziprl')
    ? payload.info.filename
    : `${payload.info.id}.ziprl`;
  anchor.href = url;
  anchor.download = safeName;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// --- IndexedDB Browser Checkpoint Store (Zero Backend Needed) ---

const DB_NAME = 'zippy-rl-browser-db';
const DB_VERSION = 1;
const STORE_CHECKPOINTS = 'checkpoints';

function openCheckpointDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_CHECKPOINTS)) {
          db.createObjectStore(STORE_CHECKPOINTS, { keyPath: 'id' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

interface StoredCheckpointRecord {
  id: string;
  createdAt: string;
  payload: ZiprlCheckpointPayload;
}

export async function saveCheckpointToBrowserDb(payload: ZiprlCheckpointPayload): Promise<void> {
  const db = await openCheckpointDb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHECKPOINTS, 'readwrite');
      const store = tx.objectStore(STORE_CHECKPOINTS);
      const record: StoredCheckpointRecord = {
        id: payload.info.id,
        createdAt: payload.info.createdAt,
        payload,
      };
      store.put(record);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}

export async function listBrowserCheckpoints(): Promise<CheckpointInfo[]> {
  const db = await openCheckpointDb();
  if (!db) return [];
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHECKPOINTS, 'readonly');
      const store = tx.objectStore(STORE_CHECKPOINTS);
      const req = store.getAll();
      req.onsuccess = () => {
        const rows = (req.result as StoredCheckpointRecord[]) ?? [];
        rows.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
        db.close();
        resolve(rows.slice(0, 30).map((r) => r.payload.info));
      };
      req.onerror = () => {
        db.close();
        resolve([]);
      };
    } catch {
      db.close();
      resolve([]);
    }
  });
}

export async function loadCheckpointFromBrowserDb(id: string): Promise<ZiprlCheckpointPayload | null> {
  const db = await openCheckpointDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHECKPOINTS, 'readonly');
      const store = tx.objectStore(STORE_CHECKPOINTS);
      const req = store.get(id);
      req.onsuccess = () => {
        const row = req.result as StoredCheckpointRecord | undefined;
        db.close();
        resolve(row ? row.payload : null);
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

export async function deleteCheckpointFromBrowserDb(id: string): Promise<void> {
  const db = await openCheckpointDb();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_CHECKPOINTS, 'readwrite');
      tx.objectStore(STORE_CHECKPOINTS).delete(id);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}
