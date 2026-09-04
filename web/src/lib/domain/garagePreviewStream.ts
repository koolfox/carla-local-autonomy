export type GarageStreamSlot = 0 | 1;

export interface GarageStreamBuffer {
  sources: [string, string];
  visible: GarageStreamSlot;
  pending: GarageStreamSlot | null;
  ready: boolean;
  lastRequested: string;
}

export interface GarageStreamFailure {
  buffer: GarageStreamBuffer;
  preservedLastFrame: boolean;
  shouldRetry: boolean;
}

export function createGarageStreamBuffer(): GarageStreamBuffer {
  return {
    sources: ['', ''],
    visible: 0,
    pending: null,
    ready: false,
    lastRequested: ''
  };
}

function slot(value: number): GarageStreamSlot {
  if (value !== 0 && value !== 1) throw new RangeError('Garage stream slot must be 0 or 1');
  return value;
}

export function stageGarageStream(
  current: GarageStreamBuffer,
  source: string
): GarageStreamBuffer {
  if (!source || source === current.lastRequested) return current;
  const target: GarageStreamSlot = current.ready
    ? (current.visible === 0 ? 1 : 0)
    : current.visible;
  const sources: [string, string] = [...current.sources];
  sources[target] = source;
  return {
    ...current,
    sources,
    pending: target,
    lastRequested: source
  };
}

export function confirmGarageStream(
  current: GarageStreamBuffer,
  rawSlot: number
): GarageStreamBuffer {
  const loaded = slot(rawSlot);
  if (!current.sources[loaded]) return current;
  const previous = current.visible;
  const sources: [string, string] = [...current.sources];
  if (previous !== loaded) sources[previous] = '';
  return {
    ...current,
    sources,
    visible: loaded,
    pending: null,
    ready: true
  };
}

export function failGarageStream(
  current: GarageStreamBuffer,
  rawSlot: number,
  busy: boolean
): GarageStreamFailure {
  const failed = slot(rawSlot);
  const isVisible = current.ready && current.visible === failed;
  if (isVisible) {
    return {
      buffer: {
        ...current,
        pending: current.pending === failed ? null : current.pending
      },
      preservedLastFrame: true,
      shouldRetry: !busy
    };
  }

  const sources: [string, string] = [...current.sources];
  sources[failed] = '';
  return {
    buffer: {
      ...current,
      sources,
      pending: current.pending === failed ? null : current.pending
    },
    preservedLastFrame: false,
    shouldRetry: !busy
  };
}
