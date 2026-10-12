import { writable } from 'svelte/store';

/** True while edited Garage settings have not yet been applied to the live preview. */
export const garageSyncPending = writable(false);

/** Set by the top bar's status to ask the inspector to open a section (e.g. 'system'). */
export const inspectorRequest = writable<string | null>(null);
