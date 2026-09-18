export type RailLabel = 'Normal' | 'Side I' | 'Side II';
export type RailResult = {
  subsystem: 'rail'; model_version: string; file_id: string; prediction: RailLabel;
  scores: Record<RailLabel, number>;
  side_energy: Record<string, Record<string, number>>;
  dominant_frequency: Record<string, Record<string, number>>;
};
export type Subsystem = 'rail' | 'door' | 'acv' | 'shm';
export type DoorCycle = { start_time: string; end_time: string; prediction: 'Normal' | 'Abnormal resistance'; operation: 'Open' | 'Close'; confidence: number; quality_flags: string[]; boundary_reason: string };
export type DoorResult = { subsystem: 'door'; model_version: string; file_id: string; cycles: DoorCycle[] };
export type ACVResult = { subsystem: 'acv'; model_version: string; file_id: string; ranked_cars: string[]; car_scores: Record<string, number>; diagnostics: Record<string, Record<string, number | null>>; schema: string };
export type SHMResult = { subsystem: 'shm'; model_version: string; file_id: string; prediction: number; cycle_count: number; equivalent_stress_amplitude: number; maximum_cycle_range: number; estimated_percentage_error: number };
export type DiagnosticResult = RailResult | DoorResult | ACVResult | SHMResult;
export type Health = { status: 'ready' | 'unavailable'; maximum_file_bytes: number; subsystems?: Partial<Record<Subsystem, boolean>> };
export const modules = {
  rail: { title: 'Rail Corrugation Diagnostics', line: 'EWL', extension: '.csv', intro: 'Upload a recording to check for corrugation on Side I or Side II.' },
  door: { title: 'Door Actuator Diagnostics', line: 'NSL', extension: '.csv', intro: 'Upload a door current stream to detect operations and abnormal resistance.' },
  acv: { title: 'ACV Ventilation Diagnostics', line: 'DTL', extension: '.xlsx', intro: 'Upload an ACV workbook to rank cars by suspected refrigerant leakage.' },
  shm: { title: 'SHM Structural Diagnostics', line: 'CCL', extension: '.csv', intro: 'Upload a stress recording to estimate fatigue damage and inspect cycle diagnostics.' },
};
export function resultLabel(result: DiagnosticResult) {
  if (result.subsystem === 'door') return `${result.cycles.filter(c => c.prediction !== 'Normal').length} / ${result.cycles.length} abnormal`;
  if (result.subsystem === 'acv') return `Top rank: ${result.ranked_cars[0]}`;
  if (result.subsystem === 'shm') return `D = ${result.prediction.toPrecision(4)}`;
  return result.prediction;
}
export function resultTone(result: DiagnosticResult) {
  if (result.subsystem === 'rail') return result.prediction === 'Normal' ? 'good' : 'bad';
  if (result.subsystem === 'door') return result.cycles.some(c => c.prediction !== 'Normal') ? 'bad' : 'good';
  return 'waiting';
}
export const DEFAULT_LIMIT = 64 * 1024 * 1024;
export const naturalOrder = (a: string, b: string) => a.localeCompare(b, 'en', { numeric: true, sensitivity: 'base' });

export function validateFiles(files: File[], existing: string[], limit: number, extension = '.csv') {
  const csv = files.filter(file => file.name.toLowerCase().endsWith(extension));
  const messages: string[] = [];
  if (csv.length !== files.length) messages.push(`${files.length - csv.length} non-${extension.slice(1).toUpperCase()} file(s) ignored.`);
  if (!csv.length) return { files: [], messages: [...messages, `Choose at least one ${extension.slice(1).toUpperCase()} recording.`] };
  const names = new Set(existing.map(name => name.toLowerCase()));
  for (const file of csv) {
    if (names.has(file.name.toLowerCase())) return { files: [], messages: [`Duplicate filename: ${file.name}. Choose recordings with unique filenames, or clear the current batch.`] };
    names.add(file.name.toLowerCase());
  }
  const accepted = csv.filter(file => {
    if (!file.size) { messages.push(`${file.name}: empty file.`); return false; }
    if (file.size > limit) { messages.push(`${file.name}: exceeds the ${limit / 1024 / 1024} MiB limit.`); return false; }
    return true;
  });
  return { files: accepted.sort((a, b) => naturalOrder(a.name, b.name)), messages };
}

export function resultsCsv(results: RailResult[]) {
  const quote = (value: string) => /[",\r\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
  return 'file_id,prediction\n' + [...results].sort((a, b) => naturalOrder(a.file_id, b.file_id))
    .map(result => `${quote(result.file_id)},${quote(result.prediction)}\n`).join('');
}

export async function readDirectory(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) return new Promise((resolve, reject) => (entry as FileSystemFileEntry).file(file => resolve([file]), reject));
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const files: File[] = [];
  while (true) {
    const entries = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
    if (!entries.length) break;
    for (const child of entries) files.push(...await readDirectory(child));
  }
  return files;
}

export async function droppedFiles(items: DataTransferItemList, fallback: FileList) {
  // Capture handles synchronously: the browser clears the drag data after drop.
  // Prefer getAsFile for ordinary files: WebKit can expose an entry whose
  // file() method is unreadable even though the actual File is available.
  const handles = Array.from(items).filter(item => item.kind === 'file').map(item => {
    let entry: FileSystemEntry | null = null;
    try { entry = item.webkitGetAsEntry?.() ?? null; } catch { /* use the File below */ }
    return { entry, file: item.getAsFile() };
  });
  const ordinaryFiles = Array.from(fallback);
  if (!handles.length) return ordinaryFiles;
  const files: File[] = [];
  for (const { entry, file } of handles) {
    if (entry?.isDirectory) files.push(...await readDirectory(entry));
    else if (file) files.push(file);
    else if (entry) files.push(...await readDirectory(entry));
  }
  return files.length ? files : ordinaryFiles;
}

export function upload(file: File, onProgress: (fraction: number) => void, signal: AbortSignal, subsystem: Subsystem = 'rail'): Promise<DiagnosticResult> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const cancel = () => xhr.abort();
    const fail = (error: Error) => { signal.removeEventListener('abort', cancel); reject(error); };
    xhr.open('POST', `/api/predict/${subsystem}`);
    xhr.timeout = 180_000;
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded / event.total); };
    xhr.onload = () => {
      signal.removeEventListener('abort', cancel);
      let body;
      try { body = JSON.parse(xhr.responseText); } catch { reject(new Error('The backend returned an unreadable response.')); return; }
      if (xhr.status < 200 || xhr.status >= 300) { reject(new Error(body.error?.message ?? 'The recording could not be analysed.')); return; }
      if (!validResult(body, subsystem) || body.file_id !== file.name) {
        reject(new Error('The backend returned an invalid prediction.')); return;
      }
      resolve(body);
    };
    xhr.onerror = () => fail(new Error('Connection lost. Check the backend and retry this file.'));
    xhr.ontimeout = () => fail(new Error('The request timed out. Retry this file.'));
    xhr.onabort = () => fail(new DOMException('Upload stopped.', 'AbortError'));
    signal.addEventListener('abort', cancel, { once: true });
    if (signal.aborted) { fail(new DOMException('Upload stopped.', 'AbortError')); return; }
    const form = new FormData(); form.append('file', file, file.name); xhr.send(form);
  });
}

export function validResult(body: DiagnosticResult, subsystem: Subsystem): boolean {
  if (!body || body.subsystem !== subsystem || typeof body.file_id !== 'string') return false;
  if (body.subsystem === 'rail') return ['Normal', 'Side I', 'Side II'].includes(body.prediction);
  if (body.subsystem === 'door') return Array.isArray(body.cycles) && body.cycles.every(c =>
    ['Normal', 'Abnormal resistance'].includes(c.prediction) && ['Open', 'Close'].includes(c.operation) &&
    typeof c.start_time === 'string' && typeof c.end_time === 'string' && Array.isArray(c.quality_flags));
  if (body.subsystem === 'acv') return Array.isArray(body.ranked_cars) && body.ranked_cars.length > 0 &&
    new Set(body.ranked_cars).size === body.ranked_cars.length && body.ranked_cars.every(car => typeof car === 'string' && Number.isFinite(body.car_scores?.[car]) && body.diagnostics?.[car]);
  return [body.prediction, body.cycle_count, body.equivalent_stress_amplitude, body.maximum_cycle_range].every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0);
}

export function exportResults(results: DiagnosticResult[], subsystem: Subsystem): string {
  const quote = (value: string | number) => `"${String(value).replaceAll('"', '""')}"`;
  const sorted = [...results].sort((a, b) => naturalOrder(a.file_id, b.file_id));
  if (subsystem === 'rail') return resultsCsv(sorted.filter((r): r is RailResult => r.subsystem === 'rail'));
  let header: string; let rows: (string | number)[][];
  if (subsystem === 'door') {
    header = 'file_id,start_time,end_time,prediction';
    rows = sorted.flatMap(r => r.subsystem === 'door' ? r.cycles.map(c => [r.file_id, c.start_time, c.end_time, c.prediction]) : []);
  } else if (subsystem === 'acv') {
    header = 'file_id,ranked_cars'; rows = sorted.flatMap(r => r.subsystem === 'acv' ? [[r.file_id, r.ranked_cars.join('|')]] : []);
  } else {
    header = 'file_id,prediction'; rows = sorted.flatMap(r => r.subsystem === 'shm' ? [[r.file_id, r.prediction]] : []);
  }
  return header + '\n' + rows.map(row => row.map(quote).join(',') + '\n').join('');
}

/** Official single-stream Door submission schema; batch exports retain file_id. */
export function exportDoorSubmission(result: DoorResult): string {
  const quote = (value: string) => `"${value.replaceAll('"', '""')}"`;
  return 'start_time,end_time,prediction\n' + result.cycles.map(c =>
    [c.start_time, c.end_time, c.prediction].map(quote).join(',') + '\n').join('');
}

export function doorTimestamp(value: string): number {
  const parts = /^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,3})$/.exec(value.trim());
  if (!parts) return Date.parse(value);
  const [year, month, day, hour, minute, second, millisecond] = parts.slice(1).map(Number);
  return Date.UTC(year, month - 1, day, hour, minute, second, millisecond);
}
export function doorClock(value: string): string {
  const timestamp = doorTimestamp(value);
  return Number.isFinite(timestamp) ? new Date(timestamp).toISOString().slice(11, 23) : value;
}
export function doorDuration(cycle: Pick<DoorCycle, 'start_time' | 'end_time'>): string {
  const duration = doorTimestamp(cycle.end_time) - doorTimestamp(cycle.start_time);
  return Number.isFinite(duration) && duration >= 0 ? `${(duration / 1000).toFixed(3)} s` : 'Not available';
}
