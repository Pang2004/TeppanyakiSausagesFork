import { useEffect, useRef, useState } from 'react';
import { type DiagnosticResult, type Subsystem, modules, upload, validateFiles } from './data';

export type Recording = {
  id: string; name: string; file?: File; size: number;
  status: 'queued' | 'uploading' | 'analysing' | 'completed' | 'failed' | 'stopped';
  progress: number; result?: DiagnosticResult; error?: string;
};

export function useBatch(subsystem: Subsystem = 'rail') {
  const [records, setRecords] = useState<Recording[]>([]);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const live = useRef<Recording[]>([]);
  const running = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const stopped = useRef(false);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; controller.current?.abort(); }; }, []);
  const update = (fn: (rows: Recording[]) => Recording[]) => {
    live.current = fn(live.current); if (mounted.current) setRecords(live.current);
  };
  const patch = (id: string, value: Partial<Recording>) => update(rows => rows.map(row => row.id === id ? { ...row, ...value } : row));
  const pump = async () => {
    if (running.current) return;
    running.current = true; stopped.current = false; setBusy(true);
    try {
      while (!stopped.current && mounted.current) {
        const next = live.current.find(row => row.status === 'queued');
        if (!next?.file) break;
        controller.current = new AbortController();
        patch(next.id, { status: 'uploading', progress: 0, error: undefined });
        try {
          const result = await upload(next.file, progress => patch(next.id, { progress, status: progress >= 1 ? 'analysing' : 'uploading' }), controller.current.signal, subsystem);
          patch(next.id, { status: 'completed', result, progress: 1, file: undefined });
        } catch (error) {
          const cancelled = error instanceof DOMException && error.name === 'AbortError';
          patch(next.id, { status: cancelled ? 'stopped' : 'failed', error: cancelled ? 'Stopped. You can retry this recording.' : (error as Error).message });
        }
      }
    } finally { running.current = false; controller.current = null; if (mounted.current) setBusy(false); }
  };
  const add = (files: File[], maximumBytes: number) => {
    if (running.current) { setMessages(['Wait for this batch to finish, or stop it before adding files.']); return; }
    const checked = validateFiles(files, live.current.map(row => row.name), maximumBytes, modules[subsystem].extension);
    setMessages(checked.messages);
    const additions: Recording[] = checked.files.map((file, index) => ({ id: `${Date.now()}-${index}-${file.name}`, name: file.name, file, size: file.size, status: 'queued', progress: 0 }));
    if (!additions.length) return;
    update(rows => [...rows, ...additions]);
    setSelected(additions[0].id);
    void pump();
  };
  const stop = () => {
    stopped.current = true;
    update(rows => rows.map(row => row.status === 'queued' ? { ...row, status: 'stopped', error: 'Stopped before upload.' } : row));
    controller.current?.abort();
  };
  const retry = () => {
    if (running.current) return;
    update(rows => rows.map(row => ['failed', 'stopped'].includes(row.status) && row.file ? { ...row, status: 'queued', progress: 0, error: undefined } : row));
    void pump();
  };
  const clear = () => { if (!running.current) { update(() => []); setSelected(null); setMessages([]); } };
  return { subsystem, records, busy, messages, setMessages, selected, setSelected, add, stop, retry, clear };
}
export type Batch = ReturnType<typeof useBatch>;
