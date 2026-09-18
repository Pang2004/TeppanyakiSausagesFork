import { useRef, useState } from 'react';
import { DEFAULT_LIMIT, doorClock, doorDuration, type DoorResult, type Health } from './data';
import type { Batch } from './useBatch';

function Time({ value }: { value?: string }) {
  const text = value ? doorClock(value) : '--:--:--.---';
  const split = text.lastIndexOf('.');
  return <>{split < 0 ? text : <>{text.slice(0, split)}<small>{text.slice(split)}</small></>}</>;
}
function Wall() {
  return <div className="door-wall" aria-hidden="true"><div className="door-livery" /><div className="door-hardware"><i /><span /><i /></div><b /></div>;
}
function Diagnosis({ result, name, message, processing, choose, disabled }: { result?: DoorResult; name?: string; message?: string; processing: boolean; choose: () => void; disabled: boolean }) {
  const [index, setIndex] = useState(0);
  const cycle = result?.cycles[index];
  const abnormal = result?.cycles.filter(c => c.prediction !== 'Normal').length ?? 0;
  const bad = !!cycle && cycle.prediction !== 'Normal';
  const state = cycle ? bad ? 'fault' : 'normal' : 'standby';
  const status = message ? 'Recording needs attention' : processing ? 'Analysing recording' : result ? 'No operations detected' : 'Awaiting CSV input';
  return <>
    <div className="door-telemetry"><div className="door-file"><strong title={name}>{name ?? 'Door cycle diagnostics'}</strong><span>DOOR</span></div>
      <div className={`door-summary ${result?.cycles.length ? abnormal ? 'bad' : 'good' : 'waiting'}`}><strong>{result?.cycles.length ? `${abnormal} abnormal resistance ${abnormal === 1 ? 'operation' : 'operations'}` : status}</strong><span>{result?.cycles.length ? `${(abnormal / result.cycles.length * 100).toLocaleString('en', { maximumFractionDigits: 1 })}% rate` : result ? '0 operations · —%' : '— operations · —%'}</span></div>
    </div>
    <div className={`door-carriage ${state}`}>
      <div className="door-overhead"><span /><i /><span /></div>
      <div className="door-body"><Wall /><div className="door-center">
      <div className="door-windows">
        <section className="door-window"><h2>{cycle ? `OPERATION #${index + 1}` : 'STANDBY'}</h2><span className="door-operation">{cycle?.operation.toUpperCase() ?? 'DOOR TELEMETRY'}</span><dl><dt>START TIME</dt><dd title={cycle?.start_time}><Time value={cycle?.start_time} /></dd><dt>END TIME</dt><dd title={cycle?.end_time}><Time value={cycle?.end_time} /></dd></dl><div className="door-window-footer">DURATION <b>{cycle ? doorDuration(cycle) : '— s'}</b></div></section>
        <section className="door-window door-verdict"><span className="eyebrow">PREDICTION STATUS</span><div className="verdict-symbol" aria-hidden="true">{cycle ? bad ? '!' : '✓' : '…'}</div><h2>{cycle ? bad ? 'ABNORMAL' : 'NORMAL' : processing ? 'ANALYSING' : 'AWAITING DATA'}</h2><p>{cycle ? bad ? 'Abnormal resistance detected' : 'Nominal door resistance' : 'No prediction yet'}</p><div className="door-window-footer"><b>{cycle ? bad ? 'FLAG' : 'PASS' : 'STANDBY'}</b><span>{cycle?.operation ?? 'DIAGNOSTIC'}</span></div></section>
      </div>
      {cycle && result ? <div className="door-cycle-list" aria-label="Detected door operations">{result.cycles.map((c, i) => <button key={i} className={`door-cycle ${index === i ? 'active' : ''} ${c.prediction === 'Normal' ? 'good' : 'bad'}`} aria-pressed={index === i} onClick={() => setIndex(i)}><span><strong>#{i + 1} <small>{c.operation}</small></strong><small>Δ {doorDuration(c)}</small><span>START {doorClock(c.start_time)}</span><span>END {doorClock(c.end_time)}</span></span><span className={c.prediction === 'Normal' ? 'good-text' : 'bad-text'}><b><i aria-hidden="true">{c.prediction === 'Normal' ? '✓' : '!'}</i>{c.prediction}</b><span aria-hidden="true">{index === i ? '▪' : '›'}</span></span></button>)}</div> : <div className="door-empty"><span className="door-empty-icon" aria-hidden="true">▤</span><h3>{status}</h3><p role="status">{message ?? (processing ? 'Reading actuator current and identifying door operations…' : result ? 'This recording contains no detected door operations. Choose another CSV to analyse.' : 'Upload actuator-current CSV files to inspect operation timings and door resistance.')}</p><button className="door-choose" disabled={disabled} onClick={choose}>Choose CSV files</button></div>}
      </div><Wall /></div>
      <div className="door-sill"><span /><span /></div><div className="door-underframe" aria-hidden="true"><i /><i /><i /></div>
    </div>
    {!!cycle?.quality_flags.length && <p className="input-messages">Recording notes: {cycle.quality_flags.join(', ')}</p>}
  </>;
}
export function DoorView({ batch, health }: { batch: Batch; health: Health | null }) {
  const input = useRef<HTMLInputElement>(null);
  const selected = batch.records.find(r => r.id === batch.selected);
  const result = selected?.result?.subsystem === 'door' ? selected.result : undefined;
  const disabled = batch.busy || !health?.subsystems?.door;
  return <section className="visual-column door-view" aria-label="Door operation diagnosis"><input ref={input} type="file" className="file-input" accept=".csv" multiple aria-label="Choose Door telemetry files" disabled={disabled} onChange={event => { if (event.target.files) batch.add(Array.from(event.target.files), health?.maximum_file_bytes ?? DEFAULT_LIMIT); event.target.value = ''; }} /><Diagnosis key={selected?.id ?? 'empty'} result={result} name={selected?.name} message={selected?.error} processing={!!selected && ['queued', 'uploading', 'analysing'].includes(selected.status)} choose={() => input.current?.click()} disabled={disabled} /></section>;
}
