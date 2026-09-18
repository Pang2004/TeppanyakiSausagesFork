import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { DEFAULT_LIMIT, droppedFiles, type Health, type RailLabel, type Subsystem, modules, resultLabel, resultTone, exportDoorSubmission, exportResults } from './data';
import { RailArtwork } from './RailArtwork';
import { DoorView, ACVView, SHMView } from './SubsystemViews';
import { TrainArtwork } from './TrainArtwork';
import { type Batch, useBatch } from './useBatch';

function Icon({ name, className = '' }: { name: 'upload' | 'file' | 'download' | 'arrow' | 'folder'; className?: string }) {
  const paths = { upload: 'M7 16a4 4 0 01-.88-7.9A5 5 0 0116 6a5 5 0 011 9.9M12 12v9m-3-6 3-3 3 3', file: 'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zm0 0v6h6M8 13h8m-8 4h6', download: 'M12 3v12m-4-4 4 4 4-4M4 16v5h16v-5', arrow: 'm9 5 7 7-7 7', folder: 'M3 7V5h6l2 2h10v13H3z' };
  return <svg className={`icon ${className}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}
function ReadyBadge({ ready }: { ready: boolean | null }) {
  return <span className={`ready-badge ${ready === false ? 'offline' : ''}`} role="status"><span className="dot" />{ready === null ? 'CONNECTING' : ready ? 'MODEL READY' : 'OFFLINE'}</span>;
}

const subsystems = [
  { key: 'rail', code: 'EW', title: 'RAIL', description: 'Corrugation wear' },
  { key: 'shm', code: 'CC', title: 'SHM', description: 'Vibration & Bogie' },
  { key: 'acv', code: 'DT', title: 'ACV', description: 'Vent / Saloon Fan' },
  { key: 'door', code: 'NS', title: 'DOOR', description: 'Actuator Current' },
];

function Home({ ready }: { ready: boolean | null }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const navigate = useNavigate();
  return <div className="home-shell page-shell">
    <header className="fleet-header"><div className="brand"><span className="mrt-mark">MRT</span><div><h1>Fleet Diagnostic</h1><p>TRAIN CONDITION MONITORING</p></div></div><ReadyBadge ready={ready} /></header>
    <main className="home-grid">
      <div className="train-stage"><TrainArtwork hovered={hovered} onHover={setHovered} onOpen={key => navigate(`/${key}`)} /></div>
      <div className="subsystem-grid" aria-label="Choose a subsystem">
        {subsystems.map(sub => <button key={sub.key} type="button" className={`subsystem-card ${sub.key} ${hovered === sub.key ? 'is-hovered' : ''}`}  onMouseEnter={() => setHovered(sub.key)} onMouseLeave={() => setHovered(null)} onFocus={() => setHovered(sub.key)} onBlur={() => setHovered(null)} onClick={() => navigate(`/${sub.key}`)}>
          
          <span className="line-pill"><span>{sub.code}</span>{sub.title}</span><span className="subsystem-description">{sub.description}</span>
          <span className="open-module">Open diagnostics <Icon name="arrow" /></span>
        </button>)}
      </div>
    </main>
    <footer className="page-footer">TeppanyakiSausages <span>·</span> PS3 Fleet Diagnostic</footer>
  </div>;
}

function ExitSign({ subsystem }: { subsystem: Subsystem }) {
  return <header className="exit-header"><Link className="exit-sign" to="/">
    <span className="exit-symbol"><b>EXIT</b><svg viewBox="0 0 36 36" fill="none" aria-hidden="true"><path d="M6 31V9a3 3 0 0 1 3-3h18a3 3 0 0 1 3 3v22M23 18.5H13m0 0 4.5-4.5M13 18.5l4.5 4.5" stroke="currentColor" strokeWidth="3.5" /></svg></span>
    <span className="exit-copy"><span className="ewl-tag">{modules[subsystem].line} · {subsystem.toUpperCase()}</span><strong>Exit to Main Page</strong><span>{modules[subsystem].title}</span></span>
  </Link></header>;
}

function UploadPanel({ batch, health, refresh }: { batch: Batch; health: Health | null; refresh: () => void }) {
  const filesInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const [foldersSupported, setFoldersSupported] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const limit = health?.maximum_file_bytes ?? DEFAULT_LIMIT;
  const available = health?.subsystems?.[batch.subsystem] ?? (batch.subsystem === 'rail' && health?.status === 'ready');
  const format = modules[batch.subsystem].extension.slice(1).toUpperCase();
  const disabled = batch.busy || collecting || !available;
  useEffect(() => {
    if (folderInput.current) {
      setFoldersSupported('webkitdirectory' in folderInput.current);
      folderInput.current.setAttribute('webkitdirectory', '');
    }
  }, []);
  const choose = (list: FileList | null) => { if (list) batch.add(Array.from(list), limit); };
  const current = batch.records.find(row => ['uploading', 'analysing'].includes(row.status));
  const completed = batch.records.filter(row => ['completed', 'failed', 'stopped'].includes(row.status)).length;
  return <section className="panel upload-panel" aria-labelledby="upload-title">
    <div className="panel-heading"><span className="eyebrow">{batch.subsystem.toUpperCase()} / {format} RECORDINGS</span><ReadyBadge ready={health ? available : null} /></div>
    <h1 id="upload-title">Analyse your recordings</h1><p className="intro">{modules[batch.subsystem].intro}</p>
    {!available && <div className="connection-message" role="status">{health ? 'The model is offline. Start the backend to analyse recordings.' : 'Connecting to the local model…'} <button className="text-button" onClick={refresh}>Check connection</button></div>}
    <div className={`drop-zone ${dragging ? 'dragging' : ''} ${disabled ? 'unavailable' : ''}`} onDragOver={event => { event.preventDefault(); if (!disabled) setDragging(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }} onDrop={async event => {
      event.preventDefault(); event.stopPropagation(); setDragging(false);
      if (disabled) return;
      setCollecting(true);
      try { const files = await droppedFiles(event.dataTransfer.items, event.dataTransfer.files); batch.add(files, limit); }
      catch { batch.setMessages(['This folder could not be read. Try Choose files instead.']); }
      finally { setCollecting(false); }
    }}>
      <span className="upload-icon"><Icon name="upload" /></span>
      <strong>{collecting ? 'Reading folder…' : batch.busy ? 'Analysing your batch' : `Drop ${format} files or a folder here`}</strong>
      <span className="drop-hint">{limit / 1024 / 1024} MiB per file · multiple recordings welcome</span>
      <div className="upload-actions"><button className="primary-button" disabled={disabled} onClick={() => filesInput.current?.click()}><Icon name="file" />Choose files</button>{foldersSupported && <button className="secondary-button" disabled={disabled} onClick={() => folderInput.current?.click()}><Icon name="folder" />Choose folder</button>}</div>
      <input className="file-input" aria-label={`Choose ${format} files`} type="file" accept={modules[batch.subsystem].extension} multiple ref={filesInput} onChange={event => { choose(event.target.files); event.target.value = ''; }} disabled={disabled} />
      <input className="file-input" aria-label={`Choose ${format} folder`} type="file" multiple ref={folderInput} onChange={event => { choose(event.target.files); event.target.value = ''; }} disabled={disabled} />
    </div>
    <p className="upload-note">Analysis starts automatically. Files are processed one at a time and removed from the server afterwards.</p>
    {batch.messages.length > 0 && <div className="input-messages" role="alert">{batch.messages.map((message, i) => <p key={i}>{message}</p>)}</div>}
    {batch.busy && <div className="batch-progress"><div className="progress-heading"><strong>{completed} / {batch.records.length} processed</strong><button className="text-button" onClick={batch.stop}>Stop batch</button></div><progress aria-label="Batch progress" max={batch.records.length} value={completed} /><p className="current-file" title={current?.name}>{current?.name}</p><p className="muted" role="status">{current?.status === 'analysing' ? `Running the ${batch.subsystem.toUpperCase()} model…` : `Uploading ${Math.round((current?.progress ?? 0) * 100)}%`}</p></div>}
  </section>;
}

function RailView({ batch }: { batch: Batch }) {
  const selected = batch.records.find(row => row.id === batch.selected);
  const prediction = selected?.result?.subsystem === 'rail' ? selected.result.prediction : undefined;
  return <section className="visual-column" aria-label="Selected recording diagnosis">
    <div className="telemetry-panel"><span className="file-icon"><Icon name="file" /></span><div className="file-label"><span className="eyebrow">{prediction ? 'FILE ANALYSED' : 'RECORDING'} <span className="mini-tag">BATCH ML</span></span><strong title={selected?.name}>{selected?.name ?? 'Awaiting file input…'}</strong></div><span className={`prediction-badge ${!prediction ? 'waiting' : prediction === 'Normal' ? 'good' : 'bad'}`}><span className="dot" />{!prediction ? (selected?.status === 'failed' ? 'Analysis failed' : 'Awaiting result') : prediction === 'Normal' ? 'Nominal Profile' : `${prediction} Corrugation`}</span></div>
    <div className="track-panel"><div className="track-head">
      {(['Side I', 'Side II'] as const).map((side, index) => <div className={`track-side ${index ? 'right' : ''}`} key={side}><strong>{index ? 'SIDE II (RIGHT) ▶' : '◀ SIDE I (LEFT)'}</strong><span className={`side-chip ${!prediction ? 'waiting' : prediction === side ? 'bad' : 'good'}`}><span className="dot" />{side}: {!prediction ? 'Waiting' : prediction === side ? 'Bad' : 'Good'}</span></div>)}
      <div className="track-axis">▲ TRACK AXIS ▲<span>CENTERLINE</span></div>
    </div>
    <div className="track-stage"><RailArtwork prediction={prediction} /></div>
    {!prediction && <p className="waiting-message">{selected?.error ?? 'Your recording’s prediction will appear here.'}</p>}
    <div className="side-cards">{(['Side I', 'Side II'] as const).map(side => <article key={side} className={`side-card ${!prediction ? 'waiting' : prediction === side ? 'bad' : 'good'}`}><div><strong>{side.toUpperCase()}</strong><span>{!prediction ? 'WAITING' : prediction === side ? 'BAD' : 'GOOD'}</span></div><h2>{!prediction ? 'Awaiting analysis' : prediction === side ? 'Corrugation Alert' : 'Smooth / Nominal'}</h2><p>{!prediction ? 'Upload a CSV recording' : prediction === side ? 'Model flags this side' : 'Not flagged by this prediction'}</p></article>)}</div>
    <p className="diagram-note">Whole-recording model classification. The diagram does not locate individual defects.</p>
    </div>
  </section>;
}

function ResultsPanel({ batch }: { batch: Batch }) {
  const successful = batch.records.flatMap(row => row.result ? [row.result] : []);
  const incomplete = batch.records.length - successful.length;
  const retryable = batch.records.some(row => ['failed', 'stopped'].includes(row.status));
  const selectedResult = batch.records.find(row => row.id === batch.selected)?.result;
  const downloadDoorSubmission = () => {
    if (selectedResult?.subsystem !== 'door') return;
    const url = URL.createObjectURL(new Blob([exportDoorSubmission(selectedResult)], { type: 'text/csv;charset=utf-8' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'door_predictions.csv'; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([exportResults(successful, batch.subsystem)], { type: 'text/csv;charset=utf-8' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${batch.subsystem}_predictions.csv`; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <section className="panel results-panel" aria-labelledby="results-title">
    <div className="results-heading"><h2 id="results-title">Recordings <span>{batch.records.length}</span></h2><button className="text-button" disabled={batch.busy || !batch.records.length} onClick={batch.clear}>Clear batch</button></div>
    {!batch.records.length ? <div className="empty-results"><Icon name="file" /><p>No recordings yet</p><span>Your batch results will appear here.</span></div> : <>
      {batch.subsystem === 'rail' && <div className="result-counts">{(['Normal', 'Side I', 'Side II'] as RailLabel[]).map(label => <span key={label}>{label}<b>{successful.filter(row => row.subsystem === 'rail' && row.prediction === label).length}</b></span>)}</div>}
      <ul className="recording-list" aria-label="Recording results">{batch.records.map(row => <li key={row.id}><button className={`recording-row ${batch.selected === row.id ? 'selected' : ''}`} aria-pressed={batch.selected === row.id} onClick={() => batch.setSelected(row.id)}><Icon name="file" /><span className="recording-name"><strong title={row.name}>{row.name}</strong><span className={row.error ? 'error-text' : ''}>{row.error ?? (row.status === 'completed' ? 'Analysis complete' : row.status === 'uploading' ? `Uploading ${Math.round(row.progress * 100)}%` : row.status === 'analysing' ? 'Analysing…' : 'Queued')}</span></span><span className={`row-status ${row.result ? resultTone(row.result) : 'waiting'}`}>{(row.result ? resultLabel(row.result) : undefined) ?? (row.status === 'failed' ? 'Error' : row.status === 'stopped' ? 'Stopped' : '···')}</span></button></li>)}</ul>
      <div className="result-actions">{retryable && <button className="secondary-button" disabled={batch.busy} onClick={batch.retry}>Retry failed / stopped files</button>}<button className="primary-button download-button" disabled={!successful.length} onClick={download}><Icon name="download" />{incomplete ? 'Download partial CSV' : 'Download predictions CSV'}</button></div>
      {batch.subsystem === 'door' && <><button className="secondary-button" disabled={selectedResult?.subsystem !== 'door'} onClick={downloadDoorSubmission}>Download selected stream submission CSV</button><p className="export-note">Submission format: start_time, end_time, prediction. Includes only the selected completed stream.</p></>}
      <p className="export-note" role="status">{successful.length} of {batch.records.length} predictions available.{incomplete ? ` ${incomplete} unfinished or failed recording(s) are omitted from the download. Select a row for details.` : (batch.subsystem === 'door' ? ' Export contains one row per operation, including its source filename.' : batch.subsystem === 'acv' ? ' Export format: file_id, ranked_cars.' : ' Export format: file_id, prediction.')}</p>
    </>}
  </section>;
}

export default function App() {
  const rail = useBatch('rail');
  const door = useBatch('door');
  const acv = useBatch('acv');
  const shm = useBatch('shm');
  const batches = { rail, door, acv, shm };
  const [health, setHealth] = useState<Health | null>(null);
  const refresh = useCallback(async () => {
    try {
      const response = await fetch('/api/health', { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error();
      setHealth(await response.json());
    } catch { setHealth({ status: 'unavailable', maximum_file_bytes: DEFAULT_LIMIT }); }
  }, []);
  useEffect(() => {
    void refresh(); const timer = setInterval(() => void refresh(), 15000);
    const prevent = (event: DragEvent) => event.preventDefault();
    window.addEventListener('dragover', prevent); window.addEventListener('drop', prevent);
    return () => { clearInterval(timer); window.removeEventListener('dragover', prevent); window.removeEventListener('drop', prevent); };
  }, [refresh]);
  return <><a className="skip-link" href="#content">Skip to content</a><div id="content"><Routes>
    <Route path="/" element={<Home ready={health ? health.status === 'ready' : null} />} />
    {(['rail', 'door', 'acv', 'shm'] as const).map(subsystem => {
      const batch = batches[subsystem];
      return <Route key={subsystem} path={`/${subsystem}`} element={<div className={`rail-shell page-shell module-${subsystem}`}><ExitSign subsystem={subsystem} /><main className="rail-workspace"><UploadPanel key={subsystem} batch={batch} health={health} refresh={() => void refresh()} />{subsystem === 'rail' ? <RailView batch={batch} /> : subsystem === 'door' ? <DoorView batch={batch} health={health} /> : subsystem === 'acv' ? <ACVView batch={batch} /> : <SHMView batch={batch} />}<ResultsPanel batch={batch} /></main><footer className="page-footer">Results stay in this tab until you clear them or refresh.</footer></div>} />;
    })}
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes></div></>;
}
