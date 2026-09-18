import { type ACVResult, type SHMResult } from './data';
import type { Batch } from './useBatch';
import acvCar from './assets/acv-car.svg';
import shmCar from './assets/shm-car.svg';

const number = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? 'Not available' : value.toLocaleString('en', { maximumFractionDigits: digits });
function FileHeading({ name, caption }: { name?: string; caption: string }) {
  return <div className="telemetry-panel"><span className="file-icon" aria-hidden="true">▤</span><div className="file-label"><span className="eyebrow">{caption}</span><strong title={name}>{name ?? 'Awaiting file input…'}</strong></div></div>;
}
function Empty({ error }: { error?: string }) {
  return <p className="module-empty" role="status">{error ?? 'Upload a recording to see its diagnosis. No prediction yet.'}</p>;
}

export { DoorView } from './DoorView';

function CarCard({ result, car, index }: { result: ACVResult; car: string; index: number }) {
  const d = result.diagnostics[car] ?? {};
  return <article className={`acv-car-card rank-${index}`} aria-label={`Rank ${index + 1}: ${car}`}>
    <div className="car-card-heading"><span className="rank-number">#{index + 1}</span><div><h2>Car {car}</h2><span className="car-score">Ranking score: {number(result.car_scores[car], 4)}</span></div><span className="rank-status">{index === 0 ? 'HIGHEST PRIORITY' : `RANK ${index + 1}`}</span></div>
    <div className="acv-car-stage"><span>{index === 0 ? 'PRIORITISE HVAC INSPECTION' : 'ACV / CAR TELEMETRY'}</span><img src={acvCar} alt={`Car ${car} ventilation illustration`} /></div>
    <dl className="car-metrics"><div><dt>Cabin temperature</dt><dd>{number(d.cabin_temperature_median, 1)}{d.cabin_temperature_median == null ? '' : ' °C'}</dd></div><div><dt>Temperature coverage</dt><dd>{d.temperature_coverage == null ? 'Not available' : `${number(d.temperature_coverage * 100, 1)}%`}</dd></div></dl>
  </article>;
}
export function ACVView({ batch }: { batch: Batch }) {
  const selected = batch.records.find(r => r.id === batch.selected);
  const result = selected?.result?.subsystem === 'acv' ? selected.result : undefined;
  return <section className="visual-column acv-view" aria-label="ACV car rankings"><FileHeading name={selected?.name} caption="ACV / RANKED CAR INSPECTION" /><div className="module-note">Cars are ordered by model score. Rankings indicate inspection priority, not a confirmed leak or a leak probability.</div>{result ? <><div className="ranking-summary"><strong>{result.ranked_cars.length} cars ranked</strong><span>Highest priority first</span></div><div className="car-rankings">{result.ranked_cars.map((car, index) => <CarCard key={car} result={result} car={car} index={index} />)}</div></> : <><div className="acv-car-stage empty-car"><img src={acvCar} alt="Ventilation carriage awaiting analysis" /></div><Empty error={selected?.error} /></>}</section>;
}
function DamageGauge({ result }: { result?: SHMResult }) {
  const d = result?.prediction;
  const percent = d == null ? 0 : Math.max(0, Math.min(100, d * 100));
  const label = d == null ? '—' : d > 0 && d * 100 < 0.01 ? '<0.01%' : `${number(d * 100, 2)}%`;
  return <div className="damage-gauge"><div className="battery-heading eyebrow">PREDICTED FATIGUE DAMAGE</div><div className="shm-train"><img src={shmCar} alt="Structural fatigue carriage illustration" /><div className={`damage-battery ${d == null ? 'waiting' : ''}`} role="meter" aria-label="Predicted fatigue damage" aria-valuemin={0} aria-valuemax={Math.max(100, (d ?? 1) * 100)} aria-valuenow={d == null ? undefined : d * 100} aria-valuetext={d == null ? 'Awaiting prediction' : label}><div className="battery-fill" style={{ width: `${percent}%` }} /><strong data-testid="damage-value">{label}</strong></div></div>{d == null && <p className="battery-awaiting">Awaiting prediction</p>}<div className="scale-labels"><span>0%</span><span>100% reference{d != null && d > 1 ? ' · battery full' : ''}</span></div><p className="module-note">Percentage of the damage reference (D = 1). Fill increases with damage.</p></div>;
}

export function SHMView({ batch }: { batch: Batch }) {
  const selected = batch.records.find(r => r.id === batch.selected);
  const result = selected?.result?.subsystem === 'shm' ? selected.result : undefined;
  const diagnostics = result ? [
    ['RAINFLOW', 'Counted stress cycles', number(result.cycle_count, 1), 'Cycle count from this recording'],
    ['STRESS', 'Equivalent stress amplitude', number(result.equivalent_stress_amplitude, 3), 'In the input stress units'],
    ['RANGE', 'Maximum cycle range', number(result.maximum_cycle_range, 3), 'In the input stress units'],
  ] : [];
  return <section className="visual-column shm-view" aria-label="Structural fatigue diagnosis"><FileHeading name={selected?.name} caption="SHM / STRUCTURAL CONDITION" /><div className="shm-panel"><div className="shm-title"><h2>Structural Car<br />Fatigue Gauge</h2><span>{result ? `D = ${number(result.prediction, 6)}` : 'AWAITING DATA'}</span></div><DamageGauge result={result} />{!result && <Empty error={selected?.error} />}<div className="influencing-panel"><h2>Recording diagnostics</h2><p>Measured summaries of the stress history.</p>{diagnostics.map(([tag, title, value, note]) => <div className="diagnostic-feature" key={tag}><span className="feature-tag">{tag}</span><div><h3>{title}</h3><p>{note}</p></div><strong>{value}</strong></div>)}</div></div></section>;
}
