import { Fragment } from 'react';
import type { RailLabel } from './data';

export function RailArtwork({ prediction }: { prediction?: RailLabel }) {
  const colors = (side: RailLabel) => !prediction ? ['#94A3B8', '#64748B', '#475569'] : prediction === side ? ['#EF4444', '#B91C1C', '#7F1D1D'] : ['#00A651', '#007A3D', '#02522B'];
  return <svg className="rail-art" viewBox="0 0 260 360" fill="none" role="img" aria-label={prediction ? `Rail diagram: ${prediction === 'Normal' ? 'both sides nominal' : `${prediction} corrugation predicted`}` : 'Rail diagram awaiting a prediction'}>
    <defs><pattern id="ballastPattern" height="16" width="16" patternUnits="userSpaceOnUse"><rect fill="#F8FAFC" height="16" width="16" /><circle cx="3" cy="3" fill="#CBD5E1" r="1" /><circle cx="11" cy="6" fill="#E2E8F0" r="1.2" /><circle cx="7" cy="12" fill="#CBD5E1" r=".9" /><circle cx="14" cy="14" fill="#94A3B8" opacity=".4" r="1.1" /></pattern></defs>
    <rect fill="url(#ballastPattern)" height="360" rx="10" width="236" x="12" />
    <line stroke="#94A3B8" strokeDasharray="6 4" strokeWidth="1.8" x1="130" x2="130" y2="360" />
    {Array.from({ length: 13 }, (_, i) => <g key={i}>{(['Side I', 'Side II'] as const).map((side, index) => {
      const [color, border] = colors(side); const x = index ? 130 : 24; const y = 14 + i * 26;
      return <g key={side}><rect x={x} y={y} width="106" height="15" rx="3" fill={color} stroke={border} strokeWidth="1.5" /><rect x={x + 2} y={y + 2} width="102" height="3" fill="white" opacity=".25" rx="1" /><rect x={x + 2} y={y + 10} width="102" height="3" fill="black" opacity=".2" rx="1" /></g>;
    })}<line x1="130" x2="130" y1={14 + i * 26} y2={29 + i * 26} stroke="white" opacity=".45" /></g>)}
    {(['Side I', 'Side II'] as const).map((side, index) => {
      const x = index * 150; const [color, border, base] = colors(side);
      return <g key={side} data-side={side} data-state={!prediction ? 'waiting' : prediction === side ? 'bad' : 'good'}>
        <rect x={44 + x} width="22" height="360" fill="#0F172A" opacity=".3" /><rect x={46 + x} width="18" height="360" fill={base} stroke={border} strokeWidth="1.5" /><rect x={49 + x} width="12" height="360" fill={color} stroke={border} strokeWidth="1.5" /><line x1={52 + x} x2={52 + x} y2="360" stroke="white" strokeWidth="2.5" opacity=".6" />
        {Array.from({ length: 13 }, (_, i) => <Fragment key={i}><rect x={42 + x} y={13 + i * 26} width="6" height="17" rx="1.5" fill="#1E293B" stroke="#475569" /><rect x={62 + x} y={13 + i * 26} width="6" height="17" rx="1.5" fill="#1E293B" stroke="#475569" /><circle cx={45 + x} cy={21.5 + i * 26} r="1.5" fill="#94A3B8" /><circle cx={65 + x} cy={21.5 + i * 26} r="1.5" fill="#94A3B8" />{prediction === side && <><rect x={49 + x} y={2 + i * 26} width="12" height="8" fill="#7F1D1D" opacity=".85" /><path d={`M${49 + x} ${4 + i * 26}h12 M${49 + x} ${8 + i * 26}h12`} stroke="#FEF08A" strokeWidth="1.5" opacity=".8" /></>}</Fragment>)}
      </g>;
    })}
  </svg>;
}
