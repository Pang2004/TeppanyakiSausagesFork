import { describe, expect, it } from 'vitest';
import { DEFAULT_LIMIT, type RailResult, droppedFiles, readDirectory, resultsCsv, validateFiles } from './data';
const file = (name: string, content = 'recording') => new File([content], name);

describe('upload selection', () => {
  it('filters non CSV files and sorts filenames naturally', () => {
    const result = validateFiles([file('Test10.csv'), file('readme.txt'), file('Test2.csv')], [], DEFAULT_LIMIT);
    expect(result.files.map(f => f.name)).toEqual(['Test2.csv', 'Test10.csv']);
    expect(result.messages[0]).toContain('1 non-CSV');
  });
  it('rejects duplicate basenames instead of overwriting predictions', () => {
    expect(validateFiles([file('same.csv'), file('same.csv')], [], DEFAULT_LIMIT).files).toEqual([]);
    expect(validateFiles([file('Test1.csv')], ['Test1.csv'], DEFAULT_LIMIT).files).toEqual([]);
  });
  it('rejects empty and oversized files while retaining valid recordings', () => {
    const result = validateFiles([file('empty.csv', ''), file('large.csv', '123456789'), file('ok.csv', '123')], [], 8);
    expect(result.files.map(f => f.name)).toEqual(['ok.csv']);
    expect(result.messages).toHaveLength(2);
  });
  it('reads every directory chunk and nested directory', async () => {
    const one = { isFile: true, file: (done: (file: File) => void) => done(file('one.csv')) } as unknown as FileSystemEntry;
    const batches = [[one], [one], []];
    const dir = { isFile: false, createReader: () => ({ readEntries: (done: (entries: FileSystemEntry[]) => void) => done(batches.shift()!) }) } as unknown as FileSystemEntry;
    expect(await readDirectory(dir)).toHaveLength(2);
    expect(batches).toHaveLength(0);
  });
});
it('exports only the official columns with natural ordering and CSV escaping', () => {
  const result = (name: string) => ({ file_id: name, prediction: 'Normal' }) as RailResult;
  expect(resultsCsv([result('Test10.csv'), result('Test2.csv')])).toBe('file_id,prediction\nTest2.csv,Normal\nTest10.csv,Normal\n');
  expect(resultsCsv([result('a,"b.csv')])).toContain('"a,""b.csv",Normal');
});

it('uses the actual dropped File when the browser entry method is unavailable', async () => {
  const expected = file('one.csv');
  const items = [{ kind: 'file', webkitGetAsEntry: () => { throw new Error('unsupported entry'); }, getAsFile: () => expected }] as unknown as DataTransferItemList;
  expect(await droppedFiles(items, [] as unknown as FileList)).toEqual([expected]);
});

it('accepts workbooks for ACV without treating CSVs as workbooks', async () => {
  expect(validateFiles([file('car.xlsx'), file('wrong.csv')], [], DEFAULT_LIMIT, '.xlsx').files.map(f => f.name)).toEqual(['car.xlsx']);
});
it('exports Door operations, ACV order and unclipped SHM damage', async () => {
  const { exportResults, validResult } = await import('./data');
  const door = { subsystem: 'door', file_id: 'door.csv', model_version: 'door-pipeline-v1', cycles: [{ start_time: '00:00:01', end_time: '00:00:03', prediction: 'Normal', operation: 'Open', confidence: .8, quality_flags: [], boundary_reason: 'test' }] } as const;
  expect(exportResults([JSON.parse(JSON.stringify(door))], 'door')).toContain('"door.csv","00:00:01","00:00:03","Normal"');
  const acv = { subsystem: 'acv' as const, file_id: 'cars.xlsx', model_version: 'v1', ranked_cars: ['08', '01'], car_scores: { '08': .8, '01': .1 }, diagnostics: { '08': {}, '01': {} }, schema: 'test' };
  expect(exportResults([acv], 'acv')).toContain('08|01');
  expect(validResult(acv, 'door')).toBe(false);
  expect(validResult({ ...acv, ranked_cars: ['08', '08'] }, 'acv')).toBe(false);
  const shm = { subsystem: 'shm' as const, file_id: 'stress.csv', model_version: 'v1', prediction: 1.258, cycle_count: 10, equivalent_stress_amplitude: 2, maximum_cycle_range: 3, estimated_percentage_error: .2 };
  expect(exportResults([shm], 'shm')).toContain('1.258');
  expect(validResult({ ...shm, prediction: NaN }, 'shm')).toBe(false);
});

it('formats official Door timestamps and measures durations across midnight', async () => {
  const { doorClock, doorDuration } = await import('./data');
  expect(doorClock('2023-7-5-0-0-3-760')).toBe('00:00:03.760');
  expect(doorDuration({ start_time: '2023-7-5-0-0-0-0', end_time: '2023-7-5-0-0-3-760' })).toBe('3.760 s');
  expect(doorDuration({ start_time: '2023-7-5-23-59-59-900', end_time: '2023-7-6-0-0-0-100' })).toBe('0.200 s');
});

it('exports only the selected Door stream using the official schema', async () => {
  const { exportDoorSubmission } = await import('./data');
  expect(exportDoorSubmission({ subsystem: 'door', model_version: 'v1', file_id: 'Test.csv', cycles: [{ start_time: '2023-7-5-0-0-0-0', end_time: '2023-7-5-0-0-3-760', prediction: 'Normal', operation: 'Close', confidence: .9, quality_flags: [], boundary_reason: 'test' }] })).toBe('start_time,end_time,prediction\n"2023-7-5-0-0-0-0","2023-7-5-0-0-3-760","Normal"\n');
});
