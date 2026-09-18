import { test, expect } from '@playwright/test';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '../../..');
const dataset = process.env.DATASET_ROOT ?? path.join(root, 'PS3/02_Datasets');
const output = process.env.EXPORT_DIR ?? path.join(root, 'app/exports');
const inputs = [
  { key: 'rail', folder: 'Rail_Corrugation/Test', extension: 'csv', count: 68 },
  { key: 'door', file: 'Door/Test.csv', extension: 'csv', count: 1 },
  { key: 'acv', folder: 'ACV/Test', extension: 'xlsx', count: 1 },
  { key: 'shm', folder: 'SHM/Test', extension: 'csv', count: 16 },
];
for (const item of inputs) {
  test(`submission export: ${item.key}`, async ({ page }) => {
    test.setTimeout(360_000);
    const files = item.file ? [path.join(dataset, item.file)] :
      (await readdir(path.join(dataset, item.folder!))).filter(f => f.endsWith(`.${item.extension}`)).map(f => path.join(dataset, item.folder!, f));
    expect(files).toHaveLength(item.count);
    await page.goto(`/${item.key}`);
    await expect(page.getByText('MODEL READY', { exact: true })).toBeVisible();
    await page.getByLabel(`Choose ${item.extension.toUpperCase()} files`, { exact: true }).setInputFiles(files);
    await expect(page.locator('.recording-row')).toHaveCount(item.count);
    await expect(page.getByRole('button', { name: 'Stop batch', exact: true })).toHaveCount(0, { timeout: 300_000 });
    await expect(page.locator('.row-status').filter({ hasText: 'Error' })).toHaveCount(0);
    const button = page.getByRole('button', { name: item.key === 'door' ? 'Download selected stream submission CSV' : 'Download predictions CSV', exact: true });
    await expect(button).toBeEnabled();
    const pending = page.waitForEvent('download'); await button.click();
    const download = await pending;
    expect(download.suggestedFilename()).toBe(`${item.key}_predictions.csv`);
    const csv = await readFile((await download.path())!, 'utf8');
    expect(csv.split('\n')[0]).toBe(item.key === 'door' ? 'start_time,end_time,prediction' : item.key === 'acv' ? 'file_id,ranked_cars' : 'file_id,prediction');
    expect(csv.trim().split('\n').length).toBeGreaterThan(1);
    if (item.key !== 'door') expect(csv.trim().split('\n')).toHaveLength(item.count + 1);
    await mkdir(output, { recursive: true });
    await writeFile(path.join(output, download.suggestedFilename()), csv);
    await writeFile(path.join(output, `${item.key}_provenance.json`), JSON.stringify({ source: 'browser submission download', inputs: files.map(f => path.basename(f)), health: await (await page.request.get('/api/health')).json() }, null, 2));
  });
}
