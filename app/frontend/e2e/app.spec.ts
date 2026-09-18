import { test, expect } from '@playwright/test';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '../../..');
const input = path.join(root, 'PS3/02_Datasets/Rail_Corrugation/Test');
const original = path.join(root, 'Optional_Items/Rail Corrugation/code/outputs/rail_predictions.csv');
const small = { name: 'demo.csv', mimeType: 'text/csv', buffer: Buffer.from('mocked by test') };
const result = (file_id: string, prediction = 'Side I') => ({ subsystem: 'rail', model_version: 'rail-pipeline-v2', file_id, prediction, scores: { Normal: .1, 'Side I': .8, 'Side II': .1 }, side_energy: {}, dominant_frequency: {} });

async function noOverflow(page: import('@playwright/test').Page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
}

test('home artwork, navigation, keyboard and responsive layouts', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  await expect(page.getByText('MODEL READY')).toBeVisible();
  await expect(page.getByRole('button', { name: /CC SHM/ })).toBeEnabled();
  await page.screenshot({ path: info.outputPath('home.png'), fullPage: true });
  await noOverflow(page);
  await page.getByRole('button', { name: /EW RAIL/ }).click();
  await expect(page).toHaveURL(/\/rail$/);
  await expect(page.getByRole('img', { name: 'Rail diagram awaiting a prediction' })).toBeVisible();
  await page.screenshot({ path: info.outputPath('rail-empty.png'), fullPage: true });
  await noOverflow(page);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Analyse your recordings' })).toBeVisible();
  await page.getByRole('link', { name: /Exit to Main Page/ }).click();
  await page.getByRole('link', { name: 'Open Rail diagnostics' }).focus();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/rail$/);
  expect(errors).toEqual([]);
});

test('real inference, batch selection, persistence and CSV download', async ({ page }, info) => {
  const source = await readFile(original, 'utf8');
  const rows = source.trim().split(/\r?\n/).slice(1).map(line => line.split(','));
  const choices = ['Normal', 'Side I', 'Side II'].map(label => rows.find(row => row[1] === label)!);
  await page.goto('/rail'); await expect(page.getByText('MODEL READY')).toBeVisible();
  await page.getByLabel('Choose CSV files', { exact: true }).setInputFiles(choices.map(row => path.join(input, row[0])));
  await expect(page.getByRole('button', { name: 'Download predictions CSV', exact: true })).toBeEnabled({ timeout: 90_000 });
  for (const [name, label] of choices) {
    await page.getByRole('button').filter({ hasText: name }).click();
    await expect(page.locator('.prediction-badge')).toHaveText(label === 'Normal' ? 'Nominal Profile' : `${label} Corrugation`);
    await expect(page.locator(`[data-side="Side I"]`)).toHaveAttribute('data-state', label === 'Side I' ? 'bad' : 'good');
    await expect(page.locator(`[data-side="Side II"]`)).toHaveAttribute('data-state', label === 'Side II' ? 'bad' : 'good');
  }
  await page.screenshot({ path: info.outputPath('rail-result.png'), fullPage: true });
  await noOverflow(page);
  await page.getByRole('link', { name: /Exit to Main Page/ }).click();
  await page.getByRole('button', { name: /EW RAIL/ }).click();
  await expect(page.locator('.recording-row')).toHaveCount(3);
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download predictions CSV', exact: true }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('rail_predictions.csv');
  const csv = await readFile((await download.path())!, 'utf8');
  expect(csv.split('\n')[0]).toBe('file_id,prediction');
  for (const [name, label] of choices) expect(csv).toContain(`${name},${label}\n`);
});

test('failed file, partial download, retry, duplicate and offline states', async ({ page }) => {
  let attempts = 0;
  await page.route('**/api/predict/rail', route => {
    attempts++;
    return attempts === 1 ? route.fulfill({ status: 422, json: { error: { message: 'Invalid Rail schema.' } } }) : route.fulfill({ json: result('demo.csv') });
  });
  await page.goto('/rail'); await expect(page.getByText('MODEL READY')).toBeVisible();
  await page.getByLabel('Choose CSV files', { exact: true }).setInputFiles(small);
  await expect(page.getByRole('button', { name: 'Retry failed / stopped files' })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Download partial CSV' })).toBeDisabled();
  await page.getByRole('button', { name: 'Retry failed / stopped files' }).click();
  await expect(page.getByRole('button', { name: 'Download predictions CSV', exact: true })).toBeEnabled();
  await page.getByLabel('Choose CSV files', { exact: true }).setInputFiles(small);
  await expect(page.getByRole('alert')).toContainText('Duplicate filename');
  expect(attempts).toBe(2);
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'unavailable', maximum_file_bytes: 67108864 } }));
  await page.reload();
  await expect(page.getByText('OFFLINE', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Choose files', exact: true })).toBeDisabled();
  await expect(page.locator('.prediction-badge')).toHaveText('Awaiting result');
});

test('stop remaining queue without submitting its files', async ({ page }) => {
  let attempts = 0;
  await page.route('**/api/predict/rail', async route => {
    attempts++;
    await new Promise(resolve => setTimeout(resolve, 2000));
    try { await route.fulfill({ json: result('demo.csv') }); } catch { /* request was cancelled */ }
  });
  await page.goto('/rail'); await expect(page.getByText('MODEL READY')).toBeVisible();
  await page.getByLabel('Choose CSV files', { exact: true }).setInputFiles([small, { ...small, name: 'second.csv' }]);
  await expect.poll(() => attempts).toBe(1);
  await page.getByRole('button', { name: 'Stop batch' }).click();
  await expect(page.getByRole('button', { name: 'Retry failed / stopped files' })).toBeEnabled();
  await expect(page.locator('.row-status').filter({ hasText: 'Stopped' })).toHaveCount(2);
  expect(attempts).toBe(1);
});

test('all 68 real files through folder selection match the existing Rail output', async ({ page }, info) => {
  test.skip(info.project.name !== 'desktop-chromium'); test.setTimeout(300_000);
  await page.goto('/rail'); await expect(page.getByText('MODEL READY')).toBeVisible();
  await page.getByLabel('Choose CSV folder', { exact: true }).setInputFiles(input);
  await expect(page.locator('.recording-row')).toHaveCount(68);
  await expect(page.getByRole('button', { name: 'Stop batch', exact: true })).toHaveCount(0, { timeout: 240_000 });
  await expect(page.locator('.row-status').filter({ hasText: 'Error' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Download predictions CSV', exact: true })).toBeEnabled({ timeout: 240_000 });
  const promise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download predictions CSV', exact: true }).click();
  const download = await promise;
  const csv = await readFile((await download.path())!, 'utf8');
  expect(csv).toBe((await readFile(original, 'utf8')).replaceAll('\r\n', '\n'));
  const exports = path.join(root, 'app/exports'); await mkdir(exports, { recursive: true });
  await writeFile(path.join(exports, 'rail_predictions.csv'), csv);
  await writeFile(path.join(exports, 'README.md'), '# App-generated predictions\n\nrail_predictions.csv was downloaded from the React Rail page after all 68 supplied test files were processed through the local API. It matches the validated model output.\n\nDoor, ACV and SHM are also integrated; their browser tests cover real model uploads and downloads.\n');
});

test('drag and drop, mixed batch partial export and narrow viewport', async ({ page }) => {
  await page.route('**/api/predict/rail', route => {
    const body = route.request().postData() ?? '';
    return body.includes('broken.csv')
      ? route.fulfill({ status: 422, json: { error: { message: 'Invalid Rail schema.' } } })
      : route.fulfill({ json: result('drop.csv', 'Normal') });
  });
  await page.goto('/rail'); await expect(page.getByText('MODEL READY')).toBeVisible();
  await page.locator('.drop-zone').evaluate(element => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(['test'], 'drop.csv', { type: 'text/csv' }));
    transfer.items.add(new File(['test'], 'broken.csv', { type: 'text/csv' }));
    element.dispatchEvent(new DragEvent('drop', { bubbles: true, dataTransfer: transfer }));
  });
  await expect(page.getByRole('button', { name: 'Retry failed / stopped files' })).toBeEnabled();
  await expect(page.getByRole('status').filter({ hasText: '1 of 2 predictions available' })).toBeVisible();
  const promise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download partial CSV' }).click();
  const download = await promise;
  expect(await readFile((await download.path())!, 'utf8')).toBe('file_id,prediction\ndrop.csv,Normal\n');
  await page.setViewportSize({ width: 320, height: 640 });
  await noOverflow(page);
  await page.getByRole('link', { name: /Exit to Main Page/ }).click();
  await noOverflow(page);
});
