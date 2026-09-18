import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve(import.meta.dirname, '../../..');
const cases = [
  { key: 'door', button: /NS DOOR/, file: 'Door/Test.csv', format: 'CSV', label: 'Detected door operations' },
  { key: 'acv', button: /DT ACV/, file: 'ACV/Test/acv_test_case.xlsx', format: 'XLSX', label: 'ACV car rankings' },
  { key: 'shm', button: /CC SHM/, file: 'SHM/Test/test01.csv', format: 'CSV', label: 'Structural fatigue diagnosis' },
];
for (const item of cases) {
  test(`${item.key}: real output, navigation, export and responsive reference layout`, async ({ page }, info) => {
    test.setTimeout(120_000);
    const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto('/');
    await page.getByRole('button', { name: item.button }).click();
    await expect(page).toHaveURL(new RegExp(`/${item.key}$`));
    await expect(page.getByRole('link', { name: /Exit to Main Page/ })).toBeVisible();
    await expect(page.getByText('MODEL READY', { exact: true })).toBeVisible();
    await page.reload();
    await expect(page.getByText('MODEL READY', { exact: true })).toBeVisible();
    const responsePromise = page.waitForResponse(r => r.url().endsWith(`/api/predict/${item.key}`));
    await page.getByLabel(`Choose ${item.format} files`, { exact: true }).setInputFiles(path.join(root, 'PS3/02_Datasets', item.file));
    const response = await responsePromise;
    expect(response.status()).toBe(200); const result = await response.json();
    await expect(page.getByRole('button', { name: 'Download predictions CSV', exact: true })).toBeEnabled();
    if (item.key === 'door') {
      await expect(page.locator('.door-cycle')).toHaveCount(result.cycles.length);
      const index = result.cycles.findIndex((c: { prediction: string }) => c.prediction === 'Abnormal resistance');
      expect(index).toBeGreaterThanOrEqual(0);
      await page.locator('.door-cycle').nth(index).click();
      await expect(page.locator('.door-verdict h2')).toHaveText('ABNORMAL');
      await page.locator('.door-cycle').first().click();
      await expect(page.locator('.door-window h2').first()).toHaveText('OPERATION #1');
      await expect(page.locator('.door-window dd').first()).toHaveText('00:00:00.000');
      await expect(page.locator('.door-window-footer').first()).toContainText('3.760 s');
      const columns = await page.locator('.door-window').all();
      const cells = await page.locator('.door-cycle').first().locator(':scope > span').all();
      for (let i = 0; i < 2; i++) {
        const window = await columns[i].boundingBox();
        const cell = await cells[i].boundingBox();
        expect(Math.abs(window!.x + window!.width / 2 - cell!.x - cell!.width / 2)).toBeLessThan(6);
        await expect(columns[i]).toHaveCSS('text-align', 'center');
        await expect(cells[i]).toHaveCSS('text-align', 'center');
      }
    } else if (item.key === 'acv') {
      await expect(page.locator('.acv-car-card')).toHaveCount(result.ranked_cars.length);
      await expect(page.locator('.acv-car-card h2').first()).toHaveText(`Car ${result.ranked_cars[0]}`);
      await expect(page.getByText('98.4% CONFIDENCE')).toHaveCount(0);
      await expect(page.getByText('Pressure mismatch', { exact: true })).toHaveCount(0);
      await expect(page.locator('.car-metrics').first().locator('dt')).toHaveCount(2);
    } else {
      await expect(page.getByTestId('damage-value')).toHaveText(result.prediction > 0 && result.prediction * 100 < 0.01 ? '<0.01%' : `${(result.prediction * 100).toLocaleString('en', { maximumFractionDigits: 2 })}%`);
      await expect(page.locator('.diagnostic-feature')).toHaveCount(3);
      await expect(page.getByText('74.2%')).toHaveCount(0);
    }
    await page.locator('.visual-column').screenshot({ path: info.outputPath(`${item.key}-result.png`) });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    const pending = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download predictions CSV', exact: true }).click();
    const download = await pending; expect(download.suggestedFilename()).toBe(`${item.key}_predictions.csv`);
    const csv = await readFile((await download.path())!, 'utf8');
    expect(csv).toContain(path.basename(item.file));
    if (item.key === 'door') expect(csv).toContain('start_time,end_time,prediction');
    if (item.key === 'acv') expect(csv).toContain(result.ranked_cars.join('|'));
    if (item.key === 'shm') expect(csv).toContain(String(result.prediction));
    await page.getByRole('link', { name: /Exit to Main Page/ }).click();
    await page.getByRole('button', { name: /EW RAIL/ }).click();
    await expect(page.locator('.recording-row')).toHaveCount(0);
    await page.getByRole('link', { name: /Exit to Main Page/ }).click();
    await page.getByRole('button', { name: item.button }).click();
    await expect(page.locator('.recording-row')).toHaveCount(1);
    await page.setViewportSize({ width: 320, height: 700 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: info.outputPath(`${item.key}-narrow.png`), fullPage: true });
    expect(errors).toEqual([]);
  });
}

test('subsystem-specific availability and keyboard navigation', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ready', maximum_file_bytes: 67108864, subsystems: { rail: true, door: false, acv: true, shm: true } } }));
  await page.goto('/door');
  await expect(page.getByText('OFFLINE', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Choose files', exact: true })).toBeDisabled();
  await page.getByRole('link', { name: /Exit to Main Page/ }).click();
  await page.getByRole('link', { name: 'Open ACV diagnostics', exact: true }).focus();
  await page.keyboard.press('Enter'); await expect(page).toHaveURL(/\/acv$/);
  await expect(page.getByText('MODEL READY', { exact: true })).toBeVisible();
});
