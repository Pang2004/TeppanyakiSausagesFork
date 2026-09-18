import { test, expect } from '@playwright/test';
const file = { name: 'state.csv', mimeType: 'text/csv', buffer: Buffer.from('test fixture') };

test('workspace sizing and complete Door standby at desktop and mobile widths', async ({ page }, info) => {
  test.setTimeout(90_000);
  for (const width of [1024, 1440, 1920, 320, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto('/');
    await expect(page.locator('.home-import')).toHaveCount(0);
    await expect(page.getByRole('button', { name: /NS DOOR/ })).toBeVisible();
    await page.screenshot({ path: info.outputPath(`home-${width}.png`), fullPage: true });
    for (const module of ['door', 'rail', 'acv', 'shm']) {
      await page.goto(`/${module}`);
      const exit = await page.locator('.exit-sign').boundingBox();
      const workspace = await page.locator('.rail-workspace').boundingBox();
      expect(Math.abs(exit!.width - workspace!.width)).toBeLessThan(1);
      await expect(page.locator('.exit-sign > .icon')).toHaveCount(0);
      if (width > 900) {
        const sidebar = await page.locator('.upload-panel').boundingBox();
        const visual = await page.locator('.visual-column').boundingBox();
        expect(sidebar!.width).toBeGreaterThanOrEqual(300);
        expect(sidebar!.width).toBeLessThanOrEqual(340);
        expect(visual!.width).toBeGreaterThan(sidebar!.width);
        expect(workspace!.width).toBe(width - 56);
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
      if (module === 'door') {
        await expect(page.locator('.door-carriage.standby')).toBeVisible();
        await expect(page.locator('.door-window')).toHaveCount(2);
        await expect(page.getByText('No prediction yet', { exact: true })).toBeVisible();
        await page.screenshot({ path: info.outputPath(`door-empty-${width}.png`), fullPage: true });
      }
    }
  }
});

test('Door upload action, processing, empty result and error retain the carriage', async ({ page }) => {
  let finish: (() => void) | undefined;
  await page.route('**/api/predict/door', async route => {
    await new Promise<void>(resolve => { finish = resolve; });
    await route.fulfill({ json: { subsystem: 'door', model_version: 'test', file_id: file.name, cycles: [] } });
  });
  await page.goto('/door');
  await expect(page.getByRole('button', { name: 'Choose CSV files', exact: true })).toBeEnabled();
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Choose CSV files', exact: true }).click();
  await (await chooser).setFiles(file);
  await expect(page.locator('.door-verdict h2')).toHaveText('ANALYSING');
  await expect(page.locator('.door-window')).toHaveCount(2);
  await expect.poll(() => !!finish).toBe(true);
  finish!();
  await expect(page.locator('.door-empty h3')).toHaveText('No operations detected');
  await page.route('**/api/predict/door', route => route.fulfill({ status: 422, json: { error: { message: 'Invalid door telemetry.' } } }));
  await page.getByLabel('Choose Door telemetry files').setInputFiles({ ...file, name: 'invalid.csv' });
  await expect(page.locator('.door-empty')).toContainText('Invalid door telemetry.');
  await expect(page.locator('.door-window')).toHaveCount(2);
});

test('SHM battery displays actual percentages while saturating only the fill', async ({ page }, info) => {
  await page.goto('/shm');
  await expect(page.getByRole('meter')).toHaveAttribute('aria-valuetext', 'Awaiting prediction');
  const values: [number, string, number][] = [[0, '0%', 0], [0.000001, '<0.01%', .0001], [.25, '25%', 25], [1, '100%', 100], [1.25, '125%', 100]];
  for (const [index, [prediction, label, fill]] of values.entries()) {
    await page.route('**/api/predict/shm', route => route.fulfill({ json: { subsystem: 'shm', model_version: 'test', file_id: `shm-${index}.csv`, prediction, cycle_count: 1, equivalent_stress_amplitude: 1, maximum_cycle_range: 1, estimated_percentage_error: 0 } }));
    await page.getByLabel('Choose CSV files', { exact: true }).setInputFiles({ ...file, name: `shm-${index}.csv` });
    await expect(page.getByTestId('damage-value')).toHaveText(label);
    await expect(page.getByRole('meter')).toHaveAttribute('aria-valuenow', String(prediction * 100));
    const fillWidth = await page.locator('.battery-fill').evaluate(el => parseFloat((el as HTMLElement).style.width));
    expect(fillWidth).toBeCloseTo(fill, 5);
    const battery = await page.getByRole('meter').boundingBox();
    const train = await page.locator('.shm-train').boundingBox();
    expect(battery!.x).toBeGreaterThan(train!.x);
    expect(battery!.y + battery!.height).toBeLessThan(train!.y + train!.height);
  }
  await page.screenshot({ path: info.outputPath('shm-battery.png'), fullPage: true });
});
