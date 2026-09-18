import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
  testDir: './e2e', timeout: 45_000, expect: { timeout: 10_000 },
  fullyParallel: false, workers: 1,
  use: { baseURL: process.env.APP_URL ?? 'http://127.0.0.1:8000', trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop-chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 1000 } } },
    { name: 'phone-chromium', use: { ...devices['Pixel 7'] } },
    { name: 'phone-webkit', use: { ...devices['iPhone 13'] } },
  ],
});
