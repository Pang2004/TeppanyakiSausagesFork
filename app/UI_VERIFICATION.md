# Subtab integration verification

The supplied Door, ACV and SHM design references were reviewed and adapted into React. The reference files have since been removed; runtime assets remain bundled in the frontend. All four tabs use the same charcoal/yellow exit component, with the appropriate line badge and a direct link to the main page. Door retains its carriage/window/operation-list visual; ACV retains ranked carriage cards; SHM retains the carriage, damage gauge and diagnostic cards.

Real-output corrections: ACV accepts XLSX, renders all model-returned cars (eight in the supplied test workbook), and shows scores rather than fabricated probabilities. SHM renders damage and cycle/stress diagnostics instead of invented remaining life, safety status or feature contributions. Door displays the official non-zero-padded timestamps as clocks and correctly computes millisecond durations. Unavailable diagnostics and empty/error/offline states are explicit.

Verification completed:

- Production TypeScript/Vite build and 9 frontend unit tests.
- 15 API tests, including real Door/ACV/SHM inference parity with the existing Python adapters, schema errors and per-model availability.
- 19 desktop/mobile Chromium browser tests passed (one duplicate 68-file mobile test intentionally skipped). The 68-file Rail upload/export matched the saved predictions before the Rail model update.
- After correcting Door timestamp formatting, all 8 new-tab desktop/mobile tests passed again.
- Real file uploads, car/operation counts, selected Door verdicts, exports, independent batches, exit navigation, keyboard links, direct-route reload, and 320px horizontal overflow checks.
- Screenshots were inspected; the supplied SHM overflow and the Door timestamp issue were corrected.

Final mobile screenshots: [Door](verification/door-mobile.png), [ACV](verification/acv-mobile.png), [SHM](verification/shm-mobile.png).

WebKit/Safari could not be verified: its Linux system dependencies are missing, and installing them requires an administrator password. Chromium coverage is not claimed as Safari coverage.

Door batch downloads include a `file_id` column to distinguish streams. The official single-stream Door submission format is still produced by the model CLI, without that extra column. No demo data or work-order dispatch is used in the app.

After Rail v3 installation, all **34 API/Rail/packaging tests** and all **10 desktop Chromium tests** passed. The complete 68-file Rail browser export now exactly matches the regenerated v3 CLI predictions. The production build and all 9 frontend unit tests pass. The previous Rail artifact and reports are retained for rollback.


## Desktop layout and reference refresh — 2026-09-19

The home upload shortcut is removed. Home and diagnostic workspaces now use the available desktop width with 28 px side margins. All diagnostic exit bars span their workspace and omit the grey right arrow. Above 900 px the upload/results sidebar is 300–340 px; diagnostics take the remaining width.

Door uses the new root reference artwork with locally bundled Chakra Petch, Space Mono and Silkscreen fonts, metallic exterior panels, synchronized lamps/windows and a continuous central seam. Window readouts and operation rows are centered on the same two column axes. The full carriage remains visible for standby, processing, errors and zero detected operations. Its empty-state upload button submits files through the real batch flow.

SHM displays fatigue damage as D × 100% inside a battery embedded in the train's central panel. The fill saturates at 100% while the displayed percentage can exceed 100%. Tiny positive values display <0.01%; raw D and CSV exports retain their original values. ACV cards now show cabin temperature and coverage only; pressure mismatch is removed from the interface.

Validation for this refresh:

- Production build and all 9 frontend unit tests passed.
- All 14 layout/subsystem checks passed in desktop and mobile Chromium, including real Door/ACV/SHM inference and exports, Door standby/processing/error/zero-operation states, and SHM percentage boundaries.
- The additional 10 existing home/Rail interaction checks passed in desktop and mobile Chromium. The unchanged 68-file model parity batch was not repeated for this presentation change.
- After the final centering change, both real-output Door browser tests passed again, including geometric column alignment checks.
- Layouts checked at 1024, 1440 and 1920 px desktop widths and 320/390 px mobile widths; no horizontal overflow.

Updated screenshots: [Home desktop](verification/home-desktop.png), [Door desktop](verification/door-desktop.png), [Door mobile](verification/door-mobile.png), [ACV desktop](verification/acv-desktop.png), [SHM desktop](verification/shm-desktop.png), [SHM mobile](verification/shm-mobile.png).
