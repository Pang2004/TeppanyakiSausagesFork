# Home + Rail application validation

## Implemented

- React/TypeScript Home and Rail pages adapted from the supplied HTML, including the original interactive train SVG, Rail SVG geometry, colours and fonts.
- Responsive desktop and phone layouts, without simulated phone status bars or hardware.
- Local FastAPI inference with the existing Rail artifact and feature extraction.
- Individual files, multiple files, folder selection and directory/file drag-and-drop; sequential processing, upload progress, per-file errors, stop/retry, selected-file visualization and official CSV export.
- Standalone app runtime and PS3 submission packaging. Door, ACV and SHM UI routes remain pending.

## Evidence

| Check | Result |
| --- | --- |
| Existing four-subsystem model tests after relocation | 43 passed |
| API validation, upload limits, model unavailability, original names and temporary-file cleanup | 8 passed |
| Packaging exclusions, flat prediction zip, missing-deliverable reporting and overwrite protection | 1 passed |
| Frontend file validation, directory chunk traversal, drag fallback and CSV serialization | 6 passed |
| Browser scenarios | Five scenarios passed in desktop Chromium, phone Chromium and phone WebKit |
| Full official Rail test folder through the browser | All 68 files completed; downloaded CSV exactly matches the validated output |
| Model preservation | All four active artifacts are byte-identical to the original artifacts |
| Standalone packaged runtime | Predicts `Normal` for uploaded `Test1.csv` with only packaged runtime on the import path |
| Frontend production build / Python lint / diff whitespace | Passed |

The five browser scenarios cover layout/navigation/reload/keyboard interaction, real inference for all three labels and CSV download, failed upload/retry/duplicate/offline handling, queue stopping, and drag-and-drop with mixed success/failure and partial export. The narrow-layout check includes a 320-pixel viewport. The 68-file folder check runs once in desktop Chromium, independently of the smaller cross-browser cases.

The tests exposed and resolved two upload differences: folder-selected files need an explicit basename in multipart uploads, and WebKit may supply an unreadable file-entry handle even when a valid `File` object is available. Files now use the direct object; directory entries still traverse all chunks recursively.

The generated browser output is `app/exports/rail_predictions.csv`. Tests and transient browser screenshots/traces remain in ignored test-result directories. Existing numerical-library and test-client dependency deprecation warnings do not affect the passing assertions.

## Practical limits

- Browser phone profiles emulate Chromium and WebKit; a physical phone on the user's Wi-Fi still needs the manual connectivity check in `app/README.md`.
- No public hosting, accounts or persistent history are configured. Refresh clears results. A model computation already started can finish after the user cancels its HTTP request; its temporary file is cleaned when processing ends.
- WebKit on this Linux machine was tested with official Ubuntu library packages unpacked into temporary storage; no system packages were changed. A new Linux setup can instead use Playwright's documented dependency installer with administrator privileges.
- The current package is an incremental Rail demo. All four attempted subsystems must eventually have app workflows and app-generated test predictions; the demo video must also be supplied before final submission.
