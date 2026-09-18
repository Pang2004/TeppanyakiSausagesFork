# PS3 methodology and implementation index

This write-up covers all four subsystems. Detailed methodology, validation protocols, retained model decisions and limitations are in each subsystem's `METHODOLOGY.md`:

- [Door](Door/METHODOLOGY.md): forward raw-stream validation and segmentation robustness.
- [ACV](ACV/METHODOLOGY.md): peer-car anomaly ranking and missing-signal checks.
- [Rail Corrugation](Rail%20Corrugation/METHODOLOGY.md): duplicate-group-aware validation, per-car wavelength energy, side mirroring and inner decision tuning.
- [SHM](SHM/METHODOLOGY.md): nested model-family selection and historical error percentiles.

The runnable app and its exact feature/prediction implementation are in `app/`. Optional training and validation code imports that implementation rather than maintaining a second feature extractor. Active artifact copies are added to each optional `model/` folder when packaging. Development datasets are intentionally excluded from the submission.

The current UI implements the supplied Home and Rail designs with responsive React, FastAPI uploads, per-recording results and official-format CSV export. Other subsystem UIs remain pending. Uploaded signals are processed locally on the app server and temporary recordings are removed. No remote model API is used.

For development commands in the repository, use `python scripts/model.py <subsystem> <train|validation|predict> ...`. In the packaged folder, the same launcher is `python Optional_Items/tools/model.py ...`; provide your own labelled data paths if running optional training or validation.
