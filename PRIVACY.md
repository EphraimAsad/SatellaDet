# Privacy and Release Hygiene

The SatellaDet source repository is intended to contain framework code only.

Do not commit:

- training or validation images;
- annotation exports that contain sensitive identifiers;
- model checkpoints or ONNX weights unless deliberately released;
- experiment logs without review;
- `.env` files, credentials, API keys, tokens, private keys, or passwords;
- local databases;
- absolute local user-directory paths;
- customer, patient, sample, employee, or organization identifiers unless intentionally public and legally cleared.

## Runtime metadata

SatellaDet checkpoints and `config.json` use basename-only representations for path-like CLI arguments so common absolute local paths are not embedded by default.

This does not guarantee that every downstream artifact is anonymous. Dataset class names, user-authored filenames, notebook output, shell history, screenshots, image metadata, and manually added files can still contain sensitive information.

## Before every public release

Run:

```bash
python tools/privacy_scan.py .
```

Then manually inspect the release archive. Automated scanning is a guardrail, not a substitute for human review.
