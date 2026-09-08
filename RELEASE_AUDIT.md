# Public Release Audit

Release target: SatellaDet v0.2.1 framework source.

The public tree is intentionally framework-only. It excludes:

- training and validation images;
- annotations and private datasets;
- `.pt`, `.pth`, `.ckpt`, and `.onnx` model artifacts;
- experiment run directories and console logs;
- local databases;
- private development notebooks and their outputs;
- private Git history;
- user-specific absolute paths.

The repository includes `tools/privacy_scan.py` to catch common secret/token shapes, user-home paths, and blocked artifact types before release.

Automated scanning cannot prove that a release contains no sensitive information. The final repository and release archive should still receive a manual review before publication.
