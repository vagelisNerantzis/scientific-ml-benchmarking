# Migration and validation report

Validation date: 2026-10-26. Python 3.11.15.

## Final repository tree

Every file is listed below; no Git metadata, caches or private artifacts are included.

```text
.github/workflows/tests.yml
.gitignore
LICENSE
README.md
analysis/analyze_errors.py
analysis/compare_spectral_domains.py
analysis/wavelength_diagnostics.py
assets/README.md
assets/generate_workflow.py
assets/workflow.svg
configs/image_benchmark_example.yaml
configs/spectral_benchmark_example.yaml
docs/data_contract.md
docs/experiment_protocol.md
docs/migration_notes.md
docs/source_provenance.json
docs/test_set_policy.md
docs/validation_report.md
examples/data/synthetic_leaked_split.csv
examples/data/synthetic_spectra.csv
examples/data/synthetic_split.csv
examples/synthetic_benchmark_example.py
examples/synthetic_leakage_example.py
experiments/spectral_window_ablation.py
pyproject.toml
requirements.txt
results/README.md
scripts/aggregate_results.py
scripts/bootstrap_metrics.py
scripts/run_image_benchmark.py
scripts/run_spectral_benchmark.py
scripts/validate_split.py
src/scientific_benchmark/__init__.py
src/scientific_benchmark/benchmark.py
src/scientific_benchmark/energy/__init__.py
src/scientific_benchmark/energy/gpu.py
src/scientific_benchmark/metrics/__init__.py
src/scientific_benchmark/metrics/calibration.py
src/scientific_benchmark/metrics/classification.py
src/scientific_benchmark/preprocessing/__init__.py
src/scientific_benchmark/preprocessing/images.py
src/scientific_benchmark/preprocessing/spectra.py
src/scientific_benchmark/provenance/__init__.py
src/scientific_benchmark/provenance/manifests.py
src/scientific_benchmark/provenance/seeds.py
src/scientific_benchmark/reporting/__init__.py
src/scientific_benchmark/reporting/aggregation.py
src/scientific_benchmark/reporting/schemas.py
src/scientific_benchmark/runtime/__init__.py
src/scientific_benchmark/runtime/timing.py
src/scientific_benchmark/splits/__init__.py
src/scientific_benchmark/splits/loader.py
src/scientific_benchmark/splits/validation.py
src/scientific_benchmark/statistics/__init__.py
src/scientific_benchmark/statistics/bootstrap.py
src/scientific_benchmark/statistics/paired.py
src/scientific_benchmark/utils/__init__.py
src/scientific_benchmark/utils/config.py
tests/test_bootstrap.py
tests/test_calibration.py
tests/test_images.py
tests/test_leakage_detection.py
tests/test_metrics.py
tests/test_result_schema.py
tests/test_split_integrity.py
```

## Source extraction and rewrites

No research script was copied unchanged. These sources were reviewed and their reusable methods rewritten. Exact input hashes are in source_provenance.json.

| Reviewed source basename | Extracted method / destination concern |
|---|---|
| `12b_train_standard_modality_benchmark.py` | images.py grid coordinates; sample aggregation; metrics/calibration; timing and NVML contracts |
| `12_train_dinov2_modality_benchmark.py` | frozen encoder methodology; not its historical MLP training recipe |
| `13_run_all_modality_experiments_v2.py` | repeated-seed and fair sample-unit reporting methodology; launcher rewritten |
| `15_run_spectral_ml_dl_benchmark.py` | spectral QC, train-only scaling, SVM baseline and metrics |
| `17_run_spectral_dl_embedding_benchmark_v2.py` | wavelength/identity contracts, calibration and telemetry; neural trainers omitted |
| `18_run_absorbance_dl_embedding_benchmark_v2.py` | absorbance domain and train-only scaling contracts; neural trainers omitted |
| `19_run_absorbance_ml_benchmark.py` | absorbance SVM baseline methodology |
| `20_analyze_transmittance_vs_absorbance.py` | seed-aligned paired sample bootstrap and exact McNemar; now stratified |
| `21_analyze_hard_spectral_samples.py` | post-hoc error summary contract; individual sample exports omitted |
| `22_wavelength_diagnostics_hard_samples.py` | train-reference deviation contract; permutation diagnostics omitted |
| `svm_window_ablation.py` | predeclared windows, train-only scaling and paired comparison methodology |

All other public package modules, configuration, entry points, synthetic fixtures, tests and documentation are newly authored. No original results were migrated.

## Intentional omissions

Raw image/spectral datasets, canonical split CSV, private manifests, predictions, embeddings, token tables, individual spectra, checkpoints, archives, caches, institutional branding and all historical metrics are excluded. Dataset-mutating tools, split generators, obsolete/recovery scripts, batch launchers and the old orchestration path are excluded. The original neural training/model zoo is not copied as an active workflow.

## Remaining paths and privacy

No private absolute source paths, usernames, local checkpoint paths or file-URL dependencies remain in scanned public text. Relative examples and environment-variable placeholders are intentional. Only synthetic sample identifiers are bundled. The canonical reference hash and aggregate counts are public validation metadata. Source basenames identify provenance and are not active imports.

## Executed checks

| Check | Exit code | Expected |
|---|---:|---:|
| compile | 0 | 0 |
| tests | 0 | 0 |
| valid_split | 0 | 0 |
| leaked_split | 2 | 2 |
| leakage_example | 2 | 2 |
| synthetic_benchmark | 0 | 0 |

- `python -m compileall -q .`: passed. Bytecode was redirected to temporary storage.
- `python -B -m unittest discover -s tests -v`: 48 tests passed, zero failures, zero skips in the validation environment.
- Valid synthetic split: accepted; 20 physical samples, 12 train / 4 validation / 4 test.
- Leaked synthetic split and leakage example: rejected with `Physical-sample leakage across splits` and exit status 2.
- Synthetic SVM validation workflow: passed; temporary outputs cleaned up.
- Image extraction: tested with generated temporary imagery and a tiny mocked frozen encoder, without downloads or training.
- Real pretrained image inference, real GPU telemetry and historical experiment reproduction were not run.
- The existing environment emits a torch warning about an installed deprecated pynvml distribution; the new repository specifies optional nvidia-ml-py and does not install or modify the research environment.

## Source preservation

All 13,231 source files retain the same names, sizes and modification times as the pre-build inventory. All 11 reviewed source content hashes match. The canonical membership SHA-256 remains unchanged.

Inventory metadata SHA-256: `a4ffe6813af1e6d478e42516c243e1da27b6c2deb131f839c6ff78c3f22f07ba`.

Canonical membership SHA-256: `a5726d3111941989407b0049e7ba67679e3b846c7e486544467fd926f99dbfc4`.

No source project file was written. No held-out images, predictions or spectral measurements were evaluated. Only source code and the canonical split byte hash were consulted during extraction.

## Scientific-integrity limitations and behavior changes

- Physical membership and class identity are validated; identity aliases without truthful lineage cannot be detected automatically.
- Default workflows evaluate validation only. Test requires an explicit flag and matching protocol; external protocol timing cannot be proven by this software.
- The textile test has already been opened. Subsequent execution is reproduction, not fresh confirmation, and test-informed model/wavelength selection remains prohibited.
- Bootstrap resamples true-class-stratified physical samples, keeps all seed/method outcomes together and reports paired B-minus-A differences. Seed SD remains descriptive.
- Image baseline is frozen timm features plus sample-level logistic regression, not original end-to-end fine-tuning. Feature pooling differs from historical probability pooling. Crop counts above nine are rejected.
- Spectral baseline is SVM-RBF with stratified physical-sample sigmoid calibration and fold-local scaling. It replaces the legacy probability interface; neural encoders and retrieval are omitted.
- Generic arrays and CSV contracts replace instrument-specific readers. Invalid probabilities/transmittance fail instead of being silently repaired.
- Post-hoc analysis is explicitly exploratory; window experiments evaluate validation only. Reduced error/wavelength helpers do not reproduce original individual-spectrum/permutation reports.
- Inventory CSV parsing reads cells even when test rows are excluded before numeric conversion; restricted test measurements should be kept in separately controlled storage.
- Optional energy measures NVIDIA board power, not whole-system energy; real telemetry was not validated here.

## Manual Git commands

Run from the new repository directory after review. No Git commands were executed by the migration.

```bash
git init
git add .
git commit -m "Initial scientific ML benchmarking framework"
```

No remote was created and nothing was pushed.


## Final portfolio improvements

The final pre-commit pass adds only the requested presentation, automation and
version changes. Existing model, preprocessing, metric and statistical behavior
is unchanged. The source project was not modified.

### Changed files

- `README.md`: adds the four design principles, embeds the workflow figure near
  Overview and updates the structure listing.
- `assets/README.md`: documents the public figure and reproduction commands.
- `assets/generate_workflow.py`: new deterministic standard-library SVG generator
  with a byte-for-byte `--check` mode.
- `assets/workflow.svg`: new generic nine-stage workflow figure.
- `.github/workflows/tests.yml`: new push/pull-request CI workflow.
- `src/scientific_benchmark/__init__.py`: exposes `__version__ = "0.1.0"`.
- `docs/validation_report.md`: updates the full tree and records this validation.

`pyproject.toml` already defines version 0.1.0 and needed no change. The package
and project versions were checked against the installed distribution.

### Validation rerun

- 48 tests passed, zero failures, zero skips using local Python 3.11.15.
- `python -m compileall -q src scripts analysis experiments examples tests assets`
  passed. Generated bytecode was redirected to temporary storage.
- A package build/install into an isolated temporary target passed without
  installing dependencies or changing the existing environment.
- Valid synthetic membership passed with exit 0.
- The leaked membership and intentional leakage example returned exit 2 and
  the expected `Physical-sample leakage across splits` diagnostic.
- The synthetic validation benchmark passed.
- SVG XML parsing passed; `python assets/generate_workflow.py --check` passed.
- Both Python-shell CI steps passed locally. Three negative controls confirmed
  that the CI leakage guard rejects a successful application command, an
  unrelated failure, and an incorrect failure status.
- CI targets Python 3.12 and installs the package with its image extra plus CPU
  PyTorch/torchvision, so optional image checks are included. unittest needs no
  third-party test runner. GitHub-hosted execution has not occurred.
- Final text scanning found no private absolute paths, usernames, local
  checkpoint paths or local file-URL dependencies. Public SVG namespace and
  dependency-service URLs are intentional.
- No Git initialization, remote creation, commit or push was performed.

Python 3.12 is within its supported lifecycle; see the
[Python version support table](https://devguide.python.org/versions/).

### CI workflow contents

```yaml
name: Tests

on: [push, pull_request]

permissions:
  contents: read

jobs:
  tests:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml
      - name: Install package and complete test dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          python -m pip install -e ".[image]"
      - name: Compile Python sources
        run: python -m compileall -q src scripts analysis experiments examples tests assets
      - name: Run complete unittest suite
        run: python -m unittest discover -s tests -v
      - name: Validate synthetic physical-sample membership
        run: python scripts/validate_split.py examples/data/synthetic_split.csv
      - name: Verify expected physical-sample leakage failure
        shell: python
        run: |
          import subprocess
          import sys

          result = subprocess.run(
              [sys.executable, "examples/synthetic_leakage_example.py"],
              capture_output=True, text=True,
          )
          output = result.stdout + result.stderr
          print(output, end="")
          expected_reason = "Physical-sample leakage across splits"
          if result.returncode != 2 or expected_reason not in output:
              raise SystemExit("Expected exit status 2 and physical-sample leakage diagnostic")
      - name: Check reproducible workflow figure
        run: python assets/generate_workflow.py --check
      - name: Check package version consistency
        shell: python
        run: |
          import pathlib
          import tomllib
          import scientific_benchmark

          project = tomllib.loads(pathlib.Path("pyproject.toml").read_text(encoding="utf-8"))
          assert scientific_benchmark.__version__ == project["project"]["version"] == "0.1.0"
```
