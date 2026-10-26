# Test-set policy

The textile held-out test set has already been opened. Its membership remains
fixed. Do not use its labels, errors, spectra, confidence, wavelength diagnostics
or reported scores to select architectures, hyperparameters, preprocessing,
wavelength windows or future operating points.

Default runners access train/validation observation values only. Membership and
inventory metadata are validated for every partition, but no test image file is
opened and test spectral feature values are not converted/evaluated by default.
Inventory CSV parsing necessarily reads stored cells into memory; keep test
measurements in separate controlled storage if even that access is prohibited.

Test evaluation requires --evaluate-test and a matching externally frozen
protocol. The template command is a reproducibility aid, not evidence of
historical pre-registration; a user can generate it after inspecting results.
The framework cannot enforce organizational access or prevent adaptive reuse.
For this already-opened case study, a later run is a locked reproduction, not
fresh confirmatory evidence. No automatic best-test-model selection exists.

Post-hoc fixed-prediction confidence intervals and paired comparisons describe
existing results. They must not feed back into training decisions. Sample-level
predictions and test imagery remain private unless separately authorized.

The original demo's test-informed Swin selection is retrospective. Projector-
level OOF routing features are not end-to-end encoder OOF. These qualifications
remain necessary if those experiments are discussed alongside this toolkit.
