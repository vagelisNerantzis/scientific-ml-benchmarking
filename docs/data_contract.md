# Data contract

The immutable membership CSV has sample_id,label,split. IDs and labels are
strings, including numeric-looking IDs; preserve leading zeros. UTF-8 BOM is
accepted. Splits are exactly train, validation, test. Each physical sample occurs
once; its label and partition never change. Hash checks apply to original bytes,
not a normalized CSV. Alias a source label column explicitly; the textile profile
uses class_code and enforces the reference hash and aggregate class counts.

Observation CSVs require sample_id,observation_id,modality,label,split. Every row
must agree with membership. An observation_id is unique within modality. Derived
crops must have their own observation IDs and retain source_observation_id or
image_id. These lineage identifiers must identify the same sample and split.
Optional seed/model columns identify prediction contexts. Physical identity is
not inferred from filenames. Correct lineage remains the data producer's duty:
no table validator can detect two aliases of the same specimen without a mapping.

Image inventories add path, resolved relative to the YAML configuration, not the
CSV. Spectral inventories add numeric feature columns whose ordered schema is
feature_columns and wavelengths_nm. No instrument-specific header parser or
implicit interpolation is supplied. Wavelengths are finite, strictly increasing
and aligned to values. Percent/fraction transmittance units are explicit;
A=-log10(T_fraction). Invalid or zero transmittance is rejected rather than clipped.
Absorbance is finite but not constrained to nonnegative instrumental values.
TrainOnlyScaler must fit train rows only and cannot silently refit.

Probability columns are prob_<label>, following the explicit class order.
Rows must be finite, nonnegative and sum to one within 1e-5; only roundoff is
renormalized. Invalid distributions fail rather than becoming uniform predictions.
Aggregate observation probabilities within a model/seed/modality/split. For
equal image weighting, call aggregate_observations(level='image'), set each
resulting observation_id to its image_id, then aggregate to sample. Direct crop
to sample averaging instead weights images by their crop counts. Keep the
chosen weighting in provenance. Modalities are not automatically fused.

Bootstrap inputs contain sample_id,label,split,seed and one prediction column,
or two paired columns. There must be exactly one prediction per sample/seed,
one partition, identical membership across seeds and consistent physical labels.
Never supply crop-level rows. Summary inputs cannot establish membership equality
across seeds; retain the private aligned prediction tables for that audit.

Result schema v1 requires model,modality,split,seed,n_samples,evaluation_unit,
accuracy,balanced_accuracy,macro_f1,ece and schema_version. The unit is
physical_sample. The runners include class metrics and confusion matrices but
omit observation/sample identifiers. A manifest records config and membership
hashes plus package versions without serializing machine directories.
