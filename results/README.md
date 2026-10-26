# Result artifacts

No historical textile metrics, private predictions, spectra, identifiers,
embeddings, checkpoints or figures are distributed here. No research results
have been fabricated. Bundled examples produce explicitly artificial results.

Benchmark runners write aggregate results.json, seed_summary.csv and a portable
manifest.json into a configured output directory. These files are ignored by
Git. Use a fresh output directory for each protocol; existing files are not
silently replaced. Keep sample-level bootstrap inputs outside this repository.

Before publishing any real aggregate, verify its producer version, immutable
membership hash, configuration, seed set, evaluation unit and permissions.
Manually inspect all fields for paths and identifiers. Schema checks do not
replace a privacy review of arbitrary custom metadata.
