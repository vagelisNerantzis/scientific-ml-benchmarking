# Migration inventory

See source_provenance.json for each reviewed source basename, content hash and
adaptation. No original project file was copied verbatim. The package and
entry points were rewritten; this is an initial portable framework, not a
claim of parity with the full historical experiment suite. Names in provenance
are identifiers for attribution, not import paths or active dependencies.

Excluded: raw datasets, canonical membership CSV, all original result artifacts,
sample-level predictions, embeddings, tokens, individual spectra, checkpoints,
archives, machine launchers, audit/rename/split-generation utilities, recovery
scripts, old trainer versions, routing pipelines, external API smoke tests,
web application and institutional branding. No historical metric is bundled.
No new partition was generated for the textile dataset; artificial examples
have unrelated IDs, labels and counts.

The private canonical hash and aggregate counts are recorded as a validation
profile. They do not disclose sample membership. The software license covers
this extracted code, not rights to the original dataset or model artifacts.
