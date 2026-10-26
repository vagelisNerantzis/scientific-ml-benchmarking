# Experiment protocol

1. Supply existing physical-sample membership. Validate the hash, labels, counts
   and all observation lineage before fitting. Never regenerate membership.
2. Declare ordered classes, modality, transforms, feature weighting, model,
   seeds, selection criterion, and telemetry scope before evaluation.
3. Fit transforms and supervised models on train physical samples only. Treat
   repeated observations as correlated, including internal calibration folds.
4. Use validation physical-sample macro F1 for any selection outside the fixed
   baseline runners. The runners perform no hyperparameter search or early stopping.
5. Freeze a protocol externally before optional test evaluation. The CLI can
   print a template but cannot attest when a protocol was frozen. Both explicit
   --evaluate-test and a matching --protocol are required to evaluate test.
6. Report per-seed metrics, sample SD (ddof=1; undefined for a single seed), and
   physical-sample bootstrap CIs. Resample within true-class strata, reuse every
   sampled physical ID across seeds and paired methods, and average seed metrics.
   This conditions on the observed class composition and chosen seed ensemble;
   it is not an estimate of all training-randomness or population-prevalence uncertainty.
7. Use exact paired McNemar separately for each seed. Do not pool seed outcomes
   as independent specimens. Multiple comparisons require separate justification.
8. Keep fixed-prediction post-hoc analyses separate from model selection.

## Migrated contracts and behavior changes

The source image trainers use patch budgets, validation checkpoint selection and
patch probabilities; the new image baseline uses a frozen public timm encoder,
mean crop features per image, equal mean image features per physical sample,
and a train-only scaled logistic classifier. Public pretrained weights may be
downloaded only when the runner is invoked. Model-specific timm normalization
is resolved explicitly. It is not a reproduction of fine-tuned ResNet50,
ConvNeXt-Tiny, Swin or the historical DINOv2 MLP head. DINOv2 encoders may be
used through timm but no architecture-specific historical recipe is implied.
The crop helper preserves the source's first nine row-major locations and
Python rounding. Counts above nine are deliberately rejected. PIL resizing
and timm evaluation transforms may differ from the original torchvision path.

The spectral runner uses sample-mean vectors, train-only StandardScaler and
SVM-RBF with fixed C/gamma. Sigmoid CalibratedClassifierCV uses stratified training-only physical-sample
folds (up to five); StandardScaler fits separately inside each calibration
fold. At least two training physical samples per class are required. This
replaces legacy SVC(probability=True) and avoids its deprecated interface. It does not reproduce the
original CNN1D, ResNet1D, transformer, contrastive or triplet-loss training,
128D embedding/retrieval outputs, augmentation, optimization, checkpointing,
or original exports. No original checkpoints or metrics are claimed here.

Calibration measures equal-width top-label ECE, not calibration fitting. It is
bin dependent; the runner uses ten bins. Macro F1 includes every declared class
with zero_division=0; balanced accuracy averages recall over classes present in
the evaluated data. Predeclare handling of absent classes.

Physical-sample stratification strengthens the generic paired bootstrap
extracted from the domain-comparison source, which resampled samples without
class strata. ECE and probability utilities now reject invalid distributions
instead of silently repairing them. Statistics require identical seed membership.
Error/wavelength analysis is reduced to aggregate error summaries and training
reference standardized deviations; no private hard-sample selection, SVM band
permutation, individual-spectrum exports or institutional figures are copied.
Window ablation is a validation-only predeclared window baseline.

Timing uses perf_counter; CUDA feature timing synchronizes the chosen device
and reports peak allocated bytes. Feature extraction is reported once; per-seed
telemetry spans CPU classifier fit plus all requested evaluations. Optional
NVML trapezoid integration measures sampled board power in Wh, includes other
GPU users/idle power and is neither process-isolated nor total system energy.
NVML index must be matched to the CUDA physical device by the operator.
