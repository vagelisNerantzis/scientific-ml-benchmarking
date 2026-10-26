"""Small functional baselines sharing the same sample-aware evaluation contract.

Image: frozen timm feature encoder + physical-sample logistic classifier.
Spectral: physical-sample features + train-only scaling + SVM-RBF.
These are portable baselines, not replicas of the full historical trainers.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .splits.loader import load_split
from .splits.validation import validate_observations
from .preprocessing.spectra import TrainOnlyScaler, validate_wavelengths, transmittance_to_absorbance
from .metrics.classification import classification_metrics
from .reporting.schemas import validate_result
from .reporting.aggregation import aggregate_seeds
from .runtime.timing import Timer
from .energy.gpu import GPUEnergyMonitor
from .provenance.manifests import sha256_file, environment_manifest
from .provenance.seeds import set_seed
from .utils.config import load_config, resolve_path

def load_inputs(config, *, evaluate_test=False):
    membership=load_split(resolve_path(config,'split_path'),
                          label_column=config.get('label_column','label'),
                          expected_sha256=config.get('split_sha256'),
                          textile=config.get('textile_case_study',False))
    frame=pd.read_csv(resolve_path(config,'observations_path'),
                      dtype={'sample_id':str,'observation_id':str,'label':str,'split':str,'modality':str},
                      keep_default_na=False)
    validate_observations(frame,membership)
    labels=config['labels']
    if len(labels)<2 or len(set(labels))!=len(labels) or set(membership.label)!=set(labels):
        raise ValueError('Explicit class schema must match membership classes')
    modality=config['modality']
    frame=frame.loc[frame.modality==modality].copy()
    allowed=['train','validation','test'] if evaluate_test else ['train','validation']
    frame=frame.loc[frame.split.isin(allowed)].copy()
    if frame.empty or not {'train','validation'}.issubset(set(frame.split)):
        raise ValueError('Train and validation observations are required')
    required=membership.loc[membership.split.isin(allowed)]
    if set(frame.sample_id)!=set(required.sample_id):
        raise ValueError('Selected modality must cover every evaluated physical sample')
    return membership,frame

def spectral_features(config, frame):
    columns=config['feature_columns']
    if not columns or len(set(columns))!=len(columns):
        raise ValueError('Unique spectral feature_columns required')
    x=frame[columns].to_numpy(dtype=float)
    validate_wavelengths(config['wavelengths_nm'],x)
    domain=config.get('input_domain','transmittance')
    target=config.get('representation',domain)
    if domain not in ['transmittance','absorbance'] or target not in ['transmittance','absorbance']:
        raise ValueError('Unsupported spectral domain')
    if domain=='transmittance':
        absorbance=transmittance_to_absorbance(x,units=config.get('transmittance_units','percent'))
        if target=='absorbance':
            x=absorbance
    elif target!='absorbance':
        raise ValueError('Inverse absorbance conversion is not implemented')
    if 'window_nm' in config:
        low,high=config['window_nm']
        if low>high:
            raise ValueError('Invalid wavelength window')
        wavelengths=np.asarray(config['wavelengths_nm'])
        mask=(wavelengths>=low)&(wavelengths<=high)
        if not mask.any():
            raise ValueError('Wavelength window contains no features')
        x=x[:,mask]
    return x

def image_features(config, frame):
    import torch
    import timm
    from timm.data import resolve_model_data_config, create_transform
    from .preprocessing.images import evaluation_crops
    # A public model identifier is mandatory; local research checkpoints are not loaded.
    if not config.get('pretrained',False):
        raise ValueError('Frozen image baseline requires pretrained=true; weights may download')
    device=torch.device(config.get('device','cpu'))
    model=timm.create_model(config['encoder'],pretrained=True,num_classes=0).to(device).eval()
    data=resolve_model_data_config(model)
    size=int(config.get('image_size',data['input_size'][-1]))
    data['input_size']=(3,size,size)
    transform=create_transform(**data,is_training=False)
    vectors=[]
    with torch.inference_mode():
        for path in frame['path']:
            local=dict(config,image_path=path)
            crops=evaluation_crops(resolve_path(local,'image_path'),size=size,
                                   fraction=float(config.get('crop_fraction',0.45)),
                                   count=int(config.get('eval_crops',9)))
            batch_size=int(config.get('image_batch_size',2))
            if batch_size<1:
                raise ValueError('image_batch_size must be positive')
            features=np.concatenate([model(torch.stack([transform(c) for c in crops[i:i+batch_size]]).to(device)).float().cpu().numpy()
                                     for i in range(0,len(crops),batch_size)])
            if features.ndim!=2 or not np.isfinite(features).all():
                raise ValueError('Encoder must return finite pooled N by D feature vectors')
            vectors.append(features.mean(0))
    return np.asarray(vectors)

def lock_protocol(config_path, config, kind):
    return {'schema_version':1,'purpose':'externally frozen evaluation protocol',
            'config_sha256':sha256_file(config_path),
            'split_sha256':sha256_file(resolve_path(config,'split_path')),
            'kind':kind,'seeds':config['seeds'], 'selection_partition':'validation'}

def run_benchmark(config_path, *, kind, evaluate_test=False, protocol_path=None):
    config=load_config(config_path)
    seeds=config.get('seeds')
    if not isinstance(seeds,list) or not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int or s<0 for s in seeds):
        raise ValueError('Provide distinct nonnegative integer seeds')
    if evaluate_test:
        if protocol_path is None:
            raise ValueError('Test evaluation requires an externally frozen --protocol JSON')
        with open(protocol_path,encoding='utf-8') as stream:
            locked=json.load(stream)
        if locked != lock_protocol(config_path,config,kind):
            raise ValueError('Protocol differs from current configuration, split, kind, or seeds')
    output=resolve_path(config,'output_dir')
    artifacts=['results.json','seed_summary.csv','manifest.json']
    if any((output/name).exists() for name in artifacts):
        raise ValueError('Result files already exist; choose a fresh output_dir')
    _,frame=load_inputs(config,evaluate_test=evaluate_test)
    labels=config['labels']
    # Feature generation/scaling accesses no held-out observation files by default.
    cuda_device=config.get('device','cpu') if kind=='image' and str(config.get('device','cpu')).startswith('cuda') else None
    set_seed(seeds[0],torch_enabled=(kind=='image'),deterministic=config.get('deterministic',False))
    with Timer(cuda_device=cuda_device) as feature_timer, GPUEnergyMonitor(enabled=config.get('gpu_energy',False),device_index=config.get('nvml_device_index',0)) as feature_energy:
        x=image_features(config,frame) if kind=='image' else spectral_features(config,frame)
    feature_columns=[f'x_{i}' for i in range(x.shape[1])]
    table=frame[['sample_id','label','split','modality']].copy()
    table[feature_columns]=x
    # Equal observation weighting within each physical sample; crops have already
    # been averaged within each image. No repeated observation is a training unit.
    table=table.groupby(['sample_id','label','split','modality'],as_index=False)[feature_columns].mean()
    train=table.loc[table.split=='train']
    if set(train.label)!=set(labels):
        raise ValueError('Training physical samples must represent every class')
    scaler=TrainOnlyScaler().fit(train[feature_columns],train.split)
    train_x=scaler.transform(train[feature_columns]) if kind=='image' else train[feature_columns].to_numpy()
    records=[]
    if kind=='spectral' and train.label.value_counts().min()<2:
        raise ValueError('Calibrated spectral baseline needs at least two training physical samples per class')
    for seed in seeds:
        set_seed(seed)
        if kind=='image':
            estimator=LogisticRegression(C=float(config.get('C',1)),max_iter=1000,random_state=seed)
        else:
            # Each training row is one physical sample. Fit scaling within each
            # calibration fold, never on its held-out physical samples.
            folds=min(5,int(train.label.value_counts().min()))
            base=make_pipeline(StandardScaler(),SVC(C=float(config.get('C',1)),
                               gamma=config.get('gamma','scale'),kernel='rbf',random_state=seed))
            estimator=CalibratedClassifierCV(base,method='sigmoid',
                      cv=StratifiedKFold(folds,shuffle=True,random_state=seed),ensemble=False)
        with Timer(cuda_device=None) as timer, GPUEnergyMonitor(enabled=config.get('gpu_energy',False),device_index=config.get('nvml_device_index',0)) as energy:
            estimator.fit(train_x,train.label)
            for split in (['validation','test'] if evaluate_test else ['validation']):
                part=table.loc[table.split==split]
                values=scaler.transform(part[feature_columns]) if kind=='image' else part[feature_columns].to_numpy()
                probs=estimator.predict_proba(values)
                order=[list(estimator.classes_).index(c) for c in labels]
                probs=probs[:,order];pred=np.asarray(labels)[probs.argmax(1)]
                metrics=classification_metrics(part.label,pred,labels=labels,probabilities=probs)
                record={'schema_version':1,'model':config.get('model_name',config.get('encoder','svm_rbf')),
                        'modality':config['modality'],'split':split,'seed':seed,
                        'n_samples':len(part),'evaluation_unit':'physical_sample',**metrics}
                records.append(record)
        # Shared seed-run telemetry is explicitly scoped, not partition-specific.
        for record in records:
            if record['seed']==seed:
                record.update(wall_seconds=timer.seconds,peak_gpu_memory_bytes=None,
                              gpu_energy_wh=energy.energy_wh,telemetry_scope='fit_plus_all_requested_evaluations')
                validate_result(record)
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(records,indent=2,allow_nan=False),encoding='utf-8')
    aggregate_seeds(pd.DataFrame(records)).to_csv(output/'seed_summary.csv',index=False)
    manifest={'schema_version':1,'kind':kind,'config_sha256':sha256_file(config_path),
              'membership_sha256':sha256_file(resolve_path(config,'split_path')),
              'test_evaluated':evaluate_test,'environment':environment_manifest(),
              'feature_extraction':{'wall_seconds':feature_timer.seconds,
                 'peak_gpu_memory_bytes':feature_timer.peak_gpu_memory_bytes,'gpu_energy_wh':feature_energy.energy_wh},
              'aggregation':'mean crop features per image; mean observation features per physical sample',
              'selection':'no automatic model selection; validation-first fixed baseline'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    return records
