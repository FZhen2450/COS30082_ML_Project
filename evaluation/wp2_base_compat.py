"""Isolated WP2 reference-interface adapter, not an approved final model loader.

Source: bbdabfd.../notebooks/wp2_selected_development_export.ipynb.
No image files, downloads, pretrained initialization, or auto-discovered assets.
Private final checkpoint loading is deliberately unavailable until evidenced.
"""
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE='multitask_convnext_base'
FEATURE_DIM=1024
EXPECTED_FINAL_CHECKPOINT_SHA256='27aba7779973b4caef68f98c627fb00c5d8afc9a9b7de58f9628954a90976b34'

class UnverifiedPrivateCheckpointError(RuntimeError):pass

def _integer(value,name):
    if isinstance(value,bool) or not isinstance(value,Integral):raise ValueError(name+' requires integer IDs without coercion')
    return int(value)

def validate_candidates(candidate_ids,pairs,*,required_count=333):
    ids=[_integer(x,'candidate_ids') for x in candidate_ids]
    if len(ids)!=required_count or ids!=sorted(set(ids)) or any(x<0 for x in ids):
        raise ValueError(f'Require {required_count} unique ascending original candidate IDs')
    if len(pairs)!=len(ids):raise ValueError('Candidate ID/pair lengths differ')
    result=[]
    for pair in pairs:
        if len(pair)!=2:raise ValueError('Each candidate requires crop/disease pair')
        c,d=(_integer(x,'candidate_pairs') for x in pair)
        if not 0<=c<55 or not 0<=d<175:raise ValueError('Candidate component ID out of bounds (55 crops/175 diseases)')
        result.append((c,d))
    if len(set(result))!=len(result):raise ValueError('Duplicate crop/disease candidate pair')
    return tuple(ids),tuple(result)

def validation_transform():
    """Exact inspected 224-square reference transform, not a native preset.

    Returns a transform only. Does not open files; caller owns input provenance.
    """
    from torchvision.transforms import v2,InterpolationMode
    return v2.Compose([v2.ToImage(),v2.Resize((224,224),interpolation=InterpolationMode.BILINEAR,antialias=True),v2.ToDtype(torch.float32,scale=True),v2.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])

def validate_output(outputs,*,expected_batch=None):
    if not isinstance(outputs,Mapping) or not {'crop_logits','disease_logits'}<=outputs.keys():raise ValueError('Require crop_logits and disease_logits mapping')
    n=None;device=None
    for key,width in [('crop_logits',55),('disease_logits',175)]:
        x=outputs[key]
        if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[0]<1 or x.shape[1]!=width or not x.is_floating_point():raise ValueError(key+f' requires finite floating [N,{width}] tensor')
        if not bool(torch.isfinite(x).all()):raise ValueError(key+' contains NaN or infinity')
        if n is not None and (x.shape[0]!=n or x.device!=device):raise ValueError('Component batch/device mismatch')
        n=x.shape[0];device=x.device
    if expected_batch is not None and n!=expected_batch:raise ValueError('Unexpected output batch count')
    if 'features' in outputs:
        x=outputs['features']
        if not isinstance(x,torch.Tensor) or x.shape!=(n,FEATURE_DIM) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):raise ValueError('features requires finite floating [N,1024] tensor')
    return n

def score_components(crop_logits,disease_logits,candidate_ids,pairs):
    """Same PyTorch FP64 arithmetic/order as the inspected Base scorer."""
    ids,pairs=validate_candidates(candidate_ids,pairs)
    validate_output(dict(crop_logits=crop_logits,disease_logits=disease_logits))
    columns=torch.tensor(pairs,dtype=torch.int64,device=crop_logits.device)
    scores=F.log_softmax(crop_logits.to(torch.float64),dim=1)[:,columns[:,0]]+F.log_softmax(disease_logits.to(torch.float64),dim=1)[:,columns[:,1]]
    if not bool(torch.isfinite(scores).all()):raise ValueError('Nonfinite canonical scores; no clipping or fallback')
    return scores

def rank_global_ids(scores,candidate_ids,*,available_k=5):
    ids=[_integer(x,'candidate_ids') for x in candidate_ids]
    if len(ids)!=333 or ids!=sorted(set(ids)) or any(x<0 for x in ids):raise ValueError('Require 333 unique ascending original candidate IDs')
    if not isinstance(scores,torch.Tensor) or scores.ndim!=2 or scores.shape[0]<1 or scores.shape[1]!=333 or scores.dtype!=torch.float64 or not bool(torch.isfinite(scores).all()):raise ValueError('Require finite native float64 [N,333] scores')
    if type(available_k) is not int or not 1<=available_k<=333:raise ValueError('Invalid ranking budget')
    index=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:available_k]
    return torch.tensor(ids,dtype=torch.int64,device=scores.device)[index]

def adapt_named_outputs(outputs,actual_records,expected_records,candidate_ids,pairs,*,expected_candidate_ids,expected_pairs):
    """No inferred names: caller must supply independently approved row identities.

    Each record is {sample_id, fold, truth_id}; both sequences must match exactly.
    Score/identity tensors are detached/copied to prevent caller-buffer mutation.
    """
    ids,pairs=validate_candidates(candidate_ids,pairs)
    expected_ids,expected_pairs=validate_candidates(expected_candidate_ids,expected_pairs)
    if (ids,pairs)!=(expected_ids,expected_pairs):raise ValueError('Candidate ordering/semantics differs from approved specification')
    def records(values):
        values=list(values);result=[]
        for row in values:
            if not isinstance(row,Mapping) or set(row)!={'sample_id','fold','truth_id'}:raise ValueError('Explicit filename/fold/truth records required')
            name=row['sample_id'];fold=row['fold'];truth=_integer(row['truth_id'],'truth_id')
            if not isinstance(name,str) or not name.strip() or fold not in ('seen','unseen') or truth not in ids:raise ValueError('Invalid filename/fold/truth identity')
            result.append(dict(sample_id=name,fold=fold,truth_id=truth))
        if not result or len({r['sample_id'] for r in result})!=len(result):raise ValueError('Missing or duplicated filenames')
        return result
    actual=records(actual_records);expected=records(expected_records)
    if actual!=expected:raise ValueError('Filename/fold/truth alignment differs from approved row order')
    validate_output(outputs,expected_batch=len(actual))
    scores=score_components(outputs['crop_logits'],outputs['disease_logits'],ids,pairs).detach().to('cpu',copy=True)
    return dict(scores=scores,top5_ids=rank_global_ids(scores,ids),candidate_ids=ids,records=actual,architecture=ARCHITECTURE,score_semantics='sum_log_softmax_v1',scope='reference_interface_only_private_final_release_unverified')

class ReferenceDualHeadBase(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone
        if not hasattr(backbone,'classifier') or getattr(backbone.classifier[-1],'in_features',None)!=FEATURE_DIM:raise ValueError('Base classifier must expose 1024 input features; Tiny fallback forbidden')
        self.backbone.classifier[-1]=nn.Identity()
        self.crop_head=nn.Linear(FEATURE_DIM,55);self.disease_head=nn.Linear(FEATURE_DIM,175)
    def forward(self,inputs):
        features=self.backbone(inputs)
        if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[1]!=FEATURE_DIM or not features.is_floating_point() or not bool(torch.isfinite(features).all()):raise ValueError('Base backbone returned unexpected/nonfinite feature tensor')
        result=dict(features=features,crop_logits=self.crop_head(features),disease_logits=self.disease_head(features));validate_output(result);return result

def create_base_model(architecture=ARCHITECTURE,*,backbone_factory=None):
    if architecture!=ARCHITECTURE:raise ValueError('Only evidenced multitask_convnext_base is supported; no Tiny fallback')
    if backbone_factory is None:
        from torchvision.models import convnext_base
        backbone_factory=convnext_base
    # No network calls, enum weights or pretrained initialization. This is a
    # reference skeleton; it is NOT a released predictor until separately loaded.
    return ReferenceDualHeadBase(backbone_factory(weights=None)).eval()

def validate_reference_checkpoint_metadata(payload,mappings):
    """Validate ONLY the explicitly evidenced historical exporter envelope.

    This must never be advertised as validation of the private final schema.
    """
    expected=dict(format_version=1,architecture=ARCHITECTURE,num_crops=55,num_diseases=175,feature_dim=1024,model_dtype='torch.float32')
    if not isinstance(payload,Mapping) or any(type(payload.get(k))!=type(v) or payload.get(k)!=v for k,v in expected.items()):raise ValueError('Incompatible evidenced reference checkpoint metadata')
    meta=payload.get('metadata',{})
    if not isinstance(meta,Mapping) or not isinstance(meta.get('selection'),Mapping):raise ValueError('Invalid reference selection metadata')
    selection=meta['selection']
    if type(meta.get('epoch')) is not int or meta.get('epoch')!=1 or type(selection.get('epoch')) is not int or selection.get('epoch')!=1 or type(selection.get('resolution')) is not int or selection.get('resolution')!=224:raise ValueError('Reference checkpoint requires epoch1 and resolution224')
    if payload.get('class_mappings')!=mappings:raise ValueError('Reference checkpoint component mappings differ')
    return dict(schema_scope='evidenced_historical_exporter_only',private_final_schema_verified=False)

def load_private_final_checkpoint(path,**_unused):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('Private WP2 checkpoint unavailable; supply owner asset and sealed release/schema; no fallback or download')
    raise UnverifiedPrivateCheckpointError('Private final checkpoint schema/release unverified; no deserialization attempted. Supply exact owner loader/schema and receipt for checkpoint '+EXPECTED_FINAL_CHECKPOINT_SHA256)
