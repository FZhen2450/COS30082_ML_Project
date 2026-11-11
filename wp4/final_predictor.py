"""Final-333 CLIP scoring with explicit separation of historical adapter training."""
from pathlib import Path
import platform

import torch
import torchvision

from final_protocol import validate_final_protocol
from .clip_predictor import (CLIPPredictor, CLIP_REVISION, CLIP_SHA256, NATIVE_PREPROCESSING,
                             catalog_contract, load_m6_checkpoint, load_native_clip,
                             tensor_hash, unit_features)

ROOT = Path(__file__).resolve().parents[1]


class FinalCLIPPredictor(CLIPPredictor):
    def __init__(self, catalog, clip_weights, adapter_checkpoint=None, device='cpu'):
        validate_final_protocol(catalog.protocol, catalog.source_csv)
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Select an available CPU or CUDA device')
        self.expected = {'ids': [row['composition_id'] for row in catalog.protocol['candidates']]}
        self.candidate_sha256 = self.model_candidate_sha256 = catalog.protocol['candidate_sha256']
        self.adapter = None
        self.provenance = {
            'variant': 'm6' if adapter_checkpoint is not None else 'frozen_clip',
            'model': 'CLIP ViT-L/14', 'clip_revision': CLIP_REVISION, 'clip_sha256': CLIP_SHA256,
            'preprocessing': NATIVE_PREPROCESSING, 'candidate_count': 333,
            'catalog_candidate_sha256': self.candidate_sha256,
            'prompt_sha256': catalog.protocol['prompt_sha256'],
            'reference_training_scope': 'full_8000',
            'adapter_training_scope': 'not_applicable_frozen_clip',
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'torchvision': str(torchvision.__version__), 'device': str(self.device)}}
        if adapter_checkpoint is not None:
            # Validate the original adapter against its actual training contract.
            # A new 333-row text bank does not relabel it as full-8,000 training.
            from .service import Catalog
            historical = Catalog(ROOT/'artifacts/data/development_seed42_reconciled_v1.json',
                                 catalog.source_csv, catalog.images)
            self.adapter, historical_text, provenance = load_m6_checkpoint(
                adapter_checkpoint, catalog_contract(historical))
            self.adapter.to(self.device)
            self.provenance.update(provenance)
            self.provenance.update({'adapter_training_scope': 'historical_development_4868',
                                   'historical_direct_text_sha256': tensor_hash(historical_text),
                                   'final_full8000_adapter_available': False})
        self.model, self.preprocess, tokenize = load_native_clip(clip_weights, self.device)
        prompts = [row['prompt'] for row in catalog.protocol['prompt_payload']['rows']]
        tokens = tokenize(prompts, context_length=77, truncate=False)
        if tuple(tokens.shape) != (333, 77):
            raise ValueError('Final tokenizer must return all 333 ordered candidate rows')
        with torch.inference_mode():
            parts = [self.model.encode_text(tokens[i:i+32].to(self.device)).cpu()
                     for i in range(0, 333, 32)]
        self.direct = unit_features(torch.cat(parts))
        if tuple(self.direct.shape) != (333, 768):
            raise ValueError('Final text bank must contain 333 native feature rows')
        self.provenance['direct_text_sha256'] = tensor_hash(self.direct)
