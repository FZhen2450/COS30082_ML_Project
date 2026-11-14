"""Final-333 CLIP scoring with explicit separation of historical adapter training."""
from pathlib import Path
import platform

import torch
import torchvision

from final_protocol import validate_final_protocol
from .clip_predictor import (CLIPPredictor, CLIP_REVISION, CLIP_SHA256, NATIVE_PREPROCESSING,
                             catalog_contract, load_m6_checkpoint, load_native_clip,
                             tensor_hash, file_hash, FULL8000_M6_SHA256)
from .text_bank import load_official_text_bank

ROOT = Path(__file__).resolve().parents[1]


class FinalCLIPPredictor(CLIPPredictor):
    def __init__(self, catalog, clip_weights, adapter_checkpoint=None, device='cpu'):
        validate_final_protocol(catalog.protocol, catalog.source_csv)
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Select an available CPU or CUDA device')
        self.direct, bank_identity = load_official_text_bank()
        self.text_bank_is_canonical = True
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
            # Both releases retain a 323-class training text contract. Final
            # inference uses the separate original 333-row official bank.
            from .service import Catalog
            historical = Catalog(ROOT/'artifacts/data/development_seed42_reconciled_v1.json',
                                 catalog.source_csv, catalog.images)
            full8000 = Path(adapter_checkpoint).is_file() and file_hash(adapter_checkpoint) == FULL8000_M6_SHA256
            options = {'full_training_sha': catalog.protocol['training_manifest_sha256']} if full8000 else {}
            self.adapter, historical_text, provenance = load_m6_checkpoint(
                adapter_checkpoint, catalog_contract(historical), **options)
            self.adapter.to(self.device)
            self.provenance.update(provenance)
            self.provenance.update({'adapter_training_scope': 'full_original_training_8000' if full8000 else 'historical_development_4868',
                                   'training_direct_text_sha256': tensor_hash(historical_text),
                                   'final_full8000_adapter_available': full8000})
            if full8000:
                self.provenance.update({'adapter_training_count': 8000,
                    'adapter_training_manifest_sha256': catalog.protocol['training_manifest_sha256']})
            else:
                self.provenance['historical_direct_text_sha256'] = tensor_hash(historical_text)
        self.model, self.preprocess, _tokenize = load_native_clip(clip_weights, self.device)
        self.provenance.update(bank_identity)
