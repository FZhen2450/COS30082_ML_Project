"""Export file and versioned tensor-state identities without modifying weights."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from checkpoint_identity import checkpoint_identity
from final_protocol import write_new_json


def export_identities(release_path, output):
    release=json.loads(release_path.read_text(encoding='utf-8'))
    if release.get('schema')!='wp1-full8000-release-v1' or release.get('training_count')!=8000:
        raise ValueError('Require the completed full-8,000 release')
    tasks={}
    for task in ('crop','disease'):
        checkpoint=release['checkpoints'][task]
        tasks[task]=checkpoint_identity(ROOT/checkpoint['path'],checkpoint['sha256'])
    result={'schema':'wp1-full8000-state-identity-v1','training_count':8000,
            'training_manifest_sha256':release['training_manifest_sha256'],
            'candidate_sha256':release['candidate_sha256'],'checkpoints':tasks,
            'weights_modified':False,'new_inference':False}
    write_new_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,default=ROOT/'artifacts/models/wp1_full8000_release.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=export_identities(args.release,args.output)
    print(json.dumps({task:{key:value for key,value in record.items() if key!='tensors'}
                      for task,record in result['checkpoints'].items()},indent=2))


if __name__=='__main__':
    main()
