"""Register a completed, verified full-training WP1 pair for GUI discovery."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from final_protocol import file_sha256, load_final_protocol, write_new_json


def register(release_path, protocol_path, source, output):
    protocol,_=load_final_protocol(protocol_path,source)
    record=json.loads(release_path.read_text(encoding='utf-8'))
    if (record.get('schema')!='wp1-full8000-release-v1' or record.get('training_scope')!='full_8000'
            or record.get('training_count')!=8000 or record.get('training_image_bytes_unchanged') is not True):
        raise ValueError('Require an actually completed and verified full-8,000 release')
    for key in ('training_manifest_sha256','candidate_sha256','source_sha256'):
        if record.get(key)!=protocol[key]:
            raise ValueError('Release/protocol identity mismatch: '+key)
    for task in ('crop','disease'):
        path=(ROOT/record['checkpoints'][task]['path']).resolve()
        if ROOT not in path.parents or file_sha256(path)!=record['checkpoints'][task]['sha256']:
            raise ValueError('Registered checkpoint path or bytes differ')
        summary=json.loads((path.parent/'summary.json').read_text(encoding='utf-8'))
        if summary['epochs_completed']!={'crop':2,'disease':3}[task] or summary['training_images_per_epoch']!=8000:
            raise ValueError('Registered task did not complete the fixed full-training budget')
    if output.exists():
        if json.loads(output.read_text(encoding='utf-8'))!=record:
            raise FileExistsError('Preserving an existing different GUI release registry')
    else:
        write_new_json(output,record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'artifacts/data/final_full8000_333_v1.json')
    parser.add_argument('--source',type=Path,default=ROOT/'DPD_Small_train.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/models/wp1_full8000_release.json')
    args=parser.parse_args()
    record=register(args.release,args.protocol,args.source,args.output)
    print(json.dumps({'registry':str(args.output),'checkpoints':record['checkpoints']},indent=2))


if __name__=='__main__':
    main()
