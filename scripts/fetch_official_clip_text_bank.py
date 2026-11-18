"""Explicit download of the shared official bank; never called by GUI startup."""
import argparse
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from wp4.text_bank import OFFICIAL_BANK_SHA256


def main():
    import hashlib
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'models/official333_clip_text_bank.npz')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserving an existing text-bank file')
    url='https://drive.google.com/uc?export=download&id=1DKvZo__aQyvW1M2QOwu661KdKOv8CHWw'
    with urllib.request.urlopen(url,timeout=30) as response:
        data=response.read(16*1024*1024+1)
    if len(data)>16*1024*1024 or hashlib.sha256(data).hexdigest()!=OFFICIAL_BANK_SHA256:
        raise ValueError('The response is not the pinned official text bank')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(data)
    print(f'Verified official bank: {args.output} SHA256={OFFICIAL_BANK_SHA256}')


if __name__=='__main__':
    main()
