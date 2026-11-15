"""Compare immutable runs, including every label pixel and scientific JSON value."""
from pathlib import Path
import argparse
import json

import numpy as np

SCIENTIFIC = ['schema_version','package_version','manifest_digest','fields','observations','edges','founders','quality',
              'founder_events','field_thresholds','label_planes','graph_scope',
              'segmenter_class','tracker_class','dependencies','python_version']

def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    a,b = [json.loads((p/'run.json').read_text()) for p in [reference,candidate]]
    differences = 0
    def diff(a,b,path):
        nonlocal differences
        if type(a) is not type(b):
            print(f'DIFF {path}: type {type(a).__name__} != {type(b).__name__}'); differences += 1
        elif isinstance(a,dict):
            for k in sorted(a.keys()|b.keys()):
                if k not in a or k not in b:
                    print(f'DIFF {path}/{k}: missing key'); differences += 1
                else: diff(a[k],b[k],f'{path}/{k}')
        elif isinstance(a,list):
            if len(a)!=len(b):
                print(f'DIFF {path}: lengths {len(a)} != {len(b)}'); differences += 1
            for i,(x,y) in enumerate(zip(a,b)): diff(x,y,f'{path}/{i}')
        elif a!=b:
            print(f'DIFF {path}: {a!r} != {b!r}'); differences += 1
    for key in SCIENTIFIC: diff(a.get(key),b.get(key),key)
    ca,cb = [json.loads(json.dumps(r['config'])) for r in [a,b]]
    for cfg in [ca,cb]:
        cfg.pop('frame_workers',None); cfg.pop('field_workers',None); cfg.pop('worker_nice',None)
        cfg.get('segmentation',{}).pop('workers',None)
    diff(ca,cb,'semantic_config')
    pa,pb = [{p.name:p for p in (folder/'labels').glob('*.npz')} for folder in [reference,candidate]]
    diff(sorted(pa),sorted(pb),'label_files')
    for name in sorted(pa.keys() & pb.keys()):
        with np.load(pa[name]) as x, np.load(pb[name]) as y:
            diff(sorted(x.files),sorted(y.files),name+'/arrays')
            for key in set(x.files)&set(y.files):
                if x[key].dtype!=y[key].dtype or x[key].shape!=y[key].shape or not np.array_equal(x[key],y[key]):
                    print(f'DIFF {name}/{key}: arrays {x[key].shape}/{x[key].dtype} != {y[key].shape}/{y[key].dtype}'); differences+=1
    summary={'equivalent':differences==0,'differences':differences,'label_planes':len(pa),
             'observations':len(a['observations']),'edges':len(a['edges']), 'founders':len(a['founders']),
             'founder_events':len(a.get('founder_events',{}))}
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Exact scientific-output comparison; ignores only run provenance and scheduling settings.')
    parser.add_argument('reference'); parser.add_argument('candidate')
    args=parser.parse_args()
    raise SystemExit(0 if compare(args.reference,args.candidate)['equivalent'] else 1)
