import numpy as np
from scipy import ndimage
from microscopy.adapters.cell_lineage import CellLineageTracker


def nodes(labels, frame):
    result=[]
    for k, box in enumerate(ndimage.find_objects(labels),1):
        if box is None: continue
        y,x=np.where(labels==k)
        result.append(dict(id=f'{frame}-{k}',field='1',frame=frame,label=k,
                           x=float(x.mean()),y=float(y.mean()),area_px=len(x),
                           bbox=[x.min(),y.min(),x.max()+1,y.max()+1]))
    return result


def link(t,a,b,frame=0):
    return t.link(nodes(a,frame),nodes(b,frame+1),previous_labels=a,current_labels=b,parameters={})


def test_drift():
    a=np.zeros((120,120),int)
    for k,(y,x) in enumerate([(10,10),(50,15),(80,80)],1): a[y:y+8,x:x+16]=k
    b=ndimage.shift(a,(4,6),order=0)
    assert len(link(CellLineageTracker(),a,b))==3


def test_contact_keeps_cell_identities():
    a=np.zeros((40,40),int);a[10:20,5:17]=1;a[10:20,20:32]=2
    b=np.zeros_like(a);b[10:20,7:20]=1;b[10:20,20:32]=2
    assert link(CellLineageTracker(),a,b)==[('0-1','1-1'),('0-2','1-2')]


def test_division():
    a=np.zeros((40,40),int);a[10:20,5:29]=1
    b=a.copy();b[10:20,17:29]=2
    assert set(link(CellLineageTracker(),a,b))=={('0-1','1-1'),('0-1','1-2')}


def test_one_frame_gap():
    a=np.zeros((40,40),int);a[10:20,5:20]=1
    z=np.zeros_like(a);t=CellLineageTracker()
    assert link(t,a,z)==[]
    assert link(t,z,a,1)==[('0-1','2-1')]
    assert link(t,a,z,2)==[]
    assert link(t,z,z,3)==[]
    assert link(t,z,a,4)==[]


def test_equal_overlap_abstains():
    a=np.zeros((40,40),int);a[10:20,5:15]=1;a[10:20,15:25]=2
    b=(a>0).astype(int)
    assert link(CellLineageTracker(),a,b)==[]


def test_persistent_division_uses_confirmation_frame():
    from types import SimpleNamespace
    from microscopy.events import summarize
    obs=[dict(id=k,field='1',frame=f,x=10,y=10) for k,f in
         [('root',0),('a',1),('b',1),('aa',2),('bb',2)]]
    data=dict(observations=obs,edges=[('root','a'),('root','b'),('a','aa'),('b','bb')],
              quality=[],quality_overrides=[],config=dict(structure_channel='phase',
              tracking=dict(adapter='cell_lineage',confirmation_frames=1)),
              founders=[dict(id='root',observation_id='root',field='1',baseline_frame=0,delayed_start=False)],event_overrides={})
    project=SimpleNamespace(manifest=[dict(field='1',frame=f,channel='phase') for f in range(3)],
                            config={},frames={('1',f,'phase'):dict(time_s=f*60) for f in range(3)})
    row=summarize(project,data)[0]
    assert (row['decision'],row['lower_frame'],row['upper_frame'])==('division',0,2)
    data['edges']=data['edges'][:2]
    assert summarize(project,data)[0]['decision']=='lost'


def test_gap_cannot_prove_no_division():
    from types import SimpleNamespace
    from microscopy.events import summarize
    data=dict(observations=[dict(id=k,field='1',frame=f,x=10,y=10) for k,f in [('root',0),('c',2)]],
              edges=[('root','c')],quality=[],quality_overrides=[],
              config=dict(structure_channel='phase',tracking=dict(adapter='cell_lineage')),
              founders=[dict(id='root',observation_id='root',field='1',baseline_frame=0,delayed_start=False)],event_overrides={})
    p=SimpleNamespace(manifest=[dict(field='1',frame=f,channel='phase') for f in range(3)],config={},
                      frames={('1',f,'phase'):dict(time_s=f) for f in range(3)})
    row=summarize(p,data)[0]
    assert row['decision']=='lost' and row['contains_tracking_gap']


def test_drift_can_move_highest_label_out_of_field():
    a=np.zeros((120,120),int)
    for k,(y,x) in enumerate([(10,10),(50,15),(80,80),(115,110)],1): a[y:y+5,x:x+10]=k
    b=ndimage.shift(a,(6,0),order=0)
    assert len(link(CellLineageTracker(),a,b))==3


def test_pipeline_bridges_an_entire_empty_frame(tmp_path):
    import json
    from microscopy.pipeline import _follow_field
    a=np.zeros((40,40),np.int32);a[10:20,5:20]=1
    (tmp_path/'prepared').mkdir();(tmp_path/'labels').mkdir();(tmp_path/'progress').mkdir()
    records=[]
    for f, labels in enumerate([a,np.zeros_like(a),a]):
        ns=nodes(labels,f)
        for n in ns:
            n['bbox']=[int(v) for v in n['bbox']]
            n['labels_file']=f'labels/0000_{f:09}.npz'
        np.savez_compressed(tmp_path/f'labels/0000_{f:09}.npz',labels=labels)
        q=dict(field='1',frame=f,channel='phase',usable=True)
        (tmp_path/f'prepared/0000_{f:09}.json').write_text(json.dumps(dict(schema_version=1,quality=q,nodes=ns,shape=list(labels.shape))))
        records.append(dict(field='1',frame=f,time_s=f))
    cfg=dict(structure_channel='phase',tracking=dict(adapter='cell_lineage'),founders=dict(edge_margin_px=0))
    _, (_,edges,_,_,_,_,_)=_follow_field((cfg,CellLineageTracker(),tmp_path,1,0,dict(field='1',records=records,threshold_used=None)))
    assert edges==[('0-1','2-1')]
