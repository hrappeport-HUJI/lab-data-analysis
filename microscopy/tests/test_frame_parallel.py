"""Scheduling contracts: completion order never determines scientific replay order."""
from concurrent.futures import Future
import numpy as np

from microscopy.demo import create
from microscopy.pipeline import run, _bounded_results, _frame_jobs, _field_plan
from microscopy.images import read_frame
from microscopy.adapters.watershed import auto_threshold


def test_bounded_queue_consumes_completed_tasks_and_round_robins(monkeypatch):
    import microscopy.pipeline as pipeline
    class Pool:
        def __init__(self):
            self.outstanding = 0
            self.peak = 0
        def submit(self, function, job):
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
            f = Future()
            f.set_result(function(job))
            return f
    pool = Pool()
    def reverse_wait(pending, **kwargs):
        # Force newest task to finish first; an ordered future.result loop would fail.
        chosen = max(pending, key=lambda f: f.result())
        pool.outstanding -= 1
        return {chosen}, pending - {chosen}
    monkeypatch.setattr(pipeline, 'wait', reverse_wait)
    assert list(_bounded_results(pool, int, range(7), 3)) == [2, 3, 4, 5, 6, 1, 0]
    assert pool.peak == 3
    plans = [{'field': f, 'records': list(range(n)), 'parameters': {}} for f,n in [('a',3),('b',1)]]
    assert [(j[1],j[2]) for j in _frame_jobs(plans,{})] == [('a',0),('b',0),('a',1),('a',2)]


def test_worker_count_exclusions_resume_and_sidecars(tmp_path):
    p = create(tmp_path / 'demo')
    p.config['tracking']['events'] = 'colony'
    p.config['quality']['excluded_structure_frames'] = {'1': [0, 3], '2': [2]}
    p.config['founders']['allow_delayed_start'] = True
    p.config['segmentation']['threshold_from_first_frame'] = True
    serial = run(p, fields=['1','2'])
    p.config['frame_workers'] = 2
    p.config['field_workers'] = 2
    parallel = run(p, fields=['1','2'])
    # Stale sidecars from another run must never be trusted on resume.
    for marker in (p.run_path(parallel['run_id'])/'prepared').glob('*.json'):
        marker.write_text('{"schema_version": -1}')
    resumed = run(p, fields=['1','2'], resume_from=parallel['run_id'])
    for result in [parallel, resumed]:
        for key in ['founders','observations','edges','quality','founder_events','field_thresholds','label_planes']:
            assert result[key] == serial[key]
        left = p.run_path(serial['run_id'])/'labels'
        right = p.run_path(result['run_id'])/'labels'
        assert {f.name for f in left.glob('*.npz')} == {f.name for f in right.glob('*.npz')}
        for plane in left.glob('*.npz'):
            with np.load(plane) as a, np.load(right/plane.name) as b:
                assert np.array_equal(a['labels'],b['labels'])
    assert not (p.run_path(parallel['run_id'])/'labels'/'0000_000000000.npz').exists()
    only_two = run(p, fields=['2'], resume_from=parallel['run_id'])
    assert all(o['field']=='2' and o['id'].startswith('f0-') for o in only_two['observations'])
    assert all(o['labels_file'].startswith('labels/0000_') for o in only_two['observations'])


def test_threshold_precedence_and_excluded_baseline(tmp_path):
    p = create(tmp_path/'demo')
    cfg = p.config
    cfg['segmentation'].update(threshold=99999, threshold_from_first_frame=True)
    cfg['quality']['excluded_structure_frames'] = {'1':[0]}
    plan = _field_plan(p,cfg,False,'1')
    expected = auto_threshold(read_frame(p,'1',0,'phase'), float(cfg['segmentation']['background_sigma_px']))
    assert plan['threshold_used'] == expected
    assert plan['parameters']['threshold_by_field']['1'] == expected
    cfg['segmentation']['threshold_by_field'] = {'1':123}
    plan = _field_plan(p,cfg,True,'1')
    assert plan['threshold_used'] is None
    assert plan['parameters']['threshold_by_field']['1'] == 123
    assert [r['frame'] for r in plan['records']] == [0]


def test_atomic_json_retries_windows_reader_and_cleans_up(tmp_path, monkeypatch):
    import microscopy.storage as storage
    from types import SimpleNamespace
    original = storage.os.replace
    failures = []
    def busy_once(source, target):
        failures.append(1)
        if len(failures) == 1:
            raise PermissionError("concurrent reader")
        original(source, target)
    # Simulate Windows without changing the shared os module or pathlib's platform.
    monkeypatch.setattr(storage, "os", SimpleNamespace(
        name="nt", fdopen=storage.os.fdopen, fsync=storage.os.fsync,
        replace=busy_once, path=storage.os.path, unlink=storage.os.unlink))
    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    storage.write_json(tmp_path/'progress.json', {"done": 2})
    assert len(failures) == 2
    assert (tmp_path/'progress.json').read_text().strip().endswith('2\n}')
    assert not list(tmp_path.glob('.write-*'))
    def always_busy(*args):
        raise PermissionError("persistent permission failure")
    storage.os.replace = always_busy
    import pytest
    with pytest.raises(PermissionError, match="persistent"):
        storage.write_json(tmp_path/'progress.json', {"done": 3})
    assert not list(tmp_path.glob('.write-*'))
