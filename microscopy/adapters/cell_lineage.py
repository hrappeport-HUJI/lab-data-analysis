"""Drift-corrected per-cell links with conservative overlap ownership and short gaps."""
import numpy as np
from microscopy.colony import field_shift, _shift
from . import register_tracker


class CellLineageTracker:
    """State belongs to one field replay; reset before starting another field.

    Only overlap-supported links are accepted. Competing parents are ranked by
    child coverage; near ties abstain. This sparse assignment never allocates an
    N-by-N matrix. Missing footprints survive one frame, behind observed cells.
    """
    def reset(self):
        self.carry = None
        self.nodes = {}

    def link(self, previous, current, *, previous_labels, current_labels, parameters):
        if not hasattr(self, 'carry'):
            self.reset()
        dy, dx = field_shift(previous_labels, current_labels,
                             parameters.get('drift_radius_px', 30))
        prior = _shift(previous_labels, dy, dx).astype(np.int64)
        parents = {n['label']: n for n in previous}
        age = {k: 0 for k in parents}
        if self.carry is not None:
            carried = _shift(self.carry, dy, dx)
            offset = int(prior.max()) + 1
            for k, node in self.nodes.items():
                key = k + offset
                parents[key] = node
                age[key] = 1
            mask = (carried > 0) & (prior == 0)
            prior[mask] = carried[mask] + offset
        children = {n['label']: n for n in current}
        both = (prior > 0) & (current_labels > 0)
        base = int(current_labels.max()) + 1
        pairs, counts = np.unique(prior[both] * base + current_labels[both], return_counts=True)
        options = {}
        for pair, count in zip(pairs, counts):
            a, b = divmod(int(pair), base)
            if a not in parents or b not in children:
                continue
            p, c = parents[a], children[b]
            coverage = count / c['area_px']
            if coverage < parameters.get('min_overlap_fraction', .2):
                continue
            if count / p['area_px'] < .08:
                continue
            options.setdefault(b, []).append((coverage, a, int(count)))
        owned = {}
        for b, opts in options.items():
            opts.sort(reverse=True)
            if len(opts) > 1:
                children[b].setdefault('tracking_flags', []).append('competing_overlap')
                if opts[0][0] - opts[1][0] < parameters.get('assignment_margin', .15):
                    for _, a, _ in opts:
                        parents[a].setdefault('tracking_flags', []).append('ambiguous_assignment')
                    continue
            _, a, count = opts[0]
            owned.setdefault(a, []).append((b, count))
        edges, linked = [], set()
        low, high = parameters.get('division_area_ratio', [.65, 1.6])
        for a, matches in owned.items():
            p = parents[a]
            accepted = []
            if len(matches) == 1:
                b, count = matches[0]
                ratio = children[b]['area_px'] / p['area_px']
                if .45 <= ratio <= 1.8:
                    accepted = [b]
            elif len(matches) == 2 and age[a] == 0:
                bs = [b for b, _ in matches]
                ratios = [children[b]['area_px'] / p['area_px'] for b in bs]
                if low <= sum(ratios) <= high and all(.2 <= r <= .85 for r in ratios):
                    accepted = bs
            if accepted:
                linked.add(a)
                for b in accepted:
                    edges.append((p['id'], children[b]['id']))
                    if age[a]:
                        children[b].setdefault('tracking_flags', []).append('gap_bridge')
        self.nodes = {}
        lut = np.zeros(max(int(prior.max()), max(parents, default=0)) + 1, np.int64)
        for a, p in parents.items():
            if a not in linked and age[a] == 0:
                lut[a] = a
                self.nodes[a] = p
        self.carry = lut[prior]
        self.carry[current_labels > 0] = 0
        return edges


register_tracker('cell_lineage', CellLineageTracker)
