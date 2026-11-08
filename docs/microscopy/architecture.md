# Architecture and adapter contract

The scientific record has four separate layers:

1. **Acquisition:** immutable image manifest with field/frame/channel, raw path, shape, dtype, timestamp, exposure and content hash. No condition-specific branching in image identity.
2. **Algorithm run:** configuration/provenance, labeled masks, observations, parent→child edges and candidate founders. Each `run` gets a new ID. Code/package/dependency versions and local Python source hashes are recorded.
3. **Review revision:** operations applied over one run. SQLite serializes writers, requires the expected revision, and keeps historical states. Edits never rewrite acquisition images or original algorithm output.
4. **Analysis snapshot:** materialized review state plus current measurement/grouping/statistical configuration. Every regeneration writes a new directory. The review revision identifies the evidence used.

Python modules keep importing, quality scoring/image reading, segmentation/tracking, review replay, event interpretation, fluorescence, statistics and presentation separate. The first public plugin interfaces are segmentation and tracking; readers, QC and analyses are ordinary replaceable Python modules rather than a speculative universal plugin protocol. The browser consumes JSON endpoints and does no scientific numerical measurement.

## Segmentation

```python
class Segmenter:
    def segment(self, image, *, context, parameters):
        # image: raw two-dimensional NumPy array
        # context: manifest image record; field/frame/channel/time/shape etc.
        # parameters: configuration['segmentation']
        return labels  # same Y,X shape; nonnegative integer IDs; 0 is background
```

Label IDs need only be unique within one image. The pipeline converts them into run-local observation IDs, records centroids/area/bounding boxes, and stores original label arrays. Coordinates are original image pixels: x=column, y=row, origin at upper left. Polygon review vertices use x,y. Geometry remains in pixels; no unverified micrometer conversion occurs.

## Tracking

```python
class Tracker:
    def link(self, previous, current, *, previous_labels,
             current_labels, parameters):
        # previous/current: observation dictionaries with id, field, frame,
        # x, y, area_px, bbox, label, time_s and labels_file.
        return [(parent_id, child_id), ...]
```

The adapter links one adjacent pair of usable structure frames. Zero edges means abstention/loss. One parent can have up to two children; each child at most one parent. Same-field forward-time edges only. The adapter must not infer condition labels or write reviews. The current interface does not accommodate global multi-frame trackers directly: wrap their output into pairwise edges, or introduce a separate sequence adapter without changing stored graph semantics. Gap-spanning manual edges are allowed, but flagged and do not establish automatic whole-recording absence of division.

Use programmatically without installation hooks:

```python
from microscopy.pipeline import run
from microscopy.adapters.classical import ThresholdSegmenter, CentroidTracker
run('/data/review-project', segmenter=ThresholdSegmenter(),
    tracker=CentroidTracker(), fields=['1'])
```

Or register a local adapter with `register_segmenter(name, AdapterClass)` / `register_tracker(name, AdapterClass)` before invoking `run`. Installed packages can expose class factories through entry points:

```toml
[project.entry-points."lab_microscopy.trackers"]
my_tracker = "my_package:Tracker"
```

Then set `tracking.adapter` to `my_tracker`. The plugin is trusted local Python code; the framework does not download models or execute code paths from experiment metadata. For custom adapters, retain the exact adapter package version/source with the project. The run records the class name and core environment, but cannot automatically fingerprint every transitive dependency or external model weight. Add model checksum/version to adapter parameters so it is preserved.

## Review operations

`microscopy.review.save(project, run_id, operation, expected_revision)` validates/replays the proposed state before committing. The browser uses the same path. Supported operations:

| Type | Required payload |
|---|---|
| `set_event` | `founder_id`, `decision`; `lower_frame`,`upper_frame` for division/lost; `full_window_verified=true` for no_division; optional `note` |
| `set_mask` | `observation_id`, `polygon` (list of x,y vertices) |
| `add_observation` | unique `observation_id` prefixed `manual-`, `field`, `frame`, `polygon` |
| `split_founder` | `founder_id`, exactly two disjoint `polygons` |
| `set_links` | parent `observation_id`, `children` (0–2 IDs); replaces outgoing links |
| `set_frame_quality` | `field`, `frame`, `channel`, boolean `usable` |
| `set_measurement_frame` | `founder_id`, `channel`, `frame` |
| `undo` | no additional fields; removes the last active operation in a new revision |
| `import` | `review` from the same run/manifest, plus optimistic revision check |

Split parent IDs remain available for audit but are excluded from measurements. Child founders start without links; baseline fluorescence is measured separately inside their polygons. Other unrelated initial founders cannot be overlapped by a split. The editor currently supports a two-way initial split, not arbitrary multi-cell colony partitioning. A complex new morphology can be supplied by a segmenter adapter.

Current storage is deliberately simple: JSON, NumPy label files, SQLite and CSV. It is suited to bounded field/time-series projects, not a multi-terabyte hosted archive. Image and measurement-mask caches are bounded; graph/review materialization currently loads a run into memory, and long review histories are replayed. Add indexed/columnar storage only when a measured project-size bottleneck warrants it. A future napari front end can use the same review API rather than maintaining a separate annotation truth.
