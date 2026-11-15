> **Historical plan.** Written on 2026-11-08 for the founder-review viewer, before the PC work was considered. Kept for reference; it has not been checked against the current `microscopy/workbench` code, so parts may already be done or superseded. See `pc_pipeline_suggestions.md` for the later suggestions.

# Microscopy viewer: implementation plan (v2)

Date: 2026-11-08. This replaces the earlier six-milestone draft.

## The request

> "I just wanted a more general version of the previous viewer (faster would've been nice but I'd settle for not super slow)."

What gets delivered: **the previous founder-review viewer, with the same layout, controls and workflow, running on the current experiment and on later projects the microscopy package produces. It must be at least as responsive as before, and existing reviews stay intact.** Everything else comes second.

Done means:

- You open Pilot_1, 2 or 3, and the viewer looks and behaves like the old one. That covers:
  - founder search and filters
  - play, step and the frame slider
  - jumps to the current call, the original call and the final frame
  - phase, PI overlay and PI alone
  - crop width, full field and click-to-recenter
  - outlines
  - the four decisions, with their timing buttons
  - notes, confidence and reviewer
  - save, save all drafts and revert
  - export and import
  - regenerating the plots and the audit PDF
- Frame stepping and founder switching feel immediate (targets in step 4). The frame label and the image on screen always match.
- None of the old viewer's KLY-specific assumptions are hardcoded.
- Reviews already in the Pilot databases survive, and new reviews go to the same place.

## Not in this plan

These are deferred until you ask for them:

- new analyses or statistics
- reworking the PI-change ("uptake") metric
- tracking accuracy, including the low division yield
- a second tracking adapter
- Napari
- features the old viewer lacked, such as choosing which frame a fluorescence measurement uses

The existing analysis scripts keep producing the results. The viewer only triggers them.

## Approach

**Keep the old viewer's front end and server request flow. Replace only its data layer.**

- **Front end:** `index.html`, `style.css`, `viewer.js` and `manual_ui.js` come back essentially unchanged.
- **Server:** the old `server.py` endpoints come back, along with `core.py`'s server-side crop rendering and caching.
- **Data layer:** the old `Project` class is the only part that knows about files. It is replaced by an adapter class with the same methods: `all_effective`, `effective`, `save_override`, `revert`, `import_reviews`, `outline_data`, `center`, `frame_info`, `raw`, `crop_arrays`, `image`, and split/undo split. The adapter reads the microscopy package's runs, manifests and masks, and writes through its SQLite review API.

Why this route:

- The old app is small: about 130 KB of code, with a UI that talks to the server through about a dozen endpoints. If those endpoints stay the same, the UI barely changes.
- Replacing one class is a narrow change, and easy to check.
- Each project keeps one review store: its current SQLite revision history. The old KLY experiment keeps its own JSON reviews and its own installed viewer. Nothing is migrated in either direction.

The new `microscopy/web/*` and the new viewer routes in `microscopy/server.py` are retired once you accept the replacement. The package's storage, review, events, measurement, images and analysis modules stay.

The work goes on a new branch off `codex/microscopy-framework` and uses copied projects on separate ports. The following stay untouched until the switch-over step: the live servers (8765, 8881–8883), the live projects and any unsaved browser drafts. Nothing is committed or merged without your say-so.

## Steps

### 1. Safety setup (small)

- Compare the live review revisions with `STATE_SNAPSHOT.json`. If anything changed, take new SQLite online backups under new filenames.
- Copy the Pilot_1 and Pilot_3 (compact run) projects as development fixtures. Raw TIFFs are only ever read.
- Time the current viewer's frame step and founder switch, so that any speedup is measured rather than claimed.

### 2. Read-only prototype, then checkpoint 1 (medium)

Run the old UI through the adapter on one Pilot_1 field, with saving switched off. It should cover:

- the founder list and filters
- playback
- phase, PI overlay and PI alone
- crop width, full field and recentering
- outlines
- the filmstrip

Time it against the step 1 baseline.

**Checkpoint 1:** you spend ten minutes with it and answer one question: "Is this the old viewer?" If the adapter route turns out to be awkward, it shows up here, before any saving code exists.

### 3. Remove the old experiment-specific assumptions (medium)

Each KLY-specific setting is replaced by project metadata:

| Old assumption | Replacement |
|---|---|
| 100 frames at a 10-minute cadence | The frame count per field and the actual timestamps, read from the manifest |
| Only phase and mCherry | The channels listed in the manifest. A channel missing at a frame (such as Pilot_3 YFP) shows "not acquired", not a black image or a zero |
| Conditions A–D and the Pos0 "diagnostic" label | The condition map and per-field notes from the project config |
| Masks on a 2× reduced grid | Full-resolution masks, with the scale taken from the run metadata |
| 100 ms PI normalization and a fixed exposure table | The exposure of each image, from metadata. This affects display only; stored measurements are unchanged |
| Images of 2048×2048 | The image shape from metadata |
| No frame exclusions | Excluded frames (Pilot_2 frame 39 onward; Pilot_1 Pos0 frames 19–20) are marked on the slider, and tracked outlines are not drawn across them |

Founder IDs are qualified by project and run, because IDs such as `f0-t0-o1` repeat across takes.

### 4. Speed (medium)

Server:

- **Load the review data once, not per request.** The run and its reviews are loaded once per review revision and kept in memory, indexed by (field, frame) and by founder. Today the server reloads and replays the whole graph for every frame and every thumbnail.
- **Send small crops.** The server renders crops at display size from cached raw images, using the old `crop_arrays` path. The full field is rendered only when you ask for it. No more full 2048×2048 PNGs to show a 160-pixel crop.
- **Cache images separately from decisions.**
  - The raw-image cache holds enough frames for a filmstrip across all channels; today it holds 12.
  - Rendered crops are cached by image, crop and display settings, separately from masks and decisions. Editing a note doesn't re-render any pixels.
- **Let the browser cache crops.** A crop never changes once rendered, but today every response is marked `no-store`.
- **Never load Pilot_3's full graph.** Pilot_3 uses the compact review run `run-3bf6b27a41c7`; the viewer never loads the 1 GB source graph. Background measurement still counts every object in each segmentation (label) image.

Browser (the old playback sequencing stays; these are the fixes needed for correctness):

- Each image request records its own founder, frame, channel, crop and display settings. Responses that arrive late are discarded, and rapid stepping is merged into a single request.
- The image, outline, frame/time label and editing target update together. "Division here" and "Loss here" use the frame actually on screen.
- Filmstrip slots are laid out in frame order before any image loads. Each slot belongs to one request, and the main image loads before the thumbnails.
- Changing opacity or contrast doesn't re-download images that are already loaded.

Targets on this Mac, with images already cached: frame step or overlay change within 200 ms (p95), and founder switch to main image within 500 ms (p95). On a cold load the viewer shows a clear loading state. Measured numbers are reported against both the current viewer and the old one.

### 5. Saving to the current review store, then checkpoint 2 (medium)

Each old action maps onto the review API:

- **Decisions:** the decision, timing frames, notes, confidence, reviewer, the "reviewed" flag and full-window verification all go through `set_event`. Fields the schema lacks (confidence, reviewer, saved crop center and width) are added without changing the meaning of existing records.
- **Revert to original call:** the store only has a global "undo last operation". A new per-founder revert operation restores that founder's original call and keeps the history. Global undo is not used as a stand-in.
- **Other actions:** "save all drafts" and export/import (which already check the run ID and manifest digest) both work. If two tabs edit the same founder, the later save is refused as stale instead of overwriting the first.
- **Drafts:** drafts stay in the browser until you press Save, as in the old viewer. The selected founder, frame and crop stay in place after a save.

**Checkpoint 2:** you review a field or two of real founders on a copy. After another backup, the viewer is then pointed at the real Pilot projects.

**Stop point:** after step 5 you have the old viewer, generalized, fast, and saving to the current store. That is the original request. Steps 6–8 bring back the correction tools and results buttons, then switch you over.

### 6. Correction tools: split, outlines and frame exclusion (medium)

Bring back three tools, using `split_founder`, `set_mask` and `set_frame_quality`, in the old viewer's expandable panel:

- splitting an initial two-cell object, and undoing the split
- drawing a cell outline on a later frame
- excluding a frame

When a founder is split, each child gets its own identity, mask and baseline PI. The parent is excluded, and neither child inherits its track or division call. This is the old viewer's behavior.

One backend change must come first. Today `set_mask`, `split_founder`, `set_links` and `set_frame_quality` (for the structure channel) **clear every confirmed event in the whole run**. With these tools in the viewer, one outline fix could silently erase all your other decisions. The fix:

- **Limit what is cleared.** Only founders whose track includes the edited cell or frame are cleared or flagged:
  - a split affects the parent
  - an outline affects that founder
  - a frame exclusion affects the founders seen in that field at that frame
- **Remeasure neighbours instead of clearing them.** If a neighbouring cell's background ring includes the edited object, its PI is remeasured, but its decision is kept.
- **Show the effect before saving.** For example: "This will reset 3 reviewed decisions in Pos4." You confirm before it is saved.
- **Test each case** on fixtures, including a check that unrelated decisions survive.

### 7. Results buttons (small)

**Regenerate plots** and **Regenerate audit PDF** run the existing pipeline on a saved review revision: the `analysis.py` result snapshots, `build_report.py` and the five-condition plotting scripts. The linear/log selectors for the PI and lag axes stay. Each results folder records the review revision it used, and older results stay where they are. The science is unchanged.

### 8. Switch-over (small)

**Checkpoint 3:** you use the new viewer on all three pilots. Then:

- Check the live sessions for unsaved drafts, and save or export them first. Browser storage is never cleared.
- Replace the launcher (`launch_review.py`, plus a double-click shortcut like the old `Start_Review.command`). The current viewer stays runnable for rollback. The old KLY viewer is left as it is.
- Update the repo skill and docs: how to open a project, where reviews and results live, and how to back up and restore.

## Acceptance checks

These are demonstrated in a real browser on all three pilots, not only through unit tests:

- **Race tests:**
  - Select founders A, then B, then C, with their responses deliberately returned in reverse order: everything on screen belongs to C.
  - Scrub or change the overlay while the filmstrip is loading: frame labels stay unique and in order, with no images from other founders or settings.
  - Press "Division here" while an image is still loading: it records the frame shown on screen.
- **Nothing written by browsing:** viewing, playing, filtering and changing contrast make no review changes.
- **Saving:** save, reopen, revert, export/import and a two-tab conflict all behave correctly, and the review history stays intact.
- **Outlines:**
  - Manual outlines land on the same raw pixels under any crop, full-field view or recentering.
  - After tracking is lost, no outline appears on a neighbouring cell, and frames shown only for context are labelled as such.
- **Edge cases:** missing YFP frames, excluded frames, irregular timestamps and the last frame are shown plainly.
- **Results:**
  - An outline edit leaves unrelated decisions alone.
  - Changes to the viewer alone leave the frozen result CSVs unchanged.
- **Speed and tests:**
  - Latency is recorded against the step 1 baseline.
  - `python -m pytest microscopy/tests plate_reader/tests` passes, and new tests reproduce the scrambled-filmstrip and stale-image bugs.

## What "general" means here

Any project produced by the microscopy package opens without code changes. Field names, frame counts, timing, channels and image size all come from metadata. This is shown on:

- the three pilots: 25, 52 and 163 frames, different cadences and exposures, and YFP on only some frames
- the package's synthetic demo, which uses different names and image sizes

The next real acquisition will be the real test, but this plan can be finished without it.

## Open questions (none block the start)

- Should the old KLY experiment eventually open in this viewer too, or stay on its own installed copy? The plan assumes it stays.
- Is there anything from the new viewer you liked and want kept? I'll ask at checkpoint 1.
