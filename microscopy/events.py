"""Separate biological event evidence from observation limits."""

from collections import defaultdict


def quality_map(data):
    result = {
        (q["field"], q["frame"], q["channel"]): q["usable"] for q in data["quality"]
    }
    result.update(
        {
            (q["field"], q["frame"], q["channel"]): q["usable"]
            for q in data["quality_overrides"]
        }
    )
    return result


def summarize(project, data):
    byid = {n["id"]: n for n in data["observations"]}
    children = defaultdict(list)
    usable = quality_map(data)
    structure = data["config"]["structure_channel"]
    for a, b in data["edges"]:
        if all(
            usable.get((byid[n]["field"], byid[n]["frame"], structure), True)
            for n in [a, b]
        ):
            children[a].append(b)
    required = int(data["config"].get("tracking", {}).get("confirmation_frames", 1))

    def supported(oid):
        for _ in range(required):
            next_ids = children[oid]
            if len(next_ids) != 1:
                return False
            oid = next_ids[0]
        return True

    colony_events = data.get("founder_events") or {}
    run_usable = {(q["field"], q["frame"], q["channel"]): q["usable"] for q in data["quality"]}
    output = []
    for founder in data["founders"]:
        root = founder["observation_id"]
        node = byid[root]
        field = founder["field"]
        frames = sorted(
            r["frame"]
            for r in project.manifest
            if r["field"] == field and r["channel"] == structure
        )
        last = frames[-1]
        current = {root}
        reachable = {root}
        division = None
        end = node["frame"]
        reason = "recording_end"
        has_gap = False
        flags = []
        if not usable.get((field, end, structure), True):
            current = set()
            reason = "baseline_unusable"
        event = colony_events.get(founder["id"]) if current else None
        if event is not None:
            late = sorted(
                q["frame"] for q in data["quality_overrides"]
                if q["field"] == field and q["channel"] == structure and not q["usable"]
                and node["frame"] <= q["frame"] <= event["upper_frame"]
                and run_usable.get((field, q["frame"], structure), True)
            )
            if late:
                # Frames excluded in review after the run: the cached automatic call used them.
                event = {**event, "decision": "lost", "tracking_end_reason": "frames_excluded_after_run",
                         "lower_frame": node["frame"], "upper_frame": late[0],
                         "last_identity_frame": node["frame"],
                         "flags": [*event.get("flags", []), "rerun_needed"]}
        if event is not None:
            # Colony-following run: the event was decided while segmenting (microscopy.colony).
            current = set()
            reason = event["tracking_end_reason"]
            end = event["last_identity_frame"]
            reachable = {root, *(i for i in event["observation_ids"] if i in byid)}
            flags = list(event.get("flags", []))
            has_gap = bool(event.get("contains_gap"))
            if event["decision"] == "no_division" and (has_gap or founder["delayed_start"]):
                # Same rule as linked tracks: absence of division needs the complete, continuous window.
                event = {**event, "decision": "lost", "tracking_end_reason": "incomplete_window"}
                reason = "incomplete_window"
            if event["decision"] == "division":
                division = (event["lower_frame"], event["upper_frame"])
        while current:
            end = min(byid[n]["frame"] for n in current)
            if end >= last:
                break
            next_ids = set()
            for oid in current:
                cs = children[oid]
                if not cs:
                    reason = "identity_lost"
                    break
                if len(cs) == 2:
                    if not all(supported(c) for c in cs):
                        reason = "unconfirmed_division_candidate"
                        break
                    if division is None:
                        division = (
                            byid[oid]["frame"],
                            max(byid[c]["frame"] for c in cs),
                        )
                has_gap = has_gap or any(
                    byid[c]["frame"] - byid[oid]["frame"]
                    != data["config"].get("expected_frame_step", 1)
                    for c in cs
                )
                next_ids.update(cs)
            if reason != "recording_end":
                break
            reachable.update(next_ids)
            current = next_ids
        decision = event["decision"] if event is not None else (
            "division"
            if division
            else (
                "no_division"
                if end == last
                and reason == "recording_end"
                and not has_gap
                and not founder["delayed_start"]
                else "lost"
            )
        )
        if event is not None:
            lower, upper = event["lower_frame"], event["upper_frame"]
        else:
            lower, upper = (
                division if division else (end, next((f for f in frames if f > end), end))
            )
        if data["config"].get("analysis_scope") == "baseline_only":
            decision = "not_assessed"
            reason = "baseline_only"
            lower = upper = founder["baseline_frame"]
        override = data["event_overrides"].get(founder["id"])
        if override:
            decision = override["decision"]
            if decision in {"division", "lost"}:
                lower, upper = override["lower_frame"], override["upper_frame"]
            elif decision == "no_division":
                lower = upper = last
        if founder.get("excluded_reason") and not override:
            decision = "exclude"

        def seconds(frame):
            return project.frames[(field, frame, structure)]["time_s"]

        lo, hi = seconds(lower), seconds(upper)
        last_nodes = [byid[n] for n in reachable if byid[n]["frame"] == end]
        last_x = (
            sum(n["x"] for n in last_nodes) / len(last_nodes)
            if last_nodes
            else node["x"]
        )
        last_y = (
            sum(n["y"] for n in last_nodes) / len(last_nodes)
            if last_nodes
            else node["y"]
        )
        row = {
            **founder,
            "last_x": last_x,
            "last_y": last_y,
            "initial_x": node["x"],
            "initial_y": node["y"],
            "contains_tracking_gap": has_gap,
            "condition": project.config.get("conditions", {}).get(field, "Unassigned"),
            "biological_replicate": project.config.get("biological_replicates", {}).get(
                field
            ),
            "decision": decision,
            "reviewed": bool(override),
            "lower_frame": lower,
            "upper_frame": upper,
            "lower_s": lo,
            "upper_s": hi,
            "division_midpoint_s": (
                (lo + hi) / 2
                if decision == "division" and lo is not None and hi is not None
                else None
            ),
            "last_identity_frame": end,
            "tracking_end_reason": reason,
            "flags": flags,
            "observation_ids": sorted(reachable, key=lambda n: (byid[n]["frame"], n)),
            "note": override.get("note", "") if override else "",
            "review_required": not bool(override),
        }
        output.append(row)
    return output
