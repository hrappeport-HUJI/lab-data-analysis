// links: "Open in ..." links on the clicked point's card.
// features.links = [{"name": "Open in review viewer", "url": "http://host:8891/?founder={id}"}]; {column} placeholders, {id} = row id.
viewerModule("links", {
  pickActions(cfg, r, head) {
    for (const l of Array.isArray(cfg) ? cfg : []) {
      const a = document.createElement("a");
      a.href = l.url.replace(/\{(\w+)\}/g, (_, k) => encodeURIComponent(k === "id" ? r.id : (r[k] ?? "")));
      a.target = "_blank"; a.rel = "noopener"; a.textContent = (l.name || "Open") + " ↗"; head.append(a);
    }
  },
});
