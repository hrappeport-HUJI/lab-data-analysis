'use strict';
const $ = id => document.getElementById(id);
let state, founder, frames = [],
    index = 0,
    selected = null,
    parent = null,
    children = [],
    drawMode = null,
    polygons = [],
    vertices = [],
    timer = null,
    renderID = 0,
    frameNodes = [],
    view = {
        x: 0,
        y: 0,
        w: 128,
        h: 96
    };
const canvas = $('canvas'),
    ctx = canvas.getContext('2d');

function message(s) {
    $('message').textContent = s || '';
}
async function api(path, body) {
    const r = await fetch(path, body ? {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Review-Token': state.token
        },
        body: JSON.stringify(body)
    } : {});
    const data = await r.json();
    if (!r.ok) throw Error(data.error || r.statusText);
    return data;
}

function guarded(fn) {
    return async (...a) => {
        try {
            await fn(...a);
        } catch (e) {
            message(e.message);
        }
    };
}

function on(id, fn) {
    $(id).onclick = guarded(fn);
}

function draftKey() {
    return `founder-draft:${location.port}:${state.run_id}:${founder.id}`;
}

function draft() {
    return {
        decision: $('decision').value,
        lower: $('lower').value,
        upper: $('upper').value,
        note: $('note').value,
        verified: $('verified').checked
    };
}

function saveDraft() {
    if (founder) localStorage.setItem(draftKey(), JSON.stringify(draft()));
}
['decision', 'lower', 'upper', 'note', 'verified'].forEach(id => $(id).oninput = saveDraft);
async function refresh(keep = true) {
    const id = keep && founder?.id;
    state = await api('/api/state');
    $('status').textContent = `${state.run_id} · saved revision ${state.revision} · ${state.founders.length} founders`;
    for (const target of ['channel', 'qualityChannel']) {
        const el = $(target),
            old = el.value;
        el.replaceChildren();
        const options = target === 'channel' ? ['None', ...state.channels] : [state.structure_channel, ...state.channels];
        for (const value of options) {
            const o = document.createElement('option');
            o.value = value;
            o.textContent = value;
            el.append(o);
        }
        el.value = options.includes(old) ? old : options[target === 'channel' && options.length > 1 ? 1 : 0];
    }
    list();
    const next = state.founders.find(f => f.id === id) || state.founders[0];
    if (next) await choose(next);
    else message('No founders detected. Inspect focus flags and segmentation in a new pilot run.');
}

function list() {
    const search = $('search').value.toLowerCase(),
        filter = $('filter').value;
    $('founders').replaceChildren();
    for (const f of state.founders) {
        if (!`${f.id} ${f.field}`.toLowerCase().includes(search) || filter === 'pending' && f.reviewed || !['all', 'pending'].includes(filter) && f.decision !== filter) continue;
        const b = document.createElement('button');
        b.className = 'founder' + (founder?.id === f.id ? ' selected' : '');
        const title = document.createElement('span');
        title.textContent = `Field ${f.field} · ${f.id}`;
        const small = document.createElement('small');
        small.textContent = `${f.condition} · ${f.decision} · ${f.reviewed?'reviewed':'candidate'}`;
        b.append(title, small);
        b.onclick = guarded(() => choose(f));
        $('founders').append(b);
    }
}
async function choose(f) {
    stop();
    founder = f;
    selected = f.observation_id;
    parent = null;
    children = [];
    cancelDraw();
    frames = state.frames.filter(r => r.field === f.field && r.channel === state.structure_channel).sort((a, b) => a.frame - b.frame);
    index = Math.max(0, frames.findIndex(r => r.frame === f.upper_frame));
    $('frame').max = frames.length - 1;
    $('title').textContent = `Field ${f.field} / ${f.id}`;
    $('details').textContent = `${f.condition} · ${f.decision} (${f.reviewed?'reviewed':'automatic candidate'}) · ${f.tracking_end_reason}${f.delayed_start?' · delayed baseline':''}`;
    $('decision').value = f.decision;
    $('lower').value = f.lower_frame;
    $('upper').value = f.upper_frame;
    $('note').value = f.note;
    $('verified').checked = f.reviewed && f.decision === 'no_division';
    const stored = localStorage.getItem(draftKey());
    if (stored) {
        const d = JSON.parse(stored);
        for (const k of ['decision', 'lower', 'upper', 'note']) $(k).value = d[k];
        $('verified').checked = d.verified;
    }
    $('draft').textContent = stored ? 'Restored unsaved draft. Saving records a new review revision.' : 'Unsaved fields are retained in this browser as a draft.';
    list();
    await render();
    await strip();
}

function imageURL(channel, frame, color = 'gray') {
    const q = new URLSearchParams({
        field: founder.field,
        frame,
        channel,
        color,
        scale: color === 'red' ? $('scale').value : 'linear'
    });
    const prefix = color === 'red' ? 'fluor' : 'phase';
    for (const key of ['Min', 'Max'])
        if ($(prefix + key).value !== '') q.set(key.toLowerCase(), $(prefix + key).value);
    return '/api/image?' + q;
}
async function loadImage(url) {
    const r = await fetch(url);
    if (!r.ok) {
        const e = await r.json();
        throw Error(e.error);
    }
    return createImageBitmap(await r.blob());
}

function imageExists(channel, frame) {
    return state.frames.some(r => r.field === founder.field && r.frame === frame && r.channel === channel);
}

function roi(shape) {
    if ($('full').checked) return {
        x: 0,
        y: 0,
        w: shape[1],
        h: shape[0]
    };
    const n = frameNodes.find(n => founder.observation_ids.includes(n.id));
    const center = n || {
        x: founder.initial_x,
        y: founder.initial_y
    };
    const w = Math.min(160, shape[1]),
        h = Math.min(108, shape[0]);
    return {
        x: Math.max(0, Math.min(shape[1] - w, center.x - w / 2)),
        y: Math.max(0, Math.min(shape[0] - h, center.y - h / 2)),
        w,
        h
    };
}

function drawOutline(target, node, region, width, height, color) {
    target.fillStyle = color;
    for (const [x, y] of node.outline) {
        target.fillRect((x - region.x) * width / region.w, (y - region.y) * height / region.h, Math.max(1, width / region.w), Math.max(1, height / region.h));
    }
}
async function paint(target, frame, region, width, height, nodes) {
    const phase = await loadImage(imageURL(state.structure_channel, frame));
    target.clearRect(0, 0, width, height);
    target.imageSmoothingEnabled = false;
    target.drawImage(phase, region.x, region.y, region.w, region.h, 0, 0, width, height);
    phase.close();
    const ch = $('channel').value;
    if (ch !== 'None' && imageExists(ch, frame)) {
        const fl = await loadImage(imageURL(ch, frame, 'red'));
        target.globalCompositeOperation = 'screen';
        target.globalAlpha = Number($('alpha').value);
        target.drawImage(fl, region.x, region.y, region.w, region.h, 0, 0, width, height);
        target.globalAlpha = 1;
        target.globalCompositeOperation = 'source-over';
        fl.close();
    }
    if ($('outlines').checked)
        for (const n of nodes) drawOutline(target, n, region, width, height, n.id === selected ? '#ffb84d' : founder.observation_ids.includes(n.id) ? '#57ff8c' : '#a1abb0');
}
async function render() {
    if (!founder) return;
    const id = ++renderID;
    const r = frames[index];
    $('frame').value = index;
    $('time').textContent = `Frame ${r.frame} · ${r.time_s===null?'time unknown':(r.time_s/60).toFixed(1)+' min'}`;
    const f = await api('/api/frame?' + new URLSearchParams({
        field: founder.field,
        frame: r.frame
    }));
    if (id !== renderID) return;
    frameNodes = f.nodes;
    view = roi(r.shape);
    const buffer = document.createElement('canvas');
    buffer.width = canvas.width;
    buffer.height = canvas.height;
    await paint(buffer.getContext('2d'), r.frame, view, buffer.width, buffer.height, frameNodes);
    if (id !== renderID) return;
    ctx.drawImage(buffer, 0, 0);
    drawPolygons();
    $('selection').textContent = selected ? `Selected: ${selected}` : 'No observation selected';
}
async function strip() {
    const saved = founder.id,
        at = frames.findIndex(r => r.frame === founder.upper_frame);
    $('strip').replaceChildren();
    for (let i = Math.max(0, at - 3); i <= Math.min(frames.length - 1, at + 3); i++) {
        const b = document.createElement('button'),
            c = document.createElement('canvas');
        c.width = 180;
        c.height = 120;
        b.append(c, document.createTextNode(`Frame ${frames[i].frame}`));
        b.onclick = guarded(async () => {
            index = i;
            await render();
        });
        $('strip').append(b);
        const f = await api('/api/frame?' + new URLSearchParams({
            field: founder.field,
            frame: frames[i].frame
        }));
        if (founder.id !== saved) return;
        await paint(c.getContext('2d'), frames[i].frame, view, c.width, c.height, f.nodes);
    }
}

function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    $('play').textContent = 'Play';
}
async function move(delta) {
    index = Math.max(0, Math.min(frames.length - 1, index + delta));
    await render();
}
async function mutate(operation) {
    stop();
    const key = draftKey();
    if (!['undo', 'set_event'].includes(operation.type)) saveDraft();
    await api('/api/review', {
        revision: state.revision,
        operation
    });
    if (['undo', 'set_event'].includes(operation.type)) localStorage.removeItem(key);
    message('Saved. Previous revisions remain in the project.');
    await refresh();
}
on('refresh', () => refresh());
on('prev', () => move(-1));
on('next', () => move(1));
on('play', () => {
    if (timer) {
        stop();
        return;
    }
    $('play').textContent = 'Pause';
    let busy = false;
    timer = setInterval(guarded(async () => {
        if (busy) return;
        if (index === frames.length - 1) {
            stop();
            return;
        }
        busy = true;
        try {
            await move(1);
        } finally {
            busy = false;
        }
    }), 500);
});
on('baseline', async () => {
    index = Math.max(0, frames.findIndex(r => r.frame === founder.baseline_frame));
    await render();
});
on('event', async () => {
    index = Math.max(0, frames.findIndex(r => r.frame === founder.upper_frame));
    await render();
});
$('frame').oninput = guarded(async () => {
    stop();
    index = Number($('frame').value);
    await render();
});
for (const id of ['channel', 'scale', 'alpha', 'full', 'outlines']) $(id).onchange = guarded(async () => {
    await render();
    await strip();
});
on('contrast', async () => {
    await render();
    await strip();
});
$('search').oninput = list;
$('filter').onchange = list;
on('setLower', () => {
    $('lower').value = frames[index].frame;
    saveDraft();
});
on('setUpper', () => {
    $('upper').value = frames[index].frame;
    saveDraft();
});
on('saveEvent', async () => {
    const key = draftKey();
    await mutate({
        type: 'set_event',
        founder_id: founder.id,
        decision: $('decision').value,
        lower_frame: Number($('lower').value),
        upper_frame: Number($('upper').value),
        full_window_verified: $('verified').checked,
        note: $('note').value
    });
    localStorage.removeItem(key);
    $('draft').textContent = 'Decision saved.';
});
on('undo', async () => {
    await mutate({
        type: 'undo'
    });
});

function cancelDraw() {
    drawMode = null;
    polygons = [];
    vertices = [];
    $('drawHelp').textContent = 'Click vertices on the image, then finish each polygon.';
}

function drawPolygons() {
    ctx.strokeStyle = '#ffb84d';
    ctx.lineWidth = 2;
    for (const p of [...polygons, vertices]) {
        if (!p.length) continue;
        ctx.beginPath();
        p.forEach(([x, y], i) => {
            const a = (x - view.x) * canvas.width / view.w,
                b = (y - view.y) * canvas.height / view.h;
            i ? ctx.lineTo(a, b) : ctx.moveTo(a, b);
        });
        if (p !== vertices) ctx.closePath();
        ctx.stroke();
    }
}

function startDraw(mode) {
    cancelDraw();
    stop();
    drawMode = mode;
    $('drawHelp').textContent = mode === 'split' ? 'Draw the first initial cell, finish it, then draw the second.' : 'Draw vertices, finish polygon, then Save outlines.';
}
on('drawMask', () => {
    if (!selected || !frameNodes.some(n => n.id === selected)) throw Error('Select an observation in the current frame first.');
    startDraw('mask');
});
on('drawNew', () => startDraw('new'));
on('drawSplit', async () => {
    index = frames.findIndex(r => r.frame === founder.baseline_frame);
    await render();
    startDraw('split');
});
on('finishPolygon', async () => {
    if (vertices.length < 3) throw Error('Draw at least three vertices.');
    polygons.push(vertices);
    vertices = [];
    await render();
});
on('cancelDraw', async () => {
    cancelDraw();
    await render();
});
canvas.onclick = guarded(async e => {
    const rect = canvas.getBoundingClientRect(),
        x = view.x + (e.clientX - rect.left) / rect.width * view.w,
        y = view.y + (e.clientY - rect.top) / rect.height * view.h;
    if (drawMode) {
        vertices.push([x, y]);
        await render();
        return;
    }
    let best = null,
        distance = Infinity;
    for (const n of frameNodes) {
        const d = Math.min(...n.outline.map(p => (p[0] - x) ** 2 + (p[1] - y) ** 2));
        if (d < distance) {
            best = n;
            distance = d;
        }
    }
    selected = distance < 225 ? best?.id : null;
    await render();
});
on('saveDrawing', async () => {
    if (vertices.length) throw Error('Finish the current polygon first.');
    let op;
    if (drawMode === 'split') {
        if (polygons.length !== 2) throw Error('Draw exactly two polygons.');
        op = {
            type: 'split_founder',
            founder_id: founder.id,
            polygons
        };
    } else {
        if (polygons.length !== 1) throw Error('Draw exactly one polygon.');
        op = {
            type: drawMode === 'mask' ? 'set_mask' : 'add_observation',
            observation_id: drawMode === 'mask' ? selected : 'manual-' + crypto.randomUUID(),
            field: founder.field,
            frame: frames[index].frame,
            polygon: polygons[0]
        };
    }
    await mutate(op);
    cancelDraw();
});
on('rememberParent', () => {
    if (!selected) throw Error('Select the parent observation.');
    parent = selected;
    children = [];
    $('parent').textContent = `Parent: ${parent}`;
});
on('addChild', () => {
    if (!selected || selected === parent) throw Error('Choose a later child observation.');
    if (!children.includes(selected)) children.push(selected);
    $('children').textContent = 'Children: ' + children.join(', ');
});
on('clearChildren', () => {
    children = [];
    $('children').textContent = 'No children selected; saving cuts outgoing links.';
});
on('saveLinks', async () => {
    if (!parent) throw Error('Select a link parent first.');
    await mutate({
        type: 'set_links',
        observation_id: parent,
        children
    });
});
async function quality(usable) {
    await mutate({
        type: 'set_frame_quality',
        field: founder.field,
        frame: frames[index].frame,
        channel: $('qualityChannel').value,
        usable
    });
}
on('badFrame', () => quality(false));
on('goodFrame', () => quality(true));
on('measurementFrame', async () => {
    if ($('channel').value === 'None') throw Error('Select a fluorescence channel.');
    await mutate({
        type: 'set_measurement_frame',
        founder_id: founder.id,
        channel: $('channel').value,
        frame: frames[index].frame
    });
});
async function results() {
    const r = await api('/api/results');
    $('regenerate').disabled = r.running;
    const box = $('resultContent');
    box.replaceChildren();
    if (r.error) message('Analysis failed: ' + r.error);
    if (r.result) {
        const s = r.result,
            p = document.createElement('p');
        p.textContent = `Result ${s.result_id} · review revision ${s.review_revision}${s.review_revision!==state.revision?' (newer reviews exist)':''} · ${s.reviewed_only?'reviewed only':'includes automatic candidates'} · ${JSON.stringify(s.counts)}`;
        box.append(p);
        for (const file of ['founders.csv', 'measurements.csv', 'summaries.csv', 'comparisons.csv', 'summary.json']) {
            const a = document.createElement('a');
            a.href = `/results/${s.result_id}/${file}`;
            a.textContent = file;
            a.download = file;
            box.append(a);
        }
        for (const plot of s.plots) {
            const im = document.createElement('img');
            im.src = `/results/${s.result_id}/${plot.file}`;
            im.alt = plot.metric;
            box.append(im);
        }
        const warning = document.createElement('p');
        warning.textContent = s.warnings.join(' ');
        box.append(warning);
    } else box.textContent = 'No analysis yet.';
    if (r.running) {
        box.prepend(document.createTextNode('Analysis running… '));
        setTimeout(guarded(results), 1500);
    }
}
on('regenerate', async () => {
    await api('/api/analyze', {
        reviewed_only: $('reviewedOnly').checked
    });
    await results();
});
$('results').ontoggle = guarded(async () => {
    if ($('results').open) await results();
});
window.addEventListener('keydown', guarded(async e => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === 'ArrowLeft') await move(-1);
    if (e.key === 'ArrowRight') await move(1);
}));
guarded(() => refresh(false))();
