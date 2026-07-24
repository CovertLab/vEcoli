import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    import numpy as np
    import polars as pl
    import altair as alt
    import anywidget
    import traitlets

    alt.data_transformers.disable_max_rows()

    return alt, anywidget, mo, np, os, pl, sys, traitlets


# ===== PinnedSelect — custom anywidget single-select with unbounded options ==
# Drop-in replacement for `mo.ui.dropdown` when the options list exceeds
# marimo's 1000-item cap. Client-side text search filters the list; only up to
# 500 items are rendered in the DOM at once (with a "refine search" hint) to
# keep the browser responsive.


@app.cell
def _(anywidget, mo, traitlets):
    _PSS_ESM = """
    function render({ model, el }) {
        const state = { open: false, query: "" };
        function currentValue() { return model.get("value") || ""; }
        function currentOptions() { return model.get("options") || []; }
        el.className = "pss-root";
        const trigger = document.createElement("div");
        trigger.className = "pss-trigger";
        trigger.tabIndex = 0;
        el.appendChild(trigger);
        const dropdown = document.createElement("div");
        dropdown.className = "pss-dropdown";
        dropdown.style.display = "none";
        el.appendChild(dropdown);
        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.placeholder = "Search...";
        searchInput.className = "pss-search";
        dropdown.appendChild(searchInput);
        const optionsPanel = document.createElement("div");
        optionsPanel.className = "pss-options-panel";
        dropdown.appendChild(optionsPanel);

        function renderTrigger() {
            const val = currentValue();
            trigger.innerHTML = "";
            const label = document.createElement("span");
            if (!val) {
                label.className = "pss-placeholder";
                label.textContent = model.get("placeholder") || "Select...";
            } else {
                label.className = "pss-value";
                label.textContent = val;
            }
            trigger.appendChild(label);
            const caret = document.createElement("span");
            caret.className = "pss-caret";
            caret.textContent = state.open ? "▴" : "▾";
            trigger.appendChild(caret);
        }

        function itemRow(item, selected) {
            const row = document.createElement("div");
            row.className = "pss-item" + (selected ? " pss-item-selected" : "");
            row.textContent = item;
            row.addEventListener("click", (e) => {
                e.stopPropagation();
                select(item);
            });
            return row;
        }

        function renderList() {
            const val = currentValue();
            const opts = currentOptions();
            const q = state.query.toLowerCase();
            const filtered = q
                ? opts.filter(o => String(o).toLowerCase().includes(q))
                : opts;
            optionsPanel.innerHTML = "";
            const hdr = document.createElement("div");
            hdr.className = "pss-header";
            hdr.textContent = "Options (" + filtered.length + " / " +
                opts.length + ")";
            optionsPanel.appendChild(hdr);
            const cap = 500;
            const shown = filtered.slice(0, cap);
            if (shown.length === 0) {
                const empty = document.createElement("div");
                empty.className = "pss-empty";
                empty.textContent = q ? "No matches" : "No options";
                optionsPanel.appendChild(empty);
            } else {
                for (const item of shown) {
                    optionsPanel.appendChild(itemRow(item, item === val));
                }
                if (filtered.length > cap) {
                    const note = document.createElement("div");
                    note.className = "pss-empty";
                    note.textContent = "…and " + (filtered.length - cap) +
                        " more (refine search)";
                    optionsPanel.appendChild(note);
                }
            }
        }

        function select(item) {
            model.set("value", item);
            model.save_changes();
            closeDropdown();
            renderTrigger();
        }

        function openDropdown() {
            state.open = true;
            dropdown.style.display = "flex";
            searchInput.value = "";
            state.query = "";
            renderList();
            setTimeout(() => searchInput.focus(), 0);
            renderTrigger();
        }
        function closeDropdown() {
            state.open = false;
            dropdown.style.display = "none";
            renderTrigger();
        }

        trigger.addEventListener("click", (e) => {
            e.stopPropagation();
            state.open ? closeDropdown() : openDropdown();
        });
        searchInput.addEventListener("input", (e) => {
            state.query = e.target.value;
            renderList();
        });
        searchInput.addEventListener("keydown", (e) => {
            if (e.key === "Escape") { closeDropdown(); trigger.focus(); }
        });

        const outsideHandler = (e) => {
            if (!el.contains(e.target)) closeDropdown();
        };
        document.addEventListener("click", outsideHandler);

        model.on("change:value", () => { renderTrigger(); });
        model.on("change:options", () => { renderTrigger(); renderList(); });

        renderTrigger();
    }
    export default { render };
    """

    _PSS_CSS = """
    .pss-root {
        position: relative;
        display: inline-block;
        min-width: 320px;
        max-width: 640px;
        font-family: inherit;
        font-size: 13px;
    }
    .pss-trigger {
        border: 1px solid #cfcfcf;
        border-radius: 4px;
        padding: 4px 8px;
        cursor: pointer;
        background: white;
        min-height: 26px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .pss-trigger:hover { border-color: #999; }
    .pss-trigger:focus { outline: 2px solid #4a90e2; outline-offset: -1px; }
    .pss-placeholder { color: #888; }
    .pss-value {
        color: #222;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex: 1;
    }
    .pss-caret { margin-left: auto; color: #888; font-size: 11px; }
    .pss-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        min-width: 100%;
        max-width: 640px;
        background: white;
        border: 1px solid #cfcfcf;
        border-radius: 4px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        z-index: 9999;
        margin-top: 2px;
        max-height: 360px;
        display: flex;
        flex-direction: column;
    }
    .pss-search {
        border: none;
        border-bottom: 1px solid #eee;
        padding: 6px 8px;
        outline: none;
        font-size: 13px;
        background: white;
    }
    .pss-search:focus { border-bottom-color: #4a90e2; }
    .pss-options-panel {
        overflow-y: auto;
        flex: 1 1 auto;
        min-height: 60px;
    }
    .pss-header {
        padding: 3px 8px;
        background: #f5f7fa;
        font-size: 10px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        border-bottom: 1px solid #eee;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .pss-item {
        padding: 4px 8px;
        cursor: pointer;
        line-height: 1.2;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pss-item:hover { background: #f0f4f8; }
    .pss-item-selected { background: #eaf2fa; font-weight: 600; }
    .pss-empty {
        padding: 8px;
        color: #888;
        font-style: italic;
        font-size: 12px;
    }
    """

    class PinnedSelect(anywidget.AnyWidget):
        _esm = _PSS_ESM
        _css = _PSS_CSS
        options = traitlets.List([]).tag(sync=True)
        value = traitlets.Unicode("").tag(sync=True)
        placeholder = traitlets.Unicode("Select...").tag(sync=True)

    def pss(options, value=None, placeholder="Select..."):
        """Searchable single-select with unbounded option count.
        Options are a flat list of strings (labels). Read the selected label
        via `pss_value(widget)`."""
        opts = list(options)
        val = value if value is not None else (opts[0] if opts else "")
        raw = PinnedSelect(
            options=opts,
            value=str(val),
            placeholder=str(placeholder),
        )
        return mo.ui.anywidget(raw)

    def pss_value(widget):
        """Extract the selected label from a PinnedSelect wrapper."""
        v = widget.value
        if isinstance(v, dict):
            return v.get("value") or ""
        return v or ""

    return PinnedSelect, pss, pss_value


@app.cell
def _(mo):
    # Browser-style navigation history overriding the picker. `history` is the
    # sequence of node IDs visited via map clicks; `cursor` is the current
    # position (-1 means "no history, use picker as root"). Back / Forward
    # move the cursor; a new map click truncates any forward history and
    # appends the newly-clicked node. Return-to-picker resets both fields.
    # allow_self_loops=True so buttons defined in the reader cell can update
    # the state and cause that same cell to re-run.
    get_nav, set_nav = mo.state(
        {"history": [], "cursor": -1}, allow_self_loops=True,
    )
    return get_nav, set_nav


@app.cell
def _(os, sys):
    wd_root = os.getcwd().split("/notebooks")[0]
    sys.path.append(wd_root)

    from ecoli.analysis.causality_network.viewer import CausalityBundle

    return CausalityBundle, wd_root


@app.cell
def _(mo, wd_root, os):
    default_bundle = os.path.join(wd_root, "out", "marimo_test", "seriesOut", "seriesOut.zip")
    bundle_path = mo.ui.text(
        value=default_bundle,
        placeholder="/path/to/seriesOut.zip",
        label="seriesOut.zip path",
        full_width=True,
    )
    bundle_path
    return (bundle_path,)


@app.cell
def _(CausalityBundle, bundle_path, mo):
    mo.stop(not bundle_path.value, mo.md("Set a path to a `seriesOut.zip` bundle above."))
    bundle = CausalityBundle(bundle_path.value)
    summary = mo.md(
        f"**Loaded**: `{bundle_path.value}` — "
        f"{len(bundle.nodes):,} nodes, {len(bundle.edges):,} edges, "
        f"{len(bundle.time)} timesteps."
    )
    summary
    return (bundle,)


@app.cell
def _(bundle, mo):
    # Default to "(all)" so the picker can reflect any currently-navigated
    # node, regardless of type — otherwise navigating to a non-Gene node via
    # the map couldn't be shown in the picker while it was filtered to Gene.
    type_filter = mo.ui.dropdown(
        options=["(all)"] + bundle.node_types,
        value="(all)",
        label="node type:",
    )
    type_filter
    return (type_filter,)


@app.cell
def _(bundle, get_nav, mo, pss, type_filter):
    # The picker's value is driven by the nav-history state so any programmatic
    # navigation (map click, Back, Forward) surfaces here. The cell re-runs on
    # nav-state changes and rebuilds the widget with the new default value.
    if type_filter.value == "(all)":
        candidates = bundle.nodes
    else:
        candidates = [n for n in bundle.nodes if n["type"] == type_filter.value]

    def _label(n):
        name = n.get("name") or ""
        return f"{n['ID']}  —  {name}" if name and name != n["ID"] else n["ID"]

    label_to_id = {_label(n): n["ID"] for n in candidates}

    _s = get_nav()
    _history = _s["history"]
    _cursor = _s["cursor"]
    _current_id = (
        _history[_cursor] if 0 <= _cursor < len(_history) else None
    )

    # If the current node isn't in the filtered candidates (e.g. user is at a
    # non-Gene node but type_filter="Gene"), splice it in so the picker can
    # display it. Otherwise the fallback to labels[0] here would be picked up
    # by the picker-watcher as a "user change" and navigate us to that wrong
    # node — silently steering the app somewhere unexpected.
    if _current_id and _current_id not in label_to_id.values():
        _extra_node = bundle.get_node(_current_id)
        if _extra_node is not None:
            _extra_label = _label(_extra_node)
            label_to_id = {_extra_label: _current_id, **label_to_id}

    id_to_label = {v: k for k, v in label_to_id.items()}
    labels = list(label_to_id.keys())

    _value = id_to_label.get(_current_id) if _current_id else None
    if _value is None:
        _value = labels[0] if labels else ""

    node_picker = pss(
        options=labels,
        value=_value,
        placeholder="Pick a node — click to search",
    )
    mo.vstack([
        mo.md(f"**Current node** ({len(candidates):,} in filter)"),
        node_picker,
    ])
    return id_to_label, label_to_id, node_picker


@app.cell
def _(get_nav, label_to_id, node_picker, pss_value, set_nav):
    # Detect user-driven picker changes and push them into history. When the
    # picker was rebuilt programmatically (e.g., after a map click), its value
    # already matches the state's current so nothing happens here. Any
    # transient exception during the reactive re-run cascade is swallowed
    # silently — otherwise marimo surfaces it as a red traceback flash before
    # the next successful run cleans it up.
    try:
        _picker_label = pss_value(node_picker)
        _picker_id = label_to_id.get(_picker_label) if _picker_label else None
        _s = get_nav()
        _history = _s["history"]
        _cursor = _s["cursor"]
        _current_id = (
            _history[_cursor] if 0 <= _cursor < len(_history) else None
        )
        if _picker_id and _picker_id != _current_id:
            _new_hist = _history[: _cursor + 1] + [_picker_id]
            set_nav({"history": _new_hist, "cursor": len(_new_hist) - 1})
    except Exception:
        pass
    return


@app.cell
def _(bundle, get_nav, label_to_id, mo, node_picker, pss_value, set_nav):
    # Current node is whatever the nav state points at. If the state hasn't
    # been seeded yet (fresh notebook load, before any user picker/map action)
    # fall back to the picker's current value so we still have something to
    # display; the picker-watcher cell will push it into history on the same
    # tick, making Back/Forward consistent from that point on.
    _nav_state = get_nav()
    _history = _nav_state["history"]
    _cursor = _nav_state["cursor"]
    _current_id = (
        _history[_cursor] if 0 <= _cursor < len(_history) else None
    )
    if _current_id is None:
        _picker_label = pss_value(node_picker)
        _current_id = (
            label_to_id.get(_picker_label) if _picker_label else None
        )
    # Silent fallback if the resolved ID isn't in the current bundle (can
    # briefly happen mid-transition when state changes before label_to_id
    # rebuilds, or if a stale ID lingers from a previous bundle). Avoids a
    # red-flashing `mo.stop` on the way to a valid state.
    node = bundle.get_node(_current_id) if _current_id else None
    if node is None:
        _current_id = bundle.nodes[0]["ID"] if bundle.nodes else None
        node = bundle.get_node(_current_id) if _current_id else None
    mo.stop(node is None, mo.md("_No nodes in bundle._"))

    url_line = f"[EcoCyc]({node['url']})" if node.get("url") else "_(no URL)_"
    synonyms = node.get("synonyms") or []
    syn_line = ", ".join(synonyms[:8]) if synonyms else "_(none)_"
    _header_md = mo.md("\n".join([
        f"### {node.get('name') or node['ID']}",
        "",
        f"- **ID**: `{node['ID']}`",
        f"- **Type**: {node['type']}  •  **Class**: {node['class']}",
        f"- **Location**: {node.get('location') or '—'}",
        f"- **Synonyms**: {syn_line}",
        f"- **Link**: {url_line}",
    ]))

    # Back / Forward through history. cursor=0 is the root of the trail — Back
    # from there is disabled (use the picker itself to move elsewhere).
    _can_back = _cursor > 0
    _can_fwd = _cursor < len(_history) - 1

    def _go_back(_v):
        _s = get_nav()
        _c = _s["cursor"]
        if _c > 0:
            set_nav({"history": _s["history"], "cursor": _c - 1})

    def _go_fwd(_v):
        _s = get_nav()
        _c = _s["cursor"]
        if _c < len(_s["history"]) - 1:
            set_nav({"history": _s["history"], "cursor": _c + 1})

    _back_btn = mo.ui.button(
        label="← back", on_click=_go_back, disabled=not _can_back,
    )
    _fwd_btn = mo.ui.button(
        label="forward →", on_click=_go_fwd, disabled=not _can_fwd,
    )
    _nav_row = mo.hstack(
        [_back_btn, _fwd_btn], justify="start", gap=0.5, align="center",
    )
    header = mo.vstack([_nav_row, _header_md])
    header
    return (node,)


@app.cell
def _(alt, bundle, mo, node, pl):
    dyn = bundle.get_dynamics(node["ID"])
    meta_by_type = {m["type"]: m for m in bundle.get_series_meta(node["ID"])}

    if not dyn:
        dyn_chart = mo.md("_No dynamics recorded for this node._")
    else:
        rows = []
        time = bundle.time
        for name, values in dyn.items():
            values = values.astype(float)
            n = min(len(time), len(values))
            unit = meta_by_type.get(name, {}).get("units", "")
            for t, v in zip(time[:n], values[:n]):
                rows.append({"series": f"{name} ({unit})" if unit else name,
                             "time": float(t), "value": float(v)})
        df = pl.DataFrame(rows)
        dyn_chart = (
            alt.Chart(df.to_pandas())
            .mark_line()
            .encode(
                x=alt.X("time:Q", title="time (s)"),
                y=alt.Y("value:Q", title="value"),
                color=alt.Color("series:N", legend=alt.Legend(orient="top")),
            )
            .properties(width=520, height=520, title="Dynamics")
        )
    return dyn, dyn_chart


@app.cell
def _(bundle):
    # ---- Helper: state-only neighborhood via process collapsing ----
    # Walks through *chains* of Process-class nodes until reaching a State,
    # so patterns like `State → Process → Process → State` (used by wcEcoli
    # for e.g. gene regulation: `TF → TF-Binding → Regulation → Gene`) are
    # rendered as one edge annotated with every intermediate process name.
    # If the current node is itself a Process, direct state neighbors are
    # returned with no collapsing.
    _MAX_PROC_CHAIN = 4  # cap chain depth to keep DFS bounded

    def state_edges_fn(node_id, direction):
        """Return {state_id: set(process_label_chain)} for `direction`."""
        n_center = bundle.get_node(node_id)
        center_is_process = (
            n_center is not None and n_center.get("class") == "Process"
        )

        def _neighbors(nid):
            return (
                bundle.outgoing.get(nid, [])
                if direction == "downstream"
                else bundle.incoming.get(nid, [])
            )

        result: dict[str, set[str]] = {}

        if center_is_process:
            for nbr in _neighbors(node_id):
                if nbr == node_id:
                    continue
                n_nbr = bundle.get_node(nbr)
                if n_nbr is not None and n_nbr.get("class") != "Process":
                    result.setdefault(nbr, set()).add("")
            return result

        def _walk(current, labels, path):
            if len(labels) > _MAX_PROC_CHAIN:
                return
            for nbr in _neighbors(current):
                if nbr == node_id or nbr in path:
                    continue
                n_nbr = bundle.get_node(nbr)
                if n_nbr is None:
                    continue
                if n_nbr.get("class") == "Process":
                    _lbl = n_nbr.get("name") or n_nbr.get("type") or nbr
                    _walk(nbr, labels + (_lbl,), path | {nbr})
                else:
                    _joined = " → ".join(labels) if labels else ""
                    result.setdefault(nbr, set()).add(_joined)

        _walk(node_id, (), {node_id})
        return result

    return (state_edges_fn,)


@app.cell
def _(bundle, mo):
    # ---- Pathway map filters ----
    # Options come from bundle-wide State-class node types (rather than just
    # types present in the current neighborhood) so the cell does NOT depend
    # on `node`. Otherwise every navigation would rebuild the widget and
    # reset the user's selection.
    _state_types = sorted({
        (_n.get("type") or "")
        for _n in bundle.nodes
        if _n.get("class") != "Process"
    } - {""})
    map_type_filter = mo.ui.multiselect(
        options=_state_types, value=_state_types, label="neighborhood nodes:",
    )
    map_depth = mo.ui.slider(
        start=1, stop=4, step=1, value=1,
        label="depth", show_value=True,
    )
    return map_depth, map_type_filter


@app.cell
def _(
    alt, bundle, dyn_chart, map_depth, map_type_filter,
    mo, node, pl, state_edges_fn,
):
    import math

    # ---- Interactive pathway map (portrait, right of the dynamics plot) ----
    # Upstream stacks above center, downstream below. Process-class nodes are
    # collapsed (state→Process→state → one edge with the process name as its
    # label). The depth slider controls how many hops out we walk. BFS visits
    # each state once, so the drawing is a spanning tree rooted at the center.
    _center_id = node["ID"]
    _max_depth = int(map_depth.value)
    _MAX_PER_LEVEL = 12
    _MAX_EDGE_LABEL = 32

    _type_ok = set(map_type_filter.value) if map_type_filter.value else None

    def _passes(nid):
        _n_p = bundle.get_node(nid)
        if _n_p is None:
            return False
        if _type_ok is not None and (_n_p.get("type") or "") not in _type_ok:
            return False
        return True

    def _lbl(nid):
        n = bundle.get_node(nid)
        if not n:
            return nid
        nm = n.get("name") or ""
        return nm if nm and nm != nid else nid

    # Per-node text alignment: right-side nodes get right-aligned labels
    # (extending toward center), left-side get left-aligned. Long names then
    # extend inward instead of off the chart edge. Evaluated as a Vega expr
    # against each datum's `x` field.
    _ALIGN_EXPR = "datum.x > 0.6 ? 'right' : datum.x < -0.6 ? 'left' : 'center'"

    def _typ(nid):
        n = bundle.get_node(nid)
        return (n.get("type") if n else "") or ""

    def _fmt_edge_label(labels_set):
        _pieces = sorted(l for l in labels_set if l)
        if not _pieces:
            return ""
        _joined = " / ".join(_pieces)
        return _joined if len(_joined) <= _MAX_EDGE_LABEL else (
            _joined[: _MAX_EDGE_LABEL - 1] + "…"
        )

    def _walk(direction):
        """BFS through state_edges_fn up to `_max_depth`. Returns:
          levels: {depth: [state ids in discovery order]}
          parent: {child_id: (parent_id, edge_label)}
          truncated: {depth: dropped_count}
        Each state is visited once — the drawing is a spanning tree so every
        node has exactly one incoming edge from its BFS parent.
        """
        visited = {_center_id}
        levels = {0: [_center_id]}
        parent = {}
        truncated = {}
        frontier = [_center_id]
        for _depth in range(1, _max_depth + 1):
            _next = []
            for src in frontier:
                edge_map = state_edges_fn(src, direction)
                for dst in sorted(edge_map.keys()):
                    if dst in visited or not _passes(dst):
                        continue
                    parent[dst] = (src, _fmt_edge_label(edge_map[dst]))
                    _next.append(dst)
                    visited.add(dst)
            if len(_next) > _MAX_PER_LEVEL:
                truncated[_depth] = len(_next) - _MAX_PER_LEVEL
                for _d in _next[_MAX_PER_LEVEL:]:
                    visited.discard(_d)
                    parent.pop(_d, None)
                _next = _next[:_MAX_PER_LEVEL]
            if not _next:
                break
            levels[_depth] = _next
            frontier = _next
        return levels, parent, truncated

    _up_levels, _up_parent, _up_trunc = _walk("upstream")
    _dn_levels, _dn_parent, _dn_trunc = _walk("downstream")

    _pos = {_center_id: (0.0, 0.0)}

    def _place(level_ids, y):
        n = len(level_ids)
        if n == 0:
            return
        if n == 1:
            _pos[level_ids[0]] = (0.0, y)
            return
        _span = 3.2  # x from -1.6 to +1.6
        _step = _span / (n - 1)
        for i, nid in enumerate(level_ids):
            _pos[nid] = (-1.6 + i * _step, y)

    for _depth, _ids in _up_levels.items():
        if _depth != 0:
            _place(_ids, float(_depth))
    for _depth, _ids in _dn_levels.items():
        if _depth != 0:
            _place(_ids, -float(_depth))

    _node_rows = [{
        "id": _center_id, "label": _lbl(_center_id),
        "type": node.get("type") or "", "role": "current",
        "x": 0.0, "y": 0.0,
    }]
    for _depth, _ids in _up_levels.items():
        if _depth == 0:
            continue
        for _nid in _ids:
            _x, _y = _pos[_nid]
            _node_rows.append({
                "id": _nid, "label": _lbl(_nid), "type": _typ(_nid),
                "role": "upstream", "x": _x, "y": _y,
            })
    for _depth, _ids in _dn_levels.items():
        if _depth == 0:
            continue
        for _nid in _ids:
            _x, _y = _pos[_nid]
            _node_rows.append({
                "id": _nid, "label": _lbl(_nid), "type": _typ(_nid),
                "role": "downstream", "x": _x, "y": _y,
            })

    # Polyline routing: each edge is 3 ordered points (parent → elbow → child)
    # rendered as a mark_line grouped by `edge_id`. Elbow is at `(dx, sy +
    # k*(dy-sy))` — the child's x-column, part-way in y between parent and
    # child. The second segment is therefore a vertical stem at `x=dx`, and
    # because every child has its own x, those stems run parallel and each
    # child gets a distinct "landing column" that makes it stand out. The
    # first (diagonal) segment leaves the parent at an angle that depends on
    # the child, so edges still fan out at distinct obtuse angles.
    _EDGE_ELBOW_K = 0.5

    _edge_rows = []
    _edge_label_rows = []
    # Directional arrow markers along each edge. Biological flow is always
    # downward in this layout (upstream drops into center, center drops into
    # downstream), so every arrow points down regardless of edge role.
    _arrow_rows = []

    def _add_edge(src_id, dst_id, label):
        sx, sy = _pos[src_id]
        dx, dy = _pos[dst_id]
        ex = dx
        ey = sy + _EDGE_ELBOW_K * (dy - sy)
        # `id` = dst so hovering the polyline OR the dst node fires the same
        # hover selection; all 3 points share id + edge_id, so the whole line
        # highlights together.
        for _i, (_x, _y) in enumerate(((sx, sy), (ex, ey), (dx, dy))):
            _edge_rows.append({
                "edge_id": dst_id, "id": dst_id,
                "x": _x, "y": _y, "order": _i,
            })
        if label:
            # Label sits 65% along the diagonal (parent → elbow) — past the
            # midpoint so it doesn't collide with the always-visible current
            # node label at depth 1, but still fanned out with the diagonal
            # rather than crammed near the child's stem.
            _label_t = 0.65
            _edge_label_rows.append({
                "id": dst_id,
                "x": sx + _label_t * (ex - sx),
                "y": sy + _label_t * (ey - sy),
                "label": label,
            })
        # Two arrow markers per edge — one 25% along the diagonal (leaves the
        # midpoint clear for the label), one at the midpoint of the vertical
        # stem. Each arrow is rotated to lie along its segment and to point
        # in the direction of biological flow (always toward decreasing y:
        # upstream cascades into the center, center cascades into downstream).
        # So the tip aligns with the slant of the segment it sits on.
        def _flow_angle(p1x, p1y, p2x, p2y):
            # Direction from higher-y endpoint toward lower-y endpoint along
            # the segment, expressed as a rotation for triangle-up: 0° keeps
            # the tip up, 180° flips it down, atan2(vx, vy) handles slants.
            if p1y >= p2y:
                _vx, _vy = p2x - p1x, p2y - p1y
            else:
                _vx, _vy = p1x - p2x, p1y - p2y
            return math.degrees(math.atan2(_vx, _vy))

        _arrow_rows.append({
            "id": dst_id,
            "x": sx + 0.25 * (ex - sx),
            "y": sy + 0.25 * (ey - sy),
            "angle": _flow_angle(sx, sy, ex, ey),
        })
        _arrow_rows.append({
            "id": dst_id,
            "x": dx, "y": (ey + dy) / 2.0,
            "angle": _flow_angle(ex, ey, dx, dy),
        })

    for dst_id, (src_id, label) in _up_parent.items():
        _add_edge(src_id, dst_id, label)
    for dst_id, (src_id, label) in _dn_parent.items():
        _add_edge(src_id, dst_id, label)

    _nodes_pdf = pl.DataFrame(_node_rows).to_pandas()
    _edges_pdf = (
        pl.DataFrame(_edge_rows).to_pandas() if _edge_rows
        else pl.DataFrame(
            schema={
                "edge_id": pl.Utf8, "id": pl.Utf8,
                "x": pl.Float64, "y": pl.Float64, "order": pl.Int64,
            }
        ).to_pandas()
    )
    _edge_labels_pdf = (
        pl.DataFrame(_edge_label_rows).to_pandas() if _edge_label_rows
        else pl.DataFrame(
            schema={
                "id": pl.Utf8, "x": pl.Float64,
                "y": pl.Float64, "label": pl.Utf8,
            }
        ).to_pandas()
    )
    _arrows_pdf = (
        pl.DataFrame(_arrow_rows).to_pandas() if _arrow_rows
        else pl.DataFrame(
            schema={
                "id": pl.Utf8, "x": pl.Float64,
                "y": pl.Float64, "angle": pl.Float64,
            }
        ).to_pandas()
    )

    _up_reach = max((d for d in _up_levels if d != 0), default=0)
    _dn_reach = max((d for d in _dn_levels if d != 0), default=0)
    _y_max = float(_up_reach) + 0.6
    _y_min = -(float(_dn_reach) + 0.6)
    _height = min(760, max(320, 90 * (_up_reach + _dn_reach + 1) + 60))

    _click = alt.selection_point(
        name="click_sel", fields=["id"], on="click", empty="none", clear=False,
    )
    # No `nearest=True` — direct-hit hover is what we want so clustered edges
    # can be distinguished. Circles are big enough to hit directly; edges are
    # widened below so pointing at a rule is easy.
    _hover = alt.selection_point(
        name="hover_sel", fields=["id"], on="mouseover",
        empty="none", clear="mouseout",
    )

    # Edges reference their own dataset by explicit `data=` kwarg on each mark
    # so the top-level LayerChart can carry `_nodes_pdf` — that lets marimo's
    # `_filter_dataframe` return the clicked node rows via `.value`.
    # mark_line with `detail=edge_id` draws one polyline per edge; `order`
    # controls the connection sequence (parent → elbow → child). All 3 points
    # of an edge share `id`, so the hover encoding lights the whole polyline.
    _edges_layer = (
        alt.Chart(_edges_pdf)
        .mark_line(cursor="pointer", strokeCap="round", strokeJoin="round")
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            detail="edge_id:N",
            order="order:Q",
            size=alt.condition(_hover, alt.value(9), alt.value(5)),
            color=alt.condition(_hover, alt.value("#e07a2b"), alt.value("#a3adb8")),
            tooltip=alt.Tooltip("id:N", title="edge"),
        )
    )

    # Directional arrows — one triangle mark per arrow, rotated per-row via
    # the `angle` encoding so the tip aligns with the segment slant and
    # points in the direction of biological flow (decreasing y). Upstream
    # segments end up with tips pointing back toward the center; downstream
    # segments have tips pointing away toward the child.
    _arrows_layer = (
        alt.Chart(_arrows_pdf)
        .mark_point(
            shape="triangle-up", filled=True,
            stroke=None, opacity=1.0,
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            angle=alt.Angle("angle:Q", scale=None),
            size=alt.condition(_hover, alt.value(260), alt.value(160)),
            color=alt.condition(
                _hover, alt.value("#e07a2b"), alt.value("#5a6470")
            ),
        )
    )

    # Edge labels hidden by default; the transform_filter on `_hover` means
    # each label row is only in the DOM while its neighbor node is hovered.
    # Two-layer halo: a wider white text drawn below, then the dark fill
    # on top — reads clearly against edges, nodes, or the plot background.
    _edge_label_outline = (
        alt.Chart(_edge_labels_pdf)
        .transform_filter(_hover)
        .mark_text(
            align="center", baseline="middle",
            fontSize=10, fontWeight="bold",
            color="white", stroke="white", strokeWidth=3.5,
            strokeJoin="round", clip=False,
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            text="label:N",
        )
    )
    _edge_label_layer = (
        alt.Chart(_edge_labels_pdf)
        .transform_filter(_hover)
        .mark_text(
            align="center", baseline="middle",
            fontSize=10, color="#222", fontWeight="bold",
            clip=False,
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            text="label:N",
        )
    )

    _nodes_layer = (
        alt.Chart(_nodes_pdf)
        .mark_circle(
            stroke="white", strokeWidth=1.5,
            cursor="pointer", opacity=1.0,
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            color=alt.Color(
                "role:N",
                scale=alt.Scale(
                    domain=["upstream", "current", "downstream"],
                    range=["#4a90e2", "#e07a2b", "#2ea36a"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            size=alt.condition(_hover, alt.value(520), alt.value(340)),
            tooltip=[
                alt.Tooltip("label:N", title="name"),
                alt.Tooltip("id:N", title="ID"),
                alt.Tooltip("type:N"),
                alt.Tooltip("role:N"),
            ],
        )
    )

    # Current node label: always visible. Halo underlay so it stays readable
    # against neighbor circles or edges that pass under it.
    _current_label_outline = (
        alt.Chart(_nodes_pdf)
        .transform_filter(alt.datum.role == "current")
        .mark_text(
            baseline="bottom", dy=-16,
            fontSize=12, fontWeight="bold",
            color="white", stroke="white", strokeWidth=4,
            strokeJoin="round", clip=False,
            align={"expr": _ALIGN_EXPR},
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            text="label:N",
        )
    )
    _current_label = (
        alt.Chart(_nodes_pdf)
        .transform_filter(alt.datum.role == "current")
        .mark_text(
            baseline="bottom", dy=-16,
            fontSize=12, fontWeight="bold", clip=False,
            align={"expr": _ALIGN_EXPR},
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            text="label:N", color=alt.value("#222"),
        )
    )

    # Neighbor labels: only rendered when hovered — filtered rows means the
    # text mark isn't in the DOM at all when not hovered, so it can't block
    # clicks on the underlying circle.
    _hover_label_outline = (
        alt.Chart(_nodes_pdf)
        .transform_filter(alt.datum.role != "current")
        .transform_filter(_hover)
        .mark_text(
            baseline="bottom", dy=-16, fontSize=11,
            color="white", stroke="white", strokeWidth=3.5,
            strokeJoin="round", clip=False,
            align={"expr": _ALIGN_EXPR},
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            text="label:N",
        )
    )
    _hover_label = (
        alt.Chart(_nodes_pdf)
        .transform_filter(alt.datum.role != "current")
        .transform_filter(_hover)
        .mark_text(
            baseline="bottom", dy=-16, fontSize=11,
            clip=False,
            align={"expr": _ALIGN_EXPR},
        )
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None),
            text="label:N", color=alt.value("#222"),
        )
    )

    _title = "Neighborhood"
    # Selections must live at the top-level LayerChart, and top-level `data`
    # must be _nodes_pdf so `mo.ui.altair_chart(...).value` filters that
    # DataFrame by the click selection (layered charts without top-level data
    # return `Undefined` and never trigger cell re-runs on selection change).
    # Skip _edge_label_layer entirely when there are no annotations — an empty
    # text layer can break Vega-Lite axis parsing in layered charts.
    # Layers, bottom → top. Each label sits directly on top of its own halo
    # underlay so the halo isn't clipped by anything drawn between the two.
    # Hover-triggered labels (edge label, neighbor label) are drawn AFTER the
    # always-on current label so they aren't blocked when they surface — the
    # halo makes both readable through any overlap.
    _layers = [_edges_layer]
    if _arrow_rows:
        _layers.append(_arrows_layer)
    _layers.append(_nodes_layer)
    _layers.append(_current_label_outline)
    _layers.append(_current_label)
    if _edge_label_rows:
        _layers.append(_edge_label_outline)
        _layers.append(_edge_label_layer)
    _layers.append(_hover_label_outline)
    _layers.append(_hover_label)
    _chart = (
        alt.layer(*_layers, data=_nodes_pdf)
        .add_params(_click, _hover)
        .properties(
            width=340, height=_height, title=_title,
            # Extra outer padding so labels marked `clip=False` have somewhere
            # to render into when they extend past the plot area (long node
            # names on the leftmost/rightmost columns).
            padding={"left": 60, "right": 60, "top": 10, "bottom": 10},
        )
    )

    _hint_bits = []
    for _d, _n in sorted(_up_trunc.items()):
        _hint_bits.append(
            f"upstream depth {_d} truncated to {_MAX_PER_LEVEL} (dropped {_n:,})"
        )
    for _d, _n in sorted(_dn_trunc.items()):
        _hint_bits.append(
            f"downstream depth {_d} truncated to {_MAX_PER_LEVEL} (dropped {_n:,})"
        )
    _hint = mo.md(f"_({' · '.join(_hint_bits)})_") if _hint_bits else mo.md("")

    # chart_selection=False so marimo doesn't add its own selection on top of
    # ours — .value is driven by click_sel / hover_sel defined above.
    pathway_chart = mo.ui.altair_chart(
        _chart,
        chart_selection=False,
        legend_selection=False,
    )
    _filter_row = mo.hstack(
        [map_type_filter, map_depth],
        justify="start", gap=1, align="start",
    )
    mo.hstack(
        [dyn_chart, mo.vstack([_filter_row, pathway_chart, _hint])],
        justify="start", gap=1, align="start",
    )
    return (pathway_chart,)


@app.cell
def _(get_nav, node, pathway_chart, set_nav):
    # Read `.value` first to establish the reactive dependency (marimo re-runs
    # this cell when the widget's stored value changes — which happens on both
    # click and hover). Then read raw click_sel from `.selections` so we
    # navigate on clicks only, ignoring hover state entirely. (`.value` is the
    # intersection of *all* selections, so a hover on a different node than
    # the clicked one would zero it out.)
    # Wrapped in try/except so any transient error during the reactive
    # cascade (stale widget references, node dict not yet populated) is
    # swallowed instead of surfacing as a red flash.
    try:
        _ = pathway_chart.value
        _sels = getattr(pathway_chart, "selections", None) or {}
        _click_state = _sels.get("click_sel") or {}
        _ids = _click_state.get("id") if isinstance(_click_state, dict) else None
        _sel_id = _ids[0] if isinstance(_ids, (list, tuple)) and _ids else None
        _cur_id = node.get("ID") if isinstance(node, dict) else None
        if _sel_id and _cur_id and _sel_id != _cur_id:
            # Push to history: truncate anything past the current cursor (like
            # a browser: a new navigation drops the "forward" trail) and append.
            _s = get_nav()
            _new_hist = _s["history"][: _s["cursor"] + 1] + [_sel_id]
            set_nav({"history": _new_hist, "cursor": len(_new_hist) - 1})
    except Exception:
        pass
    return


@app.cell
def _(bundle, mo, node):
    up_ids = sorted(bundle.upstream(node["ID"], depth=1))
    dn_ids = sorted(bundle.downstream(node["ID"], depth=1))

    def _row(nid):
        n = bundle.get_node(nid)
        return {
            "ID": nid,
            "name": (n.get("name") if n else "") or "",
            "type": (n.get("type") if n else "") or "",
            "class": (n.get("class") if n else "") or "",
        }

    up_rows = [_row(nid) for nid in up_ids]
    dn_rows = [_row(nid) for nid in dn_ids]

    _up_body = (
        mo.ui.table(up_rows, pagination=True, page_size=10)
        if up_rows else mo.md("_(none)_")
    )
    _dn_body = (
        mo.ui.table(dn_rows, pagination=True, page_size=10)
        if dn_rows else mo.md("_(none)_")
    )
    mo.accordion(
        {
            f"Upstream ({len(up_rows)})": _up_body,
            f"Downstream ({len(dn_rows)})": _dn_body,
        },
        multiple=True,
    )
    return


if __name__ == "__main__":
    app.run()
