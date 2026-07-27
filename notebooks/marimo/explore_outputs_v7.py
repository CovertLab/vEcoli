import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import sys
    import altair as alt
    import polars as pl
    from scipy.stats import pearsonr
    import anywidget
    import traitlets

    return (
        alt,
        anywidget,
        mo,
        np,
        os,
        pd,
        pearsonr,
        pickle,
        pl,
        sys,
        traitlets,
    )


# ===== PinnedMultiselect — custom anywidget with sticky-selected pattern =====
# Selected items pin at the top of the dropdown (always visible, even when the
# search string wouldn't match them) so users can deselect any number of items
# without scrolling through the full options list. Drop-in replacement for
# mo.ui.multiselect for long option lists.


@app.cell
def _(
    anywidget,
    mo,
    pms,
    pms_value,
    traitlets,
):
    _PMS_ESM = """
    function render({ model, el }) {
        const state = { open: false, query: "" };
        function currentValue() { return model.get("value") || []; }
        function currentOptions() { return model.get("options") || []; }
        function maxSel() { return model.get("max_selections") || 500; }
        function esc(s) {
            return String(s).replace(/[&<>"']/g, c => (
                {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]
            ));
        }
        el.className = "pms-root";
        const trigger = document.createElement("div");
        trigger.className = "pms-trigger";
        trigger.tabIndex = 0;
        el.appendChild(trigger);
        const dropdown = document.createElement("div");
        dropdown.className = "pms-dropdown";
        dropdown.style.display = "none";
        el.appendChild(dropdown);
        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.placeholder = "Search...";
        searchInput.className = "pms-search";
        dropdown.appendChild(searchInput);
        // "Select all" bar — opt-in via the show_select_all trait. Sits above
        // the deselect-all bar; picks up whatever is currently in the options
        // panel (search-filtered or not), capped at max_selections.
        const selectAllBar = document.createElement("div");
        selectAllBar.className = "pms-select-all";
        selectAllBar.style.display = "none";
        selectAllBar.tabIndex = 0;
        selectAllBar.setAttribute("role", "button");
        dropdown.appendChild(selectAllBar);
        // "Deselect all" bar sits above the frozen Selected panel so users
        // can clear the whole selection without individually unchecking each
        // item. Hidden entirely when nothing is selected.
        const deselectAllBar = document.createElement("div");
        deselectAllBar.className = "pms-deselect-all";
        deselectAllBar.style.display = "none";
        deselectAllBar.tabIndex = 0;
        deselectAllBar.setAttribute("role", "button");
        dropdown.appendChild(deselectAllBar);
        // Excel-style frozen panes: two independent scroll containers.
        // Selected panel stays visible (up to 5 items tall, then scrolls
        // internally) while user scrolls the options panel below.
        const selectedPanel = document.createElement("div");
        selectedPanel.className = "pms-selected-panel";
        dropdown.appendChild(selectedPanel);
        const optionsPanel = document.createElement("div");
        optionsPanel.className = "pms-options-panel";
        dropdown.appendChild(optionsPanel);

        function renderTrigger() {
            const val = currentValue();
            trigger.innerHTML = "";
            if (val.length === 0) {
                const ph = document.createElement("span");
                ph.className = "pms-placeholder";
                ph.textContent = model.get("placeholder") || "Select...";
                trigger.appendChild(ph);
            } else {
                const shown = val.slice(0, 3);
                for (const v of shown) {
                    const chip = document.createElement("span");
                    chip.className = "pms-chip";
                    chip.textContent = v;
                    trigger.appendChild(chip);
                }
                if (val.length > 3) {
                    const more = document.createElement("span");
                    more.className = "pms-count";
                    more.textContent = "+" + (val.length - 3);
                    trigger.appendChild(more);
                }
            }
            const caret = document.createElement("span");
            caret.className = "pms-caret";
            caret.textContent = state.open ? "▴" : "▾";
            trigger.appendChild(caret);
        }

        function itemRow(item, selected) {
            const row = document.createElement("div");
            row.className = "pms-item" + (selected ? " pms-item-selected" : "");
            const check = document.createElement("span");
            check.className = "pms-check";
            check.innerHTML = selected ? "&check;" : "&nbsp;";
            row.appendChild(check);
            const label = document.createElement("span");
            label.className = "pms-label";
            label.textContent = item;
            row.appendChild(label);
            row.addEventListener("click", (e) => {
                e.stopPropagation();
                toggle(item);
            });
            return row;
        }

        function renderList() {
            const val = currentValue();
            const opts = currentOptions();
            const q = state.query.toLowerCase();
            const selSet = new Set(val);
            const unselFiltered = opts.filter(
                o => !selSet.has(o) && (!q || String(o).toLowerCase().includes(q))
            );
            // ---- Select-all bar (opt-in, only when there are unselected
            // items available to add — cap by max_selections) ----
            if (model.get("show_select_all")) {
                const cap = maxSel();
                const room = Math.max(0, cap - val.length);
                const wouldAdd = Math.min(room, unselFiltered.length);
                if (wouldAdd > 0) {
                    const suffix = q ? " matching (" : " (";
                    selectAllBar.textContent =
                        "✓ Select all" + suffix + wouldAdd + ")";
                    selectAllBar.style.display = "block";
                } else {
                    selectAllBar.style.display = "none";
                }
            } else {
                selectAllBar.style.display = "none";
            }
            // ---- Deselect-all bar (only when ≥1 selected) ----
            if (val.length > 0) {
                deselectAllBar.textContent =
                    "✕ Deselect all (" + val.length + ")";
                deselectAllBar.style.display = "block";
            } else {
                deselectAllBar.style.display = "none";
            }
            // ---- Selected panel (frozen, own scroll) ----
            selectedPanel.innerHTML = "";
            if (val.length > 0) {
                const hdr = document.createElement("div");
                hdr.className = "pms-header";
                hdr.textContent = "Selected (" + val.length + ")";
                selectedPanel.appendChild(hdr);
                const body = document.createElement("div");
                body.className = "pms-selected-body";
                for (const item of val) {
                    body.appendChild(itemRow(item, true));
                }
                selectedPanel.appendChild(body);
                selectedPanel.style.display = "flex";
            } else {
                selectedPanel.style.display = "none";
            }
            // ---- Options panel (below, own scroll) ----
            optionsPanel.innerHTML = "";
            const optHdr = document.createElement("div");
            optHdr.className = "pms-header";
            optHdr.textContent = "Options (" + unselFiltered.length + ")";
            optionsPanel.appendChild(optHdr);
            const cap = 500;
            const shown = unselFiltered.slice(0, cap);
            if (shown.length === 0) {
                const empty = document.createElement("div");
                empty.className = "pms-empty";
                empty.textContent = q ? "No matches" : "All options selected";
                optionsPanel.appendChild(empty);
            } else {
                for (const item of shown) {
                    optionsPanel.appendChild(itemRow(item, false));
                }
                if (unselFiltered.length > cap) {
                    const note = document.createElement("div");
                    note.className = "pms-empty";
                    note.textContent = "…and " + (unselFiltered.length - cap) +
                        " more (refine search)";
                    optionsPanel.appendChild(note);
                }
            }
        }

        function toggle(item) {
            const val = currentValue().slice();
            const idx = val.indexOf(item);
            if (idx >= 0) {
                val.splice(idx, 1);
            } else {
                if (val.length >= maxSel()) return;
                val.push(item);
            }
            model.set("value", val);
            model.save_changes();
            renderTrigger();
            renderList();
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
        function clearAll(e) {
            e.stopPropagation();
            model.set("value", []);
            model.save_changes();
            renderTrigger();
            renderList();
        }
        deselectAllBar.addEventListener("click", clearAll);
        deselectAllBar.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") { clearAll(e); }
        });
        function selectAll(e) {
            e.stopPropagation();
            const val = currentValue().slice();
            const selSet = new Set(val);
            const q = state.query.toLowerCase();
            const opts = currentOptions();
            const cap = maxSel();
            for (const o of opts) {
                if (val.length >= cap) break;
                if (selSet.has(o)) continue;
                if (q && !String(o).toLowerCase().includes(q)) continue;
                val.push(o);
                selSet.add(o);
            }
            model.set("value", val);
            model.save_changes();
            renderTrigger();
            renderList();
        }
        selectAllBar.addEventListener("click", selectAll);
        selectAllBar.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") { selectAll(e); }
        });

        const outsideHandler = (e) => {
            if (!el.contains(e.target)) closeDropdown();
        };
        document.addEventListener("click", outsideHandler);

        model.on("change:value", () => { renderTrigger(); renderList(); });
        model.on("change:options", () => { renderTrigger(); renderList(); });

        renderTrigger();
    }
    export default { render };
    """

    _PMS_CSS = """
    .pms-root {
        position: relative;
        display: inline-block;
        min-width: 260px;
        max-width: 480px;
        font-family: inherit;
        font-size: 13px;
    }
    .pms-trigger {
        border: 1px solid #cfcfcf;
        border-radius: 4px;
        padding: 4px 8px;
        cursor: pointer;
        background: white;
        min-height: 26px;
        display: flex;
        align-items: center;
        gap: 4px;
        flex-wrap: wrap;
    }
    .pms-trigger:hover { border-color: #999; }
    .pms-trigger:focus { outline: 2px solid #4a90e2; outline-offset: -1px; }
    .pms-placeholder { color: #888; }
    .pms-chip {
        background: #eef2f7;
        color: #333;
        padding: 1px 6px;
        border-radius: 3px;
        font-size: 12px;
        max-width: 140px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pms-count { color: #666; font-size: 12px; margin-left: 2px; }
    .pms-caret { margin-left: auto; color: #888; font-size: 11px; }
    .pms-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        min-width: 100%;
        max-width: 480px;
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
    .pms-search {
        border: none;
        border-bottom: 1px solid #eee;
        padding: 6px 8px;
        outline: none;
        font-size: 13px;
        background: white;
    }
    .pms-search:focus { border-bottom-color: #4a90e2; }
    /* Select-all bar — opt-in via show_select_all. Sits above the
       deselect-all bar; hover/focus turns green to signal an additive
       action (mirroring deselect-all's destructive red). */
    .pms-select-all {
        padding: 4px 8px;
        cursor: pointer;
        background: #f5f7fa;
        color: #6a7280;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #eee;
        user-select: none;
    }
    .pms-select-all:hover,
    .pms-select-all:focus {
        background: #e9f6ee;
        color: #1f6f42;
        outline: none;
    }
    /* Deselect-all bar — sits above the frozen Selected panel. Muted
       neutral background so it doesn't compete with the selected items
       themselves; hover/focus turn it red to signal a destructive action. */
    .pms-deselect-all {
        padding: 4px 8px;
        cursor: pointer;
        background: #f5f7fa;
        color: #6a7280;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid #eee;
        user-select: none;
    }
    .pms-deselect-all:hover,
    .pms-deselect-all:focus {
        background: #fdecec;
        color: #b03030;
        outline: none;
    }
    /* Frozen "Selected" panel — sticky header + its own body scroll capped
       at ~5 items (~24px each). Border-bottom acts as the divider between
       the frozen pane and the scrollable options below. */
    .pms-selected-panel {
        display: flex;
        flex-direction: column;
        max-height: 148px;
        border-bottom: 3px solid #d5dee7;
        background: #fbfcfe;
        flex-shrink: 0;
    }
    .pms-selected-body {
        overflow-y: auto;
        max-height: 120px;
    }
    /* Options panel takes the remaining space and scrolls independently. */
    .pms-options-panel {
        overflow-y: auto;
        flex: 1 1 auto;
        min-height: 60px;
    }
    .pms-header {
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
    .pms-selected-panel .pms-header {
        background: #eef2f7;
        color: #4a5a70;
    }
    .pms-item {
        padding: 4px 8px;
        cursor: pointer;
        display: flex;
        gap: 6px;
        align-items: center;
        line-height: 1.2;
    }
    .pms-item:hover { background: #f0f4f8; }
    .pms-item-selected { background: #fafcfe; }
    .pms-item-selected:hover { background: #eaf2fa; }
    .pms-check { width: 12px; color: #4a90e2; font-weight: bold; }
    .pms-label {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .pms-empty {
        padding: 8px;
        color: #888;
        font-style: italic;
        font-size: 12px;
    }
    """

    class PinnedMultiselect(anywidget.AnyWidget):
        _esm = _PMS_ESM
        _css = _PMS_CSS
        options = traitlets.List([]).tag(sync=True)
        value = traitlets.List([]).tag(sync=True)
        max_selections = traitlets.Int(500).tag(sync=True)
        placeholder = traitlets.Unicode("Select...").tag(sync=True)
        show_select_all = traitlets.Bool(False).tag(sync=True)

    def pms(
        options, value=None, max_selections=500, placeholder="Select...",
        show_select_all=False,
    ):
        """Drop-in `mo.ui.multiselect` replacement with pinned-selected UX.
        Set `show_select_all=True` to expose a bulk-select bar above the
        deselect-all bar (capped by `max_selections`). Returns a marimo UI
        element. Read selections via `pms_value(widget)`."""
        raw = PinnedMultiselect(
            options=list(options),
            value=list(value or []),
            max_selections=int(max_selections),
            placeholder=str(placeholder),
            show_select_all=bool(show_select_all),
        )
        return mo.ui.anywidget(raw)

    def pms_value(widget):
        """Extract the selected-labels list from a PinnedMultiselect wrapper."""
        v = widget.value
        if isinstance(v, dict):
            return v.get("value") or []
        return v or []

    return PinnedMultiselect, pms, pms_value


@app.cell
def _(os, pickle, sys):
    wd_root = os.getcwd().split("/notebooks")[0]

    sys.path.append(wd_root)

    from ecoli.library.sim_data import LoadSimData
    from ecoli.library.parquet_emitter import (
        dataset_sql,
        ndlist_to_ndarray,
        read_stacked_columns,
        create_duckdb_conn,
        field_metadata,
    )
    from wholecell.utils.protein_counts import get_simulated_validation_counts

    sim_data_path = os.path.join(
        wd_root, "reconstruction", "sim_data", "kb", "simData.cPickle"
    )

    validation_data_path = os.path.join(
        wd_root, "reconstruction", "sim_data", "kb", "validationData.cPickle"
    )

    sim_data = LoadSimData(sim_data_path).sim_data

    with open(validation_data_path, "rb") as f:
        validation_data = pickle.load(f)
    return (
        LoadSimData,
        create_duckdb_conn,
        dataset_sql,
        field_metadata,
        get_simulated_validation_counts,
        ndlist_to_ndarray,
        read_stacked_columns,
        sim_data,
        sim_data_path,
        validation_data,
        wd_root,
    )


@app.cell
def _(LoadSimData):
    def get_bulk_ids(sim_data_path):
        sim_data = LoadSimData(sim_data_path).sim_data
        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
        return bulk_ids

    def get_rxn_ids(sim_data_path):
        sim_data = LoadSimData(sim_data_path).sim_data
        rxn_ids = sim_data.process.metabolism.base_reaction_ids
        return rxn_ids

    return get_bulk_ids, get_rxn_ids


@app.cell
def _(get_bulk_ids, get_rxn_ids, np, sim_data, sim_data_path):
    bulk_ids = get_bulk_ids(sim_data_path)
    bulk_ids_biocyc = [bulk_id[:-3] for bulk_id in bulk_ids]
    bulk_names_unique = list(np.unique(bulk_ids_biocyc))
    bulk_common_names = get_common_names(bulk_names_unique, sim_data)
    bulk_names2biocyc = {
        key: val for key, val in zip(bulk_common_names, bulk_names_unique)
    }
    rxn_ids = get_rxn_ids(sim_data_path)
    cistron_data = sim_data.process.transcription.cistron_data
    mrna_cistron_ids = cistron_data["id"][cistron_data["is_mRNA"]].tolist()
    mrna_gene_ids = [cistron_id.strip("_RNA") for cistron_id in mrna_cistron_ids]
    # Route common-name lookup through get_common_names so duplicates get
    # disambiguated with a [cistron_id] suffix. Without this, multiple
    # cistrons that resolve to the same common name become identical labels
    # in the picker, and `default_name_list.index(name)` always returns the
    # first match — collapsing every duplicate-labeled series onto the same
    # array column, so every selected mRNA displays identical counts.
    mrna_cistron_names = get_common_names(mrna_cistron_ids, sim_data)
    monomer_ids = sim_data.process.translation.monomer_data["id"].tolist()
    monomer_ids = [id[:-3] for id in monomer_ids]
    monomer_names = get_common_names(monomer_ids, sim_data)
    return (
        bulk_common_names,
        bulk_ids_biocyc,
        bulk_names2biocyc,
        bulk_names_unique,
        monomer_ids,
        monomer_names,
        mrna_cistron_names,
        mrna_gene_ids,
        rxn_ids,
    )


@app.cell
def _(mo):
    about_intro_md = mo.md(
        """
    Welcome to the vEcoli data explorer notebook. This notebook provides an
    interactive interface to explore, analyze and visualize the outputs of the
    E. coli whole-cell model simulations.

    By default, vEcoli uses the Parquet emitter, which saves simulation output
    in a tabular file format inside a Hive-partitioned directory structure:

    `experiment_id={}/variant={}/lineage_seed={}/generation={}/agent_id={}`

    This allows efficient organization, storage and retrieval of outputs from
    workflows that run many variants, lineage seeds, generations and agent IDs.

    Pick the **analysis type** (`single` / `multidaughter` / `multigeneration` /
    `multiseed`) to set how output is aggregated, then narrow down with the
    partition dropdowns. The selected analysis determines which partitions are
    required: e.g. `single` needs all five, `multiseed` only needs experiment
    and variant.

    **Compare datasets:** expand the *Compare datasets* section below to enable
    up to 4 additional dataset slots. Each slot has its own analysis type,
    experiment, and partition values (variant / seed / generation / agent_id
    are text inputs since they don't cascade across experiments). Enabled slots
    are plotted alongside the primary dataset in every chart tab.

    **Compare layout** (above): `overlay` distinguishes datasets with stroke
    dash on a shared y-axis; `facet` stacks one chart per dataset. Has no
    visible effect with only the primary dataset.

    The **Download** tab has a `Source dataset:` picker that controls which
    dataset is exported as TSV (compare slots included).
    """
    )
    return (about_intro_md,)


@app.cell
def _(mo):
    # User-facing label for the primary dataset (shown in chart legends and
    # in the Download tab's source picker). Kept simple ("dataset 1"); the
    # compare slots default to "dataset 2".."dataset 5".
    primary_label = mo.ui.text(value="dataset 1", placeholder="dataset 1")
    return (primary_label,)


@app.cell
def _(mo):
    analysis_select = mo.ui.dropdown(
        options=["single", "multidaughter", "multigeneration", "multiseed"],
        value="single",
    )
    # Single global toggle: controls how compare datasets are laid out in every
    # chart tab. Has no visible effect when only the primary dataset is active.
    chart_layout_mode = mo.ui.dropdown(options=["overlay", "facet"], value="overlay")
    return analysis_select, chart_layout_mode


@app.cell
def _(get_exp, mo, outdir_tree):
    _exp_options = get_exp(outdir_tree)

    def _has_history(exp_id):
        entry = outdir_tree.get(exp_id)
        if not isinstance(entry, dict):
            return False
        history = entry.get("history")
        if not isinstance(history, dict):
            return False
        return any(history.values())

    _default_exp = next(
        (e for e in _exp_options if _has_history(e)),
        _exp_options[0] if _exp_options else None,
    )
    exp_select = mo.ui.dropdown(
        options=_exp_options,
        value=_default_exp,
    )
    y_scale = mo.ui.dropdown(options=["linear", "log", "symlog"], value="linear")
    return exp_select, y_scale


@app.cell
def _(analysis_select, partition_groups, partitions_display):
    # Build (label, widget) pairs for the partition keys the chosen analysis
    # type actually needs — consumed by the toolbar in the composition cell.
    _partitions_req = partition_groups[analysis_select.value]
    _all = partitions_display()
    partition_picker_items = [(str(k), _all[k]) for k in _partitions_req]
    return (partition_picker_items,)


@app.cell
def _(exp_select, get_variants, mo, outdir_tree):
    _opts = get_variants(outdir_tree, exp_id=exp_select.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    variant_select = mo.ui.dropdown(options=_opts, value=_default)
    return (variant_select,)


@app.cell
def _(exp_select, get_seeds, mo, outdir_tree, variant_select):
    _opts = get_seeds(outdir_tree, exp_id=exp_select.value, var_id=variant_select.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    seed_select = mo.ui.dropdown(options=_opts, value=_default)
    return (seed_select,)


@app.cell
def _(exp_select, get_gens, mo, outdir_tree, seed_select, variant_select):
    _opts = get_gens(
        outdir_tree,
        exp_id=exp_select.value,
        var_id=variant_select.value,
        seed_id=seed_select.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    gen_select = mo.ui.dropdown(options=_opts, value=_default)
    return (gen_select,)


@app.cell
def _(
    exp_select,
    gen_select,
    get_agents,
    mo,
    outdir_tree,
    seed_select,
    variant_select,
):
    _opts = get_agents(
        outdir_tree,
        exp_id=exp_select.value,
        var_id=variant_select.value,
        seed_id=seed_select.value,
        gen_id=gen_select.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    agent_select = mo.ui.dropdown(options=_opts, value=_default)
    return (agent_select,)


# ----- Compare slots (up to 4 additional datasets) -----
# Cascading: each slot's variant/seed/gen/agent dropdown is in its own cell
# that depends on the slot's exp (and on prior cascade levels). Marimo's cell
# reactivity is per-cell, so changing slot N's exp only re-runs slot N's
# downstream cells — other slots' selections are preserved. Slots default to
# the first valid option at each level, so an enabled slot always has a
# fully-qualified WHERE clause (no accidental full-scans).


@app.cell
def _():
    NUM_COMPARE_SLOTS = 4
    return (NUM_COMPARE_SLOTS,)


@app.cell
def _(mo, NUM_COMPARE_SLOTS):
    # Top-level compare toggle + count. When `compare_enabled` is True, the
    # toolbar renders the first `compare_count` slot pickers. All compare
    # slots inherit the primary's analysis type (no per-slot analysis).
    compare_enabled = mo.ui.checkbox(value=False)
    compare_count = mo.ui.number(value=1, start=1, stop=NUM_COMPARE_SLOTS, step=1)
    return compare_count, compare_enabled


# Slot 0: static (label/exp) + 4 cascading dropdowns. Analysis type is
# inherited from the primary; whether the slot is rendered comes from the
# top-level `compare_enabled` + `compare_count` widgets.
@app.cell
def _(get_exp, mo, outdir_tree):
    slot_0_label = mo.ui.text(value="dataset 2", placeholder="dataset 2")
    slot_0_exp = mo.ui.dropdown(
        options=get_exp(outdir_tree), value=None, searchable=True
    )
    return slot_0_exp, slot_0_label


@app.cell
def _(get_variants, mo, outdir_tree, slot_0_exp):
    _opts = get_variants(outdir_tree, exp_id=slot_0_exp.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_0_variant = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_0_variant,)


@app.cell
def _(get_seeds, mo, outdir_tree, slot_0_exp, slot_0_variant):
    _opts = get_seeds(outdir_tree, exp_id=slot_0_exp.value, var_id=slot_0_variant.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_0_seed = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_0_seed,)


@app.cell
def _(get_gens, mo, outdir_tree, slot_0_exp, slot_0_seed, slot_0_variant):
    _opts = get_gens(
        outdir_tree,
        exp_id=slot_0_exp.value,
        var_id=slot_0_variant.value,
        seed_id=slot_0_seed.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_0_generation = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_0_generation,)


@app.cell
def _(
    get_agents,
    mo,
    outdir_tree,
    slot_0_exp,
    slot_0_generation,
    slot_0_seed,
    slot_0_variant,
):
    _opts = get_agents(
        outdir_tree,
        exp_id=slot_0_exp.value,
        var_id=slot_0_variant.value,
        seed_id=slot_0_seed.value,
        gen_id=slot_0_generation.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_0_agent_id = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_0_agent_id,)


# Slot 1
@app.cell
def _(get_exp, mo, outdir_tree):
    slot_1_label = mo.ui.text(value="dataset 3", placeholder="dataset 3")
    slot_1_exp = mo.ui.dropdown(
        options=get_exp(outdir_tree), value=None, searchable=True
    )
    return slot_1_exp, slot_1_label


@app.cell
def _(get_variants, mo, outdir_tree, slot_1_exp):
    _opts = get_variants(outdir_tree, exp_id=slot_1_exp.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_1_variant = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_1_variant,)


@app.cell
def _(get_seeds, mo, outdir_tree, slot_1_exp, slot_1_variant):
    _opts = get_seeds(outdir_tree, exp_id=slot_1_exp.value, var_id=slot_1_variant.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_1_seed = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_1_seed,)


@app.cell
def _(get_gens, mo, outdir_tree, slot_1_exp, slot_1_seed, slot_1_variant):
    _opts = get_gens(
        outdir_tree,
        exp_id=slot_1_exp.value,
        var_id=slot_1_variant.value,
        seed_id=slot_1_seed.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_1_generation = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_1_generation,)


@app.cell
def _(
    get_agents,
    mo,
    outdir_tree,
    slot_1_exp,
    slot_1_generation,
    slot_1_seed,
    slot_1_variant,
):
    _opts = get_agents(
        outdir_tree,
        exp_id=slot_1_exp.value,
        var_id=slot_1_variant.value,
        seed_id=slot_1_seed.value,
        gen_id=slot_1_generation.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_1_agent_id = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_1_agent_id,)


# Slot 2
@app.cell
def _(get_exp, mo, outdir_tree):
    slot_2_label = mo.ui.text(value="dataset 4", placeholder="dataset 4")
    slot_2_exp = mo.ui.dropdown(
        options=get_exp(outdir_tree), value=None, searchable=True
    )
    return slot_2_exp, slot_2_label


@app.cell
def _(get_variants, mo, outdir_tree, slot_2_exp):
    _opts = get_variants(outdir_tree, exp_id=slot_2_exp.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_2_variant = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_2_variant,)


@app.cell
def _(get_seeds, mo, outdir_tree, slot_2_exp, slot_2_variant):
    _opts = get_seeds(outdir_tree, exp_id=slot_2_exp.value, var_id=slot_2_variant.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_2_seed = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_2_seed,)


@app.cell
def _(get_gens, mo, outdir_tree, slot_2_exp, slot_2_seed, slot_2_variant):
    _opts = get_gens(
        outdir_tree,
        exp_id=slot_2_exp.value,
        var_id=slot_2_variant.value,
        seed_id=slot_2_seed.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_2_generation = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_2_generation,)


@app.cell
def _(
    get_agents,
    mo,
    outdir_tree,
    slot_2_exp,
    slot_2_generation,
    slot_2_seed,
    slot_2_variant,
):
    _opts = get_agents(
        outdir_tree,
        exp_id=slot_2_exp.value,
        var_id=slot_2_variant.value,
        seed_id=slot_2_seed.value,
        gen_id=slot_2_generation.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_2_agent_id = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_2_agent_id,)


# Slot 3
@app.cell
def _(get_exp, mo, outdir_tree):
    slot_3_label = mo.ui.text(value="dataset 5", placeholder="dataset 5")
    slot_3_exp = mo.ui.dropdown(
        options=get_exp(outdir_tree), value=None, searchable=True
    )
    return slot_3_exp, slot_3_label


@app.cell
def _(get_variants, mo, outdir_tree, slot_3_exp):
    _opts = get_variants(outdir_tree, exp_id=slot_3_exp.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_3_variant = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_3_variant,)


@app.cell
def _(get_seeds, mo, outdir_tree, slot_3_exp, slot_3_variant):
    _opts = get_seeds(outdir_tree, exp_id=slot_3_exp.value, var_id=slot_3_variant.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_3_seed = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_3_seed,)


@app.cell
def _(get_gens, mo, outdir_tree, slot_3_exp, slot_3_seed, slot_3_variant):
    _opts = get_gens(
        outdir_tree,
        exp_id=slot_3_exp.value,
        var_id=slot_3_variant.value,
        seed_id=slot_3_seed.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_3_generation = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_3_generation,)


@app.cell
def _(
    get_agents,
    mo,
    outdir_tree,
    slot_3_exp,
    slot_3_generation,
    slot_3_seed,
    slot_3_variant,
):
    _opts = get_agents(
        outdir_tree,
        exp_id=slot_3_exp.value,
        var_id=slot_3_variant.value,
        seed_id=slot_3_seed.value,
        gen_id=slot_3_generation.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    slot_3_agent_id = mo.ui.dropdown(options=_opts, value=_default)
    return (slot_3_agent_id,)


# Compile per-slot widgets into the compare_slots list. Direct widget
# references make marimo rebuild this cell on any inner-widget change, which
# cascades to active_datasets and the chart cells.
@app.cell
def _(
    slot_0_label,
    slot_0_exp,
    slot_0_variant,
    slot_0_seed,
    slot_0_generation,
    slot_0_agent_id,
    slot_1_label,
    slot_1_exp,
    slot_1_variant,
    slot_1_seed,
    slot_1_generation,
    slot_1_agent_id,
    slot_2_label,
    slot_2_exp,
    slot_2_variant,
    slot_2_seed,
    slot_2_generation,
    slot_2_agent_id,
    slot_3_label,
    slot_3_exp,
    slot_3_variant,
    slot_3_seed,
    slot_3_generation,
    slot_3_agent_id,
):
    _keys = ("label", "exp", "variant", "seed", "generation", "agent_id")
    _slots = [
        (
            slot_0_label,
            slot_0_exp,
            slot_0_variant,
            slot_0_seed,
            slot_0_generation,
            slot_0_agent_id,
        ),
        (
            slot_1_label,
            slot_1_exp,
            slot_1_variant,
            slot_1_seed,
            slot_1_generation,
            slot_1_agent_id,
        ),
        (
            slot_2_label,
            slot_2_exp,
            slot_2_variant,
            slot_2_seed,
            slot_2_generation,
            slot_2_agent_id,
        ),
        (
            slot_3_label,
            slot_3_exp,
            slot_3_variant,
            slot_3_seed,
            slot_3_generation,
            slot_3_agent_id,
        ),
    ]
    compare_slots = [dict(zip(_keys, _slot)) for _slot in _slots]
    return (compare_slots,)


@app.cell
def _(analysis_select, generation_range_clause, get_db_filter, partitions_dict):
    dbf_dict = partitions_dict(analysis_select.value)
    db_filter = get_db_filter(dbf_dict) + generation_range_clause

    return (db_filter,)


# ===== Generation range slider (multigeneration / multiseed) =====


@app.cell
def _(analysis_select, exp_select, outdir_tree, seed_select, variant_select):
    # Enumerate the integer generations available under the current primary
    # selection so the range slider bounds are data-driven.
    #   * multigeneration → generations for (exp, var, seed)
    #   * multiseed       → union of generations across all seeds under (exp, var)
    # Other analysis types don't span multiple generations, so we return
    # an empty list and the slider is hidden.
    def _int_or_none(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    _kind = analysis_select.value
    _gens = []
    _exp = exp_select.value
    _var = variant_select.value
    if _kind == "multigeneration":
        try:
            _seed = seed_select.value
            _folders = outdir_tree[_exp]["history"][f"experiment_id={_exp}"][
                f"variant={_var}"
            ][f"lineage_seed={_seed}"].keys()
            _gens = [_int_or_none(f.split("generation=")[1]) for f in _folders]
        except (KeyError, TypeError, AttributeError):
            _gens = []
    elif _kind == "multiseed":
        try:
            _variant_tree = outdir_tree[_exp]["history"][f"experiment_id={_exp}"][
                f"variant={_var}"
            ]
            _gen_set = set()
            for _seed_folder, _seed_children in _variant_tree.items():
                if not isinstance(_seed_children, dict):
                    continue
                for _gen_folder in _seed_children.keys():
                    if _gen_folder.startswith("generation="):
                        _gen_set.add(_int_or_none(_gen_folder.split("=")[1]))
            _gens = sorted(_gen_set)
        except (KeyError, TypeError, AttributeError):
            _gens = []

    available_generations = sorted(g for g in _gens if g is not None)
    return (available_generations,)


@app.cell
def _(available_generations, mo):
    # Only build a real range slider when there are at least 2 distinct gens
    # available. If ≥2, defaults to the full range so behavior matches v4
    # until the user narrows it. If <2 (single/multidaughter, or no data),
    # we still create a widget so downstream cells always have `.value`,
    # but hide it in the toolbar and skip the SQL clause.
    if len(available_generations) >= 2:
        _lo, _hi = available_generations[0], available_generations[-1]
        generation_range_slider = mo.ui.range_slider(
            start=_lo, stop=_hi, step=1, value=(_lo, _hi), show_value=True
        )
    else:
        # Placeholder — never rendered in the toolbar in this case.
        generation_range_slider = mo.ui.range_slider(
            start=0, stop=1, step=1, value=(0, 1), show_value=False
        )
    return (generation_range_slider,)


@app.cell
def _(analysis_select, available_generations, generation_range_slider):
    # Build the SQL clause to append to db_filter (and slot filters). Only
    # applies to aggregations that span generations; suppressed when the
    # slider is already at the full available range so the filter stays
    # minimal (and equivalent to v4 behavior).
    _kind = analysis_select.value
    _applies = (
        _kind in ("multigeneration", "multiseed") and len(available_generations) >= 2
    )
    generation_range_clause = ""
    if _applies:
        _lo, _hi = generation_range_slider.value
        _full_lo, _full_hi = available_generations[0], available_generations[-1]
        if (_lo, _hi) != (_full_lo, _full_hi):
            generation_range_clause = (
                f" AND generation BETWEEN {int(_lo)} AND {int(_hi)}"
            )
    return (generation_range_clause,)


@app.cell
def _(dataset_sql, exp_select, os, wd_root):
    # Per-chart row-stride cap (passed to `sql_downsample`). 1000 keeps each
    # chart's embedded JSON well under marimo's per-cell output cap even when
    # all 10 tabs are eagerly rendered together. Visually 1000 points is
    # indistinguishable from 2000 at typical screen widths.
    datapoints_cap = 1000

    history_sql_base, config_sql_base, _ = dataset_sql(
        os.path.join(wd_root, "out"), experiment_ids=[exp_select.value]
    )

    return config_sql_base, datapoints_cap, history_sql_base


@app.cell
def _(
    analysis_select,
    compare_count,
    compare_enabled,
    compare_slots,
    config_sql_base,
    dataset_sql,
    db_filter,
    generation_range_clause,
    history_sql_base,
    os,
    partition_groups,
    primary_label,
    wd_root,
):
    # Compile primary + the first `compare_count` compare slots into
    # [{label, history_sql, config_sql, db_filter}]. Each slot inherits the
    # primary's analysis type; the slot's WHERE clause is built from the
    # partition keys that analysis type requires.
    def _slot_filter(analysis, exp_val, var, seed, gen, agent_id):
        parts = [f"experiment_id='{exp_val}'"]
        required = partition_groups[analysis]
        _missing = (None, "N/A", "")
        if "variant" in required and var not in _missing:
            parts.append(f"variant={var}")
        if "lineage_seed" in required and seed not in _missing:
            parts.append(f"lineage_seed={seed}")
        if "generation" in required and gen not in _missing:
            parts.append(f"generation={gen}")
        if "agent_id" in required and agent_id not in _missing:
            parts.append(f"agent_id='{agent_id}'")
        return " AND ".join(parts)

    active_datasets = [
        {
            "label": primary_label.value.strip() or "dataset 1",
            "history_sql": history_sql_base,
            "config_sql": config_sql_base,
            "db_filter": db_filter,
        }
    ]
    if compare_enabled.value:
        _n = int(compare_count.value or 0)
        for _i in range(min(_n, len(compare_slots))):
            _slot = compare_slots[_i]
            _exp = _slot["exp"].value
            if not _exp:
                continue
            _slot_hist, _slot_cfg, _ = dataset_sql(
                os.path.join(wd_root, "out"), experiment_ids=[_exp]
            )
            _slot_dbf = (
                _slot_filter(
                    analysis_select.value,
                    _exp,
                    _slot["variant"].value,
                    _slot["seed"].value,
                    _slot["generation"].value,
                    _slot["agent_id"].value,
                )
                + generation_range_clause
            )
            _slot_label = _slot["label"].value.strip() or f"dataset {_i + 2}"
            active_datasets.append(
                {
                    "label": _slot_label,
                    "history_sql": _slot_hist,
                    "config_sql": _slot_cfg,
                    "db_filter": _slot_dbf,
                }
            )
    return (active_datasets,)


@app.cell
def _(get_pathways, mo, pathway_dir):
    select_pathway = mo.ui.dropdown(options=get_pathways(pathway_dir), searchable=True)
    return (select_pathway,)


@app.cell
def _():
    # Pathway picker is rendered by the toolbar in the composition cell.
    return


@app.cell
def _(mo):
    molecule_id_type = mo.ui.radio(
        options=["Common name", "BioCyc ID"], value="Common name"
    )
    # Compound display units. "counts" is the raw bulk store value. The other
    # three multiply by `listeners__enzyme_kinetics__counts_to_molar` (per-
    # timestep cell-volume-based conversion) and a unit-prefix scale.
    compound_unit = mo.ui.dropdown(options=["counts", "mM", "µM", "M"], value="counts")
    return compound_unit, molecule_id_type


@app.cell
def _(
    bulk_common_names,
    bulk_names_unique,
    bulk_override,
    mo,
    molecule_id_type,
    pms,
    select_pathway,
):
    if molecule_id_type.value == "Common name":
        molecule_id_options = bulk_common_names
    elif molecule_id_type.value == "BioCyc ID":
        molecule_id_options = bulk_names_unique

    bulk_sp_plot = pms(
        options=molecule_id_options,
        value=bulk_override(select_pathway.value),
        max_selections=500,
    )
    return (bulk_sp_plot,)


@app.cell
def _(mo):
    about_compounds_md = mo.md(
        "The **bulk** store in the vEcoli model tracks individual molecule "
        "counts of modeled compounds (transcription units, RNAs, proteins, "
        "complexes, metabolites, small molecules). Pick compounds by BioCyc "
        "ID or display name, or select a pathway above to auto-populate."
    )
    return (about_compounds_md,)


@app.cell
def _(
    active_datasets,
    bulk_ids_biocyc,
    bulk_names2biocyc,
    bulk_sp_plot,
    compound_unit,
    conn,
    datapoints_cap,
    get_plot_df_bulk_multi,
    molecule_id_type,
    pms_value,
):
    # Map unit choice to (convert_to_molar, scale). counts_to_molar gives
    # M (mol/L); scale rescales to whatever prefix the user picked.
    _unit_scale = {
        "counts": (False, 1.0),
        "M": (True, 1.0),
        "mM": (True, 1e3),
        "µM": (True, 1e6),
    }
    _convert, _scale = _unit_scale.get(compound_unit.value, (False, 1.0))

    plot_df_bulk = None
    if pms_value(bulk_sp_plot):
        plot_df_bulk = get_plot_df_bulk_multi(
            active_datasets,
            bulk_sp_plot,
            bulk_ids_biocyc,
            bulk_names2biocyc,
            datapoints_cap,
            conn,
            molecule_id_type,
            convert_to_molar=_convert,
            molar_scale=_scale,
        )
    return (plot_df_bulk,)


@app.cell
def _(
    active_datasets,
    alt,
    bulk_sp_plot,
    chart_layout_mode,
    compound_unit,
    mo,
    pl,
    plot_df_bulk,
    pms_value,
    y_scale,
):
    chart_compounds = None
    if pms_value(bulk_sp_plot) and plot_df_bulk is not None:
        _n = len(active_datasets)
        _y_title = (
            "Counts"
            if compound_unit.value == "counts"
            else f"Concentration ({compound_unit.value})"
        )
        _base = (
            alt.Chart(plot_df_bulk)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "counts:Q", scale=alt.Scale(type=y_scale.value), title=_y_title
                ),
                color=alt.Color("compound:N", legend=alt.Legend(title="Compound")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_compounds = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            # Manual facet via mo.vstack — Altair's .facet() doesn't honor
            # autosize/container-width on multi-view specs (renders empty or
            # overflows). Each per-dataset chart is a single view that
            # `width="container"` resizes correctly.
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_bulk.filter(pl.col("dataset_label") == _ds["label"])
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "counts:Q",
                            scale=alt.Scale(type=y_scale.value),
                            title=_y_title,
                        ),
                        color=alt.Color(
                            "compound:N", legend=alt.Legend(title="Compound")
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_compounds = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_compounds = _base
    return (chart_compounds,)


@app.cell
def _(mo):
    about_mrna_md = mo.md(
        "Time course plots of selected mRNA cistron counts. mRNAs may be "
        "specified by gene name or BioCyc ID. A pathway selection above "
        "auto-populates this list."
    )
    return (about_mrna_md,)


@app.cell
def _(mo):
    rna_label_type = mo.ui.radio(options=["gene name", "BioCyc ID"], value="gene name")

    y_scale_mrna = mo.ui.dropdown(options=["linear", "log", "symlog"], value="linear")

    monomer_label_type = mo.ui.radio(
        options=["common name", "BioCyc ID"], value="common name"
    )

    y_scale_monomers = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="symlog"
    )
    return monomer_label_type, rna_label_type, y_scale_monomers, y_scale_mrna


@app.cell
def _():
    # mRNA controls are rendered by the composition cell below.
    return


@app.cell
def _(
    mo,
    mrna_cistron_names,
    mrna_gene_ids,
    mrna_override,
    pms,
    rna_label_type,
    select_pathway,
):
    if rna_label_type.value == "gene name":
        rna_label_options = mrna_cistron_names
    elif rna_label_type.value == "BioCyc ID":
        rna_label_options = mrna_gene_ids

    mrna_select_plot = pms(
        options=rna_label_options,
        value=mrna_override(select_pathway.value),
        max_selections=500,
    )

    return (mrna_select_plot,)


@app.cell
def _(
    mo,
    monomer_ids,
    monomer_label_type,
    monomer_names,
    pms,
    protein_override,
    select_pathway,
):
    monomer_label_dict = {"common name": monomer_names, "BioCyc ID": monomer_ids}

    monomer_select_plot = pms(
        options=monomer_label_dict[monomer_label_type.value],
        value=protein_override(select_pathway.value),
        max_selections=500,
    )
    return (monomer_select_plot,)


@app.cell
def _(
    active_datasets,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    mrna_cistron_names,
    mrna_gene_ids,
    mrna_select_plot,
    pms_value,
    rna_label_type,
):
    plot_df_mrna = None
    if pms_value(mrna_select_plot):
        plot_df_mrna = get_plot_df_multi(
            active_datasets,
            mrna_gene_ids,
            mrna_select_plot,
            "listeners__rna_counts__full_mRNA_cistron_counts",
            "mrna_counts",
            "Genes",
            "counts",
            datapoints_cap,
            conn,
            mrna_cistron_names,
            rna_label_type,
        )
    return (plot_df_mrna,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    mrna_select_plot,
    pl,
    plot_df_mrna,
    pms_value,
    y_scale,
):
    chart_mrna = None
    if pms_value(mrna_select_plot) and plot_df_mrna is not None:
        _n = len(active_datasets)
        _base = (
            alt.Chart(plot_df_mrna)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "counts:Q", scale=alt.Scale(type=y_scale.value), title="Counts"
                ),
                color=alt.Color("Genes:N", legend=alt.Legend(title="Genes")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_mrna = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_mrna.filter(pl.col("dataset_label") == _ds["label"])
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "counts:Q",
                            scale=alt.Scale(type=y_scale.value),
                            title="Counts",
                        ),
                        color=alt.Color("Genes:N", legend=alt.Legend(title="Genes")),
                    )
                    .properties(width="container", height=220)
                )
            chart_mrna = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_mrna = _base

    return (chart_mrna,)


@app.cell
def _(mo):
    about_proteins_md = mo.md(
        "Time course of protein monomer counts. Monomers can be specified by "
        "common name or BioCyc ID; pathway selection above auto-populates."
    )
    return (about_proteins_md,)


@app.cell
def _(
    active_datasets,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    monomer_ids,
    monomer_label_type,
    monomer_names,
    monomer_select_plot,
    pms_value,
):
    plot_df_monomers = None

    if pms_value(monomer_select_plot):
        plot_df_monomers = get_plot_df_multi(
            active_datasets,
            monomer_ids,
            monomer_select_plot,
            "listeners__monomer_counts",
            "monomer_counts",
            "protein names",
            "counts",
            datapoints_cap,
            conn,
            monomer_names,
            monomer_label_type,
        )

    return (plot_df_monomers,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    monomer_select_plot,
    pl,
    plot_df_monomers,
    pms_value,
    y_scale_monomers,
):
    chart_monomers = None
    if pms_value(monomer_select_plot) and plot_df_monomers is not None:
        _n = len(active_datasets)
        _base = (
            alt.Chart(plot_df_monomers)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "counts:Q",
                    scale=alt.Scale(type=y_scale_monomers.value),
                    title="Counts",
                ),
                color=alt.Color("protein names:N", legend=alt.Legend(title="Proteins")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_monomers = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_monomers.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "counts:Q",
                            scale=alt.Scale(type=y_scale_monomers.value),
                            title="Counts",
                        ),
                        color=alt.Color(
                            "protein names:N", legend=alt.Legend(title="Proteins")
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_monomers = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_monomers = _base
    return (chart_monomers,)


@app.cell
def _(sim_data):
    # Catalyst (enzyme monomer/complex) IDs — the index `catalyst_counts` uses.
    # Different from `base_reaction_ids` (which `base_reaction_fluxes` uses),
    # so the entity picker has to switch lists when the user toggles quantity.
    catalyst_ids = list(sim_data.process.metabolism.catalyst_ids)

    # User-facing labels for catalyst-count picker: "REACTION (CATALYST)".
    # A catalyst can catalyze several reactions, so each catalyst can appear
    # multiple times (one entry per reaction-catalyst pair, ~8,300 entries).
    # `catalyst_label_to_id` maps each display label back to the underlying
    # catalyst_id; we need that for column subscripts since the parquet
    # vector is indexed by catalyst, not reaction.
    _rxn_to_cats = sim_data.process.metabolism.reaction_catalysts
    catalyst_display_labels = []
    catalyst_label_to_id = {}
    for _rxn_id, _cats in _rxn_to_cats.items():
        for _cat_id in _cats:
            _label = f"{_rxn_id} ({_cat_id})"
            catalyst_display_labels.append(_label)
            catalyst_label_to_id[_label] = _cat_id
    catalyst_display_labels = sorted(catalyst_display_labels)
    return catalyst_display_labels, catalyst_ids, catalyst_label_to_id


@app.cell
def _(mo):
    about_metabolism_md = mo.md(
        "Metabolic quantities with **separate** index lists per quantity. "
        "The entity picker auto-switches:\n\n"
        "- **Reaction flux** (`base_reaction_fluxes`, mmol/s) is indexed by "
        "  `base_reaction_ids` (~2,800 reactions). Pathway selection "
        "  auto-populates this view.\n"
        "- **Catalyst counts** (`catalyst_counts`, # of enzyme molecules) is "
        "  indexed by `catalyst_ids` (~1,500 enzyme monomers / complexes with "
        "  compartment suffixes like `[c]`, `[m]`, `[i]`). Pathway preset "
        "  doesn't apply here — search the picker manually.\n\n"
        "For a kcat-style read of how hard each enzyme is working, flip "
        "between the two quantities for related entities."
    )
    return (about_metabolism_md,)


@app.cell
def _():
    # Display name → (listener_column, dtype, y_axis_title, index_kind).
    # index_kind: "reaction" → rxn_ids; "catalyst" → catalyst_ids.
    metabolism_quantities = {
        "Reaction flux (mmol/s)": (
            "listeners__fba_results__base_reaction_fluxes",
            "FLOAT",
            "Reaction Flux (mmol/s)",
            "reaction",
        ),
        "Catalyst counts (# enzyme)": (
            "listeners__fba_results__catalyst_counts",
            "BIGINT",
            "Catalyst counts",
            "catalyst",
        ),
    }
    return (metabolism_quantities,)


@app.cell
def _(metabolism_quantities, mo):
    # Defining cell: just the dropdown. Reading `.value` in the same cell that
    # creates the widget is a marimo anti-pattern — value changes don't
    # re-trigger the defining cell, and on 0.13 it can cause the cell to
    # error and take downstream consumers (the Tabs cell) with it.
    metabolism_quantity_select = mo.ui.dropdown(
        options=list(metabolism_quantities.keys()),
        value="Reaction flux (mmol/s)",
    )
    y_scale_rxns = mo.ui.dropdown(options=["linear", "log", "symlog"], value="symlog")
    return metabolism_quantity_select, y_scale_rxns


@app.cell
def _(
    catalyst_display_labels,
    metabolism_quantities,
    metabolism_quantity_select,
    mo,
    pms,
    rxn_ids,
    rxn_override,
    select_pathway,
):
    # Downstream cell: rebuilds the multiselect whenever the quantity dropdown
    # changes, switching options + default between reaction and catalyst index.
    # Catalyst mode shows "REACTION (CATALYST)" display labels (one per
    # reaction-catalyst pair); plot_df_rxns translates these back to
    # catalyst_ids for SQL.
    _, _, _, _kind = metabolism_quantities[metabolism_quantity_select.value]
    if _kind == "reaction":
        _opts = rxn_ids
        _default = rxn_override(select_pathway.value)
    else:  # catalyst
        _opts = catalyst_display_labels
        _default = None  # pathway preset is reaction-keyed, doesn't map here
    select_rxns = pms(options=_opts, value=_default, max_selections=500)
    return (select_rxns,)


@app.cell
def _(
    active_datasets,
    catalyst_ids,
    catalyst_label_to_id,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    metabolism_quantities,
    metabolism_quantity_select,
    pms_value,
    rxn_ids,
    select_rxns,
):
    from types import SimpleNamespace as _NS

    plot_df_rxns = None
    if pms_value(select_rxns):
        _listener, _dtype, _, _kind = metabolism_quantities[
            metabolism_quantity_select.value
        ]
        if _kind == "catalyst":
            # User picked display labels like "RXN-X (CAT-A)". Translate to
            # catalyst_ids for the SQL subscripts, but preserve the original
            # display labels for the chart legend via the display_labels arg.
            _display = list(pms_value(select_rxns))
            _cat_ids = [catalyst_label_to_id[lbl] for lbl in _display]
            _fake_selector = _NS(value=_cat_ids)
            _fake_label_ui = _NS(value="BioCyc ID")
            plot_df_rxns = get_plot_df_multi(
                active_datasets,
                catalyst_ids,
                _fake_selector,
                _listener,
                "reaction_quantity",
                "reaction_id",
                "flux",
                datapoints_cap,
                conn,
                label_ui=_fake_label_ui,
                dtype=_dtype,
                display_labels=_display,
            )
        else:  # reaction
            plot_df_rxns = get_plot_df_multi(
                active_datasets,
                rxn_ids,
                select_rxns,
                _listener,
                "reaction_quantity",
                "reaction_id",
                "flux",
                datapoints_cap,
                conn,
                dtype=_dtype,
            )
    return (plot_df_rxns,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    metabolism_quantities,
    metabolism_quantity_select,
    mo,
    pl,
    plot_df_rxns,
    pms_value,
    select_rxns,
    y_scale_rxns,
):
    chart_rxns = None
    if pms_value(select_rxns) and plot_df_rxns is not None:
        _n = len(active_datasets)
        _y_title = metabolism_quantities[metabolism_quantity_select.value][2]
        _base = (
            alt.Chart(plot_df_rxns)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "flux:Q", scale=alt.Scale(type=y_scale_rxns.value), title=_y_title
                ),
                color=alt.Color(
                    "reaction_id:N", legend=alt.Legend(title="Reaction ID (BioCyc)")
                ),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_rxns = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_rxns.filter(pl.col("dataset_label") == _ds["label"])
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "flux:Q",
                            scale=alt.Scale(type=y_scale_rxns.value),
                            title=_y_title,
                        ),
                        color=alt.Color(
                            "reaction_id:N",
                            legend=alt.Legend(title="Reaction ID (BioCyc)"),
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_rxns = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_rxns = _base
    return (chart_rxns,)


# ===== Physiology tab (scalar listeners: mass, volume, machinery counts) =====


@app.cell
def _(scalar_cols):
    # Classify scalar physiology listeners into categories with intuitive
    # display labels (units included). Each category groups quantities with
    # comparable units / scale, so the user can't accidentally plot a mass
    # (fg) and a fraction (0-1) on the same axes.
    #
    # KNOWN is a curated mapping for the listeners we expect; anything not
    # in KNOWN falls through to heuristic classification so the tab still
    # works on datasets that emit columns we haven't catalogued yet.
    KNOWN = {
        # ----- Mass (absolute) — femtograms -----
        "listeners__mass__cell_mass": ("Mass (absolute)", "Cell mass (fg)"),
        "listeners__mass__dry_mass": ("Mass (absolute)", "Dry mass (fg)"),
        "listeners__mass__water_mass": ("Mass (absolute)", "Water mass (fg)"),
        "listeners__mass__protein_mass": ("Mass (absolute)", "Protein mass (fg)"),
        "listeners__mass__rna_mass": ("Mass (absolute)", "RNA mass (fg)"),
        "listeners__mass__rRna_mass": ("Mass (absolute)", "rRNA mass (fg)"),
        "listeners__mass__tRna_mass": ("Mass (absolute)", "tRNA mass (fg)"),
        "listeners__mass__mRna_mass": ("Mass (absolute)", "mRNA mass (fg)"),
        "listeners__mass__dna_mass": ("Mass (absolute)", "DNA mass (fg)"),
        "listeners__mass__smallMolecule_mass": (
            "Mass (absolute)",
            "Small molecule mass (fg)",
        ),
        "listeners__mass__inner_membrane_mass": (
            "Mass (absolute)",
            "Inner membrane mass (fg)",
        ),
        "listeners__mass__outer_membrane_mass": (
            "Mass (absolute)",
            "Outer membrane mass (fg)",
        ),
        "listeners__mass__cellWall_mass": ("Mass (absolute)", "Cell wall mass (fg)"),
        "listeners__mass__projection_mass": ("Mass (absolute)", "Projection mass (fg)"),
        "listeners__mass__pilus_mass": ("Mass (absolute)", "Pilus mass (fg)"),
        "listeners__mass__flagellum_mass": ("Mass (absolute)", "Flagellum mass (fg)"),
        # ----- Volume -----
        "listeners__mass__cell_volume": ("Volume", "Cell volume (L)"),
        # ----- Growth rates -----
        "listeners__mass__growth": ("Growth rate", "Growth (fg/s)"),
        "listeners__mass__instantaneous_growth_rate": (
            "Growth rate",
            "Instantaneous growth rate (1/s)",
        ),
        # ----- Mass fractions / ratios / fold changes -----
        "listeners__mass__protein_mass_fraction": (
            "Fractions / ratios",
            "Protein mass fraction",
        ),
        "listeners__mass__rna_mass_fraction": (
            "Fractions / ratios",
            "RNA mass fraction",
        ),
        "listeners__mass__mRna_mass_fraction": (
            "Fractions / ratios",
            "mRNA mass fraction",
        ),
        "listeners__mass__rRna_mass_fraction": (
            "Fractions / ratios",
            "rRNA mass fraction",
        ),
        "listeners__mass__tRna_mass_fraction": (
            "Fractions / ratios",
            "tRNA mass fraction",
        ),
        "listeners__mass__dna_mass_fraction": (
            "Fractions / ratios",
            "DNA mass fraction",
        ),
        "listeners__mass__smallMolecule_mass_fraction": (
            "Fractions / ratios",
            "Small molecule mass fraction",
        ),
        "listeners__mass__cellMass_fold_change": (
            "Fractions / ratios",
            "Cell mass fold change",
        ),
        "listeners__mass__proteinMass_fold_change": (
            "Fractions / ratios",
            "Protein mass fold change",
        ),
        "listeners__mass__rnaMass_fold_change": (
            "Fractions / ratios",
            "RNA mass fold change",
        ),
        "listeners__mass__dnaMass_fold_change": (
            "Fractions / ratios",
            "DNA mass fold change",
        ),
        # ----- Machinery (snapshot counts) -----
        "listeners__rnap_data__active_rnap_count": ("Machinery counts", "Active RNAPs"),
        "listeners__ribosome_data__active_ribosome_count": (
            "Machinery counts",
            "Active ribosomes",
        ),
        "listeners__ribosome_data__total_ribosomes": (
            "Machinery counts",
            "Total ribosomes",
        ),
        # ----- Event counters (per timestep) -----
        "listeners__rnap_data__didTerminate": (
            "Event counts (per step)",
            "RNAP terminations",
        ),
        "listeners__rnap_data__didStall": ("Event counts (per step)", "RNAP stalls"),
        "listeners__rnap_data__didInitialize": (
            "Event counts (per step)",
            "RNAP initiations",
        ),
        "listeners__ribosome_data__didTerminate": (
            "Event counts (per step)",
            "Ribosome terminations",
        ),
        "listeners__ribosome_data__didStall": (
            "Event counts (per step)",
            "Ribosome stalls",
        ),
        "listeners__ribosome_data__didInitialize": (
            "Event counts (per step)",
            "Ribosome initiations",
        ),
        # ----- Rates -----
        "listeners__ribosome_data__effective_elongation_rate": (
            "Elongation rates",
            "Ribosome effective elongation (aa/s)",
        ),
        "listeners__rnap_data__effective_elongation_rate": (
            "Elongation rates",
            "RNAP effective elongation (nt/s)",
        ),
        "listeners__ribosome_data__expected_rate_change": (
            "Elongation rates",
            "Ribosome expected rate change",
        ),
    }

    def _prettify(short):
        out = short.replace("_", " ").replace("Rna", "RNA").replace("Dna", "DNA")
        return out[:1].upper() + out[1:] if out else out

    def _classify(col):
        if col in KNOWN:
            return KNOWN[col]
        short = col.rsplit("__", 1)[-1]
        lower = short.lower()
        if "fraction" in lower or "ratio" in lower or "fold_change" in lower:
            return ("Fractions / ratios", _prettify(short))
        if "volume" in lower:
            return ("Volume", _prettify(short) + " (L)")
        if "mass" in lower:
            return ("Mass (absolute)", _prettify(short) + " (fg)")
        if "growth" in lower:
            return ("Growth rate", _prettify(short))
        if "elongation_rate" in lower or lower.endswith("_rate"):
            return ("Elongation rates", _prettify(short))
        if (
            "init" in lower
            or "terminat" in lower
            or "stall" in lower
            or "event" in lower
            or "abort" in lower
        ):
            return ("Event counts (per step)", _prettify(short))
        if lower.startswith(("active_", "total_", "n_")) or lower.endswith("_count"):
            return ("Machinery counts", _prettify(short))
        return ("Other", _prettify(short))

    def classify_scalar_by_prefix(prefixes):
        """Filter `scalar_cols` by prefixes, classify each with KNOWN +
        heuristic, and return (categories_in_order, classified_dict,
        label_to_col_dict). Reused by Physiology (mass only), Transcription
        (rnap_data), and Translation (ribosome_data)."""
        classified = {}
        label_to_col = {}
        for col in sorted(c for c in scalar_cols if c.startswith(prefixes)):
            cat, lbl = _classify(col)
            # Avoid label collisions across categories — append the column
            # suffix if a label is already taken.
            key = (
                lbl if lbl not in label_to_col else f"{lbl} [{col.rsplit('__', 1)[-1]}]"
            )
            classified.setdefault(cat, []).append((col, key))
            label_to_col[key] = col
        order = [
            "Mass (absolute)",
            "Fractions / ratios",
            "Volume",
            "Growth rate",
            "Machinery counts",
            "Event counts (per step)",
            "Elongation rates",
            "Other",
        ]
        categories = [c for c in order if c in classified]
        return categories, classified, label_to_col

    # Physiology tab now only surfaces `listeners__mass__` (cell mass/volume/
    # growth/fractions). RNAP scalars moved to Transcription, ribosome scalars
    # to Translation — each tab gets its own scalar picker via the same helper.
    physiology_categories, physiology_classified, physiology_label_to_col = (
        classify_scalar_by_prefix(("listeners__mass__",))
    )
    return (
        classify_scalar_by_prefix,
        physiology_categories,
        physiology_classified,
        physiology_label_to_col,
    )


@app.cell
def _(mo, physiology_categories):
    physiology_category = mo.ui.dropdown(
        options=physiology_categories,
        value=physiology_categories[0] if physiology_categories else None,
    )
    return (physiology_category,)


# ===== Scalar RNAP / ribosome sections (belong on Transcription / Translation
# tabs respectively, not Physiology, since they're the machinery for those
# processes). Reuses the physiology classifier + get_plot_df_scalar_multi. =====


@app.cell
def _(classify_scalar_by_prefix):
    # RNAP scalar quantities (active_rnap_count, didInitialize/Terminate/Stall,
    # effective_elongation_rate, ...) for the Transcription tab.
    (
        transcription_scalar_categories,
        transcription_scalar_classified,
        transcription_scalar_label_to_col,
    ) = classify_scalar_by_prefix(("listeners__rnap_data__",))
    return (
        transcription_scalar_categories,
        transcription_scalar_classified,
        transcription_scalar_label_to_col,
    )


@app.cell
def _(mo, transcription_scalar_categories):
    transcription_scalar_category = mo.ui.dropdown(
        options=transcription_scalar_categories,
        value=(
            transcription_scalar_categories[0]
            if transcription_scalar_categories
            else None
        ),
    )
    return (transcription_scalar_category,)


@app.cell
def _(
    mo,
    pms,
    transcription_scalar_category,
    transcription_scalar_classified,
):
    _items = transcription_scalar_classified.get(
        transcription_scalar_category.value, []
    )
    _labels = [lbl for (_col, lbl) in _items]
    transcription_scalar_select = pms(
        options=_labels, value=_labels[:1], max_selections=20
    )
    y_scale_transcription_scalar = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="linear"
    )
    return transcription_scalar_select, y_scale_transcription_scalar


@app.cell
def _(classify_scalar_by_prefix):
    # Ribosome scalar quantities for the Translation tab.
    (
        translation_scalar_categories,
        translation_scalar_classified,
        translation_scalar_label_to_col,
    ) = classify_scalar_by_prefix(("listeners__ribosome_data__",))
    return (
        translation_scalar_categories,
        translation_scalar_classified,
        translation_scalar_label_to_col,
    )


@app.cell
def _(mo, translation_scalar_categories):
    translation_scalar_category = mo.ui.dropdown(
        options=translation_scalar_categories,
        value=(
            translation_scalar_categories[0] if translation_scalar_categories else None
        ),
    )
    return (translation_scalar_category,)


@app.cell
def _(
    mo,
    pms,
    translation_scalar_category,
    translation_scalar_classified,
):
    _items = translation_scalar_classified.get(translation_scalar_category.value, [])
    _labels = [lbl for (_col, lbl) in _items]
    translation_scalar_select = pms(
        options=_labels, value=_labels[:1], max_selections=20
    )
    y_scale_translation_scalar = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="linear"
    )
    return translation_scalar_select, y_scale_translation_scalar


@app.cell
def _(
    mo,
    physiology_category,
    physiology_classified,
    pms,
):
    _items = physiology_classified.get(physiology_category.value, [])
    _labels = [lbl for (_col, lbl) in _items]
    physiology_select = pms(
        options=_labels,
        value=_labels[:1],  # first quantity in the category by default
        max_selections=20,
    )
    y_scale_physiology = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="linear"
    )
    return physiology_select, y_scale_physiology


@app.cell
def _(mo):
    about_physiology_md = mo.md(
        "Time course of whole-cell physiology quantities from "
        "`listeners__mass__*` — cell / dry / component masses (fg), volume "
        "(L), growth rates, and mass fractions / fold-changes "
        "(dimensionless). Use **category** to scope the quantity picker to "
        "compatible units so you don't plot a 1.0 fraction next to a 400 fg "
        "cell mass on the same axis. RNAP scalar machinery / events live in "
        "the **Transcription** tab; ribosome ones in **Translation**."
    )
    return (about_physiology_md,)


@app.cell
def _(pl):
    def get_plot_df_scalar_multi(
        active_datasets, quantities, conn, datapoints_cap=2000
    ):
        """For each dataset, select chosen scalar columns with row-stride
        downsampling (so a long simulation × many quantities doesn't blow
        through marimo's output cap when embedded in an Altair chart), melt
        to long form, tag with dataset_label, then concat. Returns None if
        no quantities or no datasets yielded data."""
        if not quantities:
            return None
        _quoted = ", ".join(f'"{q}"' for q in quantities)
        dfs = []
        for ds in active_datasets:
            _base_sql = (
                f"SELECT time, {_quoted} FROM ({ds['history_sql']}) "
                f"WHERE {ds['db_filter']}"
            )
            # Downsample by row stride: rn % CEILING(time_points * n_qty /
            # cap) so total rendered points across all quantities stays at or
            # below the cap. GREATEST(...,1) guards against zero rows.
            _sql = f"""
                WITH indexed_data AS (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY time) AS rn
                    FROM ({_base_sql})
                ),
                data_shape AS (
                    SELECT
                        COUNT(*) AS time_points,
                        time_points * {len(quantities)} AS data_points,
                        CEILING(data_points / {float(datapoints_cap)})
                            AS ds_ratio
                    FROM ({_base_sql})
                )
                SELECT time, {_quoted}
                FROM indexed_data
                CROSS JOIN data_shape
                WHERE rn % GREATEST(data_shape.ds_ratio, 1) = 0
                ORDER BY time
            """
            try:
                df = conn.sql(_sql).pl()
            except Exception:
                continue
            if len(df) == 0:
                continue
            df_long = df.unpivot(
                index="time", variable_name="quantity", value_name="value"
            )
            df_long = df_long.with_columns(pl.lit(ds["label"]).alias("dataset_label"))
            dfs.append(df_long)
        return pl.concat(dfs) if dfs else None

    return (get_plot_df_scalar_multi,)


@app.cell
def _(
    active_datasets,
    conn,
    get_plot_df_scalar_multi,
    physiology_label_to_col,
    physiology_select,
    pl,
    pms_value,
):
    plot_df_physiology = None
    if pms_value(physiology_select):
        # Translate display labels → raw listener columns for SQL.
        _cols = [
            physiology_label_to_col[lbl]
            for lbl in pms_value(physiology_select)
            if lbl in physiology_label_to_col
        ]
        plot_df_physiology = get_plot_df_scalar_multi(active_datasets, _cols, conn)
        # Rename `quantity` (column name) → display label so the chart legend
        # shows intuitive names with units.
        if plot_df_physiology is not None:
            _col_to_lbl = {v: k for k, v in physiology_label_to_col.items()}
            plot_df_physiology = plot_df_physiology.with_columns(
                pl.col("quantity").replace(_col_to_lbl)
            )
    return (plot_df_physiology,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    physiology_select,
    pl,
    plot_df_physiology,
    pms_value,
    y_scale_physiology,
):
    chart_physiology = None
    if pms_value(physiology_select) and plot_df_physiology is not None:
        _n = len(active_datasets)
        _base = (
            alt.Chart(plot_df_physiology)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_physiology.value),
                    title="Value",
                ),
                color=alt.Color("quantity:N", legend=alt.Legend(title="Quantity")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_physiology = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_physiology.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_physiology.value),
                            title="Value",
                        ),
                        color=alt.Color(
                            "quantity:N", legend=alt.Legend(title="Quantity")
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_physiology = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_physiology = _base
    return (chart_physiology,)


# ---- RNAP scalar plot_df + chart (rendered in Transcription tab) ----


@app.cell
def _(
    active_datasets,
    conn,
    get_plot_df_scalar_multi,
    pl,
    pms_value,
    transcription_scalar_label_to_col,
    transcription_scalar_select,
):
    plot_df_transcription_scalar = None
    if pms_value(transcription_scalar_select):
        _cols = [
            transcription_scalar_label_to_col[lbl]
            for lbl in pms_value(transcription_scalar_select)
            if lbl in transcription_scalar_label_to_col
        ]
        plot_df_transcription_scalar = get_plot_df_scalar_multi(
            active_datasets, _cols, conn
        )
        if plot_df_transcription_scalar is not None:
            _col_to_lbl = {v: k for k, v in transcription_scalar_label_to_col.items()}
            plot_df_transcription_scalar = plot_df_transcription_scalar.with_columns(
                pl.col("quantity").replace(_col_to_lbl)
            )
    return (plot_df_transcription_scalar,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    pl,
    plot_df_transcription_scalar,
    pms_value,
    transcription_scalar_select,
    y_scale_transcription_scalar,
):
    chart_transcription_scalar = None
    if (
        pms_value(transcription_scalar_select)
        and plot_df_transcription_scalar is not None
    ):
        _n = len(active_datasets)
        _base = (
            alt.Chart(plot_df_transcription_scalar)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_transcription_scalar.value),
                    title="Value",
                ),
                color=alt.Color("quantity:N", legend=alt.Legend(title="Quantity")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            chart_transcription_scalar = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_transcription_scalar.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_transcription_scalar.value),
                            title="Value",
                        ),
                        color=alt.Color(
                            "quantity:N", legend=alt.Legend(title="Quantity")
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_transcription_scalar = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_transcription_scalar = _base
    return (chart_transcription_scalar,)


# ---- Ribosome scalar plot_df + chart (rendered in Translation tab) ----


@app.cell
def _(
    active_datasets,
    conn,
    get_plot_df_scalar_multi,
    pl,
    pms_value,
    translation_scalar_label_to_col,
    translation_scalar_select,
):
    plot_df_translation_scalar = None
    if pms_value(translation_scalar_select):
        _cols = [
            translation_scalar_label_to_col[lbl]
            for lbl in pms_value(translation_scalar_select)
            if lbl in translation_scalar_label_to_col
        ]
        plot_df_translation_scalar = get_plot_df_scalar_multi(
            active_datasets, _cols, conn
        )
        if plot_df_translation_scalar is not None:
            _col_to_lbl = {v: k for k, v in translation_scalar_label_to_col.items()}
            plot_df_translation_scalar = plot_df_translation_scalar.with_columns(
                pl.col("quantity").replace(_col_to_lbl)
            )
    return (plot_df_translation_scalar,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    pl,
    plot_df_translation_scalar,
    pms_value,
    translation_scalar_select,
    y_scale_translation_scalar,
):
    chart_translation_scalar = None
    if pms_value(translation_scalar_select) and plot_df_translation_scalar is not None:
        _n = len(active_datasets)
        _base = (
            alt.Chart(plot_df_translation_scalar)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_translation_scalar.value),
                    title="Value",
                ),
                color=alt.Color("quantity:N", legend=alt.Legend(title="Quantity")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            chart_translation_scalar = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_translation_scalar.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_translation_scalar.value),
                            title="Value",
                        ),
                        color=alt.Color(
                            "quantity:N", legend=alt.Legend(title="Quantity")
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_translation_scalar = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_translation_scalar = _base
    return (chart_translation_scalar,)


# ===== Transcription tab (per-cistron vectors: init events, degradation,
# expected init, gene copy number) =====


@app.cell
def _():
    # Display name → listener column. All four are per-cistron (length =
    # mRNA cistron count) so the existing mRNA gene picker can index them.
    transcription_quantities = {
        "RNA init events (per cistron)": "listeners__rnap_data__rna_init_event_per_cistron",
        "RNA degradation events (per cistron)": "listeners__rna_degradation_listener__count_RNA_degraded_per_cistron",
        "Expected RNA init (per cistron)": "listeners__rna_synth_prob__expected_rna_init_per_cistron",
        "Gene copy number (per cistron)": "listeners__rna_synth_prob__gene_copy_number",
    }
    return (transcription_quantities,)


@app.cell
def _(mo, transcription_quantities):
    transcription_quantity_select = mo.ui.dropdown(
        options=list(transcription_quantities.keys()),
        value="RNA init events (per cistron)",
    )
    transcription_label_type = mo.ui.radio(
        options=["gene name", "BioCyc ID"], value="gene name"
    )
    y_scale_transcription = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="symlog"
    )
    return (
        transcription_label_type,
        transcription_quantity_select,
        y_scale_transcription,
    )


@app.cell
def _(
    mo,
    mrna_cistron_names,
    mrna_gene_ids,
    mrna_override,
    pms,
    select_pathway,
    transcription_label_type,
):
    if transcription_label_type.value == "gene name":
        _opts = mrna_cistron_names
    else:
        _opts = mrna_gene_ids
    transcription_select_plot = pms(
        options=_opts,
        value=mrna_override(select_pathway.value),
        max_selections=500,
    )
    return (transcription_select_plot,)


@app.cell
def _(mo):
    about_transcription_md = mo.md(
        "Time course of per-cistron transcription quantities — initiation "
        "events, degradation events, expected initiation, and gene copy "
        "number. Pair with the mRNA tab to see drivers of mRNA count changes "
        "(init - degradation = net synthesis). Gene selection mirrors the "
        "mRNA tab; pathway selection above auto-populates."
    )
    return (about_transcription_md,)


@app.cell
def _(
    active_datasets,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    mrna_cistron_names,
    mrna_gene_ids,
    pms_value,
    transcription_label_type,
    transcription_quantities,
    transcription_quantity_select,
    transcription_select_plot,
):
    plot_df_transcription = None
    if pms_value(transcription_select_plot):
        _listener = transcription_quantities[transcription_quantity_select.value]
        # Choose dtype based on what we know about each listener (USE_UINT16/32
        # listeners are integers; expected_rna_init is a float).
        _dtype = "FLOAT" if "expected" in _listener else "BIGINT"
        plot_df_transcription = get_plot_df_multi(
            active_datasets,
            mrna_gene_ids,
            transcription_select_plot,
            _listener,
            "transcription_counts",
            "Genes",
            "value",
            datapoints_cap,
            conn,
            mrna_cistron_names,
            transcription_label_type,
            dtype=_dtype,
        )
    return (plot_df_transcription,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    pl,
    plot_df_transcription,
    pms_value,
    transcription_quantity_select,
    transcription_select_plot,
    y_scale_transcription,
):
    chart_transcription = None
    if pms_value(transcription_select_plot) and plot_df_transcription is not None:
        _n = len(active_datasets)
        _y_title = transcription_quantity_select.value
        _base = (
            alt.Chart(plot_df_transcription)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_transcription.value),
                    title=_y_title,
                ),
                color=alt.Color("Genes:N", legend=alt.Legend(title="Genes")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_transcription = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_transcription.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_transcription.value),
                            title=_y_title,
                        ),
                        color=alt.Color("Genes:N", legend=alt.Legend(title="Genes")),
                    )
                    .properties(width="container", height=220)
                )
            chart_transcription = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_transcription = _base
    return (chart_transcription,)


# ===== Translation tab (per-monomer + per-transcript vectors) =====


@app.cell
def _(sim_data):
    # mRNA TU IDs/names for per-transcript listeners. Filters rna_data to
    # mRNAs when possible; falls back to all RNAs if the is_mRNA field is
    # missing under that exact name.
    _rna_data = sim_data.process.transcription.rna_data
    _field_names = _rna_data.dtype.names if hasattr(_rna_data, "dtype") else None
    _mrna_mask = None
    if _field_names:
        for _fname in ("is_mRNA", "isMRNA", "isMRna", "is_m_rna"):
            if _fname in _field_names:
                _mrna_mask = _rna_data[_fname]
                break
    if _mrna_mask is None:
        _ids = _rna_data["id"].tolist()
    else:
        _ids = _rna_data["id"][_mrna_mask].tolist()
    mrna_tu_ids = [i[:-3] for i in _ids]
    mrna_tu_names = [sim_data.common_names.get_common_name(i) for i in _ids]
    return mrna_tu_ids, mrna_tu_names


@app.cell
def _():
    # Display name → (listener_column, index_kind).
    # index_kind: "monomer" → uses monomer_ids/monomer_names;
    #             "transcript" → uses mrna_tu_ids/mrna_tu_names.
    translation_quantities = {
        "Ribosome init events (per monomer)": (
            "listeners__ribosome_data__ribosome_init_event_per_monomer",
            "monomer",
        ),
        "Ribosomes per transcript (per TU)": (
            "listeners__ribosome_data__n_ribosomes_per_transcript",
            "transcript",
        ),
        "Ribosomes on partial mRNA per transcript (per TU)": (
            "listeners__ribosome_data__n_ribosomes_on_partial_mRNA_per_transcript",
            "transcript",
        ),
    }
    return (translation_quantities,)


@app.cell
def _(mo, translation_quantities):
    translation_quantity_select = mo.ui.dropdown(
        options=list(translation_quantities.keys()),
        value="Ribosome init events (per monomer)",
    )
    translation_label_type = mo.ui.radio(
        options=["common name", "BioCyc ID"], value="common name"
    )
    y_scale_translation = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="symlog"
    )
    return (
        translation_label_type,
        translation_quantity_select,
        y_scale_translation,
    )


@app.cell
def _(
    mo,
    monomer_ids,
    monomer_names,
    mrna_override,
    mrna_tu_ids,
    mrna_tu_names,
    pms,
    protein_override,
    select_pathway,
    translation_label_type,
    translation_quantities,
    translation_quantity_select,
):
    _listener, _kind = translation_quantities[translation_quantity_select.value]
    if _kind == "monomer":
        _opts = (
            monomer_names
            if translation_label_type.value == "common name"
            else monomer_ids
        )
        _default = protein_override(select_pathway.value)
    else:  # "transcript" — mRNA TU index
        _opts = (
            mrna_tu_names
            if translation_label_type.value == "common name"
            else mrna_tu_ids
        )
        _default = mrna_override(select_pathway.value)
    # Pathway helpers return values keyed for monomer/cistron names; if any
    # of those don't appear in the current `_opts` list (e.g. wrong index
    # kind), drop them so the multiselect doesn't reject the default.
    _default = [v for v in (_default or []) if v in _opts]
    translation_select_plot = pms(
        options=_opts,
        value=_default,
        max_selections=500,
    )
    return (translation_select_plot,)


@app.cell
def _(mo):
    about_translation_md = mo.md(
        "Time course of translation machinery quantities. **Ribosome init "
        "events per monomer** is the supply-side companion to the **Protein** "
        "tab (initiations → monomer counts). **Ribosomes per transcript** and "
        "**Ribosomes on partial mRNA per transcript** show how loaded each "
        "mRNA TU is. The entity picker auto-switches between monomers and "
        "mRNA TUs based on the quantity chosen. Pathway selection above "
        "auto-populates; non-matching entries are dropped silently."
    )
    return (about_translation_md,)


@app.cell
def _(
    active_datasets,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    monomer_ids,
    monomer_names,
    mrna_tu_ids,
    mrna_tu_names,
    pms_value,
    translation_label_type,
    translation_quantities,
    translation_quantity_select,
    translation_select_plot,
):
    plot_df_translation = None
    if pms_value(translation_select_plot):
        _listener, _kind = translation_quantities[translation_quantity_select.value]
        if _kind == "monomer":
            _id_list = monomer_ids
            _name_list = monomer_names
        else:
            _id_list = mrna_tu_ids
            _name_list = mrna_tu_names
        plot_df_translation = get_plot_df_multi(
            active_datasets,
            _id_list,
            translation_select_plot,
            _listener,
            "translation_counts",
            "Entity",
            "value",
            datapoints_cap,
            conn,
            _name_list,
            translation_label_type,
            dtype="BIGINT",
        )
    return (plot_df_translation,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    pl,
    plot_df_translation,
    pms_value,
    translation_quantity_select,
    translation_select_plot,
    y_scale_translation,
):
    chart_translation = None
    if pms_value(translation_select_plot) and plot_df_translation is not None:
        _n = len(active_datasets)
        _y_title = translation_quantity_select.value
        _base = (
            alt.Chart(plot_df_translation)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_translation.value),
                    title=_y_title,
                ),
                color=alt.Color("Entity:N", legend=alt.Legend(title="Entity")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_translation = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_translation.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_translation.value),
                            title=_y_title,
                        ),
                        color=alt.Color("Entity:N", legend=alt.Legend(title="Entity")),
                    )
                    .properties(width="container", height=220)
                )
            chart_translation = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_translation = _base
    return (chart_translation,)


# ===== Regulation tab (TF binding per cistron / per TU) =====


@app.cell
def _(sim_data):
    # Full cistron and TU ID/name lists (unfiltered). TF-binding listeners are
    # indexed by ALL cistrons / ALL TUs — not just the mRNA-positive subset
    # the mRNA / Translation tabs use — so we need the unfiltered position
    # arrays for the column subscripts to line up.
    _cistron_data = sim_data.process.transcription.cistron_data
    _rna_data = sim_data.process.transcription.rna_data
    _cistron_ids_raw = _cistron_data["id"].tolist()
    _rna_ids_raw = _rna_data["id"].tolist()
    all_cistron_ids = [c[:-3] for c in _cistron_ids_raw]
    all_cistron_names = [
        sim_data.common_names.get_common_name(c) for c in _cistron_ids_raw
    ]
    all_tu_ids = [r[:-3] for r in _rna_ids_raw]
    all_tu_names = [sim_data.common_names.get_common_name(r) for r in _rna_ids_raw]
    return all_cistron_ids, all_cistron_names, all_tu_ids, all_tu_names


@app.cell
def _():
    # Display name → (sql_expression, index_kind).
    # `n_bound_TF_per_cistron` and `n_bound_TF_per_TU` are 2-D matrices in
    # parquet (USMALLINT[][]): the per-cistron one is (23 TFs × 4538 cistrons)
    # and the per-TU one is (3271 TUs × 23 TFs). The existing plotting
    # pipeline expects 1-D vectors, so we aggregate the TF axis in SQL
    # before it reaches the entity-subscript step:
    #   * per_cistron: sum ACROSS TFs → 4538-long per-cistron vector via
    #     element-wise list_reduce (list_zip + list_transform).
    #   * per_tu: sum INSIDE each TU's TF row → 3271-long per-TU vector via
    #     list_transform(row -> list_sum(row)).
    # Values are TOTAL TFs bound (summed over all TF species). If a per-TF
    # breakdown is ever needed, a separate quantity + TF picker can be added.
    _per_cistron_expr = (
        "list_reduce("
        "listeners__rna_synth_prob__n_bound_TF_per_cistron, "
        "(a, b) -> list_transform(list_zip(a, b), x -> x[1] + x[2]))"
    )
    _per_tu_expr = (
        "list_transform("
        "listeners__rna_synth_prob__n_bound_TF_per_TU, "
        "row -> list_sum(row))"
    )
    regulation_quantities = {
        "Total TFs bound (per cistron)": (_per_cistron_expr, "cistron"),
        "Total TFs bound (per TU)": (_per_tu_expr, "tu"),
    }
    return (regulation_quantities,)


@app.cell
def _(mo, regulation_quantities):
    regulation_quantity_select = mo.ui.dropdown(
        options=list(regulation_quantities.keys()),
        value="Total TFs bound (per cistron)",
    )
    regulation_label_type = mo.ui.radio(
        options=["common name", "BioCyc ID"], value="common name"
    )
    y_scale_regulation = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="linear"
    )
    return (
        regulation_label_type,
        regulation_quantity_select,
        y_scale_regulation,
    )


@app.cell
def _(
    all_cistron_ids,
    all_cistron_names,
    all_tu_ids,
    all_tu_names,
    mo,
    mrna_override,
    pms,
    regulation_label_type,
    regulation_quantities,
    regulation_quantity_select,
    select_pathway,
):
    _listener, _kind = regulation_quantities[regulation_quantity_select.value]
    if _kind == "cistron":
        _opts = (
            all_cistron_names
            if regulation_label_type.value == "common name"
            else all_cistron_ids
        )
    else:  # tu
        _opts = (
            all_tu_names if regulation_label_type.value == "common name" else all_tu_ids
        )
    # Pathway preset returns mRNA cistron labels; if any aren't in the
    # current (broader) option list, drop silently.
    _default = [v for v in (mrna_override(select_pathway.value) or []) if v in _opts]
    regulation_select_plot = pms(
        options=_opts,
        value=_default,
        max_selections=500,
    )
    return (regulation_select_plot,)


@app.cell
def _(mo):
    about_regulation_md = mo.md(
        "Time course of **total bound transcription-factor counts** per "
        "cistron or per TU — summed across all ~23 modeled TF species so "
        "each entity gets a single scalar per timestep. Use this view to "
        "see *why* a gene's expression changed: a jump in bound TF count "
        "typically precedes a change in transcription rate (see the "
        "**Transcription** tab) and downstream mRNA / protein counts.\n\n"
        "The entity picker spans **all** cistrons / TUs (not just mRNAs) "
        "since the underlying listeners track TF binding across the full "
        "RNA set. Pathway selection above auto-populates with mRNA-cistron "
        "entries; non-matching entries are dropped silently.\n\n"
        "*Note:* the raw parquet listeners are 2-D matrices (TFs × cistrons "
        "or TUs × TFs); we aggregate the TF axis in SQL before plotting. "
        "For a per-TF-species breakdown add a TF picker in a future revision."
    )
    return (about_regulation_md,)


@app.cell
def _(
    active_datasets,
    all_cistron_ids,
    all_cistron_names,
    all_tu_ids,
    all_tu_names,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    pms_value,
    regulation_label_type,
    regulation_quantities,
    regulation_quantity_select,
    regulation_select_plot,
):
    plot_df_regulation = None
    if pms_value(regulation_select_plot):
        _listener, _kind = regulation_quantities[regulation_quantity_select.value]
        if _kind == "cistron":
            _id_list = all_cistron_ids
            _name_list = all_cistron_names
        else:
            _id_list = all_tu_ids
            _name_list = all_tu_names
        plot_df_regulation = get_plot_df_multi(
            active_datasets,
            _id_list,
            regulation_select_plot,
            _listener,
            "tf_counts",
            "Entity",
            "value",
            datapoints_cap,
            conn,
            _name_list,
            regulation_label_type,
            dtype="BIGINT",
        )
    return (plot_df_regulation,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    mo,
    pl,
    plot_df_regulation,
    pms_value,
    regulation_quantity_select,
    regulation_select_plot,
    y_scale_regulation,
):
    chart_regulation = None
    if pms_value(regulation_select_plot) and plot_df_regulation is not None:
        _n = len(active_datasets)
        _y_title = regulation_quantity_select.value
        _base = (
            alt.Chart(plot_df_regulation)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_regulation.value),
                    title=_y_title,
                ),
                color=alt.Color("Entity:N", legend=alt.Legend(title="Entity")),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_regulation = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_regulation.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_regulation.value),
                            title=_y_title,
                        ),
                        color=alt.Color("Entity:N", legend=alt.Legend(title="Entity")),
                    )
                    .properties(width="container", height=220)
                )
            chart_regulation = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_regulation = _base
    return (chart_regulation,)


# ===== Complexation tab (per-reaction complex-assembly counts) =====


@app.cell
def _(sim_data):
    # `listeners__complexation_listener__complexation_events` is a per-
    # complexation-reaction vector (length ~1,100) matching
    # `sim_data.process.complexation.ids_reactions`. Each entry counts how
    # many of that complex were assembled in the current timestep.
    complexation_reaction_ids = list(sim_data.process.complexation.ids_reactions)
    return (complexation_reaction_ids,)


@app.cell
def _(complexation_reaction_ids, mo, pms):
    complexation_select_plot = pms(
        options=complexation_reaction_ids,
        value=None,
        max_selections=500,
    )
    y_scale_complexation = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="symlog"
    )
    return complexation_select_plot, y_scale_complexation


@app.cell
def _(mo):
    about_complexation_md = mo.md(
        "Time course of complex-assembly events per timestep, indexed by "
        "complexation reaction (`sim_data.process.complexation.ids_reactions`, "
        "~1,100 reactions). Each entry counts how many of that complex "
        "assembled during the current timestep — useful for tracking "
        "protein-complex dynamics. Pair with the **Proteins** tab to see "
        "monomer counts feeding into assembly events."
    )
    return (about_complexation_md,)


@app.cell
def _(
    active_datasets,
    complexation_reaction_ids,
    complexation_select_plot,
    conn,
    datapoints_cap,
    get_plot_df_multi,
    pms_value,
):
    plot_df_complexation = None
    if pms_value(complexation_select_plot):
        plot_df_complexation = get_plot_df_multi(
            active_datasets,
            complexation_reaction_ids,
            complexation_select_plot,
            "listeners__complexation_listener__complexation_events",
            "complexation_events",
            "Reaction",
            "value",
            datapoints_cap,
            conn,
            dtype="BIGINT",
        )
    return (plot_df_complexation,)


@app.cell
def _(
    active_datasets,
    alt,
    chart_layout_mode,
    complexation_select_plot,
    mo,
    pl,
    plot_df_complexation,
    pms_value,
    y_scale_complexation,
):
    chart_complexation = None
    if pms_value(complexation_select_plot) and plot_df_complexation is not None:
        _n = len(active_datasets)
        _y_title = "Complexation events per timestep"
        _base = (
            alt.Chart(plot_df_complexation)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "value:Q",
                    scale=alt.Scale(type=y_scale_complexation.value),
                    title=_y_title,
                ),
                color=alt.Color(
                    "Reaction:N", legend=alt.Legend(title="Complexation reaction")
                ),
            )
        )
        if _n > 1 and chart_layout_mode.value == "overlay":
            _base = _base.encode(
                strokeDash=alt.StrokeDash(
                    "dataset_label:N", legend=alt.Legend(title="Dataset")
                )
            )
            chart_complexation = _base
        elif _n > 1 and chart_layout_mode.value == "facet":
            _per_ds = []
            for _ds in active_datasets:
                _ds_df = plot_df_complexation.filter(
                    pl.col("dataset_label") == _ds["label"]
                )
                if len(_ds_df) == 0:
                    continue
                _per_ds.append(
                    alt.Chart(_ds_df, title=_ds["label"])
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "time:Q",
                            scale=alt.Scale(type="linear"),
                            axis=alt.Axis(tickCount=4),
                            title="Time (s)",
                        ),
                        y=alt.Y(
                            "value:Q",
                            scale=alt.Scale(type=y_scale_complexation.value),
                            title=_y_title,
                        ),
                        color=alt.Color(
                            "Reaction:N",
                            legend=alt.Legend(title="Complexation reaction"),
                        ),
                    )
                    .properties(width="container", height=220)
                )
            chart_complexation = mo.vstack(_per_ds) if _per_ds else None
        else:
            chart_complexation = _base
    return (chart_complexation,)


@app.cell
def _(create_duckdb_conn, os, wd_root):
    # cpus omitted → DuckDB uses all detected cores. Big win when a compare
    # slot's filter is loose enough to scan many parquet files (e.g.
    # test_scaleup with ~1,600 .pq files); single-threaded was the freeze.
    conn = create_duckdb_conn(os.path.join(wd_root, "out"), False)

    return (conn,)


@app.cell
def _(
    active_datasets,
    conn,
    ndlist_to_ndarray,
    read_stacked_columns,
):
    # Per-dataset average monomer counts. Runs the same avg-of-timesteps
    # aggregation once per active dataset (primary + comparison slots) and
    # returns a dict keyed by the dataset's user-facing label. Slots whose
    # query yields no rows are silently skipped so a bad compare partition
    # doesn't take down the whole validation tab.
    monomer_counts_by_dataset = {}
    for _ds in active_datasets:
        try:
            _history_sql_subquery = (
                f"SELECT * FROM ({_ds['history_sql']}) WHERE {_ds['db_filter']}"
            )
            _subquery = read_stacked_columns(
                _history_sql_subquery,
                ["listeners__monomer_counts"],
                order_results=False,
            )
            _sql = f"""
                WITH unnested_counts AS (
                    SELECT unnest(listeners__monomer_counts) AS counts,
                        generate_subscripts(listeners__monomer_counts, 1) AS idx,
                        experiment_id, variant, lineage_seed, generation, agent_id
                    FROM ({_subquery})
                ),
                avg_counts AS (
                    SELECT avg(counts) AS avgCounts,
                        experiment_id, variant, lineage_seed,
                        generation, agent_id, idx
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed,
                        generation, agent_id, idx
                )
                SELECT list(avgCounts ORDER BY idx) AS avgCounts
                FROM avg_counts
                GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                """
            _res = conn.sql(_sql).pl()
            if len(_res) == 0:
                continue
            monomer_counts_by_dataset[_ds["label"]] = ndlist_to_ndarray(
                _res["avgCounts"]
            )
        except Exception:
            continue
    return (monomer_counts_by_dataset,)


@app.cell
def _(
    get_simulated_validation_counts,
    get_val_ids,
    monomer_counts_by_dataset,
    sim_data,
    validation_data,
):
    # Two products here:
    # 1. `val_ids_by_experiment` — the picker's option list per proteomics
    #    dataset. Depends only on sim_data + the reference dataset, so it's
    #    identical across all active_datasets.
    # 2. `val_options_by_dataset` — the per-active-dataset {sim, data} pairs
    #    used by the scatter. Recomputed for each dataset the user has in
    #    active_datasets so comparison slots can render their own scatters.
    sim_monomer_ids = sim_data.process.translation.monomer_data["id"]
    wisniewski_ids = validation_data.protein.wisniewski2014Data["monomerId"]
    schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
    wisniewski_counts = validation_data.protein.wisniewski2014Data["avgCounts"]
    schmidt_counts = validation_data.protein.schmidt2015Data["glucoseCounts"]

    val_ids_by_experiment = {
        "Schmidt 2015": get_val_ids(schmidt_ids, sim_monomer_ids),
        "Wisniewski 2014": get_val_ids(wisniewski_ids, sim_monomer_ids),
    }

    val_options_by_dataset = {}
    for _ds_label, _mono_counts in monomer_counts_by_dataset.items():
        _sim_w, _val_w = get_simulated_validation_counts(
            wisniewski_counts, _mono_counts, wisniewski_ids, sim_monomer_ids,
        )
        _sim_s, _val_s = get_simulated_validation_counts(
            schmidt_counts, _mono_counts, schmidt_ids, sim_monomer_ids,
        )
        val_options_by_dataset[_ds_label] = {
            "Schmidt 2015": {"data": _val_s, "sim": _sim_s},
            "Wisniewski 2014": {"data": _val_w, "sim": _sim_w},
        }
    return val_ids_by_experiment, val_options_by_dataset


@app.cell
def _(mo):
    val_dataset_select = mo.ui.dropdown(
        options=["Schmidt 2015", "Wisniewski 2014"], value="Schmidt 2015"
    )
    val_label_type = mo.ui.dropdown(
        options=["Common Name", "BioCyc ID"], value="Common Name"
    )
    return val_dataset_select, val_label_type


@app.cell
def _(
    mo,
    pms,
    protein_val_override,
    select_pathway,
    val_dataset_select,
    val_ids_by_experiment,
):
    val_id_select = pms(
        options=val_ids_by_experiment[val_dataset_select.value],
        value=protein_val_override(select_pathway.value),
        max_selections=10000,
        show_select_all=True,
    )
    return (val_id_select,)


@app.cell
def _(sim_data, val_label_type):
    def get_val_ids(data_ids, sim_ids):
        sim_ids_lst = sim_ids.tolist()
        data_ids_lst = data_ids.tolist()
        overlapping_ids_set = set(sim_ids_lst) & set(data_ids_lst)
        val_ids = list(overlapping_ids_set)
        val_ids = [id[:-3] for id in val_ids]
        val_ids_mapping = {
            "Common Name": get_common_names(val_ids, sim_data),
            "BioCyc ID": val_ids,
        }
        val_ids_final = val_ids_mapping[val_label_type.value]
        return val_ids_final

    return (get_val_ids,)


@app.cell
def _(mo):
    about_validation_md = mo.md(
        "Scatter plot comparing simulated average protein counts to "
        "experimental proteomics datasets (Schmidt 2015, Wisniewski 2014), "
        "restricted to proteins present in both. Pick proteins by common "
        "name or BioCyc ID.\n\n"
        "When comparison datasets are enabled in the toolbar, each active "
        "dataset gets its own sub-tab below (primary + one per comparison "
        "slot) so you can switch between their scatters — overlays are "
        "skipped here because point clouds get illegible fast."
    )
    return (about_validation_md,)


@app.cell
def _():
    # Validation controls are rendered by the composition cell below.
    return


@app.cell
def _(
    alt,
    np,
    pearsonr,
    pl,
    pms_value,
    val_id_select,
    val_ids_by_experiment,
    val_options_by_dataset,
):
    def val_chart(dataset_name, ds_label):
        """Scatter of simulated (from active dataset `ds_label`) vs
        experimental (from `dataset_name` in {"Schmidt 2015", "Wisniewski
        2014"}) protein counts. Called once per active dataset by the
        composition cell — no overlay across datasets, since scatter overlays
        get illegible fast.

        Every point-row gets inlined into the altair spec as JSON, so with a
        full-proteome selection (~2.5k points) × several active datasets the
        combined output can trip marimo's default 20 MB output cap. Log10
        values are rounded to 3 decimals here (still ≥3 sig figs on-screen)
        to keep each row small, and the compose cell wraps the whole chart
        in `mo.lazy` so hidden sub-tabs don't pay the serialization cost."""
        _pool = val_options_by_dataset[ds_label]
        data_val = _pool[dataset_name]["data"]
        data_sim = _pool[dataset_name]["sim"]
        data_idxs = [
            val_ids_by_experiment[dataset_name].index(name)
            for name in pms_value(val_id_select)
        ]
        data_val_filtered = data_val[data_idxs]
        data_sim_filtered = data_sim[data_idxs]

        # pearsonr requires ≥2 points; show a friendlier title for single
        # selections rather than crashing with a `x and y must have length at
        # least 2` ValueError.
        if len(data_idxs) >= 2:
            _title = "Pearson r: %0.2f" % pearsonr(
                np.log10(data_sim_filtered + 1),
                np.log10(data_val_filtered + 1),
            )[0]
        else:
            _title = "(select 2+ proteins for Pearson r)"

        _log_val = np.round(np.log10(data_val_filtered + 1), 3)
        _log_sim = np.round(np.log10(data_sim_filtered + 1), 3)
        chart = (
            alt.Chart(
                pl.DataFrame(
                    {
                        dataset_name: _log_val,
                        "sim": _log_sim,
                        "protein": pms_value(val_id_select),
                    }
                )
            )
            .mark_point()
            .encode(
                x=alt.X(dataset_name, title=f"log10({dataset_name} Counts + 1)"),
                y=alt.Y("sim", title="log10(Simulation Average Counts + 1)"),
                tooltip=["protein:N"],
            )
            .properties(title=f"{ds_label} — {_title}")
        )

        # Parity line — max scale spans all experimental + sim values across
        # both proteomics datasets in the currently rendered active dataset,
        # so the line reaches the plot extent even when the active selection
        # is narrower.
        max_val = max(
            np.log10(_pool["Schmidt 2015"]["data"] + 1).max(),
            np.log10(_pool["Wisniewski 2014"]["data"] + 1).max(),
            np.log10(_pool["Schmidt 2015"]["sim"] + 1).max(),
            np.log10(_pool["Wisniewski 2014"]["sim"] + 1).max(),
        )
        parity = (
            alt.Chart(pl.DataFrame({"x": np.arange(max_val)}))
            .mark_line()
            .encode(x="x", y="x", color=alt.value("red"), strokeDash=alt.value([5, 5]))
        )

        chart_final = chart + parity

        return chart_final

    return (val_chart,)


@app.cell
def _(
    mo,
    pms_value,
    val_chart,
    val_dataset_select,
    val_id_select,
    val_options_by_dataset,
):
    # Render one scatter per active dataset. With a single active dataset the
    # result is just that scatter; with multiple, wrap them in an inner
    # `mo.ui.tabs` so the user can switch between primary / comparison
    # slots — overlays make scatter plots unreadable.
    #
    # Each per-dataset chart is wrapped in `mo.lazy` so the altair spec
    # (which inlines every point as JSON) is only generated for the
    # currently visible sub-tab. Without this, N active datasets means N
    # full-proteome specs serialized at once — a 3-slot comparison blew
    # past marimo's default 20 MB output cap.
    chart_val = None
    if pms_value(val_id_select) and val_options_by_dataset:
        _labels = list(val_options_by_dataset.keys())
        _ds_exp = val_dataset_select.value

        def _lazy_chart(_ds_label):
            # Bind ds_label at definition time via default arg — otherwise
            # every closure would capture the last loop value.
            return mo.lazy(lambda _l=_ds_label: val_chart(_ds_exp, _l))

        if len(_labels) == 1:
            chart_val = _lazy_chart(_labels[0])
        else:
            chart_val = mo.ui.tabs(
                {_ds_label: _lazy_chart(_ds_label) for _ds_label in _labels}
            )
    return (chart_val,)


@app.cell
def _(partition_groups, read_partitions):
    def partitions_dict(analysis_type):
        partitions_req = partition_groups[analysis_type]
        partitions_all = read_partitions()

        partitions_dict = {}
        for partition in partitions_req:
            partitions_dict[partition] = partitions_all[partition]
        partitions_dict["experiment_id"] = f"'{partitions_dict['experiment_id']}'"
        return partitions_dict

    def get_db_filter(partitions_dict):
        db_filter_list = []
        for key, value in partitions_dict.items():
            db_filter_list.append(str(key) + "=" + str(value))
        db_filter = " AND ".join(db_filter_list)

        return db_filter

    return get_db_filter, partitions_dict


@app.cell
def _(agent_select, exp_select, gen_select, seed_select, variant_select):
    partition_groups = {
        "multiseed": ["experiment_id", "variant"],
        "multigeneration": ["experiment_id", "variant", "lineage_seed"],
        "multidaughter": ["experiment_id", "variant", "lineage_seed", "generation"],
        "single": [
            "experiment_id",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
        ],
    }

    def partitions_display():
        partitions_list = {
            "experiment_id": exp_select,
            "variant": variant_select,
            "lineage_seed": seed_select,
            "generation": gen_select,
            "agent_id": agent_select,
        }

        return partitions_list

    def read_partitions():
        partitions_selected = {
            "experiment_id": exp_select.value,
            "variant": variant_select.value,
            "lineage_seed": seed_select.value,
            "generation": gen_select.value,
            "agent_id": agent_select.value,
        }
        return partitions_selected

    return partition_groups, partitions_display, read_partitions


@app.function
def get_common_names(bulk_names, sim_data):
    bulk_common_names = [
        sim_data.common_names.get_common_name(name) for name in bulk_names
    ]

    duplicates = []

    for item in bulk_common_names:
        if bulk_common_names.count(item) > 1 and item not in duplicates:
            duplicates.append(item)

    for dup in duplicates:
        sp_idxs = [index for index, item in enumerate(bulk_common_names) if item == dup]

        for sp_idx in sp_idxs:
            bulk_rename = str(bulk_common_names[sp_idx]) + f"[{bulk_names[sp_idx]}]"
            bulk_common_names[sp_idx] = bulk_rename

    return bulk_common_names


@app.cell
def _(
    bulk_common_names,
    bulk_names_unique,
    molecule_id_type,
    monomer_ids,
    monomer_label_type,
    monomer_names,
    mrna_cistron_names,
    mrna_gene_ids,
    np,
    os,
    pd,
    rna_label_type,
    rxn_ids,
    val_dataset_select,
    val_ids_by_experiment,
):
    pathway_dir = "pathways"

    def get_pathways(pathway_dir):
        pathway_file = os.path.join(pathway_dir, "pathways.txt")
        pathway_df = pd.read_csv(pathway_file, sep="\t")
        pathway_list = pathway_df["name"].values
        pathway_list = list(np.unique(pathway_list))
        return pathway_list

    def get_presets(preset_dir):
        preset_files = os.listdir(preset_dir)
        presets_list = [file.split(".")[0] for file in preset_files]

        return presets_list

    def read_columns(st_column):
        values = []
        for item in st_column:
            items_actual = str(item).split(" // ")
            for item_actual in items_actual:
                values.append(item_actual)
        return values

    def read_presets(pathway_name):
        preset_dict = {}
        if isinstance(pathway_name, str):
            preset_table = pd.read_csv(
                os.path.join(pathway_dir, "pathways.txt"), header=0, sep="\t"
            )
            pathway_df = preset_table[preset_table["name"] == pathway_name]

            preset_dict["reactions"] = read_columns(pathway_df["reactions"])
            preset_dict["genes"] = read_columns(pathway_df["genes"])
            preset_dict["compounds"] = read_columns(pathway_df["compounds"])

        return preset_dict

    def preset_override(preset_name):
        preset_dict = read_presets(preset_name)

        preset_final = {}

        if len(preset_dict) > 0:
            preset_final["reactions"] = np.array(preset_dict["reactions"])[
                np.isin(preset_dict["reactions"], rxn_ids)
            ].tolist()

            preset_final["genes"] = np.array(preset_dict["genes"])[
                np.isin(preset_dict["genes"], mrna_gene_ids)
            ].tolist()

            preset_final["genes"] = np.unique(preset_final["genes"]).tolist()

            if rna_label_type.value == "gene name":
                preset_gene_names = []
                for gene_id in preset_final["genes"]:
                    preset_gene_names.append(
                        mrna_cistron_names[mrna_gene_ids.index(gene_id)]
                    )
                preset_final["genes"] = preset_gene_names

            preset_final["compounds"] = np.array(preset_dict["compounds"])[
                np.isin(preset_dict["compounds"], bulk_names_unique)
            ].tolist()

            preset_final["compounds"] = np.unique(preset_final["compounds"]).tolist()

            preset_final["proteins"] = list(
                np.array(preset_final["compounds"])[
                    np.isin(preset_final["compounds"], monomer_ids)
                ]
            )

            if molecule_id_type.value == "Common name":
                preset_compound_names = []
                for name in preset_final["compounds"]:
                    preset_compound_names.append(
                        bulk_common_names[bulk_names_unique.index(name)]
                    )
                preset_final["compounds"] = preset_compound_names

            if monomer_label_type.value == "common name":
                preset_protein_names = []
                for name in preset_final["proteins"]:
                    preset_protein_names.append(monomer_names[monomer_ids.index(name)])
                preset_final["proteins"] = preset_protein_names

        return preset_final

    def bulk_override(preset_name):
        preset_dict = preset_override(preset_name)
        bulk_list = preset_dict.get("compounds")
        return bulk_list

    def rxn_override(preset_name):
        preset_dict = preset_override(preset_name)
        rxn_list = preset_dict.get("reactions")
        return rxn_list

    def mrna_override(preset_name):
        preset_dict = preset_override(preset_name)
        mrna_list = preset_dict.get("genes")
        return mrna_list

    def protein_override(preset_name):
        preset_dict = preset_override(preset_name)
        protein_list = preset_dict.get("proteins")
        return protein_list

    def protein_val_override(preset_name):
        protein_list = protein_override(preset_name)
        dataset_name = val_dataset_select.value
        protein_ids_val = val_ids_by_experiment[dataset_name]
        protein_val = list(
            np.array(protein_list)[np.isin(protein_list, protein_ids_val)]
        )
        return protein_val

    return (
        bulk_override,
        get_pathways,
        mrna_override,
        pathway_dir,
        protein_override,
        protein_val_override,
        rxn_override,
    )


@app.cell
def _(np, pl):
    def _widget_value(ui):
        """Extract the underlying list of selections from a widget whose
        `.value` is either a plain list (mo.ui.multiselect) or a dict of
        synced traits (mo.ui.anywidget/PinnedMultiselect)."""
        v = getattr(ui, "value", None)
        if isinstance(v, dict):
            return v.get("value") or []
        return v or []

    def get_plot_df_bulk(
        bulk_select_ui,
        bulk_ids_biocyc,
        bulk_names2biocyc,
        sql_base,
        db_filter,
        datapoints_cap,
        conn,
        molecule_id_ui,
        convert_to_molar=False,
        molar_scale=1.0,
    ):
        """If `convert_to_molar`, multiply each compound's summed count by
        the per-timestep `listeners__enzyme_kinetics__counts_to_molar`
        conversion factor (and `molar_scale` for unit prefix — e.g. 1000.0
        for mM, 1e6 for µM). Returns counts (BIGINT) when False, otherwise
        a float concentration."""
        _bulk_sel = _widget_value(bulk_select_ui)
        if molecule_id_ui.value == "Common name":
            bulk_sp_ids = [bulk_names2biocyc[name] for name in _bulk_sel]
        else:
            bulk_sp_ids = _bulk_sel

        sp_idxs_selected = [
            [
                f"bulk[{index + 1}]"
                for index, item in enumerate(bulk_ids_biocyc)
                if item == sp_i
            ]
            for sp_i in bulk_sp_ids
        ]

        sp_idxs_alias = [
            "+".join(sp_idxs_i) + f" as compound_{count}"
            for count, sp_idxs_i in enumerate(sp_idxs_selected)
        ]

        # Carry the per-timestep counts_to_molar through the pipeline only
        # when needed; otherwise old behavior (cast SUM to BIGINT) preserved.
        extra_select = (
            ", listeners__enzyme_kinetics__counts_to_molar AS _ctm"
            if convert_to_molar
            else ""
        )
        bulk_sql_opt = (
            f"SELECT {','.join(sp_idxs_alias)}{extra_select}, time "
            f"FROM ({sql_base}) WHERE {db_filter}"
        )

        if convert_to_molar:
            sum_clauses = ",".join(
                [
                    f" CAST(SUM(compound_{sp_idx}) AS DOUBLE) "
                    f"* first(_ctm) * {molar_scale} AS compound_{sp_idx}"
                    for sp_idx, _ in enumerate(bulk_sp_ids)
                ]
            )
        else:
            sum_clauses = ",".join(
                [
                    f" CAST (SUM(compound_{sp_idx}) AS BIGINT) AS compound_{sp_idx}"
                    for sp_idx, _ in enumerate(bulk_sp_ids)
                ]
            )

        bulk_sql_opt_sum = (
            f"SELECT{sum_clauses}, time FROM ({bulk_sql_opt}) GROUP BY time"
        )

        bulk_sql_list = (
            "SELECT ("
            + "+".join([f"[compound_{sp_idx}]" for sp_idx, _ in enumerate(bulk_sp_ids)])
            + f") AS bulk_counts, time FROM ({bulk_sql_opt_sum})"
        )

        bulk_sql_ds = sql_downsample(
            bulk_sql_list, bulk_sp_ids, "bulk_counts", datapoints_cap
        )

        df_bulk_read = conn.sql(bulk_sql_ds).pl()

        bulk_counts_mtx = np.stack(df_bulk_read["bulk_counts"])
        bulk_counts_list = [
            bulk_counts_mtx[:, col] for col in range(np.shape(bulk_counts_mtx)[1])
        ]
        bulk_plot_dict = {key: val for (key, val) in zip(_bulk_sel, bulk_counts_list)}
        bulk_plot_dict["time"] = df_bulk_read["time"].to_list()
        bulk_plot_df = pl.DataFrame(bulk_plot_dict)
        bulk_plot_df_melted = bulk_plot_df.unpivot(
            index="time", variable_name="compound", value_name="counts"
        )

        return bulk_plot_df_melted

    return (get_plot_df_bulk,)


@app.cell
def _(np, pl):
    def _widget_value(ui):
        v = getattr(ui, "value", None)
        if isinstance(v, dict):
            return v.get("value") or []
        return v or []

    def get_plot_df(
        default_id_list,
        item_selector_ui,
        listener_name,
        col_name,
        var_name,
        val_name,
        sql_base,
        db_filter,
        datapoints_cap,
        conn,
        default_name_list=None,
        label_ui=None,
        dtype="BIGINT",
        display_labels=None,
    ):
        """`display_labels` (optional): when set, used as plot_dict keys
        (which become the values in the var_name column and the chart legend
        labels) instead of the widget's selections. Lets a consumer translate
        e.g. 'RXN (CAT)' display labels → catalyst_ids before calling, while
        keeping the chart legend in the original display form."""
        _sel = _widget_value(item_selector_ui)
        if label_ui:
            if label_ui.value == "BioCyc ID":
                ids_selected = _sel
            else:
                ids_selected = [
                    default_id_list[default_name_list.index(name)] for name in _sel
                ]
        else:
            ids_selected = _sel

        idxs_selected = [
            f"{col_name}[{default_id_list.index(id) + 1}] AS {col_name}_{idx}"
            for idx, id in enumerate(ids_selected)
        ]
        col_sql_base = f"SELECT {listener_name} as {col_name}, time FROM ({sql_base}) WHERE {db_filter}"
        col_sql_sliced = f"SELECT {','.join(idxs_selected)}, time FROM ({col_sql_base})"
        col_sql_sum = (
            "SELECT"
            + ",".join(
                [
                    f" CAST (SUM({col_name}_{idx}) AS {dtype}) as {col_name}_{idx}"
                    for idx, _ in enumerate(ids_selected)
                ]
            )
            + f",time FROM ({col_sql_sliced}) GROUP BY time"
        )
        col_sql_list = (
            "SELECT ("
            + "+".join([f"[{col_name}_{idx}]" for idx, _ in enumerate(ids_selected)])
            + f") AS {col_name}, time FROM ({col_sql_sum})"
        )
        col_sql_ds = sql_downsample(
            col_sql_list, ids_selected, col_name, datapoints_cap
        )

        col_read_df = conn.sql(col_sql_ds).pl()

        counts_mtx = np.stack(col_read_df[col_name])
        counts_list = [counts_mtx[:, col] for col in range(np.shape(counts_mtx)[1])]
        _keys = display_labels if display_labels is not None else _sel
        plot_dict = {key: val for (key, val) in zip(_keys, counts_list)}
        plot_dict["time"] = col_read_df["time"].to_list()
        plot_df = pl.DataFrame(plot_dict)
        plot_df_melted = plot_df.unpivot(
            index="time", variable_name=var_name, value_name=val_name
        )

        return plot_df_melted

    return (get_plot_df,)


@app.cell
def _(get_plot_df, get_plot_df_bulk, pl):
    def _widget_value(ui):
        v = getattr(ui, "value", None)
        if isinstance(v, dict):
            return v.get("value") or []
        return v or []

    # Multi-dataset wrappers: run the existing single-dataset builder once per
    # active dataset, then concat with a dataset_label column. Slots that yield
    # empty results are skipped (e.g. invalid partition combo for that exp).
    def get_plot_df_multi(
        active_datasets,
        default_id_list,
        item_selector_ui,
        listener_name,
        col_name,
        var_name,
        val_name,
        datapoints_cap,
        conn,
        default_name_list=None,
        label_ui=None,
        dtype="BIGINT",
        display_labels=None,
    ):
        if not _widget_value(item_selector_ui):
            return None
        dfs = []
        for ds in active_datasets:
            try:
                df = get_plot_df(
                    default_id_list,
                    item_selector_ui,
                    listener_name,
                    col_name,
                    var_name,
                    val_name,
                    ds["history_sql"],
                    ds["db_filter"],
                    datapoints_cap,
                    conn,
                    default_name_list,
                    label_ui,
                    dtype,
                    display_labels=display_labels,
                )
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            dfs.append(df.with_columns(pl.lit(ds["label"]).alias("dataset_label")))
        return pl.concat(dfs) if dfs else None

    def get_plot_df_bulk_multi(
        active_datasets,
        bulk_select_ui,
        bulk_ids_biocyc,
        bulk_names2biocyc,
        datapoints_cap,
        conn,
        molecule_id_ui,
        convert_to_molar=False,
        molar_scale=1.0,
    ):
        if not _widget_value(bulk_select_ui):
            return None
        dfs = []
        for ds in active_datasets:
            try:
                df = get_plot_df_bulk(
                    bulk_select_ui,
                    bulk_ids_biocyc,
                    bulk_names2biocyc,
                    ds["history_sql"],
                    ds["db_filter"],
                    datapoints_cap,
                    conn,
                    molecule_id_ui,
                    convert_to_molar=convert_to_molar,
                    molar_scale=molar_scale,
                )
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            dfs.append(df.with_columns(pl.lit(ds["label"]).alias("dataset_label")))
        return pl.concat(dfs) if dfs else None

    return get_plot_df_bulk_multi, get_plot_df_multi


@app.function
def sql_downsample(sql_original, items_list, list_col_name, datapoints_cap=2000):
    ds_sql = f"""
    WITH 
    indexed_data AS (
      SELECT 
        *, 
        ROW_NUMBER() OVER (ORDER BY time) AS rn
    FROM ({sql_original})
    ),
    data_shape AS (
    SELECT
        COUNT(*) AS time_points,
        time_points*{len(items_list)} as data_points,
        data_points/{datapoints_cap} as ds_ratio_frac,
        CEILING(ds_ratio_frac) as ds_ratio

    FROM ({sql_original}))

    SELECT {list_col_name},time
    FROM indexed_data
    CROSS JOIN data_shape
    WHERE rn % data_shape.ds_ratio = 0
    ORDER BY time
    """

    return ds_sql


@app.cell
def _():
    from pathlib import Path

    def tree_to_dict(path):
        path = Path(path)
        # Define the dictionary structure for this level
        d = {"name": path.name}

        if path.is_dir():
            d["type"] = "directory"
            # Recursively call tree_to_dict for every child in the directory
            d["children"] = [tree_to_dict(child) for child in path.iterdir()]
        else:
            d["type"] = "file"

        return d

    def get_dir_tree(path):
        path = Path(path)

        if path.is_file():
            return None

        # Filter: Only include children that don't start with '.'
        return {
            child.name: get_dir_tree(child)
            for child in path.iterdir()
            if not child.name.startswith(".")
        }

    # Usage
    return Path, get_dir_tree


@app.cell
def _(Path, get_dir_tree, wd_root):
    outdir_path = Path(wd_root) / "out"
    outdir_tree = get_dir_tree(outdir_path)
    return (outdir_tree,)


@app.cell
def _(outdir_tree):
    def get_exp(tree):
        exp_list = list(outdir_tree.keys())
        exp_list.sort()
        return exp_list

    def get_variants(tree, exp_id):
        try:
            variant_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"].keys()

            variants = [var.split("variant=")[1] for var in variant_folders]

            variants.sort()

        except KeyError:
            variants = ["N/A"]
        return variants

    def get_seeds(tree, exp_id, var_id):
        try:
            seed_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"][
                f"variant={var_id}"
            ].keys()

            seeds = [seed.split("lineage_seed=")[1] for seed in seed_folders]

            seeds.sort()

        except KeyError:
            seeds = ["N/A"]
        return seeds

    def get_gens(tree, exp_id, var_id, seed_id):
        try:
            gen_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"][
                f"variant={var_id}"
            ][f"lineage_seed={seed_id}"].keys()

            gens = [gen.split("generation=")[1] for gen in gen_folders]

            gens.sort()

        except KeyError:
            gens = ["N/A"]

        return gens

    def get_agents(tree, exp_id, var_id, seed_id, gen_id):
        try:
            agent_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"][
                f"variant={var_id}"
            ][f"lineage_seed={seed_id}"][f"generation={gen_id}"].keys()
            agents = [agent.split("agent_id=")[1] for agent in agent_folders]
            agents.sort()
        except KeyError:
            agents = ["N/A"]
        return agents

    return get_agents, get_exp, get_gens, get_seeds, get_variants


@app.cell
def _(mo):
    about_download_md = mo.md(
        """
    Export columns from the parquet history dataset as tab-separated text.
    The current partition filter is applied automatically.

    - **Vector** columns (e.g. `listeners__monomer_counts`) are exported wide:
      one row per timestep, one column per element. Headers come from the
      saved `output_metadata__` IDs when available, otherwise numeric indices.
    - **Scalar** columns can be combined into a single file: pick one or
      more, get a `time + col1 + col2 + …` TSV.
    - **Δt > 0** keeps one row per Δt-second bucket per cell, useful for
      bringing large vector exports back into browser-download size.
    """
    )
    return (about_download_md,)


@app.cell
def _(conn, history_sql_base):
    def get_history_schema(conn, history_sql_base):
        rows = conn.sql(
            f"DESCRIBE SELECT * FROM ({history_sql_base}) LIMIT 0"
        ).fetchall()
        scalar = []
        vec1d = []
        vec2d_plus = []
        partition_cols = {
            "experiment_id",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
            "time",
        }
        for r in rows:
            name, dtype = r[0], r[1]
            if name in partition_cols:
                continue
            depth = dtype.count("[]")
            if depth == 0:
                scalar.append(name)
            elif depth == 1:
                vec1d.append(name)
            else:
                vec2d_plus.append(name)
        return scalar, vec1d, vec2d_plus

    scalar_cols, vector_cols_1d, vector_cols_2d = get_history_schema(
        conn, history_sql_base
    )
    return scalar_cols, vector_cols_1d, vector_cols_2d


@app.cell
def _(conn, config_sql_base, field_metadata):
    def get_column_metadata(col_name, fallback_len=None, config_sql=None):
        """Return (labels, source) for a vector column header.

        Calls `field_metadata` (which looks up `output_metadata__<col>`) on
        the given `config_sql` (default: primary config_sql_base; pass a slot's
        config_sql for compare downloads).
        `source` is one of:
          - "metadata"          — used the saved ID list as-is
          - "metadata-trimmed"  — saved list longer than data; truncated
          - "metadata-padded"   — saved list shorter than data; padded with idx_
          - "numeric-fallback:<reason>" — field_metadata failed; using idx_
        """
        cfg = config_sql if config_sql is not None else config_sql_base
        try:
            md = field_metadata(conn, cfg, col_name)
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            if fallback_len is None:
                return None, f"numeric-fallback:{reason}"
            return [f"idx_{i}" for i in range(fallback_len)], (
                f"numeric-fallback:{reason}"
            )

        labels = [str(x) for x in md]
        if fallback_len is None or len(labels) == fallback_len:
            return labels, "metadata"
        if len(labels) > fallback_len:
            return labels[:fallback_len], "metadata-trimmed"
        # len(labels) < fallback_len
        padded = labels + [f"idx_{i}" for i in range(len(labels), fallback_len)]
        return padded, "metadata-padded"

    def get_vector_length(col_name, history_sql_base, db_filter):
        sql = (
            f'SELECT length("{col_name}") AS n '
            f"FROM ({history_sql_base}) WHERE {db_filter} LIMIT 1"
        )
        row = conn.sql(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    return get_column_metadata, get_vector_length


@app.cell
def _():
    def _safe_alias(s):
        return (
            str(s)
            .replace('"', '""')
            .replace("\t", " ")
            .replace("\n", " ")
            .replace("\r", " ")
        )

    def _qualify_clause(time_interval):
        """ROW_NUMBER-based first-row-per-bucket filter; '' if no downsampling."""
        if not time_interval or float(time_interval) <= 0:
            return ""
        return (
            f" QUALIFY ROW_NUMBER() OVER ("
            f"PARTITION BY experiment_id, variant, lineage_seed, "
            f"generation, agent_id, FLOOR(time / {float(time_interval)}) "
            f"ORDER BY time) = 1"
        )

    def build_vector_wide_sql(
        history_sql_base, db_filter, col_name, labels, time_interval=0.0
    ):
        # Pre-filter (and optionally time-bucket) the rows we need, only
        # carrying the partition cols, time, and the one vector column.
        base = (
            f"SELECT experiment_id, variant, lineage_seed, generation, "
            f'agent_id, time, "{col_name}" '
            f"FROM ({history_sql_base}) "
            f"WHERE {db_filter}"
            f"{_qualify_clause(time_interval)}"
        )
        select_parts = ["time"] + [
            f'"{col_name}"[{i + 1}] AS "{_safe_alias(label)}"'
            for i, label in enumerate(labels)
        ]
        return (
            f"SELECT {', '.join(select_parts)} "
            f"FROM ({base}) sub "
            f"ORDER BY experiment_id, variant, lineage_seed, "
            f"generation, agent_id, time"
        )

    def build_scalar_sql(history_sql_base, db_filter, cols, time_interval=0.0):
        # time first, then partitions, then user-picked scalars
        select_cols = [
            "time",
            "experiment_id",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
        ]
        for c in cols:
            if c not in select_cols:
                select_cols.append(c)
        quoted = [f'"{c}"' for c in select_cols]
        return (
            f"SELECT {', '.join(quoted)} "
            f"FROM ({history_sql_base}) "
            f"WHERE {db_filter}"
            f"{_qualify_clause(time_interval)} "
            f"ORDER BY experiment_id, variant, lineage_seed, "
            f"generation, agent_id, time"
        )

    return build_scalar_sql, build_vector_wide_sql


@app.cell
def _(conn):
    def estimate_export_size(sql, n_data_cols, partition_cols=6):
        """Rough size estimate: row count * (data_cols + partition+time cols) * ~12 bytes.

        Uses 12 bytes/cell as an average across small ints and decimal floats
        rendered as text plus the tab separator. Real sizes vary a lot by dtype.
        """
        n_rows = conn.sql(f"SELECT count(*) FROM ({sql}) AS t").fetchone()[0]
        total_cols = n_data_cols + partition_cols
        est_bytes = int(n_rows) * total_cols * 12
        return int(n_rows), est_bytes

    return (estimate_export_size,)


@app.cell
def _(mo, vector_cols_1d, vector_cols_2d):
    dl_mode_select = mo.ui.radio(
        options=["vector column", "scalar columns"],
        value="vector column",
    )
    _note_2d = (
        f"  \n*Note: {len(vector_cols_2d)} column(s) are 2-D or deeper "
        "and are not offered for wide download.*"
        if vector_cols_2d
        else ""
    )
    dl_schema_note_md = mo.md(
        f"**{len(vector_cols_1d)} 1-D vector / scalar columns detected.**{_note_2d}"
    )
    return dl_mode_select, dl_schema_note_md


@app.cell
def _(mo, vector_cols_1d):
    dl_vector_col_select = mo.ui.dropdown(
        options=sorted(vector_cols_1d), searchable=True
    )
    dl_vector_label_mode = mo.ui.radio(
        options=["metadata IDs (when available)", "numeric indices"],
        value="metadata IDs (when available)",
    )
    return dl_vector_col_select, dl_vector_label_mode


@app.cell
def _(mo, pms, scalar_cols):
    dl_scalar_cols_select = pms(
        options=sorted(scalar_cols),
        max_selections=500,
    )
    return (dl_scalar_cols_select,)


@app.cell
def _():
    # Column picker is rendered by the composition cell below.
    return


@app.cell
def _(mo, wd_root):
    dl_filename_input = mo.ui.text(value="export.tsv", placeholder="export.tsv")
    dl_delivery_radio = mo.ui.radio(
        options=[
            "auto (browser if small, disk if large)",
            "browser download",
            "save to disk",
        ],
        value="auto (browser if small, disk if large)",
    )
    dl_disk_dir_input = mo.ui.text(
        value=f"{wd_root}/exports",
        placeholder="/absolute/path/to/output/dir",
    )
    dl_size_threshold_mb = mo.ui.number(value=50, start=1, stop=4096, step=1)
    dl_time_interval = mo.ui.number(value=0.0, start=0.0, stop=1e9, step=1.0)
    return (
        dl_delivery_radio,
        dl_disk_dir_input,
        dl_filename_input,
        dl_size_threshold_mb,
        dl_time_interval,
    )


@app.cell
def _():
    # Delivery / Δt / output-dir layout is rendered by the composition cell.
    return


@app.cell
def _(active_datasets, mo):
    # Source-dataset dropdown for the Download tab. Lists every active dataset
    # (primary + enabled compare slots) so the user picks which slice to export.
    _labels = [ds["label"] for ds in active_datasets]
    dl_source_dataset_select = mo.ui.dropdown(
        options=_labels,
        value=_labels[0] if _labels else None,
    )
    return (dl_source_dataset_select,)


@app.cell
def _(mo):
    dl_run_button = mo.ui.run_button(label="Generate TSV")
    return (dl_run_button,)


@app.cell
def _(
    active_datasets,
    build_scalar_sql,
    build_vector_wide_sql,
    conn,
    dl_delivery_radio,
    dl_disk_dir_input,
    dl_filename_input,
    dl_mode_select,
    dl_run_button,
    dl_scalar_cols_select,
    dl_size_threshold_mb,
    dl_source_dataset_select,
    dl_time_interval,
    dl_vector_col_select,
    dl_vector_label_mode,
    estimate_export_size,
    get_column_metadata,
    get_vector_length,
    mo,
    os,
    pms_value,
):
    status_md = "_Click **Generate TSV** above to export with the current settings._"
    download_widget = None
    label_source = None
    time_interval = float(dl_time_interval.value or 0.0)

    # Resolve which dataset slot to export from.
    _selected_label = dl_source_dataset_select.value
    _selected = next(
        (ds for ds in active_datasets if ds["label"] == _selected_label),
        active_datasets[0] if active_datasets else None,
    )
    dl_history_sql = _selected["history_sql"] if _selected else None
    dl_db_filter = _selected["db_filter"] if _selected else None
    dl_config_sql = _selected["config_sql"] if _selected else None

    if dl_run_button.value and _selected is not None:
        sql = None
        n_data_cols = 0

        if dl_mode_select.value == "vector column":
            col = dl_vector_col_select.value
            if not col:
                status_md = "**Please select a vector column.**"
            else:
                n_elems = get_vector_length(col, dl_history_sql, dl_db_filter)
                if n_elems == 0:
                    status_md = (
                        f"**No rows match the current filter for `{col}` "
                        f"in dataset `{_selected_label}`.**"
                    )
                else:
                    if dl_vector_label_mode.value == "metadata IDs (when available)":
                        labels, label_source = get_column_metadata(
                            col, fallback_len=n_elems, config_sql=dl_config_sql
                        )
                    else:
                        labels = [f"idx_{i}" for i in range(n_elems)]
                        label_source = "numeric (user-selected)"
                    if not labels:
                        labels = [f"idx_{i}" for i in range(n_elems)]
                        label_source = label_source or "numeric-fallback:empty-metadata"
                    sql = build_vector_wide_sql(
                        dl_history_sql,
                        dl_db_filter,
                        col,
                        labels,
                        time_interval=time_interval,
                    )
                    n_data_cols = len(labels)
        else:
            cols = list(pms_value(dl_scalar_cols_select) or [])
            if not cols:
                status_md = "**Please select at least one scalar column.**"
            else:
                sql = build_scalar_sql(
                    dl_history_sql,
                    dl_db_filter,
                    cols,
                    time_interval=time_interval,
                )
                n_data_cols = len(cols)

        if sql is not None:
            try:
                n_rows, est_bytes = estimate_export_size(sql, n_data_cols)
            except Exception as e:
                n_rows, est_bytes = 0, 0
                status_md = f"**Size estimate failed:** `{e}`"

            if n_rows == 0 and "failed" not in status_md:
                status_md = "**No rows match the current filter.**"
            elif n_rows > 0:
                threshold_bytes = float(dl_size_threshold_mb.value) * 1e6
                mode = dl_delivery_radio.value
                if mode == "auto (browser if small, disk if large)":
                    mode = (
                        "browser download"
                        if est_bytes <= threshold_bytes
                        else "save to disk"
                    )

                filename = dl_filename_input.value or "export.tsv"

                src_note = (
                    f"  \n*Header labels source:* `{label_source}`"
                    if label_source
                    else ""
                )
                ds_note = (
                    f"  \n*Time downsampling:* one row per "
                    f"`{time_interval}` s bucket per cell."
                    if time_interval > 0
                    else ""
                )
                src_note = src_note + ds_note

                if mode == "browser download":
                    df = conn.sql(sql).pl()
                    buf = df.write_csv(separator="\t").encode("utf-8")
                    download_widget = mo.download(
                        data=buf,
                        filename=filename,
                        label=f"Download {filename}",
                        mimetype="text/tab-separated-values",
                    )
                    status_md = (
                        f"Prepared **{n_rows:,} rows × "
                        f"{n_data_cols + 6} cols** (~{est_bytes / 1e6:.1f} MB "
                        f"estimated) for browser download.{src_note}"
                    )
                else:
                    out_dir = dl_disk_dir_input.value
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, filename)
                    safe_path = out_path.replace("'", "''")
                    copy_sql = (
                        f"COPY ({sql}) TO '{safe_path}' "
                        f"(FORMAT CSV, HEADER, DELIMITER E'\\t')"
                    )
                    conn.sql(copy_sql)
                    size_mb = os.path.getsize(out_path) / 1e6
                    status_md = (
                        f"Wrote **{n_rows:,} rows × "
                        f"{n_data_cols + 6} cols** to "
                        f"`{out_path}` ({size_mb:.1f} MB on disk).{src_note}"
                    )

    download_status_render = mo.vstack(
        [mo.md(status_md), download_widget]
        if download_widget is not None
        else [mo.md(status_md)]
    )
    return (download_status_render,)


@app.cell
def _(mo):
    download_memory_notes_md = mo.md(
        """
    ### Memory notes for large exports

    - **Wide vector exports** can be huge. `listeners__monomer_counts` has
      ~4,500 elements; a single cell with ~5,000 timesteps is roughly
      `5000 × 4500 × ~10 bytes ≈ 225 MB` of TSV text. A multiseed export
      multiplies that by the number of cells included.
    - **Browser download** materializes the full TSV in Python memory
      (`polars.DataFrame.write_csv → bytes`) before handing it to the
      browser. Above ~100 MB the kernel will start to feel it and the
      browser may refuse the blob.
    - **Save-to-disk mode** uses DuckDB's `COPY (...) TO file (FORMAT CSV)`,
      which streams query results to disk without building the full table
      in RAM. This is the safe choice for vectors with more than ~1,000
      elements or for any multi-cell aggregation.
    - **`auto` delivery** switches modes based on the size estimate and the
      threshold (default 50 MB). The estimate uses ~12 bytes/cell as a
      crude average; real sizes vary with dtype (large floats blow it up,
      small ints come in under).
    - **Time downsampler (Δt > 0)** keeps one row per Δt-second bucket per
      cell. Often the easiest way to drop a 200 MB export into the browser
      tier without losing the overall shape of the trace.
    - **Row count estimation** runs a `COUNT(*)` over the filtered
      subquery on every Generate click. On large multi-experiment
      datasets this scan can be the slowest part of the export.
    """
    )
    return (download_memory_notes_md,)


@app.cell
def _(
    about_intro_md,
    analysis_select,
    available_generations,
    chart_layout_mode,
    compare_count,
    compare_enabled,
    compare_slots,
    generation_range_slider,
    mo,
    partition_groups,
    partition_picker_items,
    primary_label,
    select_pathway,
):
    # Toolbar lives in its OWN cell so it always renders, even if a downstream
    # chart cell errors. The user can still pick an experiment / variant /
    # pathway from here to recover.

    # ---- primary partition row ----
    # Leads with the user-editable primary dataset label so it visually mirrors
    # the compare-slot rows below (dataset label first, then partitions).
    _partition_row = [mo.md("**dataset label:**"), primary_label]
    for _label, _widget in partition_picker_items:
        _partition_row.append(mo.md(f"**{_label}:**"))
        _partition_row.append(_widget)
    # Append the generation range slider inline with the other partition
    # widgets when the analysis type spans multiple generations. Suffix
    # shows the selected + available range so the user can see both.
    _show_gen_range = (
        analysis_select.value
        in (
            "multigeneration",
            "multiseed",
        )
        and len(available_generations) >= 2
    )
    if _show_gen_range:
        _gen_lo, _gen_hi = generation_range_slider.value
        _gen_avail_lo = available_generations[0]
        _gen_avail_hi = available_generations[-1]
        _partition_row.append(mo.md("**generations:**"))
        _partition_row.append(generation_range_slider)
        _partition_row.append(
            mo.md(
                f"**{int(_gen_lo)}–{int(_gen_hi)}** "
                f"(of {_gen_avail_lo}–{_gen_avail_hi})"
            )
        )

    # ---- compare section ----
    # Compare datasets inherit primary's analysis type, so each slot only
    # shows the partition fields that analysis actually requires (matches
    # the primary partition row exactly).
    _required = list(partition_groups[analysis_select.value])
    _slot_field_for_key = {
        "experiment_id": "exp",
        "variant": "variant",
        "lineage_seed": "seed",
        "generation": "generation",
        "agent_id": "agent_id",
    }
    _key_display = {
        "experiment_id": "experiment_id",
        "variant": "variant",
        "lineage_seed": "lineage_seed",
        "generation": "generation",
        "agent_id": "agent_id",
    }

    _compare_header = mo.hstack(
        [mo.md("**Compare datasets:**"), compare_enabled]
        + (
            [mo.md("**# of datasets:**"), compare_count]
            if compare_enabled.value
            else []
        ),
        justify="start",
        align="center",
        gap=0.5,
    )

    _slot_rows = []
    if compare_enabled.value:
        _n = int(compare_count.value or 0)
        for _i in range(min(_n, len(compare_slots))):
            _slot = compare_slots[_i]
            _row = [mo.md("**dataset label:**"), _slot["label"]]
            for _k in _required:
                _row.append(mo.md(f"**{_key_display[_k]}:**"))
                _row.append(_slot[_slot_field_for_key[_k]])
            _slot_rows.append(mo.hstack(_row, justify="start", align="center", gap=0.4))

    # Wrap the header + slot rows in a bordered box. mo.Html preserves any
    # embedded UI elements' interactivity (they're referenced by the
    # containing vstack's rendered HTML).
    _compare_section_inner = mo.vstack(
        [_compare_header] + _slot_rows if _slot_rows else [_compare_header]
    )
    _compare_section = mo.Html(
        '<div style="border: 1px solid #d0d0d0; border-radius: 6px; '
        'padding: 10px 12px; margin: 4px 0;">' + _compare_section_inner.text + "</div>"
    )

    mo.vstack(
        [
            mo.md("# vEcoli Output Explorer"),
            mo.accordion({"About this notebook": about_intro_md}),
            mo.hstack([mo.md("**analysis:**"), analysis_select], justify="start"),
            mo.hstack(_partition_row, justify="start"),
            mo.hstack(
                [
                    mo.md("**pathway:**"),
                    select_pathway,
                    mo.md("**compare layout:**"),
                    chart_layout_mode,
                ],
                justify="start",
            ),
            _compare_section,
        ]
    )
    return


# ===== Causality tab cells (ported from causality/causality_explorer.py) =====
# The causality tab reuses v6's primary-dataset dropdowns (exp_select,
# variant_select, seed_select, gen_select, agent_select) and the
# create_duckdb_conn / dataset_sql imports already loaded above.


@app.cell
def _(anywidget, mo, traitlets):
    # PinnedSelect — anywidget single-select widget for option lists that
    # exceed marimo's 1000-item cap. Kept separate from PinnedMultiselect above
    # because the causality node picker requires exclusive (single) selection
    # and its own search-then-render cycle.
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
        opts = list(options)
        val = value if value is not None else (opts[0] if opts else "")
        raw = PinnedSelect(
            options=opts,
            value=str(val),
            placeholder=str(placeholder),
        )
        return mo.ui.anywidget(raw)

    def pss_value(widget):
        v = widget.value
        if isinstance(v, dict):
            return v.get("value") or ""
        return v or ""

    return PinnedSelect, pss, pss_value


@app.cell
def _(mo):
    # Browser-style navigation state for the causality node picker. `history`
    # is the sequence of visited node IDs, `cursor` is the current position.
    # allow_self_loops=True so buttons defined in the reader cell can update
    # the state and cause that same cell to re-run.
    get_causality_nav, set_causality_nav = mo.state(
        {"history": [], "cursor": -1}, allow_self_loops=True,
    )
    return get_causality_nav, set_causality_nav


@app.cell
def _(wd_root):
    # wd_root is already appended to sys.path in the main import cell above,
    # so these imports resolve. Only the causality-specific modules are
    # imported here — create_duckdb_conn / dataset_sql come from v6.
    _ = wd_root  # explicit dep so this cell runs after path setup
    from ecoli.analysis.causality_network.viewer import CausalityBundle
    from ecoli.analysis.causality_network.build_network import BuildNetwork
    from ecoli.analysis.causality_network import read_dynamics
    from wholecell.utils import filepath as fp

    return BuildNetwork, CausalityBundle, fp, read_dynamics


@app.cell
def _(
    agent_select,
    exp_select,
    gen_select,
    os,
    seed_select,
    variant_select,
    wd_root,
):
    # Derive the per-cell Hive-style seriesOut path from the primary dataset
    # dropdowns. Guards against "N/A" (v6's placeholder when a partition level
    # is missing) — treats it as "not picked yet".
    def _bad(v):
        return v is None or v == "" or v == "N/A"

    if any(_bad(v) for v in (
        exp_select.value, variant_select.value, seed_select.value,
        gen_select.value, agent_select.value,
    )):
        causality_seriesOut_dir = None
        causality_seriesOut_zip = None
    else:
        causality_seriesOut_dir = os.path.join(
            wd_root, "out", exp_select.value, "seriesOut",
            f"variant={variant_select.value}",
            f"lineage_seed={seed_select.value}",
            f"generation={gen_select.value}",
            f"agent_id={agent_select.value}",
        )
        causality_seriesOut_zip = os.path.join(
            causality_seriesOut_dir, "seriesOut.zip",
        )
    return causality_seriesOut_dir, causality_seriesOut_zip


@app.cell
def _(causality_seriesOut_zip, mo, os):
    # Build/rebuild button + status. The vstack `causality_build_ui` is what
    # the tab displays; `causality_build_trigger` is the run_button whose
    # `.value` is a click counter that the build cell watches.
    _exists = (
        causality_seriesOut_zip is not None
        and os.path.exists(causality_seriesOut_zip)
    )
    if causality_seriesOut_zip is None:
        _status = mo.md(
            "_Pick experiment · variant · seed · generation · agent above "
            "to select a single cell._"
        )
        causality_build_trigger = mo.ui.run_button(
            label="Build seriesOut", disabled=True,
        )
    elif _exists:
        _status = mo.md("✅ `seriesOut.zip` already exists for this cell.")
        causality_build_trigger = mo.ui.run_button(label="Rebuild seriesOut")
    else:
        _status = mo.md(
            "⚠️ `seriesOut.zip` **not yet built** for this cell — "
            "click **Build** to generate it (may take several seconds)."
        )
        causality_build_trigger = mo.ui.run_button(
            label="Build seriesOut", kind="success",
        )
    causality_build_ui = mo.vstack([_status, causality_build_trigger])
    return causality_build_trigger, causality_build_ui


@app.cell
def _(
    BuildNetwork,
    agent_select,
    causality_build_trigger,
    causality_seriesOut_dir,
    causality_seriesOut_zip,
    create_duckdb_conn,
    dataset_sql,
    exp_select,
    fp,
    gen_select,
    mo,
    os,
    read_dynamics,
    seed_select,
    variant_select,
    wd_root,
):
    # Fires only on Build click (run_button.value is True once per click).
    # Mirrors BuildCausalityNetwork.run(...) in buildCausalityNetwork.py.
    if not causality_build_trigger.value or causality_seriesOut_dir is None:
        causality_build_msg = mo.md("")
    else:
        _out_dir = os.path.join(wd_root, "out")
        _sim_data_path = os.path.join(
            _out_dir, exp_select.value, "variant_sim_data",
            f"{variant_select.value}.cPickle",
        )
        if not os.path.isfile(_sim_data_path):
            causality_build_msg = mo.md(
                f"⚠️ variant sim_data not found: `{_sim_data_path}`"
            )
        else:
            fp.makedirs(causality_seriesOut_dir)
            with mo.status.spinner(title="Building causality network...") as _sp:
                _network = BuildNetwork(
                    _sim_data_path, causality_seriesOut_dir, False,
                )
                _node_list, _edge_list = _network.build_nodes_and_edges()
                _sp.update(title="Reading dynamics from parquet...")
                _conn = create_duckdb_conn(_out_dir, False, None)
                _hist_sql, _cfg_sql, _ = dataset_sql(
                    _out_dir, [exp_select.value],
                )
                _where = (
                    f"experiment_id = '{exp_select.value}' "
                    f"AND variant = {variant_select.value} "
                    f"AND lineage_seed = {seed_select.value} "
                    f"AND generation = {gen_select.value} "
                    f"AND agent_id = '{agent_select.value}'"
                )
                _hist_sql = f"SELECT * FROM ({_hist_sql}) WHERE {_where}"
                _cfg_sql = f"SELECT * FROM ({_cfg_sql}) WHERE {_where}"
                _sp.update(title="Writing seriesOut.zip...")
                read_dynamics.convert_dynamics(
                    causality_seriesOut_dir, _network.sim_data,
                    _node_list, _edge_list, exp_select.value,
                    _conn, _hist_sql, _cfg_sql,
                )
            causality_build_msg = mo.md(
                f"✅ Built `{causality_seriesOut_zip}`."
            )
    return (causality_build_msg,)


@app.cell
def _(
    CausalityBundle,
    causality_build_trigger,
    causality_seriesOut_zip,
    mo,
    os,
):
    # Bundle loader — reads `build_trigger.value` so a fresh build triggers a
    # reload here (the zip on disk changes, but marimo needs an explicit dep
    # to know to re-run this cell).
    _ = causality_build_trigger.value
    if causality_seriesOut_zip is None:
        causality_bundle = None
        causality_bundle_msg = mo.md(
            "_Pick a full partition above to load a causality bundle._"
        )
    elif not os.path.exists(causality_seriesOut_zip):
        causality_bundle = None
        causality_bundle_msg = mo.md(
            "_`seriesOut.zip` for this cell hasn't been built yet — click_ "
            "**Build seriesOut** _above._"
        )
    else:
        causality_bundle = CausalityBundle(causality_seriesOut_zip)
        causality_bundle_msg = mo.md(
            f"**Causality network loaded** — "
            f"{len(causality_bundle.nodes):,} nodes, "
            f"{len(causality_bundle.edges):,} edges, "
            f"{len(causality_bundle.time)} timesteps."
        )
    return causality_bundle, causality_bundle_msg


@app.cell
def _(causality_bundle, mo):
    # "(all)" default so the picker can reflect any currently-navigated node,
    # regardless of type — otherwise navigating to a non-Gene node via the
    # map couldn't be shown in the picker while it was filtered to Gene.
    if causality_bundle is None:
        causality_type_filter = None
    else:
        causality_type_filter = mo.ui.dropdown(
            options=["(all)"] + causality_bundle.node_types,
            value="(all)",
            label="node type:",
        )
    return (causality_type_filter,)


@app.cell
def _(causality_bundle, causality_type_filter, get_causality_nav, mo, pss):
    # Rebuild the node picker whenever the bundle / filter / nav state changes.
    # The picker's `value` is driven by nav-history so map clicks and Back /
    # Forward all surface here.
    if causality_bundle is None or causality_type_filter is None:
        causality_label_to_id = {}
        causality_id_to_label = {}
        causality_node_picker = None
        causality_picker_ui = mo.md("")
    else:
        if causality_type_filter.value == "(all)":
            _candidates = causality_bundle.nodes
        else:
            _candidates = [
                n for n in causality_bundle.nodes
                if n["type"] == causality_type_filter.value
            ]

        def _label(n):
            name = n.get("name") or ""
            return (
                f"{n['ID']}  —  {name}"
                if name and name != n["ID"] else n["ID"]
            )

        causality_label_to_id = {_label(n): n["ID"] for n in _candidates}

        _s = get_causality_nav()
        _history = _s["history"]
        _cursor = _s["cursor"]
        _current_id = (
            _history[_cursor] if 0 <= _cursor < len(_history) else None
        )
        # If the current node isn't in the filtered candidates, splice it in
        # so the picker can still display it — otherwise the fallback below
        # would be interpreted as a user change and would navigate us away.
        if _current_id and _current_id not in causality_label_to_id.values():
            _extra_node = causality_bundle.get_node(_current_id)
            if _extra_node is not None:
                _extra_label = _label(_extra_node)
                causality_label_to_id = {
                    _extra_label: _current_id, **causality_label_to_id,
                }

        causality_id_to_label = {
            v: k for k, v in causality_label_to_id.items()
        }
        _labels = list(causality_label_to_id.keys())
        _value = (
            causality_id_to_label.get(_current_id) if _current_id else None
        )
        if _value is None:
            _value = _labels[0] if _labels else ""

        causality_node_picker = pss(
            options=_labels, value=_value,
            placeholder="Pick a node — click to search",
        )
        causality_picker_ui = mo.vstack([
            mo.md(f"**Current node** ({len(_candidates):,} in filter)"),
            causality_node_picker,
        ])
    return (
        causality_id_to_label,
        causality_label_to_id,
        causality_node_picker,
        causality_picker_ui,
    )


@app.cell
def _(
    causality_label_to_id,
    causality_node_picker,
    get_causality_nav,
    pss_value,
    set_causality_nav,
):
    # Detect user-driven picker changes and push them into history. When the
    # picker was rebuilt programmatically (map click), its value already
    # matches the state's current so nothing happens here. Transient
    # exceptions during the reactive cascade are swallowed silently.
    if causality_node_picker is not None:
        try:
            _picker_label = pss_value(causality_node_picker)
            _picker_id = (
                causality_label_to_id.get(_picker_label)
                if _picker_label else None
            )
            _s = get_causality_nav()
            _history = _s["history"]
            _cursor = _s["cursor"]
            _current_id = (
                _history[_cursor] if 0 <= _cursor < len(_history) else None
            )
            if _picker_id and _picker_id != _current_id:
                _new_hist = _history[: _cursor + 1] + [_picker_id]
                set_causality_nav(
                    {"history": _new_hist, "cursor": len(_new_hist) - 1}
                )
        except Exception:
            pass
    return


@app.cell
def _(
    causality_bundle,
    causality_label_to_id,
    causality_node_picker,
    get_causality_nav,
    mo,
    pss_value,
    set_causality_nav,
):
    # Current node is whatever the nav state points at; falls back to the
    # picker's raw value on cold start. `causality_header` is the vstack of
    # back/forward buttons + a metadata card; it's rendered by the tab.
    if causality_bundle is None or causality_node_picker is None:
        causality_node = None
        causality_header = mo.md("")
    else:
        _nav_state = get_causality_nav()
        _history = _nav_state["history"]
        _cursor = _nav_state["cursor"]
        _current_id = (
            _history[_cursor] if 0 <= _cursor < len(_history) else None
        )
        if _current_id is None:
            _picker_label = pss_value(causality_node_picker)
            _current_id = (
                causality_label_to_id.get(_picker_label)
                if _picker_label else None
            )
        causality_node = (
            causality_bundle.get_node(_current_id) if _current_id else None
        )
        if causality_node is None:
            _current_id = (
                causality_bundle.nodes[0]["ID"]
                if causality_bundle.nodes else None
            )
            causality_node = (
                causality_bundle.get_node(_current_id) if _current_id else None
            )

        if causality_node is None:
            causality_header = mo.md("_No nodes in bundle._")
        else:
            _url_line = (
                f"[EcoCyc]({causality_node['url']})"
                if causality_node.get("url") else "_(no URL)_"
            )
            _syns = causality_node.get("synonyms") or []
            _syn_line = ", ".join(_syns[:8]) if _syns else "_(none)_"
            _header_md = mo.md("\n".join([
                f"### {causality_node.get('name') or causality_node['ID']}",
                "",
                f"- **ID**: `{causality_node['ID']}`",
                f"- **Type**: {causality_node['type']}  •  "
                f"**Class**: {causality_node['class']}",
                f"- **Location**: {causality_node.get('location') or '—'}",
                f"- **Synonyms**: {_syn_line}",
                f"- **Link**: {_url_line}",
            ]))
            _can_back = _cursor > 0
            _can_fwd = _cursor < len(_history) - 1

            def _go_back(_v):
                _s = get_causality_nav()
                _c = _s["cursor"]
                if _c > 0:
                    set_causality_nav(
                        {"history": _s["history"], "cursor": _c - 1}
                    )

            def _go_fwd(_v):
                _s = get_causality_nav()
                _c = _s["cursor"]
                if _c < len(_s["history"]) - 1:
                    set_causality_nav(
                        {"history": _s["history"], "cursor": _c + 1}
                    )

            _back_btn = mo.ui.button(
                label="← back", on_click=_go_back, disabled=not _can_back,
            )
            _fwd_btn = mo.ui.button(
                label="forward →", on_click=_go_fwd, disabled=not _can_fwd,
            )
            _nav_row = mo.hstack(
                [_back_btn, _fwd_btn],
                justify="start", gap=0.5, align="center",
            )
            causality_header = mo.vstack([_nav_row, _header_md])
    return causality_header, causality_node


@app.cell
def _(alt, causality_bundle, causality_node, mo, pl):
    if causality_bundle is None or causality_node is None:
        causality_dyn = None
        causality_dyn_chart = mo.md("")
    else:
        causality_dyn = causality_bundle.get_dynamics(causality_node["ID"])
        _meta_by_type = {
            m["type"]: m
            for m in causality_bundle.get_series_meta(causality_node["ID"])
        }
        if not causality_dyn:
            causality_dyn_chart = mo.md(
                "_No dynamics recorded for this node._"
            )
        else:
            _rows = []
            _time = causality_bundle.time
            for _name, _values in causality_dyn.items():
                _values = _values.astype(float)
                _n = min(len(_time), len(_values))
                _unit = _meta_by_type.get(_name, {}).get("units", "")
                for _t, _v in zip(_time[:_n], _values[:_n]):
                    _rows.append({
                        "series": f"{_name} ({_unit})" if _unit else _name,
                        "time": float(_t), "value": float(_v),
                    })
            _df = pl.DataFrame(_rows)
            causality_dyn_chart = (
                alt.Chart(_df.to_pandas())
                .mark_line()
                .encode(
                    x=alt.X("time:Q", title="time (s)"),
                    y=alt.Y("value:Q", title="value"),
                    color=alt.Color(
                        "series:N", legend=alt.Legend(orient="top"),
                    ),
                )
                .properties(width=520, height=520, title="Dynamics")
            )
    return causality_dyn, causality_dyn_chart


@app.cell
def _(causality_bundle):
    # Walk chains of Process-class nodes until reaching a State so that
    # patterns like `State → Process → Process → State` (e.g. gene regulation:
    # `TF → TF-Binding → Regulation → Gene`) render as one edge annotated
    # with every intermediate process name. If the current node is itself a
    # Process, direct state neighbors are returned with no collapsing.
    _MAX_PROC_CHAIN = 4

    def causality_state_edges_fn(node_id, direction):
        if causality_bundle is None:
            return {}
        n_center = causality_bundle.get_node(node_id)
        center_is_process = (
            n_center is not None and n_center.get("class") == "Process"
        )

        def _neighbors(nid):
            return (
                causality_bundle.outgoing.get(nid, [])
                if direction == "downstream"
                else causality_bundle.incoming.get(nid, [])
            )

        result: dict[str, set[str]] = {}

        if center_is_process:
            for nbr in _neighbors(node_id):
                if nbr == node_id:
                    continue
                n_nbr = causality_bundle.get_node(nbr)
                if n_nbr is not None and n_nbr.get("class") != "Process":
                    result.setdefault(nbr, set()).add("")
            return result

        def _walk(current, labels, path):
            if len(labels) > _MAX_PROC_CHAIN:
                return
            for nbr in _neighbors(current):
                if nbr == node_id or nbr in path:
                    continue
                n_nbr = causality_bundle.get_node(nbr)
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

    return (causality_state_edges_fn,)


@app.cell
def _(causality_bundle, mo):
    # Pathway map filters — options come from bundle-wide State-class types
    # (not just those in the current neighborhood) so the cell does NOT depend
    # on `causality_node`. Otherwise every navigation would rebuild the widget
    # and reset the user's selection.
    if causality_bundle is None:
        causality_map_type_filter = None
        causality_map_depth = None
    else:
        _state_types = sorted({
            (_n.get("type") or "")
            for _n in causality_bundle.nodes
            if _n.get("class") != "Process"
        } - {""})
        causality_map_type_filter = mo.ui.multiselect(
            options=_state_types, value=_state_types,
            label="neighborhood nodes:",
        )
        causality_map_depth = mo.ui.slider(
            start=1, stop=4, step=1, value=1,
            label="depth", show_value=True,
        )
    return causality_map_depth, causality_map_type_filter


@app.cell
def _(
    alt,
    causality_bundle,
    causality_dyn_chart,
    causality_map_depth,
    causality_map_type_filter,
    causality_node,
    causality_state_edges_fn,
    mo,
    pl,
):
    # Interactive pathway map (portrait, right of the dynamics plot). Upstream
    # stacks above center, downstream below. Process-class nodes are collapsed
    # (state→Process→state → one edge with the process name as its label). The
    # depth slider controls how many hops out we walk. BFS visits each state
    # once, so the drawing is a spanning tree rooted at the center.
    import math as _math

    if (
        causality_bundle is None or causality_node is None
        or causality_map_depth is None
        or causality_map_type_filter is None
    ):
        causality_pathway_chart = None
        causality_pathway_ui = mo.md("")
    else:
        _center_id = causality_node["ID"]
        _max_depth = int(causality_map_depth.value)
        _MAX_PER_LEVEL = 12
        _MAX_EDGE_LABEL = 32
        _type_ok = (
            set(causality_map_type_filter.value)
            if causality_map_type_filter.value else None
        )

        def _passes(nid):
            _n_p = causality_bundle.get_node(nid)
            if _n_p is None:
                return False
            if _type_ok is not None and (_n_p.get("type") or "") not in _type_ok:
                return False
            return True

        def _lbl(nid):
            _n = causality_bundle.get_node(nid)
            if not _n:
                return nid
            _nm = _n.get("name") or ""
            return _nm if _nm and _nm != nid else nid

        _ALIGN_EXPR = (
            "datum.x > 0.6 ? 'right' : datum.x < -0.6 ? 'left' : 'center'"
        )

        def _typ(nid):
            _n = causality_bundle.get_node(nid)
            return (_n.get("type") if _n else "") or ""

        def _fmt_edge_label(labels_set):
            _pieces = sorted(_l for _l in labels_set if _l)
            if not _pieces:
                return ""
            _joined = " / ".join(_pieces)
            return _joined if len(_joined) <= _MAX_EDGE_LABEL else (
                _joined[: _MAX_EDGE_LABEL - 1] + "…"
            )

        def _walk(direction):
            _visited = {_center_id}
            _levels = {0: [_center_id]}
            _parent = {}
            _truncated = {}
            _frontier = [_center_id]
            for _depth in range(1, _max_depth + 1):
                _next = []
                for _src in _frontier:
                    _edge_map = causality_state_edges_fn(_src, direction)
                    for _dst in sorted(_edge_map.keys()):
                        if _dst in _visited or not _passes(_dst):
                            continue
                        _parent[_dst] = (_src, _fmt_edge_label(_edge_map[_dst]))
                        _next.append(_dst)
                        _visited.add(_dst)
                if len(_next) > _MAX_PER_LEVEL:
                    _truncated[_depth] = len(_next) - _MAX_PER_LEVEL
                    for _d in _next[_MAX_PER_LEVEL:]:
                        _visited.discard(_d)
                        _parent.pop(_d, None)
                    _next = _next[:_MAX_PER_LEVEL]
                if not _next:
                    break
                _levels[_depth] = _next
                _frontier = _next
            return _levels, _parent, _truncated

        _up_levels, _up_parent, _up_trunc = _walk("upstream")
        _dn_levels, _dn_parent, _dn_trunc = _walk("downstream")

        _pos = {_center_id: (0.0, 0.0)}

        def _place(level_ids, y):
            _n_lvl = len(level_ids)
            if _n_lvl == 0:
                return
            if _n_lvl == 1:
                _pos[level_ids[0]] = (0.0, y)
                return
            _span = 3.2
            _step = _span / (_n_lvl - 1)
            for _i, _nid in enumerate(level_ids):
                _pos[_nid] = (-1.6 + _i * _step, y)

        for _depth, _ids in _up_levels.items():
            if _depth != 0:
                _place(_ids, float(_depth))
        for _depth, _ids in _dn_levels.items():
            if _depth != 0:
                _place(_ids, -float(_depth))

        _node_rows = [{
            "id": _center_id, "label": _lbl(_center_id),
            "type": causality_node.get("type") or "", "role": "current",
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

        _EDGE_ELBOW_K = 0.5
        _edge_rows = []
        _edge_label_rows = []
        _arrow_rows = []

        def _add_edge(src_id, dst_id, label):
            _sx, _sy = _pos[src_id]
            _dx, _dy = _pos[dst_id]
            _ex = _dx
            _ey = _sy + _EDGE_ELBOW_K * (_dy - _sy)
            for _i, (_x, _y) in enumerate(
                ((_sx, _sy), (_ex, _ey), (_dx, _dy))
            ):
                _edge_rows.append({
                    "edge_id": dst_id, "id": dst_id,
                    "x": _x, "y": _y, "order": _i,
                })
            if label:
                _label_t = 0.65
                _edge_label_rows.append({
                    "id": dst_id,
                    "x": _sx + _label_t * (_ex - _sx),
                    "y": _sy + _label_t * (_ey - _sy),
                    "label": label,
                })

            def _flow_angle(p1x, p1y, p2x, p2y):
                if p1y >= p2y:
                    _vx, _vy = p2x - p1x, p2y - p1y
                else:
                    _vx, _vy = p1x - p2x, p1y - p2y
                return _math.degrees(_math.atan2(_vx, _vy))

            _arrow_rows.append({
                "id": dst_id,
                "x": _sx + 0.25 * (_ex - _sx),
                "y": _sy + 0.25 * (_ey - _sy),
                "angle": _flow_angle(_sx, _sy, _ex, _ey),
            })
            _arrow_rows.append({
                "id": dst_id,
                "x": _dx, "y": (_ey + _dy) / 2.0,
                "angle": _flow_angle(_ex, _ey, _dx, _dy),
            })

        for _dst_id, (_src_id, _label) in _up_parent.items():
            _add_edge(_src_id, _dst_id, _label)
        for _dst_id, (_src_id, _label) in _dn_parent.items():
            _add_edge(_src_id, _dst_id, _label)

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

        _up_reach = max((_d for _d in _up_levels if _d != 0), default=0)
        _dn_reach = max((_d for _d in _dn_levels if _d != 0), default=0)
        _y_max = float(_up_reach) + 0.6
        _y_min = -(float(_dn_reach) + 0.6)
        _height = min(760, max(320, 90 * (_up_reach + _dn_reach + 1) + 60))

        _click = alt.selection_point(
            name="click_sel", fields=["id"], on="click",
            empty="none", clear=False,
        )
        _hover = alt.selection_point(
            name="hover_sel", fields=["id"], on="mouseover",
            empty="none", clear="mouseout",
        )

        _edges_layer = (
            alt.Chart(_edges_pdf)
            .mark_line(
                cursor="pointer", strokeCap="round", strokeJoin="round",
            )
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
                detail="edge_id:N",
                order="order:Q",
                size=alt.condition(_hover, alt.value(9), alt.value(5)),
                color=alt.condition(
                    _hover, alt.value("#e07a2b"), alt.value("#a3adb8"),
                ),
                tooltip=alt.Tooltip("id:N", title="edge"),
            )
        )

        _arrows_layer = (
            alt.Chart(_arrows_pdf)
            .mark_point(
                shape="triangle-up", filled=True,
                stroke=None, opacity=1.0,
            )
            .encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[-2.0, 2.0]), axis=None),
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
                angle=alt.Angle("angle:Q", scale=None),
                size=alt.condition(_hover, alt.value(260), alt.value(160)),
                color=alt.condition(
                    _hover, alt.value("#e07a2b"), alt.value("#5a6470"),
                ),
            )
        )

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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
                text="label:N", color=alt.value("#222"),
            )
        )

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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
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
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[_y_min, _y_max]), axis=None,
                ),
                text="label:N", color=alt.value("#222"),
            )
        )

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
                width=340, height=_height, title="Neighborhood",
                padding={"left": 60, "right": 60, "top": 10, "bottom": 10},
            )
        )

        _hint_bits = []
        for _d, _n_hb in sorted(_up_trunc.items()):
            _hint_bits.append(
                f"upstream depth {_d} truncated to "
                f"{_MAX_PER_LEVEL} (dropped {_n_hb:,})"
            )
        for _d, _n_hb in sorted(_dn_trunc.items()):
            _hint_bits.append(
                f"downstream depth {_d} truncated to "
                f"{_MAX_PER_LEVEL} (dropped {_n_hb:,})"
            )
        _hint = (
            mo.md(f"_({' · '.join(_hint_bits)})_")
            if _hint_bits else mo.md("")
        )

        causality_pathway_chart = mo.ui.altair_chart(
            _chart, chart_selection=False, legend_selection=False,
        )
        _filter_row = mo.hstack(
            [causality_map_type_filter, causality_map_depth],
            justify="start", gap=1, align="start",
        )
        causality_pathway_ui = mo.hstack(
            [
                causality_dyn_chart,
                mo.vstack(
                    [_filter_row, causality_pathway_chart, _hint]
                ),
            ],
            justify="start", gap=1, align="start",
        )
    return causality_pathway_chart, causality_pathway_ui


@app.cell
def _(
    causality_node,
    causality_pathway_chart,
    get_causality_nav,
    set_causality_nav,
):
    # Read `.value` first to establish the reactive dependency, then read raw
    # click_sel from `.selections` so we navigate on clicks only (ignoring
    # hover state entirely). `.value` is the intersection of *all* selections,
    # so a hover on a different node than the clicked one would zero it out.
    if causality_pathway_chart is not None and causality_node is not None:
        try:
            _ = causality_pathway_chart.value
            _sels = getattr(causality_pathway_chart, "selections", None) or {}
            _click_state = _sels.get("click_sel") or {}
            _ids = (
                _click_state.get("id")
                if isinstance(_click_state, dict) else None
            )
            _sel_id = (
                _ids[0] if isinstance(_ids, (list, tuple)) and _ids else None
            )
            _cur_id = (
                causality_node.get("ID")
                if isinstance(causality_node, dict) else None
            )
            if _sel_id and _cur_id and _sel_id != _cur_id:
                _s = get_causality_nav()
                _new_hist = _s["history"][: _s["cursor"] + 1] + [_sel_id]
                set_causality_nav(
                    {"history": _new_hist, "cursor": len(_new_hist) - 1}
                )
        except Exception:
            pass
    return


@app.cell
def _(causality_bundle, causality_node, mo):
    if causality_bundle is None or causality_node is None:
        causality_neighborhood_ui = mo.md("")
    else:
        _up_ids = sorted(
            causality_bundle.upstream(causality_node["ID"], depth=1)
        )
        _dn_ids = sorted(
            causality_bundle.downstream(causality_node["ID"], depth=1)
        )

        def _row(nid):
            _n = causality_bundle.get_node(nid)
            return {
                "ID": nid,
                "name": (_n.get("name") if _n else "") or "",
                "type": (_n.get("type") if _n else "") or "",
                "class": (_n.get("class") if _n else "") or "",
            }

        _up_rows = [_row(nid) for nid in _up_ids]
        _dn_rows = [_row(nid) for nid in _dn_ids]

        _up_body = (
            mo.ui.table(_up_rows, pagination=True, page_size=10)
            if _up_rows else mo.md("_(none)_")
        )
        _dn_body = (
            mo.ui.table(_dn_rows, pagination=True, page_size=10)
            if _dn_rows else mo.md("_(none)_")
        )
        causality_neighborhood_ui = mo.accordion(
            {
                f"Upstream ({len(_up_rows)})": _up_body,
                f"Downstream ({len(_dn_rows)})": _dn_body,
            },
            multiple=True,
        )
    return (causality_neighborhood_ui,)


@app.cell
def _(mo):
    about_causality_md = mo.md(
        """
    Interactive **causality-network explorer**: walks the whole-cell model's
    upstream/downstream dependency graph for a single cell (agent). Uses the
    primary experiment / variant / lineage_seed / generation / agent_id
    selected in the toolbar above.

    - Click **Build seriesOut** the first time to generate the bundle for the
      selected cell (needs `variant_sim_data/{variant}.cPickle`). Subsequent
      loads read the cached `seriesOut.zip`.
    - Pick any node from the searchable dropdown. The left chart shows its
      dynamics; the right chart is a portrait pathway map with upstream nodes
      above and downstream below.
    - **Click** a node in the pathway map to navigate there; use **← back** /
      **forward →** to move through history. **Hover** any node or edge to
      surface its label.
    - The bottom accordion lists depth-1 upstream/downstream neighbors as
      tables.
    """
    )
    return (about_causality_md,)


@app.cell
def _(
    about_causality_md,
    about_complexation_md,
    about_compounds_md,
    about_download_md,
    about_metabolism_md,
    about_mrna_md,
    about_physiology_md,
    about_proteins_md,
    about_regulation_md,
    about_transcription_md,
    about_translation_md,
    about_validation_md,
    bulk_sp_plot,
    causality_build_msg,
    causality_build_ui,
    causality_bundle_msg,
    causality_header,
    causality_neighborhood_ui,
    causality_pathway_ui,
    causality_picker_ui,
    causality_type_filter,
    chart_complexation,
    chart_compounds,
    chart_mrna,
    chart_monomers,
    chart_physiology,
    chart_regulation,
    chart_rxns,
    chart_transcription,
    chart_transcription_scalar,
    chart_translation,
    chart_translation_scalar,
    chart_val,
    complexation_select_plot,
    dl_delivery_radio,
    dl_disk_dir_input,
    dl_filename_input,
    dl_mode_select,
    dl_run_button,
    dl_scalar_cols_select,
    dl_schema_note_md,
    dl_size_threshold_mb,
    dl_source_dataset_select,
    dl_time_interval,
    dl_vector_col_select,
    dl_vector_label_mode,
    download_memory_notes_md,
    download_status_render,
    compound_unit,
    metabolism_quantity_select,
    mo,
    molecule_id_type,
    monomer_label_type,
    monomer_select_plot,
    physiology_category,
    physiology_select,
    plot_df_bulk,
    plot_df_complexation,
    plot_df_mrna,
    plot_df_monomers,
    plot_df_physiology,
    plot_df_regulation,
    plot_df_rxns,
    plot_df_transcription,
    plot_df_transcription_scalar,
    plot_df_translation,
    plot_df_translation_scalar,
    regulation_label_type,
    regulation_quantity_select,
    regulation_select_plot,
    rna_label_type,
    mrna_select_plot,
    select_rxns,
    transcription_label_type,
    transcription_quantity_select,
    transcription_scalar_category,
    transcription_scalar_select,
    transcription_select_plot,
    translation_label_type,
    translation_quantity_select,
    translation_scalar_category,
    translation_scalar_select,
    translation_select_plot,
    val_dataset_select,
    val_id_select,
    val_label_type,
    y_scale,
    y_scale_complexation,
    y_scale_mrna,
    y_scale_monomers,
    y_scale_physiology,
    y_scale_regulation,
    y_scale_rxns,
    y_scale_transcription,
    y_scale_transcription_scalar,
    y_scale_translation,
    y_scale_translation_scalar,
):
    # Tabs live below the toolbar in their own cell. If any chart cell errors
    # (e.g. transient parquet read), this cell shows the error in-place while
    # the toolbar above remains usable.
    def _chart_or_placeholder(chart, msg):
        return chart if chart is not None else mo.md(f"_{msg}_")

    def _plot_download(df, filename):
        """Emit a bottom-right click-to-download TSV button for a plot
        dataframe. Returns a right-aligned muted 'no data' message when the
        dataframe is empty/None so the tab layout stays consistent whether
        the chart has data or not. Button is naturally sized to its label
        (hstack with justify=end keeps it snug in the bottom-right corner)."""
        if df is None or len(df) == 0:
            _widget = mo.md("_(no plot data to download)_")
        else:
            _tsv = df.write_csv(separator="\t").encode("utf-8")
            _widget = mo.download(
                data=_tsv,
                filename=filename,
                label="Download plot data",
                mimetype="text/tab-separated-values",
            )
        return mo.hstack([_widget], justify="end", align="center")

    _compounds_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_compounds_md}),
            mo.hstack(
                [
                    mo.md("**label:**"),
                    molecule_id_type,
                    mo.md("**unit:**"),
                    compound_unit,
                    mo.md("**compounds:**"),
                    bulk_sp_plot,
                    mo.md("**scale:**"),
                    y_scale,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_compounds, "Select compounds to see the chart."
            ),
            _plot_download(plot_df_bulk, "compounds_plot_data.tsv"),
        ]
    )

    _mrna_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_mrna_md}),
            mo.hstack(
                [
                    mo.md("**label:**"),
                    rna_label_type,
                    mo.md("**genes:**"),
                    mrna_select_plot,
                    mo.md("**scale:**"),
                    y_scale_mrna,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_mrna, "Select mRNAs to see the chart."),
            _plot_download(plot_df_mrna, "mrna_plot_data.tsv"),
        ]
    )

    _proteins_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_proteins_md}),
            mo.hstack(
                [
                    mo.md("**label:**"),
                    monomer_label_type,
                    mo.md("**proteins:**"),
                    monomer_select_plot,
                    mo.md("**scale:**"),
                    y_scale_monomers,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_monomers, "Select proteins to see the chart."),
            _plot_download(plot_df_monomers, "proteins_plot_data.tsv"),
        ]
    )

    _metabolism_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_metabolism_md}),
            mo.hstack(
                [
                    mo.md("**quantity:**"),
                    metabolism_quantity_select,
                    mo.md("**reaction IDs:**"),
                    select_rxns,
                    mo.md("**scale:**"),
                    y_scale_rxns,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_rxns, "Select reactions to see the chart."),
            _plot_download(plot_df_rxns, "metabolism_plot_data.tsv"),
        ]
    )

    _validation_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_validation_md}),
            mo.hstack(
                [
                    mo.md("**dataset:**"),
                    val_dataset_select,
                    mo.md("**label type:**"),
                    val_label_type,
                    mo.md("**proteins:**"),
                    val_id_select,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_val, "Select proteins to see the scatter."),
            mo.md(
                "_(Validation scatter uses derived data; use the **Download** "
                "tab for raw counts.)_"
            ),
        ]
    )

    # Fixed-width label column so every input lines up vertically.
    def _dl_label(text):
        return mo.md(
            f'<div style="min-width: 240px; display: inline-block;"><b>{text}</b></div>'
        )

    def _dl_row(label, widget):
        return mo.hstack(
            [_dl_label(label), widget],
            justify="start",
            align="center",
            gap=0.5,
        )

    # "What to export" section: source dataset + mode + column picker(s)
    if dl_mode_select.value == "vector column":
        _picker_rows = [
            _dl_row("Vector column:", dl_vector_col_select),
            _dl_row("Header labels:", dl_vector_label_mode),
        ]
    else:
        _picker_rows = [_dl_row("Scalar columns:", dl_scalar_cols_select)]

    _what_section = mo.vstack(
        [
            mo.md("#### What to export"),
            _dl_row("Source dataset:", dl_source_dataset_select),
            _dl_row("Export mode:", dl_mode_select),
            *_picker_rows,
        ]
    )

    # "Output options" section
    _output_rows = [
        _dl_row("Filename:", dl_filename_input),
        _dl_row("Delivery:", dl_delivery_radio),
        _dl_row("Auto-switch threshold (MB):", dl_size_threshold_mb),
        _dl_row("Min Δt between samples (s):", dl_time_interval),
    ]
    if dl_delivery_radio.value != "browser download":
        _output_rows.append(_dl_row("Output directory:", dl_disk_dir_input))

    _output_section = mo.vstack([mo.md("#### Output options"), *_output_rows])

    _download_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_download_md}),
            dl_schema_note_md,
            _what_section,
            _output_section,
            mo.md("####  "),
            dl_run_button,
            download_status_render,
            mo.accordion({"Memory notes for large exports": download_memory_notes_md}),
        ]
    )

    _physiology_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_physiology_md}),
            mo.hstack(
                [
                    mo.md("**category:**"),
                    physiology_category,
                    mo.md("**quantities:**"),
                    physiology_select,
                    mo.md("**scale:**"),
                    y_scale_physiology,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_physiology, "Select physiology quantities to see the chart."
            ),
            _plot_download(plot_df_physiology, "physiology_plot_data.tsv"),
        ]
    )

    _transcription_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_transcription_md}),
            mo.md("#### Per-cistron"),
            mo.hstack(
                [
                    mo.md("**quantity:**"),
                    transcription_quantity_select,
                    mo.md("**label:**"),
                    transcription_label_type,
                    mo.md("**genes:**"),
                    transcription_select_plot,
                    mo.md("**scale:**"),
                    y_scale_transcription,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_transcription, "Select genes to see the chart."
            ),
            _plot_download(
                plot_df_transcription, "transcription_per_cistron_plot_data.tsv"
            ),
            mo.md("#### RNAP machinery / events (scalar)"),
            mo.hstack(
                [
                    mo.md("**category:**"),
                    transcription_scalar_category,
                    mo.md("**quantities:**"),
                    transcription_scalar_select,
                    mo.md("**scale:**"),
                    y_scale_transcription_scalar,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_transcription_scalar,
                "Select RNAP quantities to see the chart.",
            ),
            _plot_download(
                plot_df_transcription_scalar,
                "transcription_scalar_plot_data.tsv",
            ),
        ]
    )

    _translation_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_translation_md}),
            mo.md("#### Per-monomer / per-transcript"),
            mo.hstack(
                [
                    mo.md("**quantity:**"),
                    translation_quantity_select,
                    mo.md("**label:**"),
                    translation_label_type,
                    mo.md("**entities:**"),
                    translation_select_plot,
                    mo.md("**scale:**"),
                    y_scale_translation,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_translation, "Select entities to see the chart."
            ),
            _plot_download(plot_df_translation, "translation_per_entity_plot_data.tsv"),
            mo.md("#### Ribosome machinery / events (scalar)"),
            mo.hstack(
                [
                    mo.md("**category:**"),
                    translation_scalar_category,
                    mo.md("**quantities:**"),
                    translation_scalar_select,
                    mo.md("**scale:**"),
                    y_scale_translation_scalar,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_translation_scalar,
                "Select ribosome quantities to see the chart.",
            ),
            _plot_download(
                plot_df_translation_scalar,
                "translation_scalar_plot_data.tsv",
            ),
        ]
    )

    _regulation_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_regulation_md}),
            mo.hstack(
                [
                    mo.md("**quantity:**"),
                    regulation_quantity_select,
                    mo.md("**label:**"),
                    regulation_label_type,
                    mo.md("**entities:**"),
                    regulation_select_plot,
                    mo.md("**scale:**"),
                    y_scale_regulation,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_regulation, "Select entities to see the chart."
            ),
            _plot_download(plot_df_regulation, "regulation_plot_data.tsv"),
        ]
    )

    _complexation_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_complexation_md}),
            mo.hstack(
                [
                    mo.md("**complexation reactions:**"),
                    complexation_select_plot,
                    mo.md("**scale:**"),
                    y_scale_complexation,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_complexation, "Select reactions to see the chart."
            ),
            _plot_download(plot_df_complexation, "complexation_plot_data.tsv"),
        ]
    )

    _causality_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_causality_md}),
            causality_build_ui,
            causality_build_msg,
            causality_bundle_msg,
            (
                mo.hstack(
                    [causality_type_filter], justify="start",
                )
                if causality_type_filter is not None else mo.md("")
            ),
            causality_picker_ui,
            causality_header,
            causality_pathway_ui,
            causality_neighborhood_ui,
        ]
    )

    mo.ui.tabs(
        {
            "Physiology": _physiology_tab,
            "Compounds": _compounds_tab,
            "mRNA": _mrna_tab,
            "Transcription": _transcription_tab,
            "Regulation": _regulation_tab,
            "Proteins": _proteins_tab,
            "Translation": _translation_tab,
            "Complexation": _complexation_tab,
            "Metabolism": _metabolism_tab,
            "Validation": _validation_tab,
            "Causality": _causality_tab,
            "Download": _download_tab,
        },
    )
    return


if __name__ == "__main__":
    app.run()
