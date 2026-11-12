"""
Offline study of trigger efficiencies and per-trigger trigger scale factors (2023 postBPix).

The MC is the united output (``cf.UniteColumns``: the reduced and the production columns of the same events in one
file). Each MC dataset is read once (the first ``*`` of --mc is the dataset name), with only the columns the study
uses, and all channels, objects and variables are filled in that same pass. The data is the reduced output
(``cf.ReduceEvents`` with ``--selector-steps none``), also read once and split by channel. The QCD estimate is the ABCD
method of hbt/hist_hooks/qcd.py, per trigger category, added to the MC as process ``qcd``. Two variants are filled: the
MC with all weight columns except trigger_weight, and the MC with trigger_weight too (its ABCD reference MC is weighted
the same way, so its QCD estimate is computed with it). The plot shows both.

Objects per channel: etau: Jet, Tau, Electron (leading); mutau: Jet, Tau, Muon (leading); tautau: Jet, Tau (leading and
subleading). Each object is plotted for every variable of --vars.

The functions are plain numpy/awkward, so they run without columnflow. ``--selftest`` runs them on synthetic events.

Usage:
  python trigger_study.py --mc "<united>/*/*.parquet" --data-reduced <reduced_data.parquet> \\
      --channel etau,mutau,tautau --vars pt,eta --mc-weight <column,...> --outdir <dir> [--no-qcd] [--qcd-exclude <ids>]
  python trigger_study.py --selftest

Output: <outdir>/<channel>/<obj>_<var><index>/<category>.png, and summary.csv, processes.csv and efficiency.csv in
<outdir>.
"""
import argparse
import csv
import glob
import os
import sys
from collections import namedtuple

import awkward as ak
import numpy as np

# 2023 postBPix trigger ids (hbt/config/triggers.py, add_triggers_2023)
TRIGGER_IDS = [205, 105, 405, 304, 507, 701, 602, 605, 606, 608, 609]
GROUP_OF_ID = {205: 1, 105: 1, 405: 1, 304: 1, 507: 1, 701: 1, 602: 3, 605: 4, 606: 5, 608: 5, 609: 5}
# primary order: group order from hbt/selection/jet.py; within group 1 single lepton > cross lepton-tau >
# cross tautau > ttj (comment in jet.py); within group 5 the selection defines no order, so the id order is used
PRIORITY = [205, 105, 405, 304, 507, 701, 602, 605, 606, 608, 609]
CHANNEL_IDS = {"etau": 1, "mutau": 2, "tautau": 3}  # od.Channel ids of hbt/config/configs_hbt.py
# the plotted objects of each channel: (collection, index in the event, 0 = leading)
OBJECTS_OF_CHANNEL = {
    "etau": [("Jet", 0), ("Tau", 0), ("Electron", 0)],
    "mutau": [("Jet", 0), ("Tau", 0), ("Muon", 0)],
    "tautau": [("Jet", 0), ("Jet", 1), ("Tau", 0), ("Tau", 1)],
}
ABCD_REGIONS = ("os_iso", "os_noniso", "ss_iso", "ss_noniso")
QCD_LABEL = "qcd"

# one plotted quantity: channel (None: no channel cut), collection, variable, index of the object in the event
Spec = namedtuple("Spec", ["channel", "obj", "var", "index"])
# bin edges per variable (pt: 0-400 GeV in 40 bins, as in the attachment); a variable needs an entry here
DEFAULT_BINS = {
    "pt": np.linspace(0, 400, 41),
    "eta": np.linspace(-2.5, 2.5, 26),
    "phi": np.linspace(-np.pi, np.pi, 37),
}


def bins_for(bins, spec):
    """the bin edges of a spec: ``bins`` is one array for all variables, or a dict {variable: edges}"""
    edges = bins[spec.var] if isinstance(bins, dict) else bins
    return np.asarray(edges, dtype=np.float64)


def _ids(matched):
    return ak.values_astype(ak.drop_none(matched), np.int64)


def _present(matched, tid):
    return np.asarray(ak.any(_ids(matched) == tid, axis=1))


def primary_trigger(matched):
    """the highest-priority trigger that fired in each event (0: none), by PRIORITY"""
    ids = _ids(matched)
    primary = np.zeros(len(ids), dtype=np.int64)
    for tid in reversed(PRIORITY):  # later assignment in reversed order -> highest priority wins
        primary = np.where(_present(ids, tid), tid, primary)
    return primary


def multi_trigger_sf(events, primary):
    """
    Trigger SF of the primary trigger of each event (1.0 for events without a trigger). It is used for the events that
    pass more than one trigger, where the SF of a single trigger does not apply.
    """
    sf = np.ones(len(events), dtype=np.float64)
    for tid in TRIGGER_IDS:
        sf = np.where(primary == tid, trigger_sf(events, tid), sf)
    return sf


def categories(matched):
    """
    Trigger categories from ``matched_trigger_ids`` as boolean arrays.
    ``only_<id>`` and ``primary_<id>`` are exclusive per event (each event is in at most one of each kind).
    """
    ids = _ids(matched)
    n_hits = np.asarray(ak.num(ids, axis=1))
    cats = {"all": np.ones(len(ids), dtype=bool), "no_trigger": n_hits == 0}
    for tid in TRIGGER_IDS:
        cats[f"only_{tid}"] = _present(ids, tid) & (n_hits == 1)
    primary = primary_trigger(matched)
    for tid in TRIGGER_IDS:
        cats[f"primary_{tid}"] = primary == tid
    group1 = [t for t, g in GROUP_OF_ID.items() if g == 1]
    n_group1 = sum(np.asarray(ak.sum(ids == t, axis=1)) for t in group1)
    cats["group1_ge2"] = n_group1 >= 2
    cats["ntrig_ge2"] = n_hits >= 2  # events that pass more than one trigger (any group)
    return cats


def category_names():
    """the trigger categories, in the order of ``categories``"""
    return ["all", "no_trigger", *[f"only_{t}" for t in TRIGGER_IDS], *[f"primary_{t}" for t in TRIGGER_IDS],
            "group1_ge2", "ntrig_ge2"]


def efficiency(events, tid, weight=None, mask=None):
    """
    Efficiency of trigger ``tid``: numerator = fired (``matched_trigger_ids``), denominator = offline selection of
    the trigger (``base_pass_<id>``), which does not require the trigger to fire.
    """
    n = len(events)
    base = np.asarray(events[f"base_pass_{tid}"], dtype=bool)
    fired = _present(events.matched_trigger_ids, tid)
    w = np.ones(n) if weight is None else np.asarray(weight, dtype=np.float64)
    sel = np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    den = base & sel
    num = den & fired
    d = float(np.sum(w[den]))
    nu = float(np.sum(w[num]))
    return {"eff": nu / d if d > 0 else np.nan, "num": nu, "den": d, "n_num": int(num.sum()), "n_den": int(den.sum())}


def trigger_sf(events, tid):
    """
    Per-event scale factor of trigger ``tid``: product over the legs of data/MC for the ``*_data`` / ``*_mc`` pairs,
    times the direct scale factors (``*_sf``, ``vbf_sf``, ``tau_vbf_sf``). Sources are 1.0 where the trigger is not
    active in the channel, so the product over all channels needs no mask. The outside-region flag is not a factor.
    """
    sf = np.ones(len(events), dtype=np.float64)
    prefix = f"trigsrc_{tid}_"
    for name in events.fields:
        if not name.startswith(prefix) or name.endswith("_mc") or name.endswith("outside_corr_region"):
            continue
        if name.endswith("_data"):
            mc_name = name[: -len("_data")] + "_mc"
            sf = sf * np.asarray(events[name], dtype=np.float64) / np.asarray(events[mc_name], dtype=np.float64)
        else:
            sf = sf * np.asarray(events[name], dtype=np.float64)
    return sf


def safe_ratio(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def object_variable(events, obj="Jet", var="pt", i=0, order="pt"):
    """
    ``<obj>.<var>`` of the object with index ``i`` in each event (0 = leading), as float64, and a mask of the events
    that have such an object. ``order="pt"`` ranks the objects of each event by pT (descending) first;
    ``order="collection"`` keeps the collection order, i.e. the selection order (pT for electrons, muons and jets,
    the tau sorting key for taus). Events without it get NaN, and the mask, not the value, keeps them out of the
    histograms.
    """
    objs = events[obj]
    if order == "pt":
        objs = objs[ak.argsort(objs.pt, axis=1, ascending=False)]
    obj_i = ak.pad_none(objs[var], i + 1, axis=1)[:, i]
    valid = np.asarray(~ak.is_none(obj_i))
    return np.asarray(ak.fill_none(obj_i, np.nan), dtype=np.float64), valid


def mc_histograms(mc, bins, mc_weight, mask_mc=None, tid=None, obj="Jet", var="pt", i=0, order="pt"):
    """
    Counts per bin of ``<obj>.<var>[i]`` for MC: without trigger SF, and (if ``tid`` is given) with the SF of trigger
    ``tid``. Events without an object with index ``i`` are not counted.
    """
    x_m, valid_m = object_variable(mc, obj, var, i, order)
    w = np.asarray(mc[mc_weight], dtype=np.float64)
    mm = np.ones(len(x_m), dtype=bool) if mask_mc is None else np.asarray(mask_mc, dtype=bool)
    mm = mm & valid_m
    h_0, _ = np.histogram(x_m[mm], bins=bins, weights=w[mm])
    out = {"mc": h_0}
    if tid is not None:
        sf = trigger_sf(mc, tid)
        h_1, _ = np.histogram(x_m[mm], bins=bins, weights=(w * sf)[mm])
        out["mc_sf"] = h_1
    return out


def histograms(data, mc, bins, mc_weight, tid=None, mask_data=None, mask_mc=None, obj="Jet", var="pt", i=0,
               order="pt"):
    """
    Counts per bin of ``<obj>.<var>[i]``: data, and the MC of ``mc_histograms``. Events without an object with index
    ``i`` are not counted.
    """
    x_d, valid_d = object_variable(data, obj, var, i, order)
    md = np.ones(len(x_d), dtype=bool) if mask_data is None else np.asarray(mask_data, dtype=bool)
    md = md & valid_d
    h_d, _ = np.histogram(x_d[md], bins=bins)
    out = {"data": h_d}
    out.update(mc_histograms(mc, bins, mc_weight, mask_mc, tid, obj, var, i, order))
    return out


def abcd_regions(events):
    """
    ABCD regions as in hbt/categorization/default.py: the lepton charge (``leptons_os``: 1 = os, 0 = ss) and the
    number of isolated taus (``num_taus_iso``: iso needs 2 for tautau and 1 otherwise, noniso needs 1 for tautau and
    0 otherwise).
    """
    tautau = np.asarray(events.channel_id) == CHANNEL_IDS["tautau"]
    charge = np.asarray(events.leptons_os)
    n_taus = np.asarray(events.num_taus_iso)
    iso = n_taus >= np.where(tautau, 2, 1)
    noniso = n_taus == np.where(tautau, 1, 0)
    os_, ss = charge == 1, charge == 0
    return {"os_iso": os_ & iso, "os_noniso": os_ & noniso, "ss_iso": ss & iso, "ss_noniso": ss & noniso}


def abcd_qcd(b_data, b_mc, c_data, c_mc, d_data, d_mc):
    """
    QCD counts in the signal region (os, iso) by the ABCD method with the defaults of hbt/hist_hooks/qcd.py: the shape
    is data - MC of region B (os, noniso) per bin, the normalisation is the transfer factor C_int / D_int of the
    integrals of data - MC over region C (ss, iso) and region D (ss, noniso). A non-positive integral gives zero, and
    bins that end up <= 0 are set to zero (residual zero fill of the hook).
    """
    b_data = np.asarray(b_data, dtype=np.float64)
    c_int = float(np.sum(np.asarray(c_data, dtype=np.float64) - np.asarray(c_mc, dtype=np.float64)))
    d_int = float(np.sum(np.asarray(d_data, dtype=np.float64) - np.asarray(d_mc, dtype=np.float64)))
    if c_int <= 0 or d_int <= 0:
        return np.zeros(len(b_data))
    qcd = (b_data - np.asarray(b_mc, dtype=np.float64)) * (c_int / d_int)
    return np.where(qcd > 0, qcd, 0.0)


def _sf_tid(name):
    """the trigger whose SF is applied to a category: only the ``only_<id>`` categories"""
    return int(name.split("_")[1]) if name.startswith("only_") else None


SF_CATEGORIES = ("ntrig_ge2",)  # categories with a trigger SF besides only_<id>: the SF of the primary trigger


def has_sf(name):
    return _sf_tid(name) is not None or name in SF_CATEGORIES


def _counts(x, mask, bins):
    return np.histogram(x[mask], bins=bins)[0].astype(np.float64)


def _add_mc(store, x, valid, mask, w, sf, bins, w_tw=None):
    """adds the weighted counts of the masked events with a valid object to ``store["mc"]``, with the trigger SF
    ``sf`` (if given) to ``store["mc_sf"]``, and with the event weight ``w_tw`` (the trigger_weight variant, if given)
    to ``store["mc_tw"]``"""
    nb = len(bins) - 1
    m = mask & valid
    store["mc"] = store.get("mc", np.zeros(nb)) + np.histogram(x[m], bins=bins, weights=w[m])[0]
    if sf is not None:
        store["mc_sf"] = store.get("mc_sf", np.zeros(nb)) + np.histogram(x[m], bins=bins, weights=(w * sf)[m])[0]
    if w_tw is not None:
        store["mc_tw"] = store.get("mc_tw", np.zeros(nb)) + np.histogram(x[m], bins=bins, weights=w_tw[m])[0]


def channel_mask(events, channel):
    """the events of a channel (all events for None)"""
    if channel is None:
        return np.ones(len(events), dtype=bool)
    return np.asarray(events.channel_id) == CHANNEL_IDS[channel]


def spec_label(spec):
    return f"{spec.obj}_{spec.var}{spec.index}"


def spec_path(spec, outdir):
    """the plots of a spec go to <outdir>/<channel>/<obj>_<var><index>/ (no channel: <outdir>)"""
    if spec.channel is None:
        return outdir
    return os.path.join(outdir, spec.channel, spec_label(spec))


def study_specs(channels, variables):
    """every object of the channels (see OBJECTS_OF_CHANNEL) for every variable"""
    specs = []
    for ch in channels:
        for obj, index in OBJECTS_OF_CHANNEL[ch]:
            for var in variables:
                specs.append(Spec(ch, obj, var, index))
    return specs


def _new_acc(names, nb, with_tw=False):
    cat = {}
    for name in names:
        total = {"mc": np.zeros(nb)}
        if has_sf(name):
            total["mc_sf"] = np.zeros(nb)
        if with_tw:
            total["mc_tw"] = np.zeros(nb)
        cat[name] = {"w": 0.0, "w_tw": 0.0, "total": total, "procs": {}, "regions": {}}
    return {"cat": cat}


def new_accumulator(specs, bins, with_tw=False):
    return {"spec": {spec: _new_acc(category_names(), len(bins_for(bins, spec)) - 1, with_tw) for spec in specs},
            "eff": {}}


def accumulate(acc, mc, specs, bins, mc_weight, order="pt", region="os_iso", trigger_weight_column=None):
    """
    adds one MC dataset to the accumulators of all specs (one pass over the events; the histograms and sums are
    additive). The plotted MC of a spec is its channel, the trigger category and ``region`` (all events for None); the
    ABCD control regions are always filled per channel and category. Categories, regions, trigger SFs and the object
    variables are computed once per dataset.
    """
    cats = categories(mc.matched_trigger_ids)
    sf_multi = multi_trigger_sf(mc, primary_trigger(mc.matched_trigger_ids))
    w = np.asarray(mc[mc_weight], dtype=np.float64)
    # the second variant: the event weight times the trigger_weight column (None: not filled)
    w_tw = None if trigger_weight_column is None else w * np.asarray(mc[trigger_weight_column], dtype=np.float64)
    pid = np.asarray(mc.process_id) if "process_id" in mc.fields else np.full(len(mc), -1)
    regions = abcd_regions(mc)
    sf_cache, obj_cache, ch_masks = {}, {}, {}
    for spec in specs:
        if spec.channel not in ch_masks:
            ch_masks[spec.channel] = channel_mask(mc, spec.channel)
        ch_mask = ch_masks[spec.channel]
        okey = (spec.obj, spec.var, spec.index)
        if okey not in obj_cache:
            obj_cache[okey] = object_variable(mc, spec.obj, spec.var, spec.index, order)
        x, valid = obj_cache[okey]
        spec_bins = bins_for(bins, spec)
        sel = ch_mask if region is None else ch_mask & regions[region]
        for name, a in acc["spec"][spec]["cat"].items():
            tid = _sf_tid(name)
            sf = None
            if tid is not None:
                if tid not in sf_cache:
                    sf_cache[tid] = trigger_sf(mc, tid)
                sf = sf_cache[tid]
            elif name in SF_CATEGORIES:
                sf = sf_multi
            m_cat = cats[name] & ch_mask
            m = cats[name] & sel
            a["w"] += float(w[m].sum())
            if w_tw is not None:
                a["w_tw"] += float(w_tw[m].sum())
            _add_mc(a["total"], x, valid, m, w, sf, spec_bins, w_tw)
            for p in np.unique(pid[m_cat]):
                p = int(p)
                p_mask = pid == p
                m_p = m & p_mask
                store = a["procs"].setdefault(p, {"n_mc_events": 0})
                store["n_mc_events"] += int((m_p & valid).sum())
                _add_mc(store, x, valid, m_p, w, sf, spec_bins, w_tw)
                for r in ABCD_REGIONS:
                    _add_mc(a["regions"].setdefault((r, p), {}), x, valid, m_cat & p_mask & regions[r], w, sf,
                            spec_bins, w_tw)
    # the efficiencies do not depend on the variable: once per channel
    for ch, ch_mask in ch_masks.items():
        for tid in TRIGGER_IDS:
            for name in ("all", f"only_{tid}", f"primary_{tid}"):
                e = efficiency(mc, tid, weight=w, mask=cats[name] & ch_mask)
                s_eff = acc["eff"].setdefault((ch, tid, name), {"num": 0.0, "den": 0.0, "n_num": 0, "n_den": 0})
                for key in s_eff:
                    s_eff[key] += e[key]


def _mc_region(regions, region, key, exclude, nb):
    """MC of an ABCD region summed over the processes that are subtracted (all but ``exclude``)"""
    out = np.zeros(nb)
    for (r, p), store in regions.items():
        if r == region and p not in exclude and key in store:
            out = out + store[key]
    return out


def _check_add_up(name, parts, total, keys, nb):
    for key in keys:
        summed = np.zeros(nb)
        for part in parts.values():
            summed = summed + part[key]
        if not np.allclose(summed, total[key], rtol=1e-12, atol=1e-12):
            raise AssertionError(f"{name}: the per-process {key} histograms do not add up to the total {key}")


def plot_ratio(bins, hists, title, path, xlabel, tw_label="MC + QCD, trigger_weight"):
    """
    Data as points, the MC as stacked histograms: all MC processes merged into one histogram (``mc``, without the
    trigger_weight column) and the QCD estimate (``qcd``, if given) stacked on top. The second variant (``mc_tw``,
    ``qcd_tw``: the MC and the ABCD reference weighted with trigger_weight) is drawn as an outline. The ratio panel is
    data / (MC + QCD), and data / (MC + QCD, trigger_weight) for the outline.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.asarray(bins, dtype=np.float64)
    centers = 0.5 * (bins[1:] + bins[:-1])
    mc = np.asarray(hists["mc"], dtype=np.float64)
    qcd = np.asarray(hists.get("qcd", np.zeros(len(mc))), dtype=np.float64)
    total = mc + qcd
    import mplhep as hep
    hep.set_style(hep.style.ROOT)
    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.06, top=0.92, bottom=0.1, right=0.97)
    ax.errorbar(centers, hists["data"], yerr=np.sqrt(hists["data"]), fmt="ko", label="data")
    ax.stairs(mc, bins, fill=True, color="C0", alpha=0.6, label="MC, no trigger_weight")
    if "qcd" in hists:
        ax.stairs(total, bins, baseline=mc, fill=True, color="C2", alpha=0.6, label="QCD (ABCD)")
    rax.stairs(safe_ratio(hists["data"], total), bins, color="C0")
    if "mc_tw" in hists:
        mc_tw = np.asarray(hists["mc_tw"], dtype=np.float64)
        total_tw = mc_tw + np.asarray(hists.get("qcd_tw", np.zeros(len(mc))), dtype=np.float64)
        ax.stairs(total_tw, bins, color="C1", linewidth=2, label=tw_label)
        rax.stairs(safe_ratio(hists["data"], total_tw), bins, color="C1")
    rax.axhline(1.0, color="gray", linestyle="--")
    rax.set_ylim(0.5, 1.5)  # zoom on the data/MC deviations, values outside are clipped
    ax.set_title(title)
    ax.set_ylabel("events")
    ax.legend()
    rax.set_ylabel("data / MC")
    rax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def finish(acc, spec, data, bins, outdir, order, process_names, qcd, exclude, region):
    """
    The summary rows, the process rows and the plots of one spec. ``data`` is the data of the spec's channel and
    ``acc`` the MC accumulator of the spec (see ``accumulate``). The plotted events are the trigger category in
    ``region``; the QCD estimate uses the four ABCD regions of each category.
    """
    os.makedirs(outdir, exist_ok=True)
    nb = len(bins) - 1
    cats_d = categories(data.matched_trigger_ids)
    x_d, valid_d = object_variable(data, spec.obj, spec.var, spec.index, order)
    regions_d = abcd_regions(data)
    sel_d = np.ones(len(x_d), dtype=bool) if region is None else regions_d[region]
    rows, proc_rows = [], []
    for name, a in acc["cat"].items():
        md_cat = cats_d[name]
        md = md_cat & sel_d  # the plotted data: the trigger category in the region
        total = a["total"]
        n_d = int(md.sum())
        h = {"data": _counts(x_d, md & valid_d, bins), "mc": total["mc"]}
        if "mc_sf" in total:
            h["mc_sf"] = total["mc_sf"]
        if "mc_tw" in total:
            h["mc_tw"] = total["mc_tw"]

        # MC per process: the process histograms (without and with trigger SF) must add up to the event total
        mc_proc, n_events = {}, {}
        for p, store in sorted(a["procs"].items()):
            label = (process_names or {}).get(p, f"process_{p}")
            mc_proc[label] = {key: store[key] for key in ("mc", "mc_sf", "mc_tw") if key in store}
            n_events[label] = store["n_mc_events"]
        keys = [key for key in h if key != "data"]
        _check_add_up(name, mc_proc, h, keys, nb)

        # QCD: ABCD per trigger category, data - MC in the control regions, the same MC weighting as the total
        if qcd:
            regs = a["regions"]
            b_d, c_d, d_d = (_counts(x_d, md_cat & valid_d & regions_d[r], bins)
                             for r in ("os_noniso", "ss_iso", "ss_noniso"))

            def qcd_for(key):
                return abcd_qcd(b_d, _mc_region(regs, "os_noniso", key, exclude, nb),
                                c_d, _mc_region(regs, "ss_iso", key, exclude, nb),
                                d_d, _mc_region(regs, "ss_noniso", key, exclude, nb))

            qcd_h = {key: qcd_for(key) for key in keys}
            mc_proc[QCD_LABEL] = qcd_h
            n_events[QCD_LABEL] = np.nan
            for key in keys:
                h[key] = h[key] + qcd_h[key]
            _check_add_up(name, mc_proc, h, keys, nb)
        mc_tot = float(h["mc"].sum())

        r0 = float(h["data"].sum() / h["mc"].sum()) if h["mc"].sum() > 0 else np.nan
        row = {"category": name, "n_data": n_d, "mc_weighted": a["w"], "data_over_mc": r0,
               "qcd_hist": float(mc_proc[QCD_LABEL]["mc"].sum()) if qcd else np.nan}
        if "mc_sf" in h:
            row["data_over_mc_sf"] = float(h["data"].sum() / h["mc_sf"].sum()) if h["mc_sf"].sum() > 0 else np.nan
        if "mc_tw" in h:
            row["mc_tw_weighted"] = a["w_tw"]
            row["data_over_mc_tw"] = float(h["data"].sum() / h["mc_tw"].sum()) if h["mc_tw"].sum() > 0 else np.nan
            row["qcd_tw_hist"] = float(mc_proc[QCD_LABEL]["mc_tw"].sum()) if qcd else np.nan
        rows.append(row)

        if mc_proc:
            for label, part in mc_proc.items():
                part_mc = float(part["mc"].sum())
                proc_rows.append({"category": name, "process": label, "n_mc_events": n_events[label],
                                  "mc_hist": part_mc,
                                  "mc_sf_hist": float(part["mc_sf"].sum()) if "mc_sf" in part else np.nan,
                                  "mc_tw_hist": float(part["mc_tw"].sum()) if "mc_tw" in part else np.nan,
                                  "fraction": part_mc / mc_tot if mc_tot > 0 else np.nan})
            proc_rows.append({"category": name, "process": "total",
                              "n_mc_events": sum(v for lab, v in n_events.items() if lab != QCD_LABEL),
                              "mc_hist": mc_tot,
                              "mc_sf_hist": float(h["mc_sf"].sum()) if "mc_sf" in h else np.nan,
                              "mc_tw_hist": float(h["mc_tw"].sum()) if "mc_tw" in h else np.nan, "fraction": 1.0})

        if h["mc"].sum() > 0:  # one plot per trigger category (no_trigger included)
            plot_h = {"data": h["data"], "mc": total["mc"]}  # all MC processes merged, without QCD
            if qcd:
                plot_h["qcd"] = mc_proc[QCD_LABEL]["mc"]
            if "mc_tw" in total:
                plot_h["mc_tw"] = total["mc_tw"]
                if qcd:
                    plot_h["qcd_tw"] = mc_proc[QCD_LABEL]["mc_tw"]
            title = f"{name}, {region or 'all'}, {spec.channel or 'no channel cut'} (n_data={n_d})"
            plot_ratio(bins, plot_h, title, os.path.join(outdir, f"{name}.png"),
                       f"{spec.obj}.{spec.var}[{spec.index}]")
    return rows, proc_rows


def efficiency_rows(acc, data_by_channel, specs):
    """the trigger efficiencies in data and MC of every channel of the specs"""
    rows = []
    for ch in dict.fromkeys(spec.channel for spec in specs):
        data = data_by_channel[ch]
        cats_d = categories(data.matched_trigger_ids)
        for tid in TRIGGER_IDS:
            for name in ["all", f"only_{tid}", f"primary_{tid}"]:
                e_mc = acc["eff"][(ch, tid, name)]
                eff_mc = e_mc["num"] / e_mc["den"] if e_mc["den"] > 0 else np.nan
                e_d = efficiency(data, tid, mask=cats_d[name])
                rows.append({"channel": ch, "trigger": tid, "category": name, "eff_data": e_d["eff"],
                             "eff_mc": eff_mc, "n_den_data": e_d["n_den"], "n_den_mc": e_mc["n_den"]})
    return rows


STEP_LABELS = ["nominal", *[f"+{tid}" for tid in TRIGGER_IDS]]  # cumulative steps of the one-by-one SF study
FINAL_STEP = STEP_LABELS[-1]  # all trigger SFs applied step by step
COLUMN_LABEL = "column"  # nominal times the trigger_weight column, compared with FINAL_STEP
ALL_LABELS = [*STEP_LABELS, COLUMN_LABEL]
WEIGHT_BINS = np.linspace(0.8, 1.2, 40)  # bins of the weight histograms

# np.linspace(0, 400, 41)   
def step_weights(events, weight):
    """
    The nominal weight, then the nominal weight times the trigger SF of each trigger in turn (cumulative, in the order of
    TRIGGER_IDS). The SF of a trigger is 1.0 for the events where it did not fire, so every step is defined for all events.
    """
    steps = [("nominal", weight)]
    cum = weight
    for tid in TRIGGER_IDS:
        cum = cum * trigger_sf(events, tid)
        steps.append((f"+{tid}", cum))
    return steps


def new_step_accumulator(bins):
    nb = len(bins) - 1
    return {name: {label: np.zeros(nb) for label in ALL_LABELS} for name in category_names()}


def accumulate_steps(store, mc, spec, bins, mc_weight, column, order="pt", region="os_iso"):
    """
    One MC dataset: the histogram of ``spec`` after each cumulative step, and with the ``column`` weight (the production
    trigger weight) in place of the steps, per trigger category (region as in accumulate).
    """
    cats = categories(mc.matched_trigger_ids)
    w = np.asarray(mc[mc_weight], dtype=np.float64)
    x, valid = object_variable(mc, spec.obj, spec.var, spec.index, order)
    regions = abcd_regions(mc)
    ch = channel_mask(mc, spec.channel)
    sel = ch if region is None else ch & regions[region]
    steps = step_weights(mc, w) + [(COLUMN_LABEL, w * np.asarray(mc[column], dtype=np.float64))]
    for name in category_names():
        m = cats[name] & sel & valid
        for label, sw in steps:
            store[name][label] += np.histogram(x[m], bins=bins, weights=sw[m])[0]


def plot_steps(bins, data_hist, steps, title, path, xlabel):
    """
    The MC after each cumulative SF step (lines) against the nominal MC (filled) and the data (points); the ratio panel
    shows each step divided by the nominal.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.asarray(bins, dtype=np.float64)
    centers = 0.5 * (bins[1:] + bins[:-1])
    nominal = np.asarray(steps["nominal"], dtype=np.float64)
    fig, (ax, rax, cax) = plt.subplots(3, 1, sharex=True, figsize=(7, 9), gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    ax.errorbar(centers, data_hist, yerr=np.sqrt(data_hist), fmt="ko", label="data")
    ax.stairs(nominal, bins, fill=True, color="C0", alpha=0.3, label="MC nominal")
    labels = [label for label in steps if label not in ("nominal", COLUMN_LABEL)]
    cmap = plt.get_cmap("viridis")
    for i, label in enumerate(labels):
        h = np.asarray(steps[label], dtype=np.float64)
        color = cmap(i / max(len(labels) - 1, 1))
        ax.stairs(h, bins, color=color, linewidth=1.2, label=f"+ SF {label[1:]}")
        rax.stairs(safe_ratio(h, nominal), bins, color=color, linewidth=1.2)
    rax.axhline(1.0, color="gray", linestyle="--")
    rax.set_ylim(0.5, 1.5)
    if COLUMN_LABEL in steps:
        column = np.asarray(steps[COLUMN_LABEL], dtype=np.float64)
        ax.stairs(column, bins, color="k", linestyle="--", linewidth=1.2, label="trigger_weight column")
        rax.stairs(safe_ratio(column, nominal), bins, color="k", linestyle="--", linewidth=1.2)
        cax.stairs(safe_ratio(column, np.asarray(steps[FINAL_STEP], dtype=np.float64)), bins, color="k", linewidth=1.2)
        cax.axhline(1.0, color="gray", linestyle="--")
        cax.set_ylim(0.5, 1.5)
        cax.set_ylabel("column / steps")
    ax.set_title(title)
    ax.set_ylabel("events")
    ax.legend(fontsize=6, ncol=2)
    rax.set_ylabel("step / nominal")
    rax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compare_column_to_steps(cat):
    """
    The MC of one trigger category with the trigger_weight column against the MC with all trigger SFs applied step by
    step (the last cumulative step): the two totals, their ratio, and the largest absolute difference of a bin.
    """
    column = np.asarray(cat[COLUMN_LABEL], dtype=np.float64)
    steps = np.asarray(cat[FINAL_STEP], dtype=np.float64)
    total_col, total_steps = float(column.sum()), float(steps.sum())
    return {"mc_column": total_col, "mc_steps": total_steps,
            "column_over_steps": total_col / total_steps if total_steps > 0 else np.nan,
            "max_abs_bin_diff": float(np.max(np.abs(column - steps))) if len(column) else 0.0}


def write_steps(store, spec, data, bins, outdir, order, region):
    """
    The one-by-one trigger SF study of ``spec``: per trigger category the MC after each cumulative step, the data, and
    the plots. Writes ``steps.csv`` and one plot per category into ``outdir``.
    """
    os.makedirs(outdir, exist_ok=True)
    cats_d = categories(data.matched_trigger_ids)
    x_d, valid_d = object_variable(data, spec.obj, spec.var, spec.index, order)
    regions_d = abcd_regions(data)
    sel_d = np.ones(len(x_d), dtype=bool) if region is None else regions_d[region]
    rows, check_rows = [], []
    for name in category_names():
        h_d = np.histogram(x_d[cats_d[name] & sel_d & valid_d], bins=bins)[0].astype(np.float64)
        nominal = store[name]["nominal"]
        for label in ALL_LABELS:
            h = store[name][label]
            rows.append({"channel": spec.channel, "obj": spec.obj, "var": spec.var, "index": spec.index,
                         "category": name, "step": label, "n_data": int(h_d.sum()), "mc_weighted": float(h.sum()),
                         "ratio_to_nominal": float(h.sum() / nominal.sum()) if nominal.sum() > 0 else np.nan,
                         "data_over_mc": float(h_d.sum() / h.sum()) if h.sum() > 0 else np.nan})
        check_rows.append({"category": name, **compare_column_to_steps(store[name])})
        if nominal.sum() > 0 or h_d.sum() > 0:
            plot_steps(bins, h_d, {label: store[name][label] for label in ALL_LABELS},
                       f"{name}, {region or 'all'}, {spec.channel}: {spec_label(spec)}",
                       os.path.join(outdir, f"{name}.png"), f"{spec.obj}.{spec.var}[{spec.index}]")
    write_csv(os.path.join(outdir, "steps.csv"), rows)
    write_csv(os.path.join(outdir, "trigger_weight_check.csv"), check_rows)
    shown = [r for r in check_rows if r["mc_steps"] > 0 or r["mc_column"] > 0]
    print(f"\n=== trigger_weight column vs step-by-step SFs: {spec.channel} {spec_label(spec)} ===")
    _print_table(shown, ["category", "mc_column", "mc_steps", "column_over_steps", "max_abs_bin_diff"])
    return rows


def accumulate_weight_hists(store, dataset, mc, weight_columns, nominal):
    """one MC dataset: histograms of the nominal weight, of each weight column and of each trigger SF (all events)"""
    values = {"nominal": np.asarray(nominal, dtype=np.float64)}
    for column in weight_columns:
        values[column] = np.asarray(mc[column], dtype=np.float64)
    for tid in TRIGGER_IDS:
        if any(field.startswith(f"trigsrc_{tid}_") for field in mc.fields):
            values[f"sf_{tid}"] = trigger_sf(mc, tid)
    for name, v in values.items():
        s = store.setdefault(name, {}).setdefault(
            dataset, {"hist": np.zeros(len(WEIGHT_BINS) - 1), "n": 0, "sum": 0.0, "sum2": 0.0})
        s["hist"] += np.histogram(v, bins=WEIGHT_BINS)[0]
        s["n"] += len(v)
        s["sum"] += float(v.sum())
        s["sum2"] += float((v * v).sum())


def write_weight_hists(store, outdir):
    """one log-log histogram per weight (one line per dataset) and the mean and std per dataset in weights.csv"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = os.path.join(outdir, "weights")
    os.makedirs(folder, exist_ok=True)
    rows = []
    for name, per_dataset in store.items():
        if 'sf_' not in name:continue
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for dataset, s in sorted(per_dataset.items()):
            ax.stairs(s["hist"], WEIGHT_BINS, label=dataset)
            mean = s["sum"] / s["n"]
            std = float(np.sqrt(max(s["sum2"] / s["n"] - mean * mean, 0.0)))
            rows.append({"weight": name, "dataset": dataset, "n_events": s["n"], "mean": mean, "std": std})
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("events")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"{name}.png"))
        plt.close(fig)
    write_csv(os.path.join(folder, "weights.csv"), rows)


def study(data_by_channel, mc_datasets, specs, bins, mc_weight, outdir, order="pt", process_names=None, qcd=True,
          qcd_exclude=(), region="os_iso", step_specs=(), trigger_weight_column=None):
    """
    Reads the MC datasets once for all ``specs`` (one pass fills every channel, object and variable), then the summary
    rows, the process rows and the plots of each spec, and the efficiencies of each channel. ``data_by_channel`` maps
    each channel (None: no channel cut) to its data. Returns the summary rows, the efficiency rows and the process rows;
    each row carries the channel, object, variable and index.
    """
    os.makedirs(outdir, exist_ok=True)
    if step_specs and trigger_weight_column is None:
        raise ValueError("the step study needs the trigger weight column (trigger_weight_column)")
    acc = new_accumulator(specs, bins, with_tw=trigger_weight_column is not None)
    steps_acc = {spec: new_step_accumulator(bins_for(bins, spec)) for spec in step_specs}
    for mc in mc_datasets:
        accumulate(acc, mc, specs, bins, mc_weight, order=order, region=region,
                   trigger_weight_column=trigger_weight_column)
        for spec in step_specs:
            accumulate_steps(steps_acc[spec], mc, spec, bins_for(bins, spec), mc_weight, trigger_weight_column,
                             order, region)
    exclude = {int(p) for p in qcd_exclude}
    summary, procs = [], []
    for spec in specs:
        rows, proc_rows = finish(acc["spec"][spec], spec, data_by_channel[spec.channel], bins_for(bins, spec),
                                 spec_path(spec, outdir), order, process_names, qcd, exclude, region)
        for row in rows + proc_rows:
            row.update(channel=spec.channel, obj=spec.obj, var=spec.var, index=spec.index)
        summary += rows
        procs += proc_rows
    for spec in step_specs:
        write_steps(steps_acc[spec], spec, data_by_channel[spec.channel], bins_for(bins, spec),
                    os.path.join(outdir, "steps", f"{spec.channel}_{spec_label(spec)}"), order, region)
    return summary, efficiency_rows(acc, data_by_channel, specs), procs


def run_study(data, mc_datasets, bins, mc_weight, outdir, obj="Jet", var="pt", i=0, order="pt", process_names=None,
              qcd=True, qcd_exclude=(), region="os_iso", trigger_weight_column=None):
    """one object and variable without a channel cut, the plots go to ``outdir``; returns the rows of ``study``"""
    if isinstance(mc_datasets, ak.Array):
        mc_datasets = [mc_datasets]
    return study({None: data}, mc_datasets, [Spec(None, obj, var, i)], bins, mc_weight, outdir, order=order,
                 process_names=process_names, qcd=qcd, qcd_exclude=qcd_exclude, region=region,
                 trigger_weight_column=trigger_weight_column)


def load(reduced_glob, produced_glob=None, reduced_columns=None, produced_columns=None):
    """reads the reduced output and adds the trigger sources of the production output, joined on ``event``; with
    ``*_columns`` only these columns are read (memory), default all"""

    red = ak.concatenate([ak.from_parquet(p, columns=reduced_columns) for p in sorted(glob.glob(reduced_glob))],
                         axis=0)
    if produced_glob is None:
        return red
    prod = ak.concatenate([ak.from_parquet(p, columns=produced_columns) for p in sorted(glob.glob(produced_glob))],
                          axis=0)
    if len(prod) != len(red):
        raise ValueError("reduced and produced outputs do not contain the same events in the same order")
    for field in prod.fields:
        # trigger SF sources, and produced weights (e.g. trigger_weight) that are not in the reduced output
        # if field.startswith("trigsrc_") or (field.endswith("_weight") and field not in red.fields):
        red = ak.with_field(red, prod[field], field)
    return red


def present_columns(pattern, wanted=(), prefixes=()):
    """
    The columns of the parquet files matched by ``pattern`` that are in ``wanted`` (a nested ``Obj.field`` is kept when
    ``Obj`` exists) or start with one of ``prefixes``. Only these are read by ``load``, which keeps the memory small.
    """
    import pyarrow.dataset as pds

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"no parquet files match {pattern}")
    names = set(pds.dataset(paths[0], format="parquet").schema.names)
    keep = [n for n in names if n in wanted or n.startswith(tuple(prefixes))]
    keep += [c for c in wanted if "." in c and c.split(".")[0] in names]
    return sorted(set(keep))


def mc_per_dataset(pattern, wanted, prefixes=()):
    """
    Yields ``(dataset, events)`` one dataset at a time from the united MC output (reduced and production columns in
    one file): the first ``*`` of the pattern is the dataset name. Only the ``wanted`` columns and the columns starting
    with one of ``prefixes`` are read.
    """
    prefix = pattern.partition("*")[0]
    datasets = sorted({path[len(prefix):].split("/")[0] for path in glob.glob(pattern)})
    if not datasets:
        raise ValueError(f"no MC output matches {pattern}")
    for dataset in datasets:
        dataset_glob = pattern.replace("*", dataset, 1)
        columns = present_columns(dataset_glob, wanted, prefixes)
        # without the trigger sources trigger_sf would silently return 1.0 for every event
        if prefixes and not any(c.startswith(tuple(prefixes)) for c in columns):
            raise ValueError(f"{dataset_glob}: no {'/'.join(prefixes)}* columns (trigger SF sources) in the united output")
        yield dataset, load(dataset_glob, reduced_columns=columns)


def _print_table(rows, keys):
    print("  ".join(f"{k:>18}" for k in keys))
    for row in rows:
        # not every row has every key (e.g. the SF ratio only exists for the only_<id> categories), print "-" then
        print("  ".join(
            f"{row[k]:>18.4g}" if isinstance(row.get(k), float) else f"{str(row.get(k, '-')):>18}" for k in keys
        ))


def write_csv(path, rows):
    """writes the rows (dicts) to csv, with the union of their keys as the columns"""
    if not rows:
        return
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def selftest():
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    matched = ak.Array([
        [205],  # e0: group 1 only, etau
        [205, 405],  # e1: two group-1 hits, ee channel for the 205 ee source
        [608],  # e2: group 5 only, no base pass for 205
        [602, 605],  # e3: groups 3 and 4, primary 602
        [],  # e4: nothing fired
        [701, 205, 608],  # e5: group-1 hits and group 5, primary 205
        [507],  # e6: tautau trigger 507
        [105],  # e7: single mu
    ])
    n = len(matched)
    base205 = ak.Array([True, True, True, False, True, True, False, False])
    e_data = ak.Array([0.9, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0])
    e_mc = ak.Array([0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0, 1.0])
    ee_sf = ak.Array([1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    tau_data = ak.Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.72, 1.0])
    tau_mc = ak.Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 1.0])
    # base_pass_<id> exists for every config trigger; all False except the ones set here
    base_cols = {f"base_pass_{tid}": ak.Array([False] * n) for tid in TRIGGER_IDS}
    base_cols["base_pass_205"] = base205
    base_cols["base_pass_507"] = ak.Array([False] * 6 + [True, False])
    # ABCD regions: channel etau (1), num_taus_iso >= 1 is iso, == 0 is noniso; leptons_os 1 = os, 0 = ss
    region_cols = {
        "channel_id": np.ones(n, dtype=np.int64),
        "leptons_os": np.array([1, 1, 0, 0, 1, 0, 1, 0]),
        "num_taus_iso": np.array([1, 0, 1, 0, 1, 1, 0, 0]),
    }
    var = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0])
    jet = ak.Array([[{"pt": float(v)}] for v in var])  # one jet per event, Jet.pt[:, 0] = var
    mc = ak.Array({
        "event": np.arange(n),
        "matched_trigger_ids": matched,
        **base_cols,
        **region_cols,
        "weight": ak.Array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        "Jet": jet,
        "process_id": np.array([1, 1, 2, 2, 1, 2, 1, 1]),
        "trigsrc_205_etau_e_data": e_data,
        "trigsrc_205_etau_e_mc": e_mc,
        "trigsrc_205_ee_e_sf": ee_sf,
        "trigsrc_507_tautau_tau_data": tau_data,
        "trigsrc_507_tautau_tau_mc": tau_mc,
    })
    data = ak.Array({"event": np.arange(n), "matched_trigger_ids": matched, **base_cols, **region_cols, "Jet": jet})

    cats = categories(matched)
    check(list(np.flatnonzero(cats["only_205"])) == [0], "only_205: only event e0")
    check(list(np.flatnonzero(cats["primary_205"])) == [0, 1, 5], "primary_205: e0, e1, e5 (group 1 priority)")
    check(list(np.flatnonzero(cats["primary_602"])) == [3], "primary_602: e3 (group 3 before group 4)")
    check(list(np.flatnonzero(cats["primary_608"])) == [2], "primary_608: e2 only")
    check(list(np.flatnonzero(cats["group1_ge2"])) == [1, 5], "group1_ge2: e1 and e5")
    check(list(np.flatnonzero(cats["no_trigger"])) == [4], "no_trigger: e4")
    check(list(cats) == category_names(), "the category names are the ones of categories()")
    check(list(bins_for({"pt": [0, 1], "eta": [0, 2, 3]}, Spec(None, "Jet", "eta", 0))) == [0.0, 2.0, 3.0],
          "bins_for picks the bin edges of the variable of a spec")
    check(list(np.flatnonzero(cats["ntrig_ge2"])) == [1, 3, 5],
          "ntrig_ge2: the events that pass more than one trigger (e1, e3, e5)")
    sf_multi = multi_trigger_sf(mc, primary_trigger(matched))
    check(np.allclose(sf_multi, [1.125, 1.5, 1.0, 1.0, 1.0, 1.125, 0.9, 1.0]),
          f"SF of the primary trigger per event {list(np.round(sf_multi, 4))} (expected 205 for e0, e1, e5; 507 for e6)")
    steps_m = step_weights(mc, mc["weight"])
    check(steps_m[0][0] == "nominal" and len(steps_m) == 1 + len(TRIGGER_IDS),
          "step study: the nominal weight, then one cumulative step per trigger")
    check(np.allclose(np.asarray(steps_m[-1][1]), [1.125, 1.5, 2.0, 1.0, 1.0, 1.125, 0.9, 1.0]),
          "step study: the last cumulative step is the nominal weight times all trigger SFs")
    bins_t = np.array([0.0, 20.0, 40.0, 80.0])
    prod_all = np.asarray(step_weights(mc, np.ones(n))[-1][1])
    store_ok = new_step_accumulator(bins_t)
    accumulate_steps(store_ok, ak.with_field(mc, prod_all, "trigger_weight"), Spec(None, "Jet", "pt", 0), bins_t,
                     "weight", "trigger_weight", region=None)
    cmp_ok = compare_column_to_steps(store_ok["all"])
    check(cmp_ok["max_abs_bin_diff"] < 1e-12 and abs(cmp_ok["column_over_steps"] - 1.0) < 1e-12,
          "check: a trigger_weight column equal to the step-by-step SFs gives identical MC histograms")
    store_bad = new_step_accumulator(bins_t)
    accumulate_steps(store_bad, ak.with_field(mc, np.ones(n), "trigger_weight"), Spec(None, "Jet", "pt", 0), bins_t,
                     "weight", "trigger_weight", region=None)
    cmp_bad = compare_column_to_steps(store_bad["all"])
    check(cmp_bad["max_abs_bin_diff"] > 0.1,
          f"check: a trigger_weight column of 1.0 differs from the step-by-step SFs (max bin diff {cmp_bad['max_abs_bin_diff']:.3f})")

    eff = efficiency(mc, 205, weight=mc["weight"])
    check(abs(eff["den"] - 6.0) < 1e-9 and abs(eff["num"] - 3.0) < 1e-9,
          f"efficiency denominator from base_pass (weighted den={eff['den']}, num={eff['num']}, expected 6 and 3)")
    check(abs(eff["eff"] - 0.5) < 1e-9, f"efficiency 205 = {eff['eff']} (expected 0.5)")
    eff_nounweighted = efficiency(mc, 205)
    check(eff_nounweighted["n_den"] == 5 and eff_nounweighted["n_num"] == 3,
          "unweighted counts: den 5 (e0, e1, e2, e4, e5), num 3 (e0, e1, e5)")

    sf205 = trigger_sf(mc, 205)
    check(abs(sf205[0] - 0.9 / 0.8) < 1e-9, f"SF 205 in etau event e0 = data/mc of the leg ({sf205[0]:.6f})")
    check(abs(sf205[1] - 1.5) < 1e-9, f"SF 205 in ee event e1 = ee direct SF ({sf205[1]:.6f})")
    check(abs(sf205[4] - 1.0) < 1e-9, "SF 205 is 1.0 where the trigger did not match (e4)")
    check(abs(sf205[2] - 1.0) < 1e-9, "SF 205 is 1.0 where the trigger did not match (e2, base pass only)")
    sf507 = trigger_sf(mc, 507)
    check(abs(sf507[6] - 0.72 / 0.8) < 1e-9, f"SF 507 tautau object level data/mc at e6 ({sf507[6]:.6f})")
    check(np.allclose(np.delete(sf507, 6), 1.0), "SF 507 is 1.0 for all other events")

    h = histograms(data, mc, bins=[0, 20, 40, 80], mc_weight="weight", tid=205,
                   mask_data=cats["only_205"], mask_mc=cats["only_205"], obj="Jet", var="pt", i=0)
    check(int(h["data"].sum()) == 1, "data histogram counts the only_205 event")
    check(abs(h["mc"].sum() - 1.0) < 1e-9 and abs(h["mc_sf"].sum() - 1.125) < 1e-9,
          "MC only_205 histogram without (1.0) and with the SF (1.125)")

    # the object index: leading and subleading jet; an event without a subleading jet is not counted
    two = ak.Array({"Jet": ak.Array([[{"pt": 30.0}, {"pt": 70.0}], [{"pt": 50.0}], []]), "weight": [1.0, 1.0, 1.0]})
    h_lead = histograms(two, two, [0, 40, 80], "weight", obj="Jet", var="pt", i=0)
    h_sub = histograms(two, two, [0, 40, 80], "weight", obj="Jet", var="pt", i=1)
    # default order="pt": event 0 has 70 and 30 GeV, event 1 has 50 GeV, event 2 has no jet
    check(list(h_lead["data"]) == [0, 2] and list(h_sub["data"]) == [1, 0],
          "pT-ordered leading jet [0, 2], subleading jet [1, 0]; the event without a subleading jet is not counted: "
          f"{list(h_lead['data'])}, {list(h_sub['data'])}")

    # ordering: a tau collection is sorted by a tau key, not by pT; order="pt" ranks by pT, "collection" keeps the key
    taus = ak.Array({"Tau": ak.Array([[{"pt": 30.0}, {"pt": 61.0}], [{"pt": 45.0}]]), "weight": [1.0, 1.0]})
    t_pt = object_variable(taus, "Tau", "pt", 0, order="pt")[0]
    t_col = object_variable(taus, "Tau", "pt", 0, order="collection")[0]
    check(list(t_pt) == [61.0, 45.0] and list(t_col) == [30.0, 45.0],
          f"leading tau by pT {list(t_pt)}, by collection order {list(t_col)} (expected [61, 45] and [30, 45])")

    # ABCD arithmetic: shape from B (data - MC per bin), transfer C_int / D_int = 40 / 20 = 2
    q = abcd_qcd([10, 20], [4, 5], [30, 30], [10, 10], [20, 20], [10, 10])
    check(np.allclose(q, [12, 30]), f"ABCD: shape from B times transfer C/D ({list(q)}, expected [12, 30])")
    q_neg = abcd_qcd([10], [4], [5], [9], [30], [10])  # C_int = -4 -> zero
    check(np.allclose(q_neg, [0]), "ABCD: a negative integral gives zero")
    q_res = abcd_qcd([3, 20], [5, 5], [30], [10], [30], [10])  # bin 0: data - MC = -2 -> zero, bin 1 kept
    check(np.allclose(q_res, [0, 15]), f"ABCD: bins <= 0 are set to zero ({list(q_res)}, expected [0, 15])")

    rows, eff_rows, proc_rows = run_study(data, mc, [0, 20, 40, 80], "weight", outdir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "trigger_study_selftest"), qcd=False, region=None)
    total_all = [r["mc_hist"] for r in proc_rows if r["category"] == "all" and r["process"] == "total"][0]
    sum_all = sum(r["mc_hist"] for r in proc_rows if r["category"] == "all" and r["process"] != "total")
    check(abs(total_all - sum_all) < 1e-9 and abs(total_all - 9.0) < 1e-9,
          f"MC per process adds up to the MC total ({sum_all} vs {total_all}, expected 9)")
    only = [r for r in proc_rows if r["category"] == "only_205" and r["process"] != "total"]
    check(len(only) == 1 and only[0]["process"] == "process_1" and only[0]["n_mc_events"] == 1,
          "only_205: its single MC event belongs to process_1")
    only_tot = [r for r in proc_rows if r["category"] == "only_205" and r["process"] == "total"][0]
    check(abs(only_tot["mc_sf_hist"] - 1.125) < 1e-9,
          f"only_205: MC with trigger SF per process adds up to the total ({only_tot['mc_sf_hist']}, expected 1.125)")
    check(any(r["category"] == "only_205" and "data_over_mc_sf" in r for r in rows),
          "run_study writes the only_205 row with and without SF")
    png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trigger_study_selftest", "only_205.png")
    check(os.path.exists(png) and os.path.getsize(png) > 0, "plot file written for only_205")

    # QCD added as process "qcd": the MC total is the os_iso event MC plus the QCD, and the per-process add-up holds
    qdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trigger_study_selftest_qcd")
    _, _, proc_q = run_study(data, mc, [0, 20, 40, 80], "weight", outdir=qdir, qcd=True)
    tot_q = [r["mc_hist"] for r in proc_q if r["category"] == "all" and r["process"] == "total"][0]
    qcd_all = [r["mc_hist"] for r in proc_q if r["category"] == "all" and r["process"] == QCD_LABEL][0]
    check(abs(tot_q - (2.0 + qcd_all)) < 1e-9 and qcd_all >= 0,
          f"os_iso MC 2.0 (e0, e4) + QCD {qcd_all:.4f} = {tot_q:.4f}")

    # the MC split into two datasets (read one at a time) gives the same numbers as one array
    _, _, proc_split = run_study(data, [mc[:4], mc[4:]], [0, 20, 40, 80], "weight",
                                 outdir=os.path.join(qdir, "split"), qcd=True)
    same = [(r["category"], r["process"], round(r["mc_hist"], 9)) for r in proc_q] == \
        [(r["category"], r["process"], round(r["mc_hist"], 9)) for r in proc_split]
    check(same, "MC read per dataset (two chunks) gives the same per-process rows as one array")

    # two channels in one pass: the channel mask is applied per spec. All MC events are etau (channel_id 1), so the
    # etau spec gets the full weight 9.0 (no region cut) and the mutau spec gets no MC
    specs = [Spec("etau", "Jet", "pt", 0), Spec("mutau", "Jet", "pt", 0)]
    data_by_channel = {"etau": data, "mutau": data[channel_mask(data, "mutau")]}
    s_both, _, _ = study(data_by_channel, [mc], specs, [0, 20, 40, 80], "weight", os.path.join(qdir, "channels"),
                         qcd=False, region=None)
    s_etau = [r for r in s_both if r["channel"] == "etau" and r["category"] == "all"][0]
    s_mutau = [r for r in s_both if r["channel"] == "mutau" and r["category"] == "all"][0]
    check(abs(s_etau["mc_weighted"] - 9.0) < 1e-9 and s_mutau["mc_weighted"] == 0.0,
          f"channel mask per spec: etau MC {s_etau['mc_weighted']} (expected 9.0), mutau MC {s_mutau['mc_weighted']} "
          "(expected 0.0)")

    _, _, proc_os = run_study(data, mc, [0, 20, 40, 80], "weight", outdir=os.path.join(qdir, "os_iso_only"), qcd=False)
    tot_os = [r["mc_hist"] for r in proc_os if r["category"] == "all" and r["process"] == "total"][0]
    check(abs(tot_os - 2.0) < 1e-9, f"plotted MC is the os_iso events only (e0, e4: {tot_os}, expected 2.0)")

    # the trigger_weight variant: the plotted MC and the ABCD reference MC are weighted with the trigger_weight column too.
    # Data at 25 GeV (bin 0): os_iso 1 event, region B (os, noniso) 2, C (ss, iso) 3, D (ss, noniso) 2.
    # MC at 25 GeV, weight 1, one event per region; trigger_weight 0.5 for os_iso and B, 1.0 for C and D.
    # Without the column: B = 1, C = 1, D = 1, QCD = (2 - 1) * (3 - 1) / (2 - 1) = 2, MC + QCD = 1 + 2 = 3.
    # With the column: B = 0.5, QCD = (2 - 0.5) * 2 / 1 = 3, MC + QCD = 0.5 + 3 = 3.5.
    def abcd_events(lep, taus, weight=None, tw=None):
        k = len(lep)
        cols = {
            "event": np.arange(k),
            "matched_trigger_ids": ak.Array([[205] for _ in range(k)]),
            "channel_id": np.ones(k, dtype=np.int64),
            "leptons_os": np.array(lep, dtype=np.int64),
            "num_taus_iso": np.array(taus, dtype=np.int64),
            "Jet": ak.Array([[{"pt": 25.0}] for _ in range(k)]),
            **{f"base_pass_{tid}": ak.Array([False] * k) for tid in TRIGGER_IDS},
        }
        if weight is not None:
            cols.update(weight=np.asarray(weight, dtype=np.float64), process_id=np.ones(k, dtype=np.int64),
                        trigger_weight=np.asarray(tw, dtype=np.float64))
        return ak.Array(cols)

    d_abcd = abcd_events([1, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 1, 1, 0, 0])
    m_abcd = abcd_events([1, 1, 0, 0], [1, 0, 1, 0], weight=[1.0] * 4, tw=[0.5, 0.5, 1.0, 1.0])
    rows_tw, _, proc_tw = run_study(d_abcd, m_abcd, [0, 50, 100], "weight",
                                    outdir=os.path.join(qdir, "trigger_weight"), qcd=True,
                                    trigger_weight_column="trigger_weight")

    def value(process, key):
        return [r[key] for r in proc_tw if r["category"] == "all" and r["process"] == process][0]

    check(abs(value("total", "mc_hist") - 3.0) < 1e-9 and abs(value(QCD_LABEL, "mc_hist") - 2.0) < 1e-9,
          f"ABCD without trigger_weight: MC + QCD {value('total', 'mc_hist')} (expected 3), "
          f"QCD {value(QCD_LABEL, 'mc_hist')} (expected 2)")
    check(abs(value("total", "mc_tw_hist") - 3.5) < 1e-9 and abs(value(QCD_LABEL, "mc_tw_hist") - 3.0) < 1e-9,
          f"ABCD with trigger_weight in the reference MC: MC + QCD {value('total', 'mc_tw_hist')} (expected 3.5), "
          f"QCD {value(QCD_LABEL, 'mc_tw_hist')} (expected 3)")
    row_all = [r for r in rows_tw if r["category"] == "all"][0]
    check(abs(row_all["mc_weighted"] - 1.0) < 1e-9 and abs(row_all["mc_tw_weighted"] - 0.5) < 1e-9,
          f"summary: MC weighted {row_all['mc_weighted']} (expected 1), with trigger_weight "
          f"{row_all['mc_tw_weighted']} (expected 0.5)")
    check(os.path.exists(os.path.join(qdir, "trigger_weight", "all.png")), "plot with the trigger_weight variant written")

    print()
    print("FAILURES:", failures if failures else "none")
    sys.exit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mc", default="/pnfs/desy.de/cms/tier2/store/user/milee/hbt_store/analysis_hbt/cf.UniteColumns/23post_v14_limited/*/nominal/calib__default/sel__default/red__default/prod__default/prod28_trg/",
                        help="glob of the united MC output (cf.UniteColumns: reduced and production columns in one file); the first * is the dataset")
    parser.add_argument("--data-reduced", default="/pnfs/desy.de/cms/tier2/store/user/roward/hbt_store/analysis_hbt/cf.ReduceEvents/23post_v14/data_*/nominal/calib__default/sel__default/red__default/prod28_trg/", help="glob of the reduced data output (selector steps none)")
    parser.add_argument("--vars", default="pt", help="comma-separated variables of the objects (default pt); plotted: obj.var[index]")
    parser.add_argument("--order", choices=["pt", "collection"], default="pt",
                        help="rank the objects by pt (default), or keep the collection order (for taus: the selection "
                        "order of the tau pair, not pt)")
    parser.add_argument("--mc-weight", default="normalization_weight,normalized_pu_weight,normalized_isr_weight,normalized_fsr_weight,normalized_njet_btag_weight_pnet,electron_id_weight,electron_reco_weight,muon_id_weight,muon_iso_weight,tau_weight",
                        help="comma-separated weight columns, multiplied. For a comparison with PlotVariables1D, use "
                        "all event weight columns of the config except trigger_weight, and compare the MC line only; "
                        "trigger_weight is the second variant (--trigger-weight-column)")
    parser.add_argument("--channel", default="etau,mutau,tautau",
                        help="comma-separated channels (default: etau,mutau,tautau)")
    parser.add_argument("--process-names", default="",
                        help="optional labels for the process ids, e.g. 1:hh_ggf,2:dy (default: process_<id>)")
    parser.add_argument("--no-qcd", action="store_true",
                        help="do not add the ABCD QCD estimate (hbt --hist-hook qcd) to the MC")
    parser.add_argument("--qcd-exclude", default="",
                        help="comma-separated process ids not subtracted in the ABCD control regions, e.g. the "
                        "signal (hbt excludes signal processes); default: none")
    parser.add_argument("--region", choices=["os_iso", "all"], default="os_iso",
                        help="plot and summary rows for the os_iso events (default), or all events; the QCD estimate "
                        "always uses the ABCD regions")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--steps-spec", default="",
                        help="channel:obj:var:index (e.g. etau:Jet:pt:0): one-by-one study of the trigger SFs for this "
                        "object (comma-separated list allowed): the MC after each cumulative SF step, and the check against "
                        "the trigger_weight column (writes <outdir>/steps/)")
    parser.add_argument("--trigger-weight-column", default="trigger_weight",
                        help="production trigger weight column: the second MC variant (MC and ABCD QCD weighted with it), "
                        "also compared with the step-by-step SFs in the step study")
    parser.add_argument("--weight-hists", action="store_true",
                        help="histograms of each weight, of the nominal product and of each trigger SF (<outdir>/weights/)")
    args = parser.parse_args()
    bins = DEFAULT_BINS
    weights = [column.strip() for column in args.mc_weight.split(",")]
    channels = [c.strip() for c in args.channel.split(",") if c.strip()]
    variables = [v.strip() for v in args.vars.split(",") if v.strip()]
    unknown = [v for v in variables if v not in DEFAULT_BINS]
    if unknown:
        parser.error(f"no bin edges for {unknown}: add them to DEFAULT_BINS")
    names = {int(k): v for k, v in (item.split(":") for item in args.process_names.split(",") if item)}
    excluded = [int(x) for x in args.qcd_exclude.split(",") if x.strip()]
    specs = study_specs(channels, variables)
    step_specs = []
    for text in (t.strip() for t in args.steps_spec.split(",") if t.strip()):
        ch, obj, var, index = text.split(":")
        step = Spec(ch, obj, var, int(index))
        if step not in specs:
            parser.error(f"--steps-spec {text} is not one of the plotted specs (check --channel and --vars)")
        step_specs.append(step)
    if args.trigger_weight_column in weights:
        print(f"note: {args.trigger_weight_column} is not in the nominal product: it is the second variant, removed from --mc-weight")
        weights = [w for w in weights if w != args.trigger_weight_column]

    # only the columns the study uses are read: the rest of the ntuple is not in memory
    objects = sorted({spec.obj for spec in specs})
    obj_fields = [f"{o}.pt" for o in objects] + [f"{o}.{v}" for o in objects for v in variables]
    red_wanted = ["event", "channel_id", "leptons_os", "num_taus_iso", "process_id", "matched_trigger_ids",
                  "trigger_weight", *[f"base_pass_{tid}" for tid in TRIGGER_IDS], *weights, *obj_fields]

    # the data is read once, and split by channel
    data = load(args.data_reduced, reduced_columns=present_columns(args.data_reduced, red_wanted))
    data_by_channel = {ch: data[channel_mask(data, ch)] for ch in channels}
    del data

    weight_store = {}

    def mc_chunks():
        # one dataset at a time, read once: all channels, objects and variables are filled from it
        for dataset, events in mc_per_dataset(args.mc, red_wanted, ["trigsrc_"]):
            if args.trigger_weight_column not in events.fields:
                raise SystemExit(f"{dataset}: no column {args.trigger_weight_column} in the MC (--trigger-weight-column)")
            weight = np.ones(len(events), dtype=np.float64)
            for column in weights:
                weight = weight * np.asarray(events[column], dtype=np.float64)
            print(f"{dataset}: {len(events)} events", flush=True)
            if args.weight_hists:
                accumulate_weight_hists(weight_store, dataset, events, weights, weight)
            yield ak.with_field(events, weight, "mc_weight_study")

    summary, eff_rows, proc_rows = study(
        data_by_channel, mc_chunks(), specs, bins, "mc_weight_study", args.outdir, order=args.order,
        process_names=names, qcd=not args.no_qcd, qcd_exclude=excluded,
        region=None if args.region == "all" else args.region, step_specs=step_specs,
        trigger_weight_column=args.trigger_weight_column)

    if args.weight_hists:
        write_weight_hists(weight_store, args.outdir)
    write_csv(os.path.join(args.outdir, "summary.csv"), summary)
    write_csv(os.path.join(args.outdir, "processes.csv"), proc_rows)
    write_csv(os.path.join(args.outdir, "efficiency.csv"), eff_rows)
    for spec in specs:
        print()
        print(f"=== {spec.channel}: {spec.obj} {spec.var}[{spec.index}] ===")
        mine = [r for r in summary if (r["channel"], r["obj"], r["var"], r["index"]) == spec]
        _print_table(mine, ["category", "n_data", "mc_weighted", "data_over_mc", "data_over_mc_tw", "data_over_mc_sf",
                            "qcd_hist"])


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    else:
        main()