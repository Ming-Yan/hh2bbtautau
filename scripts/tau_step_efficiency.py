#!/usr/bin/env python3
# coding: utf-8

"""
Per-step efficiency of true hadronic taus from H->tautau, read from cf.SelectEvents outputs.

Needs the columns written by hbt/selection/tau_eff.py ("gen_tau_eff", "gen_htt_channel_id", "tau_eff_event_sel") and
"channel_id" from the lepton selection. They are stored in columns_*.parquet for *all* events (no event filter), so
taus lost at any step are included.

Per gen channel (etau, mutau, tautau) it produces
  - step tables and plots (eff vs gen visible tau_h pT and vs gen dR between the two visible taus), for all tau_h
    and, in tautau, separately for the leading and subleading gen tau_h (by visible pT)
  - per trigger kind: efficiency up to trig_fired, trig_offline and trig_matched (cumulative and relative to the
    previous step), with plots per step (all triggers overlaid) and per trigger (the three trigger steps overlaid),
    each also in slices of gen visible tau_h pT; optionally only for taus passing the offline selection
    (--trigger-denominator top_iso) to isolate the trigger effect
  - losses to the isolation ranking and a split by gen decay
and, per reco channel (channel_id assigned by the selection),
  - the gen -> reco channel migration (after the lepton step and after the full selection)
  - the purity of the leading selected taus (fraction matched to a true tau_h from H->tautau)

Example:

    STORE=/pnfs/desy.de/cms/tier2/store/user/$USER/hbt_store/analysis_hbt/cf.SelectEvents
    python scripts/tau_step_efficiency.py \\
        "$STORE/24_v15/hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/<version>/columns_*.parquet" \\
        --fiducial --outdir tau_eff

Efficiencies carry binomial errors sqrt(eff (1 - eff) / N), with N the effective number of entries
(sum w)^2 / sum w^2. All plotted values are also written as .csv next to the plots, and the printed tables to
summary.txt.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import awkward as ak


CHANNELS = {1: "etau", 2: "mutau", 3: "tautau"}
RECO_CHANNELS = {0: "none", 1: "etau", 2: "mutau", 3: "tautau", 4: "ee", 5: "mumu", 6: "emu"}
N_LEAD = {1: 1, 2: 1, 3: 2}

STEPS = [
    ("reco", "reco tau within dR<0.2 (NanoAOD Tau exists)"),
    ("acc", "pt > 20, |eta| < 2.5"),
    ("id", "+ |dz| < 0.2, PNet vsJet >= VVVLoose"),
    ("ch_id", "+ vsE/vsMu (channel WP), dR>0.5 to sel. e/mu"),
    ("top_iso", "+ among most isolated (1 for e/mutau, 2 for tautau)"),
    ("trig_fired", "+ a channel trigger fired"),
    ("trig_offline", "+ trigger offline pt cut"),
    ("trig_matched", "+ trigger object matching"),
    ("selected", "+ leading selected tau, lepton step passed"),
    ("selected_same_channel", "+ event in the same channel as gen decay"),
    ("event_selected", "+ full event selection (incl. jets)"),
]
# steps counted by gen_tau_eff.ntrig_<kind> (same order as TAU_EFF_STEPS in hbt/selection/lepton.py)
TRIG_COUNT_STEPS = ("acc", "id", "ch_id", "top_iso", "trig_fired", "trig_offline", "trig_matched")
TRIGGER_KINDS = {
    "etau": ("single_e", "cross_e_tau", "cross_e_vbf", "cross_vbf"),
    "mutau": ("single_mu", "cross_mu_tau", "cross_mu_vbf", "cross_vbf"),
    "tautau": (
        "cross_tau_tau", "cross_tau_tau_jet", "cross_tau_tau_vbf", "cross_tau_vbf", "cross_quadjet", "cross_vbf",
    ),
}

# plot settings
PT_BINS = np.linspace(20, 200, 19)
# default gen visible tau_h pT slices for the per-trigger efficiency vs dR(l, l) (override with --pt-slices)
PT_SLICES = (20.0, 40.0, 60.0, 100.0, float("inf"))
DR_BINS = np.linspace(0, 5, 26)
PLOT_STEPS = ("reco", "acc", "id", "ch_id", "top_iso", "trig_matched", "selected")
# categorical colors in fixed order, plus distinct markers so series do not rely on color alone
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
MARKERS = ("o", "s", "^", "v", "D", "P", "X")
NEUTRAL = "#52514e"
TRIGGER_STEPS = ("trig_fired", "trig_offline", "trig_matched")
# per-trigger plots: offline-selection reference (top_iso) and the three trigger steps, (color, marker)
TRIGGER_STEP_STYLE = {
    "top_iso": ("#e87ba4", "D"),
    "trig_fired": ("#2a78d6", "<"),
    "trig_offline": ("#eda100", ">"),
    "trig_matched": ("#008300", "P"),
}
DR_LABELS = {
    "etau": r"gen $\Delta R(e, \tau_h)$ (visible)",
    "mutau": r"gen $\Delta R(\mu, \tau_h)$ (visible)",
    "tautau": r"gen $\Delta R(\tau_h, \tau_h)$ (visible)",
}


class Report:
    """
    Prints and keeps the lines to write them to summary.txt.
    """

    def __init__(self):
        self.lines = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self.lines) + "\n")


out = Report()


#
# loading
#

def load(patterns: list[str]) -> ak.Array:
    import pyarrow.parquet as pq

    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no files found for {patterns}")
    available = set(pq.read_schema(files[0]).names)
    wanted = ["gen_tau_eff", "gen_htt_channel_id", "channel_id", "tau_eff_event_sel", "mc_weight"]
    missing = [c for c in wanted if c not in available]
    if "gen_tau_eff" in missing or "gen_htt_channel_id" in missing:
        raise SystemExit(f"columns {missing} not found, was cf.SelectEvents run with hbt/selection/tau_eff.py?")
    if missing:
        print(f"note: columns {missing} not found, the corresponding parts are skipped")
    cols = [f"{c}.*" if c == "gen_tau_eff" else c for c in wanted if c in available]
    arrs = [ak.from_parquet(f, columns=cols) for f in files]
    print(f"read {len(files)} file(s), {sum(len(a) for a in arrs)} events")
    # make sure the tau dimension is variable-length (boolean masks on regular arrays would flatten)
    return ak.from_regular(ak.concatenate(arrs), axis=None)


def event_weights(events: ak.Array, unweighted: bool) -> np.ndarray:
    if unweighted or "mc_weight" not in events.fields:
        return np.ones(len(events), dtype=np.float64)
    return ak.to_numpy(events.mc_weight).astype(np.float64)


def prepare_taus(events: ak.Array, w_ev: np.ndarray, fiducial: bool) -> tuple[ak.Array, np.ndarray]:
    """
    Adds dR between the two visible gen taus ("dr_ll") and the pT rank among the gen tau_h ("rank", 0 = leading)
    to each gen tau, selects hadronic (and optionally fiducial) taus and flattens them. Returns (taus, weights).
    """
    taus = events.gen_tau_eff

    pair = ak.pad_none(taus, 2, axis=1)
    dphi = (pair.phi[:, 0] - pair.phi[:, 1] + np.pi) % (2 * np.pi) - np.pi
    dr_ll = ak.fill_none(np.sqrt((pair.eta[:, 0] - pair.eta[:, 1])**2 + dphi**2), np.nan)
    taus = ak.with_field(taus, ak.broadcast_arrays(dr_ll, taus.pt)[0], "dr_ll")

    is_had = taus.decay == 15
    key = ak.where(is_had, taus.pt, -1.0)
    taus = ak.with_field(taus, ak.argsort(ak.argsort(key, axis=1, ascending=False), axis=1), "rank")

    keep = is_had
    if fiducial:
        keep = keep & (taus.pt > 20) & (abs(taus.eta) < 2.5)
    taus = taus[keep]
    w = np.repeat(w_ev, ak.to_numpy(ak.num(taus.pt, axis=1)))
    return ak.flatten(taus), w


#
# efficiencies
#

def cumulative_flags(taus: ak.Array) -> dict[str, np.ndarray]:
    # enforce a monotonic chain even if a later flag is set without an earlier one
    flags, cum = {}, np.ones(len(taus), dtype=bool)
    for step, _ in STEPS:
        if step not in taus.fields:
            continue
        cum = cum & ak.to_numpy(taus[step])
        flags[step] = cum
    return flags


def trigger_flags(taus: ak.Array, kind: str) -> dict[str, np.ndarray] | None:
    field = f"ntrig_{kind}"
    if field not in taus.fields:
        return None
    n = ak.to_numpy(taus[field])
    reco = ak.to_numpy(taus.reco)
    return {step: reco & (n > i) for i, step in enumerate(TRIG_COUNT_STEPS)}


def binned_efficiency(
    x: np.ndarray,
    passed: np.ndarray,
    w: np.ndarray,
    bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficiency per bin with binomial errors, sqrt(eff * (1 - eff) / N). For weighted events, N is the effective
    number of entries (sum w)^2 / sum w^2, which reduces to the plain count for unit weights.
    Returns (eff, err, n_eff).
    """
    den = np.histogram(x, bins, weights=w)[0]
    num = np.histogram(x[passed], bins, weights=w[passed])[0]
    den_w2 = np.histogram(x, bins, weights=w**2)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = num / den
        n_eff = den**2 / den_w2
        e = np.clip(eff, 0.0, 1.0)
        err = np.sqrt(e * (1.0 - e) / n_eff)
    return eff, err, n_eff


def frac(w: np.ndarray, mask: np.ndarray) -> float:
    total = w.sum()
    return w[mask].sum() / total if total else np.nan


#
# tables
#

def step_table(taus: ak.Array, w: np.ndarray, title: str) -> None:
    n0 = w.sum()
    out(f"\n=== {title}: {len(taus)} true tau_h (sum w = {n0:.1f}) ===")
    out(f"{'step':24s} {'cum. eff':>9s} {'step eff':>9s} {'lost here':>10s}   description")
    prev = n0
    flags = cumulative_flags(taus)
    for step, desc in STEPS:
        if step not in flags:
            continue
        n = w[flags[step]].sum()
        step_eff = n / prev if prev else np.nan
        out(f"{step:24s} {n / n0:9.3f} {step_eff:9.3f} {(prev - n) / n0:10.3f}   {desc}")
        prev = n


def trigger_table(taus: ak.Array, w: np.ndarray, ch: str, title: str) -> list[tuple[str, dict]]:
    """
    Cumulative efficiency per trigger kind (each kind evaluated on its own) for the trigger-related steps.
    Returns the kinds that occur, with their flags.
    """
    kinds = []
    for kind in TRIGGER_KINDS[ch]:
        flags = trigger_flags(taus, kind)
        if flags is not None and flags["acc"].any():
            kinds.append((kind, flags))
    if not kinds:
        return kinds
    combined = cumulative_flags(taus)
    cols = ("top_iso", "trig_fired", "trig_offline", "trig_matched")
    out(f"\n--- {title}: per trigger kind (cum. eff of true tau_h, each trigger on its own) ---")
    out(f"{'trigger':20s} " + " ".join(f"{c:>13s}" for c in cols) + f" {'only this':>10s}")
    for kind, flags in kinds:
        others = np.zeros(len(w), dtype=bool)
        for k2, f2 in kinds:
            if k2 != kind:
                others |= f2["trig_matched"]
        only = flags["trig_matched"] & ~others
        out(f"{kind:20s} " + " ".join(f"{frac(w, flags[c]):13.3f}" for c in cols) + f" {frac(w, only):10.3f}")
    out(f"{'any (combined)':20s} " + " ".join(f"{frac(w, combined[c]):13.3f}" for c in cols))
    out("'only this': true tau_h passing trig_matched for this trigger kind and for no other kind")

    # effect of each trigger step on its own: efficiency relative to the previous step
    rel = [("trig_fired", "top_iso"), ("trig_offline", "trig_fired"), ("trig_matched", "trig_offline")]
    heads = [f"{cur[5:]}|{prev.replace('trig_', '')}" for cur, prev in rel]
    out(f"\n--- {title}: per trigger kind, efficiency of each trigger step relative to the previous one ---")
    out(f"{'trigger':20s} " + " ".join(f"{h:>16s}" for h in heads))
    for kind, flags in kinds + [("any (combined)", combined)]:
        vals = [frac(w[flags[prev]], flags[cur][flags[prev]]) for cur, prev in rel]
        out(f"{kind:20s} " + " ".join(f"{v:16.3f}" for v in vals))
    return kinds


def ranking_losses(taus: ak.Array, w: np.ndarray) -> None:
    f = cumulative_flags(taus)
    lost_rank = f["ch_id"] & ~f["top_iso"]
    out(f"lost to iso ranking (passes ch_id, out-ranked by another tau): {frac(w, lost_rank):.3%}")


def by_gen_dm(taus: ak.Array, w: np.ndarray) -> None:
    f = cumulative_flags(taus)
    nc, npi0 = ak.to_numpy(taus.n_charged), ak.to_numpy(taus.n_pi0)
    out("by gen decay (n charged hadrons, n pi0) -- cum. eff after id / ch_id / trig_matched / selected")
    for c in (1, 3):
        for p0 in (0, 1, 2):
            m = (nc == c) & ((npi0 == p0) if p0 < 2 else (npi0 >= 2))
            if not m.any():
                continue
            effs = " ".join(
                f"{w[m & f[s]].sum() / w[m].sum():6.3f}" for s in ("id", "ch_id", "trig_matched", "selected")
            )
            out(f"  {c}-prong + {p0}{'+' if p0 == 2 else ''} pi0  (n={m.sum():6d})  {effs}")


def channel_migration(events: ak.Array, w_ev: np.ndarray, outdir: str | None) -> None:
    if "channel_id" not in events.fields:
        return
    gen = ak.to_numpy(events.gen_htt_channel_id)
    reco = ak.to_numpy(events.channel_id)
    gen_rows = [(1, "etau"), (2, "mutau"), (3, "tautau"), (0, "other")]
    stages = [("lepton step", reco)]
    if "tau_eff_event_sel" in events.fields:
        stages.append(("full selection", np.where(ak.to_numpy(events.tau_eff_event_sel), reco, 0)))

    matrices = []
    corner = "gen \\ reco"
    for stage, reco_ch in stages:
        m = np.zeros((len(gen_rows), len(RECO_CHANNELS)))
        for i, (g, _) in enumerate(gen_rows):
            row = gen == g
            for j, r in enumerate(RECO_CHANNELS):
                m[i, j] = frac(w_ev[row], reco_ch[row] == r) if row.any() else np.nan
        matrices.append((stage, m))
        out(f"\n=== gen -> reco channel migration after {stage} (row-normalized, 'none' = not selected) ===")
        out(f"{corner:12s} " + " ".join(f"{n:>8s}" for n in RECO_CHANNELS.values()))
        for i, (_, gname) in enumerate(gen_rows):
            out(f"{gname:12s} " + " ".join(f"{v:8.3f}" for v in m[i]))

    if outdir:
        with open(os.path.join(outdir, "channel_migration.csv"), "w") as f:
            f.write("stage,gen_channel," + ",".join(RECO_CHANNELS.values()) + "\n")
            for stage, m in matrices:
                for i, (_, gname) in enumerate(gen_rows):
                    f.write(f"{stage},{gname}," + ",".join(f"{v:.6g}" for v in m[i]) + "\n")
        plot_migration(matrices, [g for _, g in gen_rows], os.path.join(outdir, "channel_migration.png"))


def purity(events: ak.Array, w_ev: np.ndarray) -> None:
    """
    Per reco channel: fraction of the leading selected taus that are true tau_h from H->tautau.
    """
    if "channel_id" not in events.fields:
        return
    reco = ak.to_numpy(events.channel_id)
    n_true = ak.to_numpy(ak.sum(events.gen_tau_eff.selected & (events.gen_tau_eff.decay == 15), axis=1))
    stages = [("lepton step", np.ones(len(events), dtype=bool))]
    if "tau_eff_event_sel" in events.fields:
        stages.append(("full selection", ak.to_numpy(events.tau_eff_event_sel)))
    out("\n=== purity of the leading selected taus per reco channel ===")
    out(f"{'reco channel':12s} {'stage':16s} {'events':>9s} {'tau purity':>11s} {'all lead. true':>15s}")
    for r, rname in CHANNELS.items():
        for stage, sel in stages:
            m = (reco == r) & sel
            if not m.any():
                continue
            n_lead = N_LEAD[r]
            pur = (w_ev[m] * n_true[m]).sum() / (w_ev[m].sum() * n_lead)
            out(f"{rname:12s} {stage:16s} {m.sum():9d} {pur:11.3f} {frac(w_ev[m], n_true[m] == n_lead):15.3f}")


#
# plots
#

def plot_efficiencies(
    taus: ak.Array,
    w: np.ndarray,
    series: list[tuple[str, np.ndarray, str, str]],
    ch: str,
    title: str,
    path: str,
) -> None:
    """
    Cumulative efficiency vs gen visible tau_h pT and vs gen dR(l, l), one column each, with the gen distribution of
    the same variable in a panel below. *series* holds (label, passed mask, color, marker).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variables = [
        ("pt", ak.to_numpy(taus.pt), PT_BINS, r"gen visible $\tau_h$ $p_T$ [GeV]"),
        ("dr_ll", ak.to_numpy(taus.dr_ll), DR_BINS, DR_LABELS[ch]),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 6.5), sharex="col", gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.2},
    )
    rows = []
    for col, (name, x, bins, xlabel) in enumerate(variables):
        ax_eff, ax_dist = axes[0, col], axes[1, col]
        centers = 0.5 * (bins[1:] + bins[:-1])
        half = 0.5 * np.diff(bins)
        valid = np.isfinite(x)

        for label, passed, color, marker in series:
            eff, err, n_eff = binned_efficiency(x[valid], passed[valid], w[valid], bins)
            ax_eff.errorbar(
                centers, eff, yerr=err, xerr=half, fmt=marker, ms=4, lw=1, elinewidth=1, capsize=0,
                color=color, label=label,
            )
            rows += [
                (name, lo, hi, label, e, de, n)
                for lo, hi, e, de, n in zip(bins[:-1], bins[1:], eff, err, n_eff)
            ]

        ax_eff.set_ylim(0, 1.05)
        ax_eff.set_ylabel("cumulative efficiency")
        ax_eff.grid(alpha=0.25, lw=0.5)
        ax_eff.set_title(title, loc="left", fontsize=10)

        # gen distribution of the same variable (denominator), normalized to unit area
        dist = np.histogram(x[valid], bins, weights=w[valid])[0]
        if dist.sum() > 0:
            dist = dist / dist.sum()
        ax_dist.stairs(dist, bins, fill=True, color="#d9d8d3", edgecolor=NEUTRAL, lw=1)
        ax_dist.set_ylabel("gen dist.\n(a.u.)", fontsize=9)
        ax_dist.set_xlabel(xlabel)
        ax_dist.set_xlim(bins[0], bins[-1])
        ax_dist.grid(alpha=0.25, lw=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.17)
    fig.savefig(path, dpi=130)
    plt.close(fig)

    csv_path = os.path.splitext(path)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("variable,bin_lo,bin_hi,series,eff,err,n_eff\n")
        for r in rows:
            f.write(",".join(str(v) if isinstance(v, str) else f"{v:.6g}" for v in r) + "\n")
    print(f"saved {path} (+ .csv)")


def plot_in_pt_slices(
    taus: ak.Array,
    w: np.ndarray,
    series: list[tuple[str, np.ndarray, str, str]],
    ch: str,
    title: str,
    path: str,
    pt_edges: tuple[float, ...],
) -> None:
    """
    Efficiency vs gen dR(l, l) in slices of the gen visible pT of the tau_h itself, one column per slice, with the
    gen dR distribution of that slice below. Separates the dR dependence from the pT dependence (boosted HH decays
    have high pT *and* small dR). *series* holds (label, passed mask, color, marker).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = ak.to_numpy(taus.dr_ll)
    pt = ak.to_numpy(taus.pt)
    bins = DR_BINS
    centers = 0.5 * (bins[1:] + bins[:-1])
    half = 0.5 * np.diff(bins)
    n_slices = len(pt_edges) - 1

    fig, axes = plt.subplots(
        2, n_slices, figsize=(4.0 * n_slices + 1.0, 6.5), sharex=True, sharey="row", squeeze=False,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.08},
    )
    rows = []
    total_w = w[np.isfinite(x)].sum()
    for i in range(n_slices):
        lo, hi = pt_edges[i], pt_edges[i + 1]
        ax_eff, ax_dist = axes[0, i], axes[1, i]
        in_slice = (pt >= lo) & (pt < hi) & np.isfinite(x)
        slice_label = f"{lo:g} <= pT < {hi:g} GeV" if np.isfinite(hi) else f"pT >= {lo:g} GeV"
        tex_label = (
            rf"${lo:g} \leq p_T^{{vis}} < {hi:g}$ GeV" if np.isfinite(hi) else rf"$p_T^{{vis}} \geq {lo:g}$ GeV"
        )
        share = w[in_slice].sum() / total_w if total_w else np.nan

        for label, passed, color, marker in series:
            eff, err, n_eff = binned_efficiency(x[in_slice], passed[in_slice], w[in_slice], bins)
            ax_eff.errorbar(
                centers, eff, yerr=err, xerr=half, fmt=marker, ms=4, lw=1, elinewidth=1, capsize=0,
                color=color, label=label,
            )
            rows += [
                (slice_label, b_lo, b_hi, label, e, de, n)
                for b_lo, b_hi, e, de, n in zip(bins[:-1], bins[1:], eff, err, n_eff)
            ]
        ax_eff.set_ylim(0, 1.05)
        ax_eff.grid(alpha=0.25, lw=0.5)
        ax_eff.set_title(f"{tex_label} ({share:.0%} of $\\tau_h$)", fontsize=10)

        dist = np.histogram(x[in_slice], bins, weights=w[in_slice])[0]
        if dist.sum() > 0:
            dist = dist / dist.sum()
        ax_dist.stairs(dist, bins, fill=True, color="#d9d8d3", edgecolor=NEUTRAL, lw=1)
        ax_dist.set_xlabel(DR_LABELS[ch], fontsize=10)
        ax_dist.set_xlim(bins[0], bins[-1])
        ax_dist.grid(alpha=0.25, lw=0.5)

    axes[0, 0].set_ylabel("cumulative efficiency")
    axes[1, 0].set_ylabel("gen dist.\n(a.u.)", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2)
    fig.savefig(path, dpi=130)
    plt.close(fig)

    csv_path = os.path.splitext(path)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("pt_slice,dr_lo,dr_hi,series,eff,err,n_eff\n")
        for r in rows:
            f.write(",".join(str(v) if isinstance(v, str) else f"{v:.6g}" for v in r) + "\n")
    print(f"saved {path} (+ .csv)")


def trigger_plots(taus: ak.Array, w: np.ndarray, kinds: list[tuple[str, dict]], ch: str, title: str, args) -> None:
    """
    Writes to <outdir>/<ch>_triggers/:
      - <step>.png / <step>_pt_slices.png for step in trig_fired, trig_offline, trig_matched: all trigger kinds and
        their combination overlaid
      - <kind>.png / <kind>_pt_slices.png: top_iso and the three trigger steps for one trigger kind (and for the
        combination of all kinds, "any_trigger"), showing how much each trigger step costs
    With --trigger-denominator top_iso, only true tau_h passing the offline selection up to top_iso are used.
    """
    outdir = os.path.join(args.outdir, f"{ch}_triggers")
    os.makedirs(outdir, exist_ok=True)

    combined = cumulative_flags(taus)
    if args.trigger_denominator == "top_iso":
        den = combined["top_iso"]
        suffix = ", taus passing top_iso"
    else:
        den = np.ones(len(taus), dtype=bool)
        suffix = ""
    t, wd = taus[den], w[den]
    slices_title = ", in gen visible tau_h pT slices"

    for step in TRIGGER_STEPS:
        series = [(kind, flags[step][den], c, mk) for (kind, flags), c, mk in zip(kinds, COLORS, MARKERS)]
        series.append(("any trigger", combined[step][den], NEUTRAL, "*"))
        step_title = f"{title}: up to {step}, per trigger{suffix}"
        plot_efficiencies(t, wd, series, ch, step_title, os.path.join(outdir, f"{step}.png"))
        plot_in_pt_slices(
            t, wd, series, ch, step_title + slices_title, os.path.join(outdir, f"{step}_pt_slices.png"),
            args.pt_slices,
        )

    for kind, flags in kinds + [("any_trigger", combined)]:
        series = [(s, flags[s][den], color, marker) for s, (color, marker) in TRIGGER_STEP_STYLE.items()]
        kind_title = f"{title}: {kind.replace('_', ' ') if kind == 'any_trigger' else kind}{suffix}"
        plot_efficiencies(t, wd, series, ch, kind_title, os.path.join(outdir, f"{kind}.png"))
        plot_in_pt_slices(
            t, wd, series, ch, kind_title + slices_title, os.path.join(outdir, f"{kind}_pt_slices.png"),
            args.pt_slices,
        )


def plot_migration(matrices: list[tuple[str, np.ndarray]], gen_names: list[str], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # single-hue sequential ramp (light -> dark blue)
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    reco_names = list(RECO_CHANNELS.values())
    fig, axes = plt.subplots(1, len(matrices), figsize=(6.2 * len(matrices), 3.6), squeeze=False)
    for ax, (stage, m) in zip(axes[0], matrices):
        ax.imshow(np.nan_to_num(m), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]) and m[i, j] >= 0.0005:
                    ax.text(
                        j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.55 else "#0b0b0b",
                    )
        ax.set_xticks(range(len(reco_names)))
        ax.set_xticklabels(reco_names)
        ax.set_yticks(range(len(gen_names)))
        ax.set_yticklabels(gen_names)
        ax.set_xlabel("reco channel")
        ax.set_ylabel("gen channel")
        ax.set_title(f"after {stage} (row-normalized)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved {path} (+ .csv)")


#
# main
#

def analyse_gen_channel(events: ak.Array, w_ev: np.ndarray, ch_id: int, ch: str, args) -> None:
    sel = ak.to_numpy(events.gen_htt_channel_id) == ch_id
    taus, w = prepare_taus(events[sel], w_ev[sel], args.fiducial)
    if len(taus) == 0:
        return
    title = f"gen {ch}" + (" (fiducial)" if args.fiducial else "")

    def path(name):
        return os.path.join(args.outdir, f"{ch}_{name}.png")

    # steps, all tau_h and (tautau) leading / subleading
    subsets = [("", np.ones(len(taus), dtype=bool))]
    if ch == "tautau":
        rank = ak.to_numpy(taus.rank)
        subsets += [("leading", rank == 0), ("subleading", rank == 1)]
    for label, m in subsets:
        sub_title = f"{title}, {label} gen tau_h" if label else title
        step_table(taus[m], w[m], sub_title)
        if args.outdir:
            f = cumulative_flags(taus[m])
            series = [(s, f[s], c, mk) for s, c, mk in zip(PLOT_STEPS, COLORS, MARKERS) if s in f]
            plot_efficiencies(taus[m], w[m], series, ch, sub_title, path(f"steps_{label}" if label else "steps"))

    # per trigger kind
    kinds = trigger_table(taus, w, ch, title)
    if args.outdir and kinds:
        trigger_plots(taus, w, kinds, ch, title, args)

    out("")
    ranking_losses(taus, w)
    by_gen_dm(taus, w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="columns_*.parquet files or glob patterns")
    ap.add_argument("--channel", choices=["etau", "mutau", "tautau", "all"], default="all", help="gen channel(s)")
    ap.add_argument("--fiducial", action="store_true", help="denominator: gen visible pt > 20, |eta| < 2.5")
    ap.add_argument("--unweighted", action="store_true", help="ignore mc_weight")
    ap.add_argument("--outdir", default=None, help="directory for plots, csv files and summary.txt")
    ap.add_argument(
        "--pt-slices", default=",".join(f"{v:g}" for v in PT_SLICES),
        type=lambda v: tuple(float(x) for x in v.split(",")),
        help="gen visible tau_h pT edges for the per-trigger plot vs dR, e.g. 20,40,60,100,inf (default: %(default)s)",
    )
    ap.add_argument(
        "--trigger-denominator", choices=["all", "top_iso"], default="all",
        help="denominator of the trigger plots: all true tau_h (cumulative, default) or only those passing the offline "
        "tau selection up to top_iso, to isolate the trigger effect",
    )
    args = ap.parse_args()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    events = load(args.files)
    w_ev = event_weights(events, args.unweighted)

    for ch_id, ch in CHANNELS.items():
        if args.channel in ("all", ch):
            analyse_gen_channel(events, w_ev, ch_id, ch, args)

    # per reco channel
    channel_migration(events, w_ev, args.outdir)
    purity(events, w_ev)

    if args.outdir:
        out.save(os.path.join(args.outdir, "summary.txt"))
        print(f"saved {os.path.join(args.outdir, 'summary.txt')}")


if __name__ == "__main__":
    main()