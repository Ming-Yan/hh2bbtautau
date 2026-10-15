#!/usr/bin/env python3
# coding: utf-8

"""
Compare the pT of the reco tau matched to each true tau_h from H->tautau with its gen visible pT, using outputs that
already exist: gen_tau_eff.reco_idx from cf.SelectEvents and the calibrated Tau.pt from cf.CalibrateEvents.

Both tasks process the same input file(s) per branch and write one row per input event, in the same order, so row i
of SelectEvents' columns_<b>.parquet and CalibrateEvents' calib_<b>.parquet is the same event and reco_idx indexes
the calibrated Tau collection directly. This is the tau pT the selection actually used (after the tau energy
calibration). The script checks that the row counts and the tau multiplicities are consistent.

Example:

    STORE=/pnfs/desy.de/cms/tier2/store/user/$USER/hbt_store/analysis_hbt
    DS=24_v15/hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default
    python scripts/tau_reco_vs_gen_pt.py \\
        --select-dir "$STORE/cf.SelectEvents/$DS/sel__default/<sel_version>" \\
        --calib-dir "$STORE/cf.CalibrateEvents/$DS/<calib_version>" \\
        --fiducial --outdir tau_pt_response

(use `law run cf.CalibrateEvents ... --print-output 0` to find the calibration output path)
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import awkward as ak


PT_BINS = np.linspace(20, 200, 19)
RESP_BINS = np.linspace(0.0, 2.0, 81)
# gen decay groups (n charged hadrons, n pi0 selector), colors and markers in fixed order
DECAYS = [
    ("1-prong", lambda nc, np0: (nc == 1) & (np0 == 0)),
    (r"1-prong + $\pi^0$s", lambda nc, np0: (nc == 1) & (np0 >= 1)),
    ("3-prong", lambda nc, np0: (nc == 3) & (np0 == 0)),
    (r"3-prong + $\pi^0$s", lambda nc, np0: (nc == 3) & (np0 >= 1)),
]
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "v")
NEUTRAL = "#52514e"


def branch_files(directory: str, prefixes: tuple[str, ...]) -> dict[int, str]:
    """
    Maps branch number -> file for <prefix>_<branch>.parquet files in *directory*, trying *prefixes* in order.
    """
    for prefix in prefixes:
        files = {}
        for path in glob.glob(os.path.join(directory, f"{prefix}_*.parquet")):
            m = re.search(rf"{prefix}_(\d+)\.parquet$", path)
            if m:
                files[int(m.group(1))] = path
        if files:
            return files
    raise SystemExit(f"no {' or '.join(p + '_<branch>.parquet' for p in prefixes)} files found in {directory}")


def load(select_dir: str, calib_dir: str) -> ak.Array:
    # SelectEvents writes columns_<branch>.parquet, CalibrateEvents calib_<branch>.parquet
    sel_files = branch_files(select_dir, ("columns",))
    cal_files = branch_files(calib_dir, ("calib", "columns"))
    branches = sorted(set(sel_files) & set(cal_files))
    if not branches:
        raise SystemExit("no branch numbers in common between the two directories")
    import pyarrow.parquet as pq
    if "Tau" not in pq.read_schema(cal_files[branches[0]]).names:
        raise SystemExit(f"no Tau.pt in {cal_files[branches[0]]}, the calibrator did not produce calibrated taus")
    if set(sel_files) != set(cal_files):
        print(f"note: using the {len(branches)} branches present in both directories")

    taus = []
    for b in branches:
        sel = ak.from_parquet(sel_files[b], columns=["gen_tau_eff.*", "gen_htt_channel_id"])
        cal = ak.from_parquet(cal_files[b], columns=["Tau.pt"])
        if len(sel) != len(cal):
            raise SystemExit(f"branch {b}: {len(sel)} rows in SelectEvents but {len(cal)} in CalibrateEvents")
        g = ak.from_regular(sel.gen_tau_eff, axis=None)
        idx = g.reco_idx
        n_tau = ak.num(cal.Tau.pt, axis=1)
        if ak.any(idx >= ak.broadcast_arrays(n_tau, idx)[0]):
            raise SystemExit(f"branch {b}: reco_idx out of range, outputs are not aligned (different versions?)")
        safe = ak.fill_none(ak.pad_none(cal.Tau.pt, 1, axis=1), np.nan)
        reco_pt = ak.where(idx >= 0, safe[ak.where(idx >= 0, idx, 0)], np.nan)
        g = ak.with_field(g, reco_pt, "reco_pt_calib")
        g = ak.with_field(g, ak.broadcast_arrays(sel.gen_htt_channel_id, g.pt)[0], "gen_channel")
        taus.append(ak.flatten(g))
    taus = ak.concatenate(taus)
    print(f"read {len(branches)} branch(es), {len(taus)} gen taus")
    return taus


def plot(taus: ak.Array, outdir: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    gen = ak.to_numpy(taus.pt).astype(float)
    reco = ak.to_numpy(taus.reco_pt_calib).astype(float)
    resp = reco / gen
    nc, np0 = ak.to_numpy(taus.n_charged), ak.to_numpy(taus.n_pi0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1) reco vs gen visible pT
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
    h, xe, ye = np.histogram2d(gen, reco, bins=[PT_BINS, PT_BINS])
    mesh = axes[0].pcolormesh(xe, ye, h.T, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(h.max(), 2)))
    axes[0].plot([PT_BINS[0], PT_BINS[-1]], [PT_BINS[0], PT_BINS[-1]], color=NEUTRAL, lw=1, ls="--")
    fig.colorbar(mesh, ax=axes[0], label="taus")
    axes[0].set_xlabel(r"gen visible $\tau_h$ $p_T$ [GeV]")
    axes[0].set_ylabel(r"matched reco $\tau_h$ $p_T$ (calibrated) [GeV]")
    axes[0].set_title(title, loc="left", fontsize=10)

    # 2) response per gen decay
    for (label, sel_fn), color, marker in zip(DECAYS, COLORS, MARKERS):
        m = sel_fn(nc, np0)
        if not m.any():
            continue
        counts = np.histogram(resp[m], RESP_BINS)[0]
        dens = counts / max(counts.sum(), 1) / np.diff(RESP_BINS)
        centers = 0.5 * (RESP_BINS[1:] + RESP_BINS[:-1])
        axes[1].plot(
            centers, dens, color=color, lw=1.5, marker=marker, ms=3,
            label=f"{label} (median {np.median(resp[m]):.3f})",
        )
    axes[1].axvline(1.0, color=NEUTRAL, lw=1, ls="--")
    axes[1].set_xlabel(r"$p_T^{reco} / p_T^{gen, vis}$")
    axes[1].set_ylabel("normalized")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("response by gen decay", loc="left", fontsize=10)

    # 3) median response and resolution (half 68% interval) vs gen visible pT
    centers = 0.5 * (PT_BINS[1:] + PT_BINS[:-1])
    half = 0.5 * np.diff(PT_BINS)
    med, res = np.full(len(centers), np.nan), np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(PT_BINS[:-1], PT_BINS[1:])):
        r = resp[(gen >= lo) & (gen < hi)]
        if len(r) >= 20:
            q16, q50, q84 = np.percentile(r, [16, 50, 84])
            med[i], res[i] = q50, 0.5 * (q84 - q16)
    axes[2].errorbar(centers, med, yerr=res, xerr=half, fmt="o", ms=4, color=COLORS[0], lw=1,
                     label="median, bars: half 68% interval")
    axes[2].axhline(1.0, color=NEUTRAL, lw=1, ls="--")
    axes[2].set_ylim(0.5, 1.5)
    axes[2].set_xlabel(r"gen visible $\tau_h$ $p_T$ [GeV]")
    axes[2].set_ylabel(r"$p_T^{reco} / p_T^{gen, vis}$")
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].set_title("response vs gen pT", loc="left", fontsize=10)

    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    path = os.path.join(outdir, "tau_pt_reco_vs_gen.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)

    with open(os.path.join(outdir, "tau_pt_response_vs_gen_pt.csv"), "w") as f:
        f.write("gen_pt_lo,gen_pt_hi,median_response,half_68_interval\n")
        for lo, hi, m_, r_ in zip(PT_BINS[:-1], PT_BINS[1:], med, res):
            f.write(f"{lo:g},{hi:g},{m_:.6g},{r_:.6g}\n")
    print(f"saved {path} (+ csv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select-dir", required=True, help="directory with cf.SelectEvents columns_<branch>.parquet")
    ap.add_argument("--calib-dir", required=True, help="directory with cf.CalibrateEvents calib_<branch>.parquet")
    ap.add_argument("--channel", choices=["etau", "mutau", "tautau", "all"], default="all", help="gen channel")
    ap.add_argument("--fiducial", action="store_true", help="gen visible pt > 20, |eta| < 2.5")
    ap.add_argument("--outdir", default="tau_pt_response")
    args = ap.parse_args()

    taus = load(args.select_dir, args.calib_dir)
    m = (taus.decay == 15) & (taus.reco_idx >= 0)
    if args.channel != "all":
        m = m & (taus.gen_channel == {"etau": 1, "mutau": 2, "tautau": 3}[args.channel])
    if args.fiducial:
        m = m & (taus.pt > 20) & (abs(taus.eta) < 2.5)
    taus = taus[m]

    resp = ak.to_numpy(taus.reco_pt_calib / taus.pt)
    print(f"{len(taus)} matched true tau_h")
    print(f"response pT_reco / pT_gen,vis: median {np.median(resp):.3f}, "
          f"half 68% interval {0.5 * np.subtract(*np.percentile(resp, [84, 16])):.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    title = f"gen {args.channel}" + (" (fiducial)" if args.fiducial else "") + ", matched true tau_h"
    plot(taus, args.outdir, title)


if __name__ == "__main__":
    main()