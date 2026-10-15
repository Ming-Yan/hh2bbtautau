# coding: utf-8

"""
Gen-anchored tau efficiency bookkeeping during the selection.

For every generator-level tau from the H->tautau decay, the closest reco tau (dR < 0.2) is looked up
and the cumulative tau-level selection steps accumulated in the lepton selection are evaluated on it,
using the step masks of the channel that corresponds to the generator-level decay.

The result is stored as a ragged column "gen_tau_eff" (one entry per gen tau, so [event, 2] for
HH->bbtautau) plus the event-level columns "gen_htt_channel_id" and "tau_eff_event_sel". Per trigger kind,
"gen_tau_eff.ntrig_<kind>" holds the number of consecutive tau steps passed for that trigger (step i passed if > i).

Since SelectEvents writes produced columns for *all* events (no event filter applied), these columns can be read
directly from cf.SelectEvents' columns_*.parquet to compute efficiencies, including events that fail later steps.
"""

from __future__ import annotations

from columnflow.production import Producer, producer
from columnflow.production.cms.gen_particles import gen_higgs_lookup
from columnflow.production.util import lv_mass
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from hbt.selection.lepton import TAU_EFF_STEPS, TAU_EFF_CHANNELS, TAU_EFF_TRIGGER_KINDS

np = maybe_import("numpy")
ak = maybe_import("awkward")


# maximum dR between visible gen tau and reco tau
MATCH_DR = 0.2


def _nested_delta_r(a_eta, a_phi, b_eta, b_phi):
    """
    dR between all pairs of two jagged collections, returned with shape [event, a, b].
    """
    ae, be = ak.unzip(ak.cartesian([a_eta, b_eta], axis=1, nested=True))
    ap, bp = ak.unzip(ak.cartesian([a_phi, b_phi], axis=1, nested=True))
    dphi = (ap - bp + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((ae - be)**2 + dphi**2)


def _take(mask, idx):
    """
    Returns mask[idx] per event for a jagged boolean *mask* [event, reco] and integer *idx* [event, gen],
    with idx == -1 mapping to False.
    """
    safe_mask = ak.fill_none(ak.pad_none(mask, 1, axis=1), False)
    vals = safe_mask[ak.where(idx >= 0, idx, 0)]
    return (idx >= 0) & vals


def _take_int(values, idx):
    """
    Same as _take for jagged integer *values*, with idx == -1 mapping to 0.
    """
    safe = ak.fill_none(ak.pad_none(values, 1, axis=1), 0)
    vals = safe[ak.where(idx >= 0, idx, 0)]
    return ak.where(idx >= 0, vals, 0)


def gen_htt_taus(gen_higgs: ak.Array) -> ak.Array:
    """
    Visible generator taus from the H->tautau decay (always the last higgs in gen_higgs), shape [event, 2].
    """
    w_ch = gen_higgs[:, -1].tau_w_children  # [event, tau, decay products]
    apdg = abs(w_ch.pdgId)
    is_nu = (apdg == 12) | (apdg == 14) | (apdg == 16)
    vis = lv_mass(w_ch[~is_nu]).sum(axis=-1)
    first = ak.fill_none(ak.firsts(apdg, axis=2), 0)
    is_e, is_mu = first == 11, first == 13
    return ak.zip({
        "pt": ak.values_astype(vis.pt, np.float32),
        "eta": ak.values_astype(vis.eta, np.float32),
        "phi": ak.values_astype(vis.phi, np.float32),
        "decay": ak.values_astype(ak.where(is_e, 11, ak.where(is_mu, 13, 15)), np.int8),  # 11 e, 13 mu, 15 had
        "n_charged": ak.values_astype(ak.sum((apdg == 211) | (apdg == 321), axis=2), np.int8),
        "n_pi0": ak.values_astype(ak.sum(apdg == 111, axis=2), np.int8),
    })


@producer(
    uses={gen_higgs_lookup, "Tau.{pt,eta,phi}"},  # channel_id is created in the lepton selection
    produces={"gen_tau_eff.*", "gen_htt_channel_id", "tau_eff_event_sel"},
    exposed=False,
)
def gen_tau_eff(
    self: Producer,
    events: ak.Array,
    tau_eff_masks: dict[str, dict[str, ak.Array]],
    tau_eff_trig: dict[str, dict[str, ak.Array]],
    sel_tau_indices: ak.Array,
    lepton_sel: ak.Array,
    event_sel: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    :param tau_eff_masks: per-channel, per-step tau masks from the lepton selection aux data
    :param tau_eff_trig: per-channel, per-trigger-kind number of tau steps passed, from the lepton selection aux data
    :param sel_tau_indices: indices of the finally selected taus (results.objects.Tau.Tau)
    :param lepton_sel: event mask of the lepton step
    :param event_sel: full event selection mask
    """
    events = self[gen_higgs_lookup](events, **kwargs)
    gen = gen_htt_taus(events.gen_higgs)
    is_had = gen.decay == 15

    # gen channel per event
    ch_ids = {ch: self.config_inst.get_channel(ch).id for ch in TAU_EFF_CHANNELS}
    n_had = ak.sum(is_had, axis=1)
    n_e = ak.sum(gen.decay == 11, axis=1)
    n_mu = ak.sum(gen.decay == 13, axis=1)
    gen_ch = np.zeros(len(events), dtype=np.int8)
    gen_ch = np.where(ak.to_numpy((n_had == 2)), ch_ids["tautau"], gen_ch)
    gen_ch = np.where(ak.to_numpy((n_had == 1) & (n_e == 1)), ch_ids["etau"], gen_ch)
    gen_ch = np.where(ak.to_numpy((n_had == 1) & (n_mu == 1)), ch_ids["mutau"], gen_ch)

    # match each hadronic gen tau to the closest reco tau
    dr = _nested_delta_r(gen.eta, gen.phi, events.Tau.eta, events.Tau.phi)  # [event, gen, reco]
    best = ak.argmin(dr, axis=2, keepdims=True)
    best_dr = ak.fill_none(ak.firsts(dr[best], axis=2), 99.0)
    matched = is_had & (best_dr < MATCH_DR)
    reco_idx = ak.where(matched, ak.fill_none(ak.firsts(best, axis=2), -1), -1)

    cols = {
        **{f: gen[f] for f in gen.fields},
        "reco_dr": ak.values_astype(best_dr, np.float32),
        "reco_idx": ak.values_astype(reco_idx, np.int16),
        "reco": matched,
    }

    # cumulative step flags, evaluated in the channel of the gen decay
    for step in TAU_EFF_STEPS:
        flag = ak.zeros_like(matched)
        for ch in TAU_EFF_CHANNELS:
            ch_flag = _take(tau_eff_masks[ch][step], reco_idx)
            flag = ak.where(gen_ch == ch_ids[ch], ch_flag, flag)
        cols[step] = flag

    # per trigger kind: number of consecutive tau steps passed, in the channel of the gen decay
    for kind in TAU_EFF_TRIGGER_KINDS:
        n = ak.values_astype(ak.zeros_like(reco_idx), np.int8)
        for ch in TAU_EFF_CHANNELS:
            if kind in tau_eff_trig[ch]:
                n = ak.where(gen_ch == ch_ids[ch], _take_int(tau_eff_trig[ch][kind], reco_idx), n)
        cols[f"ntrig_{kind}"] = ak.values_astype(n, np.int8)

    def contained(idx, collection):
        r, c = ak.unzip(ak.cartesian([idx, collection], axis=1, nested=True))
        return ak.fill_none(ak.any(r == c, axis=2), False) & (idx >= 0)

    # is the matched reco tau in the final Tau collection of an event passing the lepton step, and is it one of the
    # leading (most isolated) taus that form the lepton pair (1 for etau/mutau, 2 for tautau)
    n_lead = np.where(ak.to_numpy(events.channel_id) == ch_ids["tautau"], 2, 1)
    lead_tau_indices = sel_tau_indices[ak.local_index(sel_tau_indices) < n_lead]
    cols["in_tau_collection"] = contained(reco_idx, sel_tau_indices) & lepton_sel
    cols["selected"] = contained(reco_idx, lead_tau_indices) & lepton_sel
    cols["selected_same_channel"] = cols["selected"] & (events.channel_id == gen_ch)
    cols["event_selected"] = cols["selected"] & event_sel

    events = set_ak_column(events, "gen_tau_eff", ak.zip(cols))
    events = set_ak_column(events, "gen_htt_channel_id", gen_ch, value_type=np.int8)
    events = set_ak_column(events, "tau_eff_event_sel", np.asarray(event_sel, dtype=bool))

    return events
