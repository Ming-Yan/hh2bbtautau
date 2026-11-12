# coding: utf-8

"""
Exemplary selection methods.
"""

from __future__ import annotations

import operator

from columnflow.categorization import Categorizer, categorizer
from columnflow.columnar_util import attach_coffea_behavior, full_like, ak_concatenate_safe
from columnflow.util import maybe_import

from hbt.production.jet import jet_multiplicity, bjet_multiplicity
from hbt.util import MET_COLUMN, stack_lvectors, create_lvector_xyz, rotate_px_py

ak = maybe_import("awkward")


# helpers
all_true = lambda events: full_like(events.event, True, dtype=bool)

hhbjet_multiplicity = bjet_multiplicity.derive("hhbjet_multiplicity", cls_dict={"jet_name": "HHBJet"})


#
# dummy selector
#

@categorizer(uses={"event"})
def cat_all(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # keep all events
    return events, all_true(events)


#
# lepton channels
#

@categorizer(uses={"channel_id"})
def cat_etau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.etau.id


@categorizer(uses={"channel_id"})
def cat_mutau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.mutau.id


@categorizer(uses={"channel_id"})
def cat_tautau(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.tautau.id


@categorizer(uses={"channel_id"})
def cat_ee(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.ee.id


@categorizer(uses={"channel_id"})
def cat_mumu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.mumu.id


@categorizer(uses={"channel_id"})
def cat_emu(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.channel_id == self.config_inst.channels.n.emu.id


#
# QCD regions
#

@categorizer(uses={"leptons_os"})
def cat_os(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # oppositive sign leptons
    return events, events.leptons_os == 1


@categorizer(uses={"leptons_os"})
def cat_ss(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # same sign leptons
    return events, events.leptons_os == 0


@categorizer(uses={"num_taus_iso", "channel_id"})
def cat_iso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # isolated region should contain exactly 1 iso taus in etau/mutau
    # isolated region should contain exactly 2 iso taus in tautau
    ch_tautau = self.config_inst.channels.n.tautau
    return events, events.num_taus_iso >= ak.where(events.channel_id == ch_tautau.id, 2, 1)


@categorizer(uses={"num_taus_iso", "channel_id"})
def cat_noniso(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # non-isolated region should contain exactly 0 iso taus in etau/mutau
    # non-isolated region should contain exactly 1 iso tau for tautau channel
    ch_tautau = self.config_inst.channels.n.tautau
    return events, events.num_taus_iso == ak.where(events.channel_id == ch_tautau.id, 1, 0)


#
# additional special regions
#

@categorizer(uses={"HHBJet.{mass,pt,eta,phi}"})
def cat_jetpt50(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    mask = ak.all(events.HHBJet.pt>50,axis=-1)
    return events, mask

@categorizer(uses={"single_triggered"})
def cat_single_triggered(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, events.single_triggered == 1


#
# trigger-based regions, based on matched_trigger_ids of the 2023 trigger study
#

# study trigger ids sorted by priority (highest first), same order as in the trigger study
TRIGGER_STUDY_IDS = [205, 105, 405, 304, 507, 701, 602, 605, 606, 608, 609]

# priority group of each study trigger in 2023, mirrors the trigger tag groups in hbt/selection/jet.py
TRIGGER_STUDY_GROUPS = {
    205: 1, 105: 1, 405: 1, 304: 1, 507: 1, 701: 1,  # group 1: e/mu single, cross e/mu-tau, cross tau-tau
    # group 2 (cross quadjet) has no study trigger in 2023
    602: 3,  # group 3: cross tau-tau vbf
    605: 4,  # group 4: cross vbf
    606: 5, 608: 5, 609: 5,  # group 5: cross e/mu/tau vbf
}


def _trigger_study_fired(events) -> dict[int, ak.Array]:
    # per-event booleans, whether each study trigger is among the matched trigger ids
    return {
        tid: ak.fill_none(ak.any(events.matched_trigger_ids == tid, axis=1), False)
        for tid in TRIGGER_STUDY_IDS
    }


def _trigger_study_any(events, fired: dict[int, ak.Array], trigger_ids: list[int]) -> ak.Array:
    # per-event boolean, whether at least one of the given study triggers fired
    mask = full_like(events.event, False, dtype=bool)
    for tid in trigger_ids:
        mask = mask | fired[tid]
    return mask


def _trigger_study_count(events, fired: dict[int, ak.Array], trigger_ids: list[int]) -> ak.Array:
    # per-event number of fired study triggers among the given ones
    count = full_like(events.event, 0, dtype=int)
    for tid in trigger_ids:
        count = count + fired[tid] * 1
    return count


def _trigger_study_group(group: int) -> list[int]:
    return [tid for tid in TRIGGER_STUDY_IDS if TRIGGER_STUDY_GROUPS.get(tid) == group]


@categorizer(uses={"matched_trigger_ids", "event"}, trigger_id=None)
def cat_only_trigger(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # only this study trigger fired, no other study trigger
    fired = _trigger_study_fired(events)
    n_fired = _trigger_study_count(events, fired, TRIGGER_STUDY_IDS)
    return events, fired[self.trigger_id] & (n_fired == 1)


@categorizer(uses={"matched_trigger_ids", "event"}, trigger_id=None)
def cat_primary_trigger(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # this study trigger fired and no study trigger of higher priority fired; others may also fire
    fired = _trigger_study_fired(events)
    higher = TRIGGER_STUDY_IDS[:TRIGGER_STUDY_IDS.index(self.trigger_id)]
    return events, fired[self.trigger_id] & ~_trigger_study_any(events, fired, higher)


@categorizer(uses={"matched_trigger_ids", "event"}, group=None)
def cat_only_group(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least one trigger of the group fired and no trigger of any other group
    fired = _trigger_study_fired(events)
    in_group = _trigger_study_group(self.group)
    others = [tid for tid in TRIGGER_STUDY_IDS if tid not in in_group]
    mask = _trigger_study_any(events, fired, in_group) & ~_trigger_study_any(events, fired, others)
    return events, mask


@categorizer(uses={"matched_trigger_ids", "event"}, group=None, min_fired=1)
def cat_group_min(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least min_fired triggers of the group fired
    fired = _trigger_study_fired(events)
    count = _trigger_study_count(events, fired, _trigger_study_group(self.group))
    return events, count >= self.min_fired


@categorizer(uses={"matched_trigger_ids", "event"}, groups=None)
def cat_groups_all(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # at least one trigger of every listed group fired (chained with +)
    fired = _trigger_study_fired(events)
    mask = full_like(events.event, True, dtype=bool)
    for group in self.groups:
        mask = mask | _trigger_study_any(events, fired, _trigger_study_group(group))
    return events, mask


# only_<id>: only this study trigger fired
cat_only_205 = cat_only_trigger.derive("cat_only_205", cls_dict={"trigger_id": 205})
cat_only_105 = cat_only_trigger.derive("cat_only_105", cls_dict={"trigger_id": 105})
cat_only_405 = cat_only_trigger.derive("cat_only_405", cls_dict={"trigger_id": 405})
cat_only_304 = cat_only_trigger.derive("cat_only_304", cls_dict={"trigger_id": 304})
cat_only_507 = cat_only_trigger.derive("cat_only_507", cls_dict={"trigger_id": 507})
cat_only_701 = cat_only_trigger.derive("cat_only_701", cls_dict={"trigger_id": 701})
cat_only_602 = cat_only_trigger.derive("cat_only_602", cls_dict={"trigger_id": 602})
cat_only_605 = cat_only_trigger.derive("cat_only_605", cls_dict={"trigger_id": 605})
cat_only_606 = cat_only_trigger.derive("cat_only_606", cls_dict={"trigger_id": 606})
cat_only_608 = cat_only_trigger.derive("cat_only_608", cls_dict={"trigger_id": 608})
cat_only_609 = cat_only_trigger.derive("cat_only_609", cls_dict={"trigger_id": 609})

# primary_<id>: this study trigger fired, no study trigger of higher priority fired
cat_primary_205 = cat_primary_trigger.derive("cat_primary_205", cls_dict={"trigger_id": 205})
cat_primary_105 = cat_primary_trigger.derive("cat_primary_105", cls_dict={"trigger_id": 105})
cat_primary_405 = cat_primary_trigger.derive("cat_primary_405", cls_dict={"trigger_id": 405})
cat_primary_304 = cat_primary_trigger.derive("cat_primary_304", cls_dict={"trigger_id": 304})
cat_primary_507 = cat_primary_trigger.derive("cat_primary_507", cls_dict={"trigger_id": 507})
cat_primary_701 = cat_primary_trigger.derive("cat_primary_701", cls_dict={"trigger_id": 701})
cat_primary_602 = cat_primary_trigger.derive("cat_primary_602", cls_dict={"trigger_id": 602})
cat_primary_605 = cat_primary_trigger.derive("cat_primary_605", cls_dict={"trigger_id": 605})
cat_primary_606 = cat_primary_trigger.derive("cat_primary_606", cls_dict={"trigger_id": 606})
cat_primary_608 = cat_primary_trigger.derive("cat_primary_608", cls_dict={"trigger_id": 608})
cat_primary_609 = cat_primary_trigger.derive("cat_primary_609", cls_dict={"trigger_id": 609})

# only_group<N>: at least one trigger of group N fired and no trigger of any other group
cat_only_group1 = cat_only_group.derive("cat_only_group1", cls_dict={"group": 1})
cat_only_group2 = cat_only_group.derive("cat_only_group2", cls_dict={"group": 2})
cat_only_group3 = cat_only_group.derive("cat_only_group3", cls_dict={"group": 3})
cat_only_group4 = cat_only_group.derive("cat_only_group4", cls_dict={"group": 4})
cat_only_group5 = cat_only_group.derive("cat_only_group5", cls_dict={"group": 5})

# group1_ge2: at least two triggers of group 1 fired
cat_group1_ge2 = cat_group_min.derive("cat_group1_ge2", cls_dict={"group": 1, "min_fired": 2})

# group combinations: at least one trigger of every listed group fired
# group 2 has no study trigger in 2023, so the combinations skip it
cat_group1 = cat_groups_all.derive("cat_group1", cls_dict={"groups": (1,)})
cat_group1_group3 = cat_groups_all.derive("cat_group1_group3", cls_dict={"groups": (1, 3)})
cat_group1_group3_group4 = cat_groups_all.derive("cat_group1_group3_group4", cls_dict={"groups": (1, 3, 4)})
cat_group1_group3_group4_group5 = cat_groups_all.derive(
    "cat_group1_group3_group4_group5", cls_dict={"groups": (1, 3, 4, 5)},
)


#
# kinematic regions
#

@categorizer(uses={"event"})
def cat_incl(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # fully inclusive selection
    return events, all_true(events)


@categorizer(uses={jet_multiplicity})
def cat_eq0j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 0


@categorizer(uses={jet_multiplicity})
def cat_eq1j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 1


@categorizer(uses={jet_multiplicity})
def cat_eq2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 2


@categorizer(uses={jet_multiplicity})
def cat_eq3j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 3


@categorizer(uses={jet_multiplicity})
def cat_eq4j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 4


@categorizer(uses={jet_multiplicity})
def cat_eq5j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) == 5


@categorizer(uses={jet_multiplicity})
def cat_ge2j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 2


@categorizer(uses={jet_multiplicity})
def cat_ge3j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 3


@categorizer(uses={jet_multiplicity})
def cat_ge4j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 4


@categorizer(uses={jet_multiplicity})
def cat_ge5j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 5


@categorizer(uses={jet_multiplicity})
def cat_ge6j(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[jet_multiplicity](events, **kwargs) >= 6


@categorizer(uses={bjet_multiplicity})
def cat_eq0b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 0


@categorizer(uses={bjet_multiplicity})
def cat_eq1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 1


@categorizer(uses={bjet_multiplicity})
def cat_eq2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 2


@categorizer(uses={bjet_multiplicity})
def cat_eq3b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) == 3


@categorizer(uses={bjet_multiplicity})
def cat_ge0b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 0


@categorizer(uses={bjet_multiplicity})
def cat_ge1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 1


@categorizer(uses={bjet_multiplicity})
def cat_ge2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 2


@categorizer(uses={bjet_multiplicity})
def cat_ge3b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, self[bjet_multiplicity](events, **kwargs) >= 3


@categorizer(uses={"HHBJet.{mass,pt,eta,phi}"})
def di_bjet_mass_window(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    di_bjet_mass = events.HHBJet.sum(axis=1).mass
    mask = (
        (di_bjet_mass >= 40) &
        (di_bjet_mass <= 270)
    )
    return events, mask


@categorizer(uses={"Tau.{mass,pt,eta,phi}"})
def di_tau_mass_window(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leptons = [events.Electron * 1, events.Muon * 1, events.Tau * 1]
    di_tau_mass = ak_concatenate_safe(leptons, axis=1)[:, :2].sum(axis=1).mass
    mask = (
        (di_tau_mass >= 15) &
        (di_tau_mass <= 130)
    )
    return events, mask


@categorizer(
    uses={di_bjet_mass_window, di_tau_mass_window, hhbjet_multiplicity},
)
def cat_res1b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    events, bjet_mass_mask = self[di_bjet_mass_window](events, **kwargs)
    mask = (
        (self[hhbjet_multiplicity](events, **kwargs) == 1) &
        tau_mass_mask &
        bjet_mass_mask
    )
    return events, mask


@categorizer(
    uses={di_bjet_mass_window, di_tau_mass_window, hhbjet_multiplicity},
)
def cat_res2b(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, tau_mass_mask = self[di_tau_mass_window](events, **kwargs)
    events, bjet_mass_mask = self[di_bjet_mass_window](events, **kwargs)
    mask = (
        (self[hhbjet_multiplicity](events, **kwargs) >= 2) &
        tau_mass_mask &
        bjet_mass_mask
    )
    return events, mask


@categorizer(
    uses={
        cat_res1b, cat_res2b, di_tau_mass_window,
        "FatJet.{pt,phi,msoftdrop,particleNet_XbbVsQCD,mass,eta}",
    },
)
def cat_boosted(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # exclude res1b or res2b, and exactly one selected fat jet that should also pass a tighter pt cut
    mask = (
        ~self[cat_res1b](events, **kwargs)[1] &
        ~self[cat_res2b](events, **kwargs)[1] &
        (ak.num(events.FatJet, axis=1) == 1) &
        (ak.sum(events.FatJet.pt > 350, axis=1) == 1) &
        (ak.sum(events.FatJet.particleNet_XbbVsQCD > 0.75, axis=1) >= 1) &  # wp defined by cclub
        self[di_tau_mass_window](events, **kwargs)[1] &
        ak.any(events.FatJet.msoftdrop >= 30, axis=1) &
        ak.any(events.FatJet.msoftdrop <= 450, axis=1)
    )
    return events, mask


@categorizer(uses={"vbf_dnn_moe_hh_vbf"})
def cat_vbf_0p5(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    return events, (events.vbf_dnn_moe_hh_vbf > 0.5)


@categorizer(uses={cat_res1b, cat_vbf_0p5})
def cat_res1b_novbf(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, res1b_mask = self[cat_res1b](events, **kwargs)
    events, vbf_mask = self[cat_vbf_0p5](events, **kwargs)
    return events, (res1b_mask & ~vbf_mask)


@categorizer(uses={cat_res2b, cat_vbf_0p5})
def cat_res2b_novbf(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, res2b_mask = self[cat_res2b](events, **kwargs)
    events, vbf_mask = self[cat_vbf_0p5](events, **kwargs)
    return events, (res2b_mask & ~vbf_mask)


@categorizer(uses={cat_boosted, cat_vbf_0p5})
def cat_boosted_novbf(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    events, boosted_mask = self[cat_boosted](events, **kwargs)
    events, vbf_mask = self[cat_vbf_0p5](events, **kwargs)
    return events, (boosted_mask & ~vbf_mask)


@categorizer(
    uses={"FatJet.{pt,phi,msoftdrop,particleNet_XbbVsQCD,mass,eta}"},
)
def cat_boosted_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        (ak.num(events.FatJet, axis=1) == 1) &
        ak.any(events.FatJet.msoftdrop >= 80, axis=1) &
        ak.any(events.FatJet.msoftdrop <= 170, axis=1) &
        (ak.sum(events.FatJet.pt > 300, axis=1) == 1) &
        (ak.sum(events.FatJet.particleNet_XbbVsQCD > 0.75, axis=1) >= 1)  # wp defined by cclub
    )
    return events, mask


@categorizer(uses={cat_boosted_cc, "vbf_dnn_moe_hh_vbf"})
def cat_vbf_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        ~self[cat_boosted_cc](events, **kwargs)[1] &
        (events.vbf_dnn_moe_hh_vbf > 0.5)
    )
    return events, mask


@categorizer(
    uses={"{Electron,Muon,Tau,HHBJet}.{mass,pt,eta,phi}", "reg_dnn_moe_nu{1,2}_p{x,y,z}", "channel_id"},
    # channel dependent mean and std values for the regressed tautau system mass
    tautau_window={
        # regressed values
        "etau": (111.0, 38.0),
        "mutau": (110.0, 45.0),
        "tautau": (130.0, 36.0),
        # values actually used for svfit based tautau system
        "ee": (119.0, 57.0),
        "mumu": (116.0, 61.0),
        "emu": (116.0, 61.0),  # values from mumu
    },
    # channel dependent mean and std values for the bb (from HHBJet) system mass
    bb_window={
        "etau": (118.0, 225.0),
        "mutau": (118.0, 215.0),
        "tautau": (125.0, 222.0),
        "ee": (109.0, 232.0),
        "mumu": (114.0, 228.0),
        "emu": (114.0, 228.0),  # values from mumu
    },
)
def cat_hh_reg_mass_window_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # build bb system
    events = attach_coffea_behavior(events, {"HHBJet": "Jet"})
    bb = events.HHBJet[:, :2].sum(axis=1)
    # build visible tautau system
    tautau_vis = stack_lvectors([events.Electron, events.Muon, events.Tau])[..., :2].sum(axis=1)
    # get regressed neutrinos and rotate them
    ref_phi = tautau_vis.phi
    nu1 = create_lvector_xyz(
        *rotate_px_py(events.reg_dnn_moe_nu1_px, events.reg_dnn_moe_nu1_py, ref_phi),
        events.reg_dnn_moe_nu1_pz,
    )
    nu2 = create_lvector_xyz(
        *rotate_px_py(events.reg_dnn_moe_nu2_px, events.reg_dnn_moe_nu2_py, ref_phi),
        events.reg_dnn_moe_nu2_pz,
    )
    # build regressed tautau system
    tautau_reg = stack_lvectors([nu1, nu2, tautau_vis]).sum(axis=-1)
    # get masses
    m_bb = bb.mass
    m_tautau = tautau_reg.mass

    # apply channel dependent mass window cuts
    def mass_window(channel_name: str) -> ak.Array:
        channel_mask = events.channel_id == self.config_inst.get_channel(channel_name).id
        chi_tautau = (m_tautau - self.tautau_window[channel_name][0]) / self.tautau_window[channel_name][1]
        chi_bb = (m_bb - self.bb_window[channel_name][0]) / self.bb_window[channel_name][1]
        return channel_mask & ((chi_tautau**2 + chi_bb**2) <= 1.0)

    mass_mask = (
        full_like(events.event, False, dtype=bool) |
        mass_window("etau") |
        mass_window("mutau") |
        mass_window("tautau") |
        mass_window("ee") |
        mass_window("mumu") |
        mass_window("emu")
    )
    return events, mass_mask


@categorizer(
    uses={cat_boosted_cc, cat_vbf_cc, hhbjet_multiplicity, cat_hh_reg_mass_window_cc},
    n_btags=None,  # needs to be set
    n_btags_op=None,  # needs to be set
    reject_boosted=True,
    reject_vbf=True,
)
def _cat_res_cc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    mask = (
        self[cat_hh_reg_mass_window_cc](events, **kwargs)[1] &
        (~self[cat_boosted_cc](events, **kwargs)[1] if self.reject_boosted else True) &
        (~self[cat_vbf_cc](events, **kwargs)[1] if self.reject_vbf else True) &
        (self.n_btags_op(self[hhbjet_multiplicity](events, **kwargs), self.n_btags))
    )
    return events, mask


cat_res1b_cc = _cat_res_cc.derive("cat_res1b_cc", cls_dict={"n_btags": 1, "n_btags_op": operator.eq})
cat_res2b_cc = _cat_res_cc.derive("cat_res2b_cc", cls_dict={"n_btags": 2, "n_btags_op": operator.ge})
cat_res1b_inclvbf_cc = cat_res1b_cc.derive("cat_res1b_inclvbf_cc", cls_dict={"reject_vbf": False})
cat_res2b_inclvbf_cc = cat_res2b_cc.derive("cat_res2b_inclvbf_cc", cls_dict={"reject_vbf": False})


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_mll40(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leps = ak_concatenate_safe([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    return events, leps.sum(axis=1).mass > 40.0


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_dy(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # e/mu driven DY region: mll > 40 and met < 30 (to supress tau decays into e/mu)
    leps = ak_concatenate_safe([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]
    mask = (
        (leps.sum(axis=1).mass > 40) &
        (events[self.config_inst.x.met_name].pt < 30)
    )
    return events, mask


@cat_dy.init
def cat_dy_init(self: Categorizer, **kwargs) -> None:
    super(cat_dy, self).init_func(**kwargs)
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")


@categorizer(
    uses={
        "{Electron,Muon,Tau}.{pt,eta,phi,mass}",
        MET_COLUMN("{pt,phi}"),
    },
)
def cat_dyc(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    leps = ak_concatenate_safe([events.Electron * 1, events.Muon * 1, events.Tau * 1], axis=1)[:, :2]

    mask_cclub = (
        (leps.sum(axis=1).mass >= 70) &
        (leps.sum(axis=1).mass <= 110) &
        (events[self.config_inst.x.met_name].pt < 45)
    )

    return events, mask_cclub


@categorizer(uses={"{Electron,Muon,Tau}.{pt,eta,phi,mass}"})
def cat_tt(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
    # tt region: met > 30 (due to neutrino presence in leptonic w decays)
    mask = events[self.config_inst.x.met_name].pt > 30
    return events, mask


@cat_tt.init
def cat_tt_init(self: Categorizer, **kwargs) -> None:
    super(cat_tt, self).init_func(**kwargs)
    self.uses.add(f"{self.config_inst.x.met_name}.{{pt,phi}}")
