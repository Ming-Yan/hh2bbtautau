# coding: utf-8

"""
Per-trigger trigger scale factor sources.

The trigger weight producers in :py:mod:`hbt.production.trigger_sf` multiply the efficiencies and scale factors of all
trigger legs of an event into one weight per channel. The producer in this module instead stores the ingredients of
the trigger scale factors separately for every trigger, e.g. the data and simulation efficiencies of the electron leg
of the single electron trigger, so that the trigger scale factors of individual triggers can be studied in dedicated
phase spaces. The ingredients are not combined here, the combination (e.g. the correlated ditrigger formula) is left
to the analysis.

For every trigger and every (channel, source) pair a column ``trigsrc_<trigger_id>_<channel>_<source>`` is produced.
Its value is the source value for events that are in the channel and in which the trigger was matched (see
``matched_trigger_ids``), and 1.0 otherwise, so that the columns can be multiplied freely. Sources are either
efficiencies (``*_data``, ``*_mc``) or scale factors (``*_sf``) as produced by the trigger weight producers.

Additionally, ``trigsrc_<trigger_id>_etau_vbf_sf_outside_corr_region`` is produced for the cross_e_vbf trigger. It is
True for events in which the vbf electron scale factor is not applied, because its input variables are outside of the
range covered by the correction (2023 only), in which case the scale factor is set to 1.0.

The producer is only active when the config auxiliary entry ``store_trigger_sf_sources`` is set (2023 postBPix).
"""

from __future__ import annotations

from collections.abc import Callable

from columnflow.production import Producer, producer
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from hbt.config.util import Trigger
from hbt.production.trigger_sf import trigger_weight, mask_corrector_inputs_vbf_e_jet

ak = maybe_import("awkward")
np = maybe_import("numpy")


# name of the source flag for the cross_e_vbf trigger, see module docstring
VBF_E_OUTSIDE_REGION_SOURCE = "vbf_sf_outside_corr_region"


def _column(name: str) -> Callable:
    """
    Returns a getter for the per-event values of an event-level column.
    """
    return lambda events: events[name]


def _object_product(name: str) -> Callable:
    """
    Returns a getter for the per-event product over the objects of an object-level column.
    """
    return lambda events: ak.prod(events[name], axis=1, mask_identity=False)


def get_trigger_sf_source_specs(trigger: Trigger) -> list[tuple[str, str, Callable]]:
    """
    Returns the trigger scale factor sources of a *trigger* as a list of ``(channel_name, source_name, getter)``
    tuples. The getter maps the events to the per-event values of the source, before the matching and channel masks
    are applied. The sources are assigned based on the trigger tags, following the leg structure used in
    :py:mod:`hbt.production.trigger_sf`.
    """
    specs = []

    # single lepton triggers
    if trigger.has_tag("single_e"):
        specs += [
            ("etau", "e_data", _column("single_trigger_e_data_effs")),
            ("etau", "e_mc", _column("single_trigger_e_mc_effs")),
            ("ee", "e_sf", _column("ee_trigger_weight")),
            ("emu", "e_sf", _column("emu_e_trigger_weight")),
        ]
    if trigger.has_tag("single_mu"):
        specs += [
            ("mutau", "mu_data", _column("single_trigger_mu_data_effs")),
            ("mutau", "mu_mc", _column("single_trigger_mu_mc_effs")),
            ("mumu", "mu_sf", _column("mumu_trigger_weight")),
            ("emu", "mu_sf", _column("emu_mu_trigger_weight")),
        ]

    # cross lepton-tau triggers
    if trigger.has_tag("cross_e_tau"):
        specs += [
            ("etau", "e_data", _column("cross_trigger_e_data_effs")),
            ("etau", "e_mc", _column("cross_trigger_e_mc_effs")),
            ("etau", "tau_data", _object_product("tau_trigger_eff_data_etau")),
            ("etau", "tau_mc", _object_product("tau_trigger_eff_mc_etau")),
        ]
    if trigger.has_tag("cross_mu_tau"):
        specs += [
            ("mutau", "mu_data", _column("cross_trigger_mu_data_effs")),
            ("mutau", "mu_mc", _column("cross_trigger_mu_mc_effs")),
            ("mutau", "tau_data", _object_product("tau_trigger_eff_data_mutau")),
            ("mutau", "tau_mc", _object_product("tau_trigger_eff_mc_mutau")),
        ]

    # tau-tau triggers
    if trigger.has_tag("cross_tau_tau"):
        specs += [
            ("tautau", "tau_data", _object_product("tau_trigger_eff_data_tautau")),
            ("tautau", "tau_mc", _object_product("tau_trigger_eff_mc_tautau")),
        ]
    if trigger.has_tag("cross_tau_tau_jet"):
        specs += [
            ("tautau", "tau_data", _object_product("tau_trigger_eff_data_tautaujet")),
            ("tautau", "tau_mc", _object_product("tau_trigger_eff_mc_tautaujet")),
            ("tautau", "jet_data", _object_product("jet_trigger_eff_data")),
            ("tautau", "jet_mc", _object_product("jet_trigger_eff_mc")),
        ]

    # vbf triggers (scale factors only)
        
    if trigger.has_tag("cross_e_vbf"):
        specs += [
            ("etau", "vbf_sf", _column("vbf_e_jet_trigger_sf")),
        ]
    if trigger.has_tag("cross_mu_vbf"):
        specs += [
            ("mutau", "vbf_sf", _column("vbf_mu_jet_trigger_sf")),
        ]
    if trigger.has_tag("cross_vbf"):
        specs += [
            ("etau", "vbf_sf", _column("vbf_incl_jet_trigger_sf_etau")),
            ("tautau", "vbf_sf", _column("vbf_incl_jet_trigger_sf_tautau")),
        ]
    if trigger.has_tag("cross_tau_tau_vbf"):
        specs += [
            ("tautau", "tau_vbf_sf", _object_product("tau_trigger_sf_tautauvbf")),
            ("tautau", "vbf_sf", _column("vbf_ditau_jet_trigger_sf")),
        ]
    if trigger.has_tag("cross_tau_vbf"):
        specs += [
            ("tautau", "tau_vbf_sf", _object_product("tau_trigger_sf_tauvbf")),
            ("tautau", "vbf_sf", _column("vbf_tau_jet_trigger_sf")),
        ]

    return specs


def trigger_sf_source_column(trigger: Trigger, channel_name: str, source_name: str) -> str:
    """
    Returns the name of the output column for a trigger scale factor source.
    """
    return f"trigsrc_{trigger.id}_{channel_name}_{source_name}"


def get_vbf_e_outside_corr_region(producer_inst: Producer, events: ak.Array) -> ak.Array:
    """
    Returns a per-event mask that is True where the vbf electron scale factor is not applied because its input
    variables are outside of the range covered by the correction. The inputs are built in the same way as in
    :py:class:`hbt.production.jet.vbfjet_trigger_efficiencies`.
    """
    vbfjets = events.VBFJet[events.VBFJet.pt > 0]
    vbfjet1 = ak.firsts(vbfjets[:, :1], axis=1)
    vbfjet2 = ak.firsts(vbfjets[:, 1:2], axis=1)
    electrons = events.Electron[events.Electron.pt > 0]
    inputs = {
        "lep_pt": ak.firsts(electrons[:, :1].pt, axis=1),
        "mjj": (vbfjet1 + vbfjet2).mass,
    }
    outside_mask = mask_corrector_inputs_vbf_e_jet(producer_inst, inputs)
    if outside_mask is None:
        return ak.zeros_like(events.event, dtype=bool)
    return ak.fill_none(outside_mask, False)


@producer(
    uses={
        # the trigger weights are created first, their intermediate ingredients are read from the events
        trigger_weight,
        "channel_id", "matched_trigger_ids", "Electron.pt", "VBFJet.{pt,eta,phi,mass}",
    },
    mc_only=True,
)
def trigger_sf_sources(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Produces the individual trigger scale factor sources for every trigger of the config, see the module docstring.
    """
    # only active when requested in the config, consistent with the registered output columns
    if not self.config_inst.has_aux("store_trigger_sf_sources"):
        return events

    for trigger in self.config_inst.x.triggers:
        trigger_matched = np.any(events.matched_trigger_ids == trigger.id, axis=-1)

        for channel_name, source_name, getter in get_trigger_sf_source_specs(trigger):
            channel_mask = events.channel_id == self.config_inst.get_channel(channel_name).id
            values = ak.where(channel_mask & trigger_matched, getter(events), 1.0)
            events = set_ak_column(
                events,
                trigger_sf_source_column(trigger, channel_name, source_name),
                values,
                value_type=np.float32,
            )

        if trigger.has_tag("cross_e_vbf"):
            outside_flag = (
                (events.channel_id == self.config_inst.get_channel("etau").id) &
                trigger_matched &
                get_vbf_e_outside_corr_region(self, events)
            )
            events = set_ak_column(
                events,
                trigger_sf_source_column(trigger, "etau", VBF_E_OUTSIDE_REGION_SOURCE),
                outside_flag,
                value_type=bool,
            )

    return events


@trigger_sf_sources.init
def trigger_sf_sources_init(self: Producer, **kwargs) -> None:
    super(trigger_sf_sources, self).init_func(**kwargs)

    # the produced columns depend on the triggers of the config, so they are registered here
    if not self.config_inst.has_aux("store_trigger_sf_sources"):
        return

    for trigger in self.config_inst.x.triggers:
        for channel_name, source_name, _ in get_trigger_sf_source_specs(trigger):
            self.produces.add(trigger_sf_source_column(trigger, channel_name, source_name))
        if trigger.has_tag("cross_e_vbf"):
            self.produces.add(trigger_sf_source_column(trigger, "etau", VBF_E_OUTSIDE_REGION_SOURCE))
