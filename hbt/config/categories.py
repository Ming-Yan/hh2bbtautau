# coding: utf-8

"""
Definition of categories.
"""

from __future__ import annotations

import functools

import law
import order as od

from columnflow.config_util import (
    add_category, create_category_combinations, CategoryGroup, CategoryIDCache,
)
from columnflow.types import Any


logger = law.logger.get_logger(__name__)


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    # root category (-1 has special meaning in cutflow)
    root_cat = add_category(config, name="all", id=-1, selection="cat_all", label="")

    # helper to assign categories to config with cached ids
    id_cache_location = law.util.rel_path(__file__, "data", f"category_ids_{config.name}.json")
    id_cache = CategoryIDCache(id_cache_location).open()
    def _add_category(**kwargs) -> od.Category:
        kwargs.setdefault("id", id_cache)
        return add_category(config, parent=root_cat, **kwargs)

    # lepton channels
    _add_category(name="etau", id=1, selection="cat_etau", label=config.channels.n.etau.label)
    _add_category(name="mutau", id=2, selection="cat_mutau", label=config.channels.n.mutau.label)
    _add_category(name="tautau", id=3, selection="cat_tautau", label=config.channels.n.tautau.label)
    _add_category(name="ee", id=4, selection="cat_ee", label=config.channels.n.ee.label)
    _add_category(name="mumu", id=5, selection="cat_mumu", label=config.channels.n.mumu.label)
    _add_category(name="emu", id=6, selection="cat_emu", label=config.channels.n.emu.label)

    # qcd regions
    _add_category(name="os", selection="cat_os", label="OS", tags={"os"})
    _add_category(name="ss", selection="cat_ss", label="SS", tags={"ss"})
    _add_category(name="iso", selection="cat_iso", label=r"iso", tags={"iso"})
    _add_category(name="noniso", selection="cat_noniso", label=r"non-iso", tags={"noniso"})  # noqa: E501

    # kinematic categories
    _add_category(name="incl", selection="cat_incl", label="inclusive")
    _add_category(name="eq0j", selection="cat_eq0j", label="0 jets")
    _add_category(name="eq1j", selection="cat_eq1j", label="1 jet")
    _add_category(name="eq2j", selection="cat_eq2j", label="2 jets")
    _add_category(name="ge2j", selection="cat_ge2j", label=r"$\geq$2 jets")
    _add_category(name="eq3j", selection="cat_eq3j", label="3 jets")
    _add_category(name="eq4j", selection="cat_eq4j", label="4 jets")
    _add_category(name="ge4j", selection="cat_ge4j", label=r"$\geq$4 jets")
    _add_category(name="eq5j", selection="cat_eq5j", label="5 jets")
    _add_category(name="ge5j", selection="cat_ge5j", label=r"$\geq$5 jets")
    _add_category(name="ge6j", selection="cat_ge6j", label=r"$\geq$6 jets")
    _add_category(name="eq0b", selection="cat_eq0b", label="0 b-tags")
    _add_category(name="ge0b", selection="cat_ge0b", label="")
    _add_category(name="eq1b", selection="cat_eq1b", label="1 b-tag")
    _add_category(name="eq2b", selection="cat_eq2b", label="2 b-tags")
    _add_category(name="ge2b", selection="cat_ge2b", label=r"$\geq$2 b-tags")
    _add_category(name="ge3b", selection="cat_ge3b", label=r"$\geq$3 b-tags")
    _add_category(name="dy", selection="cat_dy", label="DY enriched")
    _add_category(name="dyc", selection="cat_dyc", label="DY region")
    _add_category(name="dy_st", selection=["cat_dy", "cat_single_triggered"], label="DY enriched, ST")
    _add_category(name="tt", selection="cat_tt", label=r"$t\bar{t}$ enriched")
    _add_category(name="mll40", selection="cat_mll40", label=r"$m_{ll} > 40$")

    _add_category(name="res1b", selection="cat_res1b", label="res1b")
    _add_category(name="res2b", selection="cat_res2b", label="res2b")
    _add_category(name="boosted", selection="cat_boosted", label="boosted")
    # VBF cleaned
    _add_category(name="res1b_novbf", selection="cat_res1b_novbf", label="res1b, VBF cleaned", tags={"skip_cutflow"})  # noqa: E501
    _add_category(name="res2b_novbf", selection="cat_res2b_novbf", label="res2b, VBF cleaned", tags={"skip_cutflow"})  # noqa: E501
    _add_category(name="boosted_novbf", selection="cat_boosted_novbf", label="boosted, VBF cleaned", tags={"skip_cutflow"})  # noqa: E501
    # new cclub based categories
    _add_category(name="res1b_cc", selection="cat_res1b_cc", label="Resolved, ggF, 1 b-tag", tags={"skip_cutflow"})
    _add_category(name="res2b_cc", selection="cat_res2b_cc", label=r"Resolved, ggF, $\geq$2 b-tags", tags={"skip_cutflow"})  # noqa: E501
    _add_category(name="vbf_cc", selection="cat_vbf_cc", label="Resolved, VBF", tags={"skip_cutflow"})
    _add_category(name="boosted_cc", selection="cat_boosted_cc", label="Boosted", tags={"skip_cutflow"})
    _add_category(name="res1b_inclvbf_cc", selection="cat_res1b_inclvbf_cc", label="Resolved, ggF, 1 b-tag", tags={"skip_cutflow"})  # noqa: E501
    _add_category(name="res2b_inclvbf_cc", selection="cat_res2b_inclvbf_cc", label=r"Resolved, ggF, $\geq$2 b-tags", tags={"skip_cutflow"})  # noqa: E501
    ### fix_pnet
    _add_category(name="res1b_invH_cc", selection="cat_invHres1b_cc", label="Resolved, ggF,inv $m_H$, 1 b-tag", tags={"skip_cutflow"})
    _add_category(name="res2b_invH_cc", selection="cat_invHres2b_cc", label=r"Resolved, ggF, inv $m_H$,$\geq$2 b-tags", tags={"skip_cutflow"})  # noqa: E501
    _add_category(name="invH2b",selection="cat_invH_2b", label="~ $m_{\tau\tau}$+$m_{bb}$ mass, 2b ")
    _add_category(name="invH1b",selection="cat_invH_1b", label="~ $m_{\tau\tau}$+$m_{bb}$ mass, 1b")
    _add_category(name="invb1b",selection="cat_invb_1b", label="~$m_{bb}$  mass, 1b ")
    _add_category(name="invt1b",selection="cat_invtau_1b", label="~$m_{\\tau\\tau}$ mass, 1b")
    _add_category(name="mll501b",selection="cat_mll50_1b", label="$m_{\\tau\\tau}$>50, 1b")
    _add_category(name="2b",selection="cat_2b", label="2b ")
    _add_category(name="1b",selection="cat_1b", label="1b")
    _add_category(name="mixed",selection="cat_dmorigin_mixed", label="PNet recovered 1 taus")
    _add_category(name="pnet",selection="cat_dmorigin_pnet", label="PNet recovered 2 taus")
    _add_category(name="hps",selection="cat_dmorigin_orig", label="HPS taus")
    # _add_category(name="lep35",selection="cat_dilep35", label="lep pT>35 GeV")
    ## test
    DM_PAIRS = [
    (0, 0), (0, 1), (0, 10), (0, 11),
    (1, 1), (1, 10), (1, 11),
    (10, 10), (10, 11),
    (11, 11),
    ]
    DM_LABELS = {0: "1h", 1: "1h1$\\pi_0$", 10: "3h", 11: "3h1$\\pi_0$"}

    for i, (dm1, dm2) in enumerate(DM_PAIRS):
        _add_category(
            name=f"dm{dm1}_dm{dm2}",
            selection=f"cat_tau_dm{dm1}_dm{dm2}",
            label=f"dm{dm1}_dm{dm2}",
        )
    # _add_category(name="cross_vbf",selection="cat_cross_vbf",label = "cross vbf", tags={"skip_cutflow"})
    # _add_category(name="cross_tautaujet",selection="cat_cross_tautaujet",label = "cross tautau jet", tags={"skip_cutflow"})
    # _add_category(name="cross_tautau",selection="cat_cross_tautau",label = "cross tautau", tags={"skip_cutflow"})
    #
    # build groups
    #

    def name_fn(categories: dict[str, od.Category]) -> str:
        return "__".join(cat.name for cat in categories.values() if cat)

    def kwargs_fn(categories: dict[str, od.Category], add_qcd_group: bool = True) -> dict[str, Any]:
        # build auxiliary information
        aux = {}
        if add_qcd_group and not ({"sign", "tau2"} - set(categories.keys())):
            aux["qcd_group"] = name_fn({
                name: cat for name, cat in categories.items()
                if name not in {"sign", "tau2"}
            })
        # return the desired kwargs
        return {
            # just increment the category id
            "id": id_cache,
            # join all tags
            "tags": set.union(*[cat.tags for cat in categories.values() if cat]),
            # auxiliary information
            "aux": aux,
            # label
            "label": ", ".join([
                cat.label
                for cat in categories.values()
                if cat.label and cat.name != "os" and cat.name != "iso"  # os and iso are the defaults
            ]) or None,
        }
    dm= [f"dm{dm1}_dm{dm2}" for dm1, dm2 in DM_PAIRS]
    
    # main analysis categories
    main_categories = {
        # channels first
        "channel": CategoryGroup(["etau", "mutau", "tautau"], is_complete=False, has_overlap=False),
        # kinematic regions in the middle (to be extended)
        "kin": CategoryGroup(
            [
                "incl", "res1b", "res2b", "boosted", "res1b_novbf", "res2b_novbf", "boosted_novbf",
                "res1b_cc", "res2b_cc", "vbf_cc", "boosted_cc", "res1b_inclvbf_cc", "res2b_inclvbf_cc",
                "mixed","pnet","hps"
                # "invH2b","invH1b","1b","2b","res1b_invH_cc","res2b_invH_cc","invt1b","invb1b","mll501b"# invmass
                # *dm,"lep35"
                # "eq0b", "eq1b", "eq2b", "ge2b", "ge3b",#prod28

            #    "taudm0","taudm2","taudm20","taudm22","taudm1","taudm10","taudm110","taudm101","taudm12","taudm21",
            #    "cross_tautau","cross_tautaujet","cross_vbf" #
            ],
            is_complete=True,
            has_overlap=True,
        ),
        "nb":CategoryGroup(["eq0b", "eq1b", "eq2b", "ge0b", "ge2b"], is_complete=True, has_overlap=True),
        # qcd regions last
        "sign": CategoryGroup(["os", "ss"], is_complete=True, has_overlap=False),
        "tau2": CategoryGroup(["iso", "noniso"], is_complete=True, has_overlap=False),
    }
    def skip_fn(categories: dict[str, od.Category]) -> bool:
        ch, dm = categories.get("channel"), categories.get("kin")
        # drop any dm combination that isn't in tautau
        if ch=="tautau":return False
        if "dm" in dm.name and ch.name != "tautau":return True
        if "mixed" in dm.name and ch.name != "tautau":return True
        else: return False

    create_category_combinations(
        config=config,
        categories=main_categories,
        name_fn=name_fn,
        parent_mode="safe",
        kwargs_fn=functools.partial(kwargs_fn, add_qcd_group=True),
        skip_existing=False,
        skip_fn=skip_fn,
    )
       # _add_category(
    #     name="mutau__incl",
    #     label= r"$\mu\tau$, inclusive (no iso,charge selection)",
    #     categories=[
    #         config.get_category("mutau__incl__os__iso"),
    #         config.get_category("mutau__incl__os__noniso"),
    #         config.get_category("mutau__incl__ss__iso"),
    #         config.get_category("mutau__incl__ss__noniso"),
    #     ],
    # ))



    # control categories
    control_categories = {
        # channels
        "channel": CategoryGroup(["ee", "mumu", "emu"], is_complete=False, has_overlap=False),
        # kinematic regions
        "kin": CategoryGroup(["incl", "dy", "dyc", "tt", "dy_st", "mll40"], is_complete=True, has_overlap=True),
        "jets": CategoryGroup(["eq2j", "eq3j", "eq4j", "ge2j", "ge4j", "eq5j", "ge6j"], is_complete=True, has_overlap=True),  # noqa: E501
        "tags": CategoryGroup(["eq0b", "eq1b", "eq2b", "ge0b", "ge2b"], is_complete=True, has_overlap=True),
        # relative sign
        "sign": CategoryGroup(["os"], is_complete=False, has_overlap=False),
    }

    def skip_fn_ctrl(categories: dict[str, od.Category]) -> bool:
        if "channel" not in categories or "kin" not in categories:
            return False
        ch_cat = categories["channel"]
        kin_cat = categories["kin"]
        # skip dy in emu
        if kin_cat.name.startswith("dy") and ch_cat.name == "emu":
            return True
        # skip tt in ee/mumu
        if kin_cat.name == "tt" and ch_cat.name in {"ee", "mumu"}:
            return True
        return False

    create_category_combinations(
        config=config,
        categories=control_categories,
        name_fn=name_fn,
        parent_mode="safe",
        kwargs_fn=functools.partial(kwargs_fn, add_qcd_group=False),
        skip_existing=False,
        skip_fn=skip_fn_ctrl,
    )

    # additional combined categories
    # _add_category(
    #     name="etau__incl",
    #     label= r"e$\tau$, inclusive (no iso,charge selection)",
    #     categories=[
    #         config.get_category("etau__incl__os__iso"),
    #         config.get_category("etau__incl__os__noniso"),
    #         config.get_category("etau__incl__ss__iso"),
    #         config.get_category("etau__incl__ss__noniso"),
    #     ],
    # )
    # _add_category(
    #     name="mutau__incl",
    #     label= r"$\mu\tau$, inclusive (no iso,charge selection)",
    #     categories=[
    #         config.get_category("mutau__incl__os__iso"),
    #         config.get_category("mutau__incl__os__noniso"),
    #         config.get_category("mutau__incl__ss__iso"),
    #         config.get_category("mutau__incl__ss__noniso"),
    #     ],
    # )
    # _add_category(
    #     name="tautau__incl",
    #     label= r"$\tau\tau$, inclusive (no iso,charge selection)",
    #     categories=[
    #         config.get_category("tautau__incl__os__iso"),
    #         config.get_category("tautau__incl__os__noniso"),
    #         config.get_category("tautau__incl__ss__iso"),
    #         config.get_category("tautau__incl__ss__noniso"),
    #     ],
    # )
    # _add_category(
    #     name="incl__os__iso",
    #     label=r"$e\tau/\mu\tau/\tau\tau$, inclusive",
    #     categories=[
    #         config.get_category("etau__incl__os__iso"),
    #         config.get_category("mutau__incl__os__iso"),
    #         config.get_category("tautau__incl__os__iso"),
    #     ],
    # )

    id_cache.close()
