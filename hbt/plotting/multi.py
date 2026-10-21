# coding: utf-8

from __future__ import annotations

import law
import order as od

from columnflow.plotting import supports_multi_variable, supports_multi_category#, supports_multi_version

from columnflow.plotting.plot_all import plot_all
from columnflow.plotting.plot_util import prepare_style_config, remove_residual_axis, use_flow_bins
from columnflow.util import maybe_import
from columnflow.types import TYPE_CHECKING, Any
from columnflow.plotting.plot_functions_1d import plot_variable_stack
if TYPE_CHECKING:
    hist = maybe_import("hist")
    plt = maybe_import("matplotlib.pyplot")


@supports_multi_variable
def multi_var_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category,
    variable_insts: list[od.Variable] | dict[str, list[od.Variable]],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    # emulate multi-variable input
    if isinstance(list(hists)[0], od.Process):
        assert isinstance(variable_insts, list)
        assert len(variable_insts) == 1
        hists = {variable_insts[0].name: hists}
        variable_insts = {variable_insts[0].name: variable_insts}

    # make sure all variable lists are 1D
    for var_name, var_insts in variable_insts.items():
        if len(var_insts) != 1:
            raise NotImplementedError(f"multi_plot only supports 1D histograms, but got variables {var_insts}")
        variable_insts[var_name] = var_insts[0]

    # remove shift axis
    hists = {var_name: remove_residual_axis(proc_hists, "shift") for var_name, proc_hists in hists.items()}

    # unique process instances
    unique_proc_insts = set(sum((list(v.keys()) for v in hists.values()), []))

    # helper to create legend labels
    def create_legend_label(proc_inst: od.Process, var_inst: od.Variable) -> str:
        return var_inst.x_title_short if len(unique_proc_insts) == 1 else f"{proc_inst.label}: {var_inst.x_title_short}"

    # create plot instructions for plot_all
    plot_config = {
        **{
            f"multi_var_plot__{proc_inst.name}__{var_name}": {
                "method": "draw_hist",
                "hist": move_flow(h, variable_insts[var_name]),
                "kwargs": dict(
                    label=create_legend_label(proc_inst, variable_insts[var_name]),
                    color=config_inst.x.get_color_from_sequence(i),
                    linestyle=config_inst.x.get_line_style_from_sequence(j),
                    error_type="variance",
                    linewidth=1.5,
                ),
            }
            for i, (var_name, proc_hists) in enumerate(hists.items())
            for j, (proc_inst, h) in enumerate(proc_hists.items())
        },
    }

    # combine the style config
    style_config = law.util.merge_dicts(
        # inferred from config, category, variable
        prepare_style_config(
            config_inst=config_inst,
            category_inst=category_inst,
            variable_inst=list(variable_insts.values())[0],
        ),
        # passed styles
        style_config,
        (variable_settings or {}),
        # overwrite with hardcoded settings
        {
            "annotate_cfg": {"xy": (0.035, 0.95)},
            "cms_label_cfg": {"loc": 0},
            "legend_cfg": {"loc": "upper right", "fontsize": 18},
        },
        deep=True,
    )

    # update kwargs for plot_all
    kwargs = kwargs | {
        "skip_ratio": True,
        "skip_legend": False,
    }

    return plot_all(plot_config, style_config, **kwargs)


@supports_multi_category
def multi_cat_plot(
    hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: od.Category | list[od.Category],
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift],
    style_config: dict | None = None,
    variable_settings: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    # emulate multi-category input
    if isinstance(list(hists)[0], od.Process):
        assert isinstance(category_inst, list)
        assert len(category_inst) == 1
        hists = {category_inst[0].name: hists}
        category_inst = [category_inst]

    # prepare objects
    category_insts = {cat_inst.name: cat_inst for cat_inst in law.util.make_list(category_inst)}
    assert all(cat_name in hists for cat_name in category_insts)
    assert len(variable_insts) == 1
    variable_inst = variable_insts[0]

    # remove shift axis
    hists = {cat_name: remove_residual_axis(proc_hists, "shift") for cat_name, proc_hists in hists.items()}

    # unique process instances
    unique_proc_insts = set(sum((list(v.keys()) for v in hists.values()), []))

    # helper to create legend labels
    def create_legend_label(proc_inst: od.Process, cat_inst: od.Category) -> str:
        return cat_inst.label_short if len(unique_proc_insts) == 1 else f"{proc_inst.label}: {cat_inst.label_short}"

    # create plot instructions for plot_all
    plot_config = {
        **{
            f"multi_cat_plot__{proc_inst.name}__{cat_name}": {
                "method": "draw_hist",
                "hist": move_flow(h, variable_inst),
                "kwargs": dict(
                    label=create_legend_label(proc_inst, category_insts[cat_name]),
                    color=config_inst.x.get_color_from_sequence(i),
                    linestyle=config_inst.x.get_line_style_from_sequence(j),
                    error_type="variance",
                    linewidth=1.5,
                ),
            }
            for i, (cat_name, proc_hists) in enumerate(hists.items())
            for j, (proc_inst, h) in enumerate(proc_hists.items())
        },
    }

    # combine the style config
    style_config = law.util.merge_dicts(
        # inferred from config, category, variable
        prepare_style_config(
            config_inst=config_inst,
            variable_inst=variable_inst,
        ),
        # passed styles
        style_config,
        (variable_settings or {}),
        # overwrite with hardcoded settings
        {
            "cms_label_cfg": {"loc": 0},
            "legend_cfg": {"loc": "upper right", "fontsize": 18},
        },
        deep=True,
    )

    # update kwargs for plot_all
    kwargs = kwargs | {
        "skip_ratio": True,
        "skip_legend": False,
    }

    return plot_all(plot_config, style_config, **kwargs)

# @supports_multi_version
# def multi_ver_plot(
#     hists: dict[od.Process, hist.Hist] | dict[str, dict[od.Process, hist.Hist]],
#     config_inst: od.Config,
#     category_inst: od.Category,
#     variable_insts: list[od.Variable],
#     shift_insts: list[od.Shift],
#     style_config: dict | None = None,
#     variable_settings: dict[str, Any] | None = None,
#     **kwargs,
# ) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
#     # emulate multi-version input
#     if isinstance(list(hists)[0], od.Process):
#         hists = {category_inst.name: hists}  # single fake "version" key, label is irrelevant here since len==1

#     # prepare objects
#     assert len(variable_insts) == 1
#     variable_inst = variable_insts[0]

#     # remove shift axis
#     hists = {ver_name: remove_residual_axis(proc_hists, "shift") for ver_name, proc_hists in hists.items()}

#     # unique process instances
#     unique_proc_insts = set(sum((list(v.keys()) for v in hists.values()), []))

#     # helper to create legend labels
#     def create_legend_label(proc_inst: od.Process, ver_name: str) -> str:
#         return ver_name if len(unique_proc_insts) == 1 else f"{proc_inst.label}: {ver_name}"

#     # create plot instructions for plot_all
#     plot_config = {
#         **{
#             f"multi_ver_plot__{proc_inst.name}__{ver_name}": {
#                 "method": "draw_hist",
#                 "hist": move_flow(h, variable_inst),
#                 "kwargs": dict(
#                     label=create_legend_label(proc_inst, ver_name),
#                     color=config_inst.x.get_color_from_sequence(i),
#                     linestyle=config_inst.x.get_line_style_from_sequence(j),
#                     error_type="variance",
#                     linewidth=1.5,
#                 ),
#             }
#             for i, (ver_name, proc_hists) in enumerate(hists.items())
#             for j, (proc_inst, h) in enumerate(proc_hists.items())
#         },
#     }

#     # combine the style config
#     style_config = law.util.merge_dicts(
#         # inferred from config, category, variable
#         prepare_style_config(
#             config_inst=config_inst,
#             category_inst=category_inst,
#             variable_inst=variable_inst,
#         ),
#         # passed styles
#         style_config,
#         (variable_settings or {}),
#         # overwrite with hardcoded settings
#         {
#             "cms_label_cfg": {"loc": 0},
#             "legend_cfg": {"loc": "upper right", "fontsize": 18},
#         },
#         deep=True,
#     )

#     # update kwargs for plot_all
#     kwargs = kwargs | {
#         "skip_ratio": True,
#         "skip_legend": False,
#     }

#     return plot_all(plot_config, style_config, **kwargs)

# helper to move flow bins into first/last ones
def move_flow(h: hist.Hist, variable_inst: od.Variable) -> hist.Hist:
    overflow = variable_inst.x("overflow", False)
    underflow = variable_inst.x("underflow", False)
    if overflow or underflow:
        h = use_flow_bins(h, variable_inst.name, underflow=underflow, overflow=overflow)
    return h

# at the top of multi.py, with the other imports
import functools
from columnflow.plotting.plot_functions_1d import plot_variable_stack

# genPartFlav value -> (legend label, colour)
GENFLAV_STYLE = {
    5: (r"genuine $\tau_h$", "#3f90da"),
    1: ("prompt e", "#ffa90e"),
    2: (r"prompt $\mu$", "#bd1f01"),
    3: (r"$\tau \to e$", "#94a4a2"),
    4: (r"$\tau \to \mu$", "#832db6"),
    0: ("unmatched (jet fake)", "#a96b59"),
}

import re

GENFLAV_RE = re.compile(r"__genflav(\d+)")


def split_genflav_name(name: str) -> tuple[int | None, str]:
    # "tautau__incl__eq1b__genflav1__os__iso" -> (1, "tautau__incl__eq1b__os__iso")
    # "tautau__incl__eq1b__os__iso" -> (None, "tautau__incl__eq1b__os__iso")
    match = GENFLAV_RE.search(name)
    if match is None:
        return None, name
    return int(match.group(1)), GENFLAV_RE.sub("", name, count=1)


@supports_multi_category
def multi_cat_genflav_stack(
    hists: dict[str, dict[od.Process, hist.Hist]],
    config_inst: od.Config,
    category_inst: list[od.Category],
    variable_insts: list[od.Variable],
    shift_insts: list[od.Shift],
    **kwargs,
):
    # same preparation as multi_cat_plot
    hists = {cat_name: remove_residual_axis(proc_hists, "shift") for cat_name, proc_hists in hists.items()}
    cats = {cat.name: cat for cat in law.util.make_list(category_inst)}
    variable_inst = variable_insts[0]
    
    parsed = {name: split_genflav_name(name) for name in hists.keys()}
    # from IPython import embed; embed()  # noqa: E800
    # exactly one parent region (a category without a genflav suffix) must be requested
    parent_names = {parent for flav, parent in parsed.values() if flav is None}
    if len(parent_names) != 1:
        raise ValueError(f"expected exactly one parent region in --categories, got {sorted(parent_names)}")
    parent_name = parent_names.pop()

    # genflav children of that parent, keyed by genPartFlav value
    genflav_names = {
        flav: name
        for name, (flav, parent) in parsed.items()
        if flav is not None and parent == parent_name
    }

    plot_hists = {}
    for proc_inst, h in hists[parent_name].items():
        if proc_inst.is_data:
            plot_hists[proc_inst] = move_flow(h, variable_inst)

    # one stack component per genflav value: the sum of its MC processes
    for flav, (label, color) in GENFLAV_STYLE.items():
        cat_name = genflav_names.get(flav)
        if cat_name is None:
            continue
        mc_hists = [h for proc_inst, h in hists[cat_name].items() if not proc_inst.is_data]
        if not mc_hists:
            continue
        proc = od.Process(name=f"genflav{flav}", id="+", label=label)
        proc.color1 = color  # set the colour the same way your config sets process colours
        plot_hists[proc] = move_flow(functools.reduce(lambda a, b: a + b, mc_hists), variable_inst)

    return plot_variable_stack(
        plot_hists,
        config_inst=config_inst,
        variable_insts=variable_insts,
        category_inst=cats[parent_name],
        shift_insts=shift_insts,
        **kwargs,
    )