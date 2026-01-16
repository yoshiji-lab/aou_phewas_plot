#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phewas_plot.py  – Manhattan & Volcano plots for PheWAS results

Updated:
- Global font now uses a Linux-friendly fallback stack:
  Liberation Sans (closest to Arial) -> Arial -> Helvetica -> Nimbus Sans -> DejaVu Sans
- Optional: embed TrueType fonts in PDF/PS for consistent rendering
"""

from __future__ import annotations
import datetime, os
from pathlib import Path

# ── Matplotlib defaults ───────────────────────────────────────────
import matplotlib as mpl
import logging
from matplotlib import font_manager as fm

# Silence "findfont" warnings (still shows real errors)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

def _pick_first_available_font(candidates: list[str]) -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

# Prefer Arial-like fonts on Linux VMs; fall back to DejaVu Sans
FONT_CANDIDATES = [
    "Liberation Sans",  # closest to Arial (often installed on Linux)
    "Arial",
    "Helvetica",
    "Nimbus Sans",
    "DejaVu Sans",      # usually always available with matplotlib
]

chosen_font = _pick_first_available_font(FONT_CANDIDATES)

if chosen_font is not None:
    mpl.rcParams["font.family"] = chosen_font
else:
    # absolute last-resort fallback
    mpl.rcParams["font.family"] = "sans-serif"

# Optional: make PDF/PS output more consistent across viewers
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# (Optional) print to confirm once
# print("Using font:", mpl.rcParams["font.family"])

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mc

# ── Data stack ────────────────────────────────────────────────────
import numpy as np
import polars as pl
import adjustText


# ╭────────────────────────────────────────────────────────────────╮
# │ Global helper                                                  │
# ╰────────────────────────────────────────────────────────────────╯
def set_global_font_size(size: int | float) -> None:
    """Bump ALL common font rcParams to *size* pt."""
    keys = ["font.size", "axes.titlesize", "axes.labelsize",
            "xtick.labelsize", "ytick.labelsize",
            "legend.fontsize", "figure.titlesize"]
    for k in keys:
        mpl.rcParams[k] = size


# ╭────────────────────────────────────────────────────────────────╮
# │ Plot class                                                     │
# ╰────────────────────────────────────────────────────────────────╯
class Plot:
    # ───────────────────────── constructor ────────────────────────
    def __init__(self,
                 phewas_result_csv_path: str | os.PathLike,
                 converged_only: bool = True,
                 bonferroni: float | None = None,
                 phecode_version: str | None = None,
                 color_palette: tuple[str, ...] | None = None) -> None:

        self.phewas_result = pl.read_csv(
            phewas_result_csv_path,
            dtypes={"phecode": str, "converged": bool},
        )

        # Bonferroni
        self.bonferroni = (-np.log10(0.05 / len(self.phewas_result))
                           if bonferroni is None else bonferroni)

        # keep converged rows
        if converged_only:
            self.phewas_result = self.phewas_result.filter(
                pl.col("converged") == True
            )

        # proxy for +∞
        max_non_inf = (
            self.phewas_result.filter(pl.col("neg_log_p_value") != np.inf)
            .sort("neg_log_p_value", descending=True)["neg_log_p_value"][0]
        )
        if max_non_inf < self.phewas_result["neg_log_p_value"].max():
            self.inf_proxy = max_non_inf * 1.2
            self.phewas_result = self.phewas_result.with_columns(
                pl.when(pl.col("neg_log_p_value") == np.inf)
                .then(self.inf_proxy)
                .otherwise(pl.col("neg_log_p_value"))
                .alias("neg_log_p_value")
            )
        else:
            self.inf_proxy = None

        self.nominal_significance = -np.log10(0.05)
        self.phecode_version = phecode_version.upper() if phecode_version else "X"

        # colour palette
        if color_palette is None:
            color_palette = (
                "blue", "indianred", "darkcyan", "goldenrod", "darkblue",
                "magenta", "green", "red", "darkturquoise", "olive",
                "black", "royalblue", "maroon", "darkolivegreen",
                "coral", "purple", "gray"
            )
        self.color_palette = color_palette
        self.phecode_categories = (
            self.phewas_result["phecode_category"].unique().sort().to_list()
        )
        self.color_dict = {
            self.phecode_categories[i]: self.color_palette[i % len(self.color_palette)]
            for i in range(len(self.phecode_categories))
        }
        self.phewas_result = self.phewas_result.with_columns(
            pl.col("phecode_category").replace(self.color_dict).alias("label_color")
        )

        self.positive_betas: pl.DataFrame | None = None
        self.negative_betas: pl.DataFrame | None = None
        self.offset = 9  # x-axis padding

    # ───────────────────── internal helpers ───────────────────────
    @staticmethod
    def _create_phecode_index(df: pl.DataFrame) -> pl.DataFrame:
        if "phecode_index" in df.columns:
            df = df.drop("phecode_index")
        return (
            df.sort(by=["phecode_category", "neg_log_p_value"],
                    descending=[False, True])
              .with_columns(pl.Series("phecode_index",
                                      range(1, len(df) + 1)))
              .with_columns((15 * np.exp(pl.col("beta")))
                            .alias("marker_size_by_beta"))
        )

    @staticmethod
    def _split_by_beta(df: pl.DataFrame,
                       marker_size_by_beta=False) -> tuple[pl.DataFrame, pl.DataFrame]:
        if marker_size_by_beta and "_marker_size" not in df.columns:
            df = df.with_columns((18*pl.col("beta").abs()).alias("_marker_size"))
        pos = df.filter(pl.col("beta") >= 0).sort("beta", descending=True)
        neg = df.filter(pl.col("beta") < 0).sort("beta", descending=False)
        return pos, neg

    # ──────────────────────────────────────────────────────────────
    #  Draw coloured category names on the X-axis
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _x_ticks(plot_df, sel_color_dict,
                 tick_size: int | float | None = None):
        import matplotlib.pyplot as plt
        if tick_size is None:
            tick_size = mpl.rcParams["xtick.labelsize"]

        xt = (
            plot_df[["phecode_category", "phecode_index"]]
            .group_by("phecode_category")
            .mean()
        )

        plt.xticks(
            xt["phecode_index"],
            xt["phecode_category"],
            rotation=45,
            ha="right",
            size=tick_size,
        )

        for lbl, clr in zip(
            sorted(plt.gca().get_xticklabels(), key=lambda l: l.get_text()),
            sel_color_dict.values(),
        ):
            lbl.set_color(clr)

    # ──────────────────────────────────────────────────────────────
    #  Scatter layer for Manhattan
    # ──────────────────────────────────────────────────────────────
    def _manhattan_scatter(self, ax,
                           marker_size_by_beta,
                           scale_factor: float = 1.0):
        sp = sn = None
        if marker_size_by_beta:
            sp = self.positive_betas["_marker_size"] * scale_factor
            sn = self.negative_betas["_marker_size"] * scale_factor

        ax.scatter(self.positive_betas["phecode_index"],
                   self.positive_betas["neg_log_p_value"],
                   s=sp,
                   c=self.positive_betas["label_color"],
                   marker="^", alpha=0.7)

        ax.scatter(self.negative_betas["phecode_index"],
                   self.negative_betas["neg_log_p_value"],
                   s=sn,
                   c=self.negative_betas["label_color"],
                   marker="v", alpha=0.3)

    # ──────────────────────────────────────────────────────────────
    #  Annotate selected Manhattan points
    # ──────────────────────────────────────────────────────────────
    def _manhattan_label(
        self,
        plot_df: pl.DataFrame,
        label_values="p_value",
        label_count: int = 10,
        label_categories=None,
        label_text_column="phecode_string",
        label_value_threshold=0,
        label_split_threshold=30,
        label_color="label_color",
        label_size=8,
        label_weight="normal",
        y_col="neg_log_p_value",
        x_col="phecode_index",
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import adjustText

        if isinstance(label_values, str):
            label_values = [label_values]

        data_to_label = pl.DataFrame(schema=plot_df.schema)
        pos, neg = self.positive_betas.clone(), self.negative_betas.clone()
        if "_marker_size" in pos.columns:
            pos = pos.drop("_marker_size"); neg = neg.drop("_marker_size")

        for item in label_values:
            if item == "positive_beta":
                chunk = pos.filter(pl.col("beta") >= label_value_threshold)
            elif item == "negative_beta":
                chunk = neg.filter(pl.col("beta") <= label_value_threshold)
            elif item == "p_value":
                chunk = plot_df.sort("p_value").filter(
                    pl.col("neg_log_p_value") >= label_value_threshold)
            else:
                chunk = plot_df.filter(pl.col("phecode") == item)

            if label_categories is not None:
                chunk = chunk.filter(pl.col("phecode_category").is_in(label_categories))
            data_to_label = pl.concat([data_to_label, chunk]).head(label_count)

        def _split_text(s: str, thresh: int = 30) -> str:
            words, out, ln = s.split(" "), "", 0
            for w in words:
                out += w; ln += len(w)
                if ln >= thresh and w != words[-1]:
                    out += "\n"; ln = 0
                else:
                    out += " "
            return out

        texts = []
        for i in range(len(data_to_label)):
            clr = (label_color if mcolors.is_color_like(label_color)
                   else data_to_label[label_color][i])
            texts.append(
                plt.text(
                    float(data_to_label[x_col][i]),
                    float(data_to_label[y_col][i]),
                    _split_text(data_to_label[label_text_column][i],
                                label_split_threshold),
                    color=clr,
                    size=label_size,
                    weight=label_weight,
                    bbox=dict(facecolor="white", edgecolor="none",
                              boxstyle="round", alpha=0.5, lw=0.5),
                )
            )
        if texts:
            adjustText.adjust_text(
                texts,
                arrowprops=dict(arrowstyle="simple", color="gray",
                                lw=0.5, mutation_scale=2),
            )

    # ──────────────────────────────────────────────────────────────
    #  Guide lines & legend helpers
    # ──────────────────────────────────────────────────────────────
    def _lines(self, ax, plot_type, plot_df, x_col,
               nominal_significance_line=False, bonferroni_line=False,
               infinity_line=False, y_threshold_line=False,
               y_threshold_value=None,
               x_positive_threshold_line=False, x_positive_threshold_value=None,
               x_negative_threshold_line=False, x_negative_threshold_value=None):
        extra = 1 if plot_type == "manhattan" else 0.05
        if nominal_significance_line:
            ax.hlines(self.nominal_significance,
                      plot_df[x_col].min()-self.offset-extra,
                      plot_df[x_col].max()+self.offset+extra,
                      colors="red", lw=1)
        if bonferroni_line:
            ax.hlines(self.bonferroni,
                      plot_df[x_col].min()-self.offset-extra,
                      plot_df[x_col].max()+self.offset+extra,
                      colors="green", lw=1)
        if infinity_line and self.inf_proxy is not None:
            ax.yaxis.get_major_ticks()[-2].set_visible(False)
            ax.hlines(self.inf_proxy*0.98,
                      plot_df[x_col].min()-self.offset-extra,
                      plot_df[x_col].max()+self.offset+extra,
                      colors="blue", linestyle="dashdot", lw=1)
        if y_threshold_line:
            ax.hlines(y_threshold_value,
                      plot_df[x_col].min()-self.offset-extra,
                      plot_df[x_col].max()+self.offset+extra,
                      colors="gray", linestyles="dashed", lw=1)
        if x_positive_threshold_line:
            ax.vlines(x_positive_threshold_value,
                      plot_df["neg_log_p_value"].min()-self.offset,
                      plot_df["neg_log_p_value"].max()+self.offset+5,
                      colors="orange", linestyles="dashed")
        if x_negative_threshold_line:
            ax.vlines(x_negative_threshold_value,
                      plot_df["neg_log_p_value"].min()-self.offset,
                      plot_df["neg_log_p_value"].max()+self.offset+5,
                      colors="lightseagreen", linestyles="dashed")

    def _manhattan_legend(self, ax, legend_marker_size):
        elements = [
            Line2D([0], [0], color="blue", lw=1, linestyle="dashdot", label="Infinity"),
            Line2D([0], [0], color="green", lw=1, label="Bonferroni\nCorrection"),
            Line2D([0], [0], color="red", lw=1, label="Nominal\nSignificance"),
            Line2D([0], [0], marker="^", color="white",
                   markerfacecolor="blue", markersize=legend_marker_size,
                   alpha=0.7, label="Increased\nRisk"),
            Line2D([0], [0], marker="v", color="white",
                   markerfacecolor="blue", markersize=legend_marker_size,
                   alpha=0.3, label="Decreased\nRisk"),
        ]
        ax.legend(handles=elements, handlelength=2, loc="center left",
                  bbox_to_anchor=(1, 0.5), fontsize=legend_marker_size)

    # ═════════════════════ Manhattan plot ════════════════════════
    def manhattan(self, label_values="p_value", label_value_threshold=0,
                  label_count=10, label_size=8,
                  label_text_column="phecode_string",
                  label_color="label_color", label_weight="normal",
                  label_split_threshold=30,
                  marker_size_by_beta=False, marker_scale_factor=1,
                  phecode_categories=None,
                  title=None, title_text_size=10,
                  y_limit=None, axis_text_size=8,
                  fig_width: float | None = None,
                  fig_height: float | None = None,
                  dpi=150, show_legend=True, legend_marker_size=6,
                  save_plot=True, output_file_name=None,
                  output_file_type="pdf"):

        if phecode_categories:
            if isinstance(phecode_categories, str):
                phecode_categories = [phecode_categories]
            phecode_categories.sort()
            sel_dict = {k: self.color_dict[k] for k in phecode_categories}
            plot_df = self._create_phecode_index(
                self.phewas_result.filter(
                    pl.col("phecode_category").is_in(phecode_categories)))
        else:
            sel_dict = self.color_dict
            plot_df = self._create_phecode_index(self.phewas_result)

        if fig_width is None or fig_height is None:
            fig_width = 12 * (len(sel_dict) / 10)
            fig_height = 7

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        if title:
            plt.title(title, weight="bold", size=title_text_size)
        if y_limit:
            ax.set_ylim(-0.2, y_limit)
        ax.set_ylabel(r"$-\log_{10}$(p-value)", size=axis_text_size)

        self.positive_betas, self.negative_betas = self._split_by_beta(
            plot_df, marker_size_by_beta)

        plt.xlim(float(plot_df["phecode_index"].min()) - self.offset - 1,
                 float(plot_df["phecode_index"].max()) + self.offset + 1)
        self._x_ticks(plot_df, sel_dict, tick_size=axis_text_size)
        self._manhattan_scatter(ax, marker_size_by_beta, marker_scale_factor)
        self._lines(ax, "manhattan", plot_df, "phecode_index",
                    nominal_significance_line=True, bonferroni_line=True,
                    infinity_line=True)
        self._manhattan_label(
            plot_df              = plot_df,
            label_values         = label_values,
            label_count          = label_count,
            label_categories     = phecode_categories,
            label_text_column    = label_text_column,
            label_value_threshold= label_value_threshold,
            label_split_threshold= label_split_threshold,
            label_color          = label_color,
            label_size           = label_size,
            label_weight         = label_weight,
        )
        if show_legend:
            self._manhattan_legend(ax, legend_marker_size)

        if save_plot:
            self._save_plot("manhattan", output_file_name, output_file_type)

    # ═════════════════════ Save helper ═══════════════════════════
    @staticmethod
    def _save_plot(plot_type="plot", out_name=None, out_type="pdf",
                   tight: bool = False):
        if out_name is None or "." not in out_name:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"{plot_type}_{stamp}.{out_type}"
        if tight:
            plt.savefig(out_name, bbox_inches="tight")
        else:
            plt.savefig(out_name)
        print(f"Plot saved ➜ {Path(out_name).resolve()}")
