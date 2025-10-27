#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------
# Color palette and heatmap UI
# ----------------------------

def p_color(P: Optional[int]) -> str:
    cmap = {
        5: "#b30000",  # muito alta
        4: "#e6550d",  # alta
        3: "#fd8d3c",  # moderada
        2: "#fdae6b",  # baixa
        1: "#fdd0a2",  # muito baixa
    }
    return cmap.get(P, "#fdd0a2")

def draw_heatmap_P(p_by_section: Dict[str, Optional[int]],
                   n_sections: int,
                   nodata_color: str,
                   title: str,
                   save: Optional[str] = None,
                   dpi: int = 140) -> None:
    order = [f"S{i}" for i in range(1, n_sections + 1)]
    fig, ax = plt.subplots(figsize=(max(10, n_sections * 0.6), 2.8))

    for i, s in enumerate(order):
        P = p_by_section.get(s, None)
        color = p_color(P) if P is not None else nodata_color
        rect = patches.Rectangle((i, 0), 1, 1, lw=1, ec="black", fc=color)
        if P == 1:
            risk = "Muito baixo"
        elif P == 2:
            risk = "Baixo"
        elif P == 3:
            risk = "Moderado"
        elif P == 4:
            risk = "Alto"
        elif P == 5:
            risk = "Crítico"
        else:
            risk = "Sem dados"
        ax.add_patch(rect)
        label = f"{s}\n{('P'+str(P)) if P is not None else 'NA'}\n{risk}"
        ax.text(i + 0.5, 0.5, label, ha="center", va="center", fontsize=5, weight="bold")

    ax.set_xlim(0, n_sections)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold")

    handles = [
        patches.Patch(facecolor=p_color(5), edgecolor='black', label='P5 - Crítico'),
        patches.Patch(facecolor=p_color(4), edgecolor='black', label='P4 - Alto'),
        patches.Patch(facecolor=p_color(3), edgecolor='black', label='P3 - Moderado'),
        patches.Patch(facecolor=p_color(2), edgecolor='black', label='P2 - Baixo'),
        patches.Patch(facecolor=p_color(1), edgecolor='black', label='P1 - Muito baixo'),
        patches.Patch(facecolor=nodata_color, edgecolor='black', label='Sem dados'),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=6, bbox_to_anchor=(1, 1.35), frameon=False, fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.show()

# ---------------
# Helper routines
# ---------------

SECTION_RE = re.compile(r"^\s*S(\d+)\s*$", re.IGNORECASE)

def section_index(s: str) -> Optional[int]:
    """Extract numeric index from a section label like 'S12'."""
    if not isinstance(s, str):
        return None
    m = SECTION_RE.match(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def normalize_section(s: str) -> Optional[str]:
    """Return normalized section label 'S#' or None if not parseable."""
    idx = section_index(str(s))
    return f"S{idx}" if idx and idx > 0 else None

def parse_risk(x) -> Optional[int]:
    """Parse risk as int in [1..5]; accept 'P3', '3', 3; return None if NA."""
    if pd.isna(x):
        return None
    try:
        # strings like 'P3', 'p4', ' 3 '
        s = str(x).strip().upper()
        s = s[1:] if s.startswith("P") else s
        v = int(float(s))  # handles '3.0' if present
        if 1 <= v <= 5:
            return v
    except Exception:
        pass
    return None

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have the expected columns and types: Section, SpanID, RiskClass."""
    

    sec_col = "Section"
    span_col = "SpanID"
    risk_col = "RiskClass_supervised"

    

    # Normalize and clean
    out = pd.DataFrame()
    out["Section"] = df[sec_col].map(normalize_section)
    out["SpanID"] = df[span_col].astype(str).str.strip()
    out["RiskClass"] = df[risk_col].map(parse_risk)

    # Drop rows with invalid Section labels
    out = out[~out["Section"].isna()].copy()
    return out

def infer_n_sections(df_norm: pd.DataFrame) -> int:
    """Infer n_sections from the largest S# present."""
    max_idx = 0
    for s in df_norm["Section"].dropna().unique():
        idx = section_index(s)
        if idx and idx > max_idx:
            max_idx = idx
    # Fallback: at least the count of unique sections
    return max(max_idx, len(df_norm["Section"].unique()))

# -----
# Main
# -----

def main():
    ap = argparse.ArgumentParser(description="Bridge section risk heatmaps (by section and by span max risk)")
    ap.add_argument("--csv", required=True, help="Path to the CSV (must contain Section, RiskClass, [SpanID])")
    ap.add_argument("--n_sections", type=int, default=None, help="Total number of sections (S1..SN). If not set, inferred.")
    ap.add_argument("--nodata-color", default="#9e9e9e", help="Color for 'no data' boxes")
    ap.add_argument("--dpi", type=int, default=140, help="DPI for saved figures")
    ap.add_argument("--save-prefix", default=None, help="If set, saves PNGs as <prefix>_by_section.png and <prefix>_by_span.png")
    args = ap.parse_args()

    # Load and normalize
    df = pd.read_csv(args.csv)
    df = coerce_columns(df)

    # Consolidate duplicates: for each Section keep the MAX risk found
    # (if your pipeline emits multiple rows per section/time/window)
    df_sec = df.groupby("Section", as_index=False)["RiskClass"].max()

    # Build mapping for Heatmap 1 (by section)
    p_by_section = {row["Section"]: (int(row["RiskClass"]) if not pd.isna(row["RiskClass"]) else None)
                    for _, row in df_sec.iterrows()}

    # n_sections
    n_sections = args.n_sections or infer_n_sections(df)

    # Heatmap 1: each section -> its own RiskClass
    draw_heatmap_P(
        p_by_section=p_by_section,
        n_sections=n_sections,
        nodata_color=args.nodata_color,
        title="Risco por seção (cada seção com sua RiskClass)",
        save=(f"{args.save_prefix}_by_section.png" if args.save_prefix else None),
        dpi=args.dpi
    )

    # Heatmap 2: sections colored by the MAX RiskClass among all sections sharing the same SpanID
    # Compute max risk per span
    df_span_max = df.groupby("SpanID", as_index=False)["RiskClass"].max()
    span_to_max = {row["SpanID"]: (int(row["RiskClass"]) if not pd.isna(row["RiskClass"]) else None)
                   for _, row in df_span_max.iterrows()}

    # Map each section to its span's max risk
    sec_to_span = df.groupby("Section", as_index=False)["SpanID"].first()
    sec_to_span = dict(zip(sec_to_span["Section"], sec_to_span["SpanID"]))

    p_by_section_spanmax = {}
    for i in range(1, n_sections + 1):
        s = f"S{i}"
        span = sec_to_span.get(s)
        P = span_to_max.get(span, None) if span is not None else None
        p_by_section_spanmax[s] = P

    draw_heatmap_P(
        p_by_section=p_by_section_spanmax,
        n_sections=n_sections,
        nodata_color=args.nodata_color,
        title="Risco por vão (cada seção recebe o maior RiskClass do seu SpanID)",
        save=(f"{args.save_prefix}_by_span.png" if args.save_prefix else None),
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()
