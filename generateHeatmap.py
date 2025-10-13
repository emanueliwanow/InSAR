#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera Probabilidade de falha (P) por seção e um heatmap 1×N.

Entrada:
  - CSV com 1 coluna de data (Date/Data/Timestamp) e colunas S1..S{N}
  - Datas em qualquer formato parseável pelo pandas

Parâmetros:
  --csv <arquivo.csv>            caminho do CSV
  --n_sections <N>               número total de seções (S1..SN) a representar no heatmap
  --nodata-color <hex>           cor para "sem dados" (default: #9e9e9e)
  --date-col <nome>              (opcional) nome explícito da coluna de data
  --save <arquivo.png>           (opcional) salva o heatmap em arquivo
  --dpi <int>                    (opcional) resolução ao salvar (default: 140)

Critério de P:
  - Base |vel|: <0.5→1; <1.0→2; <2.0→3; <3.0→4; ≥3.0→5
  - +1 se |Δ90d| > 2 mm
  - +1 se |acc|  > 1 mm/ano²
  - Clamped em [1,5]

Δ90d:
  - Último valor menos o primeiro registro >= (t_final - 90 dias)
  - Se não houver amostra na janela, usa o primeiro valor da série como fallback

Observações:
  - Se uma seção S_k não existir no CSV ou tiver <4 pontos válidos, ela entra como "sem dados".
  - A tabela é ordenada por P (desc) e |vel| (desc).
"""

import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DATE_CANDIDATES = ("date", "data", "timestamp")

# ---------- Cálculos por seção ----------
def detect_date_column(df: pd.DataFrame, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Coluna de data '{explicit}' não encontrada no CSV.")
        return explicit
    for c in df.columns:
        if c.lower().strip() in DATE_CANDIDATES:
            return c
    raise ValueError(f"Não encontrei coluna de data. Aceito: {DATE_CANDIDATES} ou use --date-col.")

def series_metrics(df: pd.DataFrame, date_col: str, col: str
                   ) -> Optional[Tuple[float, float, float, float]]:
    sub = df[[date_col, col]].dropna()
    if len(sub) < 4:
        return None
    # Tempo em anos desde a primeira observação
    x = (sub[date_col] - sub[date_col].min()).dt.days.values / 365.25
    y = sub[col].astype(float).values

    # Velocidade (mm/ano)
    vel = np.polyfit(x, y, 1)[0]
    # Aceleração (mm/ano²) = 2*b2
    b2, b1, b0 = np.polyfit(x, y, 2)
    acc = 2.0 * b2

    # Δ90d
    t_end = sub[date_col].max()
    ref = t_end - pd.Timedelta(days=90)
    y_now = sub.iloc[-1][col]
    if any(sub[date_col] >= ref):
        y_past = sub.loc[sub[date_col] >= ref, col].iloc[0]
    else:
        y_past = sub.iloc[0][col]
    d90 = float(y_now - y_past)

    return float(vel), float(acc), d90, float(y_now)

def probability_P(vel: float, acc: float, d90: float) -> int:
    av = abs(vel)
    if av < 0.5: P = 1
    elif av < 1.0: P = 2
    elif av < 2.0: P = 3
    elif av < 3.0: P = 4
    else: P = 5
    if abs(d90) > 2.0: P += 1
    if abs(acc) > 1.0: P += 1
    return max(1, min(5, int(P)))

def p_color(P: Optional[int]) -> str:
    # Paleta usada em nossas análises anteriores
    cmap = {
        5: "#b30000",  # muito alta
        4: "#e6550d",  # alta
        3: "#fd8d3c",  # moderada
        2: "#fdae6b",  # baixa
        1: "#fdd0a2",  # muito baixa
    }
    return cmap.get(P, "#fdd0a2")

def p_label(P: Optional[int]) -> str:
    if P is None: return "Sem dados"
    return {5:"P5 Muito alta",4:"P4 Alta",3:"P3 Moderada",2:"P2 Baixa",1:"P1 Muito baixa"}[P]

# ---------- Heatmap ----------
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
        ax.add_patch(rect)
        label = f"{s}\n{('P'+str(P)) if P is not None else 'NA'}"
        ax.text(i + 0.5, 0.5, label, ha="center", va="center", fontsize=8, weight="bold")

    ax.set_xlim(0, n_sections)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold")

    # Legenda compacta
    handles = [
        patches.Patch(facecolor=p_color(5), edgecolor='black', label='P5'),
        patches.Patch(facecolor=p_color(4), edgecolor='black', label='P4'),
        patches.Patch(facecolor=p_color(3), edgecolor='black', label='P3'),
        patches.Patch(facecolor=p_color(2), edgecolor='black', label='P2'),
        patches.Patch(facecolor=p_color(1), edgecolor='black', label='P1'),
        patches.Patch(facecolor=nodata_color, edgecolor='black', label='Sem dados'),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=6, bbox_to_anchor=(1, 1.35), frameon=False, fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.show()

# ---------- Pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Calcula P por seção e gera heatmap 1×N.")
    ap.add_argument("--csv", required=True, help="Caminho do CSV de série temporal")
    ap.add_argument("--n_sections", required=True, type=int, help="Número total de seções (S1..SN) no heatmap")
    ap.add_argument("--nodata-color", default="#9e9e9e", help="Cor para 'sem dados' (default: #9e9e9e)")
    ap.add_argument("--date-col", default=None, help="Nome da coluna de data (se quiser forçar)")
    ap.add_argument("--save", default=None, help="Arquivo de saída para o heatmap (opcional)")
    ap.add_argument("--dpi", default=140, type=int, help="DPI ao salvar (default: 140)")
    args = ap.parse_args()

    # Carrega dados
    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    date_col = detect_date_column(df, args.date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Lista de seções alvo
    target_sections = [f"S{i}" for i in range(1, args.n_sections + 1)]

    # Calcula métricas e P
    rows = []
    p_map: Dict[str, Optional[int]] = {}
    for s in target_sections:
        if s not in df.columns:
            # Sem dados para esta seção no CSV
            p_map[s] = None
            continue
        mets = series_metrics(df, date_col, s)
        if mets is None:
            p_map[s] = None
            continue
        vel, acc, d90, cur = mets
        P = probability_P(vel, acc, d90)
        p_map[s] = P
        rows.append({
            "ROI": s,
            "Veloc_mm_ano": vel,
            "Accel_mm_ano2": acc,
            "Delta90d_mm": d90,
            "Atual_mm": cur,
            "P": P,
            "Classe_P": p_label(P),
        })

    # Tabela ordenada por P desc e |vel| desc
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["P"], ascending=False, kind="mergesort")
        out = out.iloc[out["Veloc_mm_ano"].abs().sort_values(ascending=False).index]
        for col in ["Veloc_mm_ano", "Accel_mm_ano2", "Delta90d_mm", "Atual_mm"]:
            out[col] = out[col].astype(float).round(3)
        print("\nProbabilidade por seção (critério explícito):\n")
        print(out.to_string(index=False))
    else:
        print("Aviso: nenhuma seção com dados suficientes para cálculo.")

    # Heatmap
    title = f"Heatmap de Probabilidade (P) – {args.csv}"
    draw_heatmap_P(p_map, args.n_sections, args.nodata_color, title, save=args.save, dpi=args.dpi)

if __name__ == "__main__":
    main()
