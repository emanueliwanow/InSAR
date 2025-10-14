#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gera Probabilidade de falha (P) por seção e um heatmap 1×N.

Entrada:
  - Dois CSVs com 1 coluna de data (Date/Data/Timestamp) e colunas S1..S{N}
    * --csv-total : série completa
    * --csv-90d   : janela dos últimos ~90 dias (mesmo layout)
  - Datas em qualquer formato parseável pelo pandas

Parâmetros:
  --csv-total <arquivo.csv>      caminho do CSV da série total
  --csv-90d <arquivo.csv>        caminho do CSV da janela dos últimos 90 dias
  --n_sections <N>               número total de seções (S1..SN) a representar no heatmap
  --nodata-color <hex>           cor para "sem dados" (default: #9e9e9e)
  --date-col <nome>              (opcional) nome explícito da coluna de data (se ambos usarem o mesmo nome)
  --save <arquivo.png>           (opcional) salva o heatmap em arquivo
  --dpi <int>                    (opcional) resolução ao salvar (default: 140)
  --pillars <list>               (opcional) lista de tuplas com seções em que ha um pilar no meio das seções (ex: "S1,S2") (default: None)

Critério de P (novo):
  - Base |vel_90d|: <0.5→1; <1.0→2; <2.0→3; <3.0→4; ≥3.0→5
  - +1 se |acc_90d| > 1 mm/ano²
  - +1 se agravamento vs. histórico: (mesmo sinal entre vel_total e vel_90d e |vel_90d| ≥ 1.2*|vel_total|) ou |acc_total| > 1
  - Clamped em [1,5]

Observações:
  - Se uma seção S_k inexistir em qualquer CSV ou tiver <4 pontos válidos (após eventual fallback), ela entra como "sem dados".
  - A tabela é ordenada por P (desc) e |vel_90d| (desc).
"""

import argparse
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# ---------- Cálculos por seção ----------

def _fit_metrics(x_years: np.ndarray, y_mm: np.ndarray) -> Tuple[float, float]:
    # Velocidade (mm/ano) via regressão linear; Aceleração (mm/ano²) via termo quadrático
    vel = np.polyfit(x_years, y_mm, 1)[0]
    b2, b1, b0 = np.polyfit(x_years, y_mm, 2)
    acc = 2.0 * b2
    return float(vel), float(acc)

def series_metrics(df_total: pd.DataFrame, date_col_total: str,
                   df_90d: pd.DataFrame, date_col_90d: str,
                   col: str
                   ) -> Optional[Tuple[float, float, float, float]]:
    """
    Retorna tupla: (vel_90d, acc_90d, vel_total, acc_total)
    Fallback: se a série 90d não tiver dados suficientes, recorta os últimos 90 dias da série total.
    """
    # ---- Série total ----
    if col not in df_total.columns:
        return None
    subT = df_total[[date_col_total, col]].dropna()
    if len(subT) < 4:
        return None

    # Tempo em anos desde a primeira observação total
    xT = (subT[date_col_total] - subT[date_col_total].min()).dt.days.values / 365.25
    yT = subT[col].astype(float).values
    vel_total, acc_total = _fit_metrics(xT, yT)

    # ---- Série 90d (preferencialmente do CSV dedicado) ----
    has_recent = (col in df_90d.columns)
    subR = df_90d[[date_col_90d, col]].dropna() if has_recent else pd.DataFrame(columns=[date_col_90d, col])

    # Se insuficiente, recorta dos 90 dias finais do total
    if len(subR) < 4:
        t_end = subT[date_col_total].max()
        ref = t_end - pd.Timedelta(days=90)
        subR = subT.loc[subT[date_col_total] >= ref, [date_col_total, col]].copy()

    # Se ainda insuficiente, aborta
    if len(subR) < 4:
        return None

    # Tempo em anos desde a primeira observação da janela
    xR = (subR.iloc[:, 0] - subR.iloc[:, 0].min()).dt.days.values / 365.25
    yR = subR.iloc[:, 1].astype(float).values
    vel_90d, acc_90d = _fit_metrics(xR, yR)

    return vel_90d, acc_90d, vel_total, acc_total

def probability_P(vel_90d: float, acc_90d: float, vel_total: float, acc_total: float) -> int:
    # Base por |vel_90d|
    av = abs(vel_90d)
    if av < 0.5: P = 1
    elif av < 1.0: P = 2
    elif av < 2.0: P = 3
    elif av < 3.0: P = 4
    else: P = 5

    # Aceleração recente forte
    if abs(acc_90d) > 1.0:
        P += 1

    # Agravamento vs histórico (mesmo sinal e ganho de pelo menos 20%) ou aceleração histórica alta
    same_sign = (vel_90d == 0) or (vel_total == 0) or (np.sign(vel_90d) == np.sign(vel_total))
    worsened = same_sign and (abs(vel_90d) >= 1.2 * abs(vel_total))
    if worsened or (abs(acc_total) > 1.0):
        P += 1

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
    ap.add_argument("--csv-total", required=True, help="Caminho do CSV de série temporal TOTAL")
    ap.add_argument("--csv-90d", required=True, help="Caminho do CSV da janela dos últimos 90 dias")
    ap.add_argument("--n_sections", required=True, type=int, help="Número total de seções (S1..SN) no heatmap")
    ap.add_argument("--nodata-color", default="#9e9e9e", help="Cor para 'sem dados' (default: #9e9e9e)")
    ap.add_argument("--date-col", default=None, help="Nome da coluna de data (se quiser forçar)")
    ap.add_argument("--save", default=None, help="Arquivo de saída para o heatmap (opcional)")
    ap.add_argument("--dpi", default=140, type=int, help="DPI ao salvar (default: 140)")
    ap.add_argument("--pillars", default=None, help="Lista de tuplas com seções em que ha um pilar entre as seções (ex: ['S1,S2','S3,S4'])")
    args = ap.parse_args()

    # Carrega dados (TOTAL)
    df_total = pd.read_csv(args.csv_total)
    df_total.columns = df_total.columns.str.strip()
    date_col_total = 'date'
    df_total[date_col_total] = pd.to_datetime(df_total[date_col_total], errors="coerce")

    # Carrega dados (90d)
    df_90d = pd.read_csv(args.csv_90d)
    df_90d.columns = df_90d.columns.str.strip()
    # Se o usuário passou --date-col, assume o mesmo nome; senão detecta para o CSV de 90d
    date_col_90d = 'date'
    df_90d[date_col_90d] = pd.to_datetime(df_90d[date_col_90d], errors="coerce")

    # Lista de seções alvo
    target_sections = [f"S{i}" for i in range(1, args.n_sections + 1)]

    # Parse pillars argument
    pillar_positions = []
    if args.pillars:
        print(f"Pillars: {args.pillars}")
        pillar_list = eval(args.pillars)
        for pillar_pair in pillar_list:
            s1, s2 = pillar_pair.split(',')
            # Extract section numbers
            num1 = int(s1.strip().replace('S', ''))
            num2 = int(s2.strip().replace('S', ''))
            # Pillar is between these sections, so boundary is after the smaller number
            pillar_positions.append(min(num1, num2))
        pillar_positions = sorted(set(pillar_positions))  # Remove duplicates and sort

    # Calcula métricas e P
    rows = []
    p_map: Dict[str, Optional[int]] = {}
    for s in target_sections:
        if s not in df_total.columns:
            p_map[s] = None
            continue
        mets = series_metrics(df_total, date_col_total, df_90d, date_col_90d, s)
        if mets is None:
            p_map[s] = None
            continue
        vel_90d, acc_90d, vel_total, acc_total = mets
        P = probability_P(vel_90d, acc_90d, vel_total, acc_total)
        p_map[s] = P
        rows.append({
            "ROI": s,
            "Vel_90d_mm_ano": vel_90d,
            "Acc_90d_mm_ano2": acc_90d,
            "Vel_total_mm_ano": vel_total,
            "Acc_total_mm_ano2": acc_total,
            "P": P,
            "Classe_P": p_label(P),
        })

    # Apply pillar logic: sections between two pillars get the same P
    if args.pillars and pillar_positions:
        print(f"Pillar boundaries: {pillar_positions}")
        segments = []
        start = 1
        for boundary in pillar_positions:
            if start <= boundary:
                segments.append((start, boundary))
                start = boundary + 1
        if start <= args.n_sections:
            segments.append((start, args.n_sections))
        print(f"Segments: {segments}")
        for seg_start, seg_end in segments:
            segment_sections = [f"S{i}" for i in range(seg_start, seg_end + 1)]
            segment_P_values = [p_map[s] for s in segment_sections if s in p_map and p_map[s] is not None]
            if segment_P_values:
                max_P = max(segment_P_values)
                for s in segment_sections:
                    if s in p_map and p_map[s] is not None:
                        p_map[s] = max_P
                        for row in rows:
                            if row["ROI"] == s:
                                row["P"] = max_P
                                row["Classe_P"] = p_label(max_P)

    # Tabela ordenada por P desc e |vel_90d| desc
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["P"], ascending=False, kind="mergesort")
        out = out.iloc[out["Vel_90d_mm_ano"].abs().sort_values(ascending=False).index]
        for col in ["Vel_90d_mm_ano", "Acc_90d_mm_ano2", "Vel_total_mm_ano", "Acc_total_mm_ano2"]:
            out[col] = out[col].astype(float).round(3)
        print("\nProbabilidade por seção (critério aceler./veloc.):\n")
        print(out.to_string(index=False))
    else:
        print("Aviso: nenhuma seção com dados suficientes para cálculo.")

    # Heatmap
    title = f"Heatmap de Probabilidade (P)"
    draw_heatmap_P(p_map, args.n_sections, args.nodata_color, title, save=args.save, dpi=args.dpi)

if __name__ == "__main__":
    main()
