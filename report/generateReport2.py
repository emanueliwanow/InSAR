#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build an InSAR project PDF report from 4 standard figures using PdfPages.

Inputs:
  - Project name (e.g., "Tocantins24meses")
  - Project title (free text)
  - Analysis period START_DATE and END_DATE (YYYY-MM-DD)

It looks for these images under /insar-data/<project>/report/:
  <project>_heatmap2_base.png
  <project>_heatmap2_same_probability_for_sections_between_pillars.png
  <project>_Plot.png
  <project>_velocity_sections.png

Example:
  python make_report.py \
    --project Tocantins24meses \
    --title "Tocantins – 24 meses (Sentinel-1 InSAR)" \
    --start 2022-12-01 --end 2024-12-01 \
    --out Tocantins24meses_report.pdf
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from textwrap import fill

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def add_cover_page(pp, project, title, start_date, end_date):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape in inches
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Add logo at the top
    logo_path = Path("/insar-data/InSAR/pictures/SAW_Logo.jpg")
    if logo_path.exists():
        try:
            logo = mpimg.imread(str(logo_path))
            # Create a larger inset axis for the logo at the top center
            logo_ax = fig.add_axes([0.25, 0.80, 0.5, 0.15])
            logo_ax.imshow(logo)
            logo_ax.axis("off")
        except Exception as e:
            print(f"Warning: Could not load logo: {e}")

    header = title if title else project
    period = f"Analysis period: {start_date} → {end_date}"
    subtitle = f"Project: {project}"
    gen = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    ax.text(0.5, 0.70, header, ha="center", va="center", fontsize=28, fontweight="bold")
    ax.text(0.5, 0.58, period, ha="center", va="center", fontsize=16)
    ax.text(0.5, 0.52, subtitle, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.12, gen, ha="center", va="center", fontsize=10, alpha=0.7)

    # Simple footer
    ax.text(0.5, 0.05, "InSAR Report (ISCE2 / Miaplpy / MintPy)", ha="center", va="center", fontsize=10, alpha=0.6)
    pp.savefig(fig, dpi=200)
    plt.close(fig)


def add_image_page(pp, img_path: Path, caption: str):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.80])  # generous margins

    if img_path.exists():
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.axis("off")
            # caption area
            fig.text(0.03, 0.04, fill(caption, 130), fontsize=11)
        except Exception as e:
            ax.axis("off")
            fig.text(0.5, 0.55, "Error loading image", ha="center", va="center", fontsize=16, color="red")
            fig.text(0.5, 0.48, str(e), ha="center", va="center", fontsize=10)
            fig.text(0.03, 0.04, fill(caption, 130), fontsize=11)
    else:
        ax.axis("off")
        fig.text(0.5, 0.55, "Missing image", ha="center", va="center", fontsize=16, color="red")
        fig.text(0.5, 0.48, str(img_path), ha="center", va="center", fontsize=10)
        fig.text(0.03, 0.04, fill(caption, 130), fontsize=11)

    pp.savefig(fig, dpi=200)
    plt.close(fig)

def add_heatmap_probability_explanation_page(pp):
    """Adiciona uma página explicando como o P (probabilidade) é calculado."""
    fig = plt.figure(figsize=(11.69, 16.54))  # A4 landscape
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Título
    fig.text(0.5, 0.92, "Como o P do heatmap é calculado",
             ha="center", va="center", fontsize=18, fontweight="bold")

    # Helper para escrever blocos com quebra automática
    def write_block(x, y, text, size=11, bold=False, mono=False, width=120, dy=0.01):
        fontweight = "bold" if bold else "normal"
        family = "monospace" if mono else None
        wrapped = fill(text, width)
        for i, line in enumerate(wrapped.splitlines()):
            fig.text(x, y - i*dy, line, fontsize=size, fontweight=fontweight, fontfamily=family)
        # retorna y final após o bloco
        lines = len(wrapped.splitlines())
        return y - lines*dy - 0.010  # espaço extra após o parágrafo

    x = 0.05
    y = 0.86

    # 1) Entradas
    y = write_block(x, y, "1) Entradas", size=13, bold=True)
    y = write_block(x, y, "- Dois CSVs por seção (S_k) com deslocamentos em mm:")
    y = write_block(x, y, "  1) Série TOTAL (todo o histórico).  2) Janela RECENTE (~90 dias).")
    y = write_block(x, y, "- Coluna de data (datetime) e colunas S1..SN. Remoção de nulos. "
                          "Exigência de ≥4 amostras por ajuste. Para 90d com <4 pontos, faz-se fallback "
                          "recortando 90d do total; persistindo <4, a seção fica 'Sem dados'.")

    # 2) Métricas por regressão
    y = write_block(x, y, "2) Métricas estimadas por regressão", size=13, bold=True)
    y = write_block(x, y, "Tempo em anos desde a primeira amostra da janela considerada.")
    y = write_block(x, y, "Velocidade (mm/ano): ajuste linear y = a·x + b → vel = a.")
    y = write_block(x, y, "Aceleração (mm/ano²): ajuste quadrático y = c·x² + d·x + e → acc = 2c.")
    y = write_block(x, y, "Resultado por seção: vel_total, acc_total (histórico) e vel_90d, acc_90d (recentes).")

    # 3) Cálculo do P
    y = write_block(x, y, "3) Cálculo de P (1–5)", size=13, bold=True)

    # 3.1 Base por |vel_total|
    y = write_block(x, y, "3.1 Base por |vel_total| (mm/ano):", bold=True)
    tabela = (
        "  |vel_total| < 0.5           →  P = 1\n"
        "  0.5 ≤ |vel_total| < 1.0     →  P = 2\n"
        "  1.0 ≤ |vel_total| < 2.0     →  P = 3\n"
        "  2.0 ≤ |vel_total| < 4.0     →  P = 4\n"
        "  |vel_total| ≥ 4.0           →  P = 5"
    )
    y = write_block(x, y, tabela, mono=True)

    # 3.2 Curvatura de longo prazo
    y = write_block(x, y, "3.2 Curvatura (histórico):", bold=True)
    y = write_block(x, y, "+1 nível se |acc_total| > 1.0 mm/ano² (indica curvatura sustentada).")

    # 3.3 Modulação pelos últimos 90 dias
    y = write_block(x, y, "3.3 Modulação pelos últimos 90 dias (não substitui o histórico):", bold=True)
    y = write_block(x, y, "- Agravamento consistente (sobe 1): mesmo sinal entre vel_90d e vel_total "
                          "E |vel_90d| ≥ 1.25·|vel_total| E |vel_90d| ≥ 0.5 mm/ano.")
    y = write_block(x, y, "- Contradição recente forte (desce 1): sinais opostos entre vel_90d e vel_total "
                          "E |vel_90d| > 1.0 mm/ano.")
    y = write_block(x, y, "- Janela recente 'calma' (desce 1): |vel_90d| < 0.5 E |acc_90d| < 0.3 mm/ano².")

    # 3.4 Limites
    y = write_block(x, y, "3.4 Limites:", bold=True)
    y = write_block(x, y, "Após os ajustes, aplica-se clamp: P ∈ {1, 2, 3, 4, 5}.")

    # 4) Casos de borda
    y = write_block(x, y, "4) Casos de borda", size=13, bold=True)
    y = write_block(x, y, "- vel_total = 0 ou vel_90d = 0: o teste de 'mesmo sinal' é tratado de forma tolerante "
                          "(considera 'mesmo') para não penalizar ruído muito baixo.")
    y = write_block(x, y, "- Se alguma métrica não puder ser estimada (poucos pontos), a seção fica 'Sem dados'.")
    y = write_block(x, y, "- Opcional: regra de pilares pode uniformizar P entre seções contidas no mesmo vão.")

    # 5) Saída
    y = write_block(x, y, "5) Saída", size=13, bold=True)
    y = write_block(x, y, "Para cada seção: Vel_90d_mm_ano, Acc_90d_mm_ano2, Vel_total_mm_ano, "
                          "Acc_total_mm_ano2, P, Classe_P; além de um heatmap 1×N colorido por P.")

    # 6) Intuição
    y = write_block(x, y, "6) Intuição resumida", size=13, bold=True)
    y = write_block(x, y, "O longo prazo comanda (robusto ao ruído); curvatura histórica eleva risco; "
                          "os 90 dias apenas modulam (agravam se coerentes e mais rápidos; aliviam se contradizem "
                          "forte ou se estão calmos).")

    # Rodapé
    fig.text(0.5, 0.04, "Esta lógica prioriza estabilidade do histórico sem perder sensibilidade a pioras recentes.",
             ha="center", va="center", fontsize=9, alpha=0.7)

    pp.savefig(fig, dpi=200)
    plt.close(fig)


def add_images_from_folder_page(pp, folder_path: Path, caption: str = "", title: str = ""):
    """
    Add a single PDF page with all images from a folder arranged in one column.
    The page size adapts to fit all images.
    
    Parameters:
    -----------
    pp : PdfPages
        The PDF writer object
    folder_path : Path
        Path to the folder containing images
    caption : str, optional
        Caption text to display at the bottom of the page
    title : str, optional
        Title text to display at the top of the page
    """
    # Find all image files in the folder
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    image_files = []
    
    if folder_path.exists() and folder_path.is_dir():
        for ext in image_extensions:
            image_files.extend(sorted(folder_path.glob(f"*{ext}")))
            image_files.extend(sorted(folder_path.glob(f"*{ext.upper()}")))
        # Remove duplicates and sort
        image_files = sorted(list(set(image_files)))
    
    if not image_files:
        # No images found - show message on standard page
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, f"No images found in:\n{folder_path}", 
                ha="center", va="center", fontsize=14, color="red")
        if title:
            fig.text(0.5, 0.95, title, ha="center", va="top", fontsize=14, fontweight="bold")
        if caption:
            fig.text(0.03, 0.04, fill(caption, 130), fontsize=11)
        pp.savefig(fig, dpi=200)
        plt.close(fig)
        return
    
    # Load all images and calculate their aspect ratios
    images_data = []
    for img_path in image_files:
        try:
            img = mpimg.imread(str(img_path))
            height, width = img.shape[:2]
            aspect_ratio = height / width if width > 0 else 1.0
            images_data.append({
                'path': img_path,
                'img': img,
                'aspect_ratio': aspect_ratio,
                'error': None
            })
        except Exception as e:
            images_data.append({
                'path': img_path,
                'img': None,
                'aspect_ratio': 0.5,  # default aspect for error display
                'error': str(e)
            })
    
    # Calculate page dimensions
    page_width = 8.27  # A4 width in inches
    left_margin = 0.5  # inches
    right_margin = 0.5  # inches
    image_width = page_width - left_margin - right_margin
    
    # Space for title and caption
    title_space = 0.8 if title else 0.3  # inches
    caption_space = 0.6 if caption else 0.3  # inches
    image_spacing = 0.3  # inches between images
    label_space = 0.25  # space for filename labels
    
    # Calculate total height needed
    total_image_height = sum(
        img_data['aspect_ratio'] * image_width + label_space 
        for img_data in images_data
    )
    total_spacing = image_spacing * (len(images_data) - 1) if len(images_data) > 1 else 0
    page_height = title_space + total_image_height + total_spacing + caption_space
    
    # Create figure with calculated height
    fig = plt.figure(figsize=(page_width, page_height))
    fig.patch.set_facecolor("white")
    
    # Add title if provided
    if title:
        title_y = 1 - (title_space / 2) / page_height
        fig.text(0.5, title_y, title, ha="center", va="center", 
                fontsize=14, fontweight="bold")
    
    # Calculate starting position for first image
    current_y = page_height - title_space
    
    # Add images in a single column
    for img_data in images_data:
        img_height = img_data['aspect_ratio'] * image_width
        
        # Calculate position in figure coordinates (0 to 1)
        bottom = (current_y - img_height) / page_height
        left = left_margin / page_width
        width = image_width / page_width
        height = img_height / page_height
        
        ax = fig.add_axes([left, bottom, width, height])
        
        if img_data['error'] is None:
            # Display image
            ax.imshow(img_data['img'])
            ax.axis("off")
        else:
            # Display error message
            ax.axis("off")
            ax.text(0.5, 0.5, f"Error loading:\n{img_data['path'].name}\n{img_data['error']}", 
                   ha="center", va="center", fontsize=10, color="red",
                   transform=ax.transAxes)
        
        # Add filename label below image
        label_y = (current_y - img_height - label_space/2) / page_height
        fig.text(0.5, label_y, img_data['path'].name, 
                ha="center", va="center", fontsize=9)
        
        # Move to next position
        current_y -= (img_height + label_space + image_spacing)
    
    # Add caption if provided
    if caption:
        caption_y = caption_space / 2 / page_height
        fig.text(0.5, caption_y, fill(caption, 100), 
                ha="center", va="center", fontsize=11)
    
    pp.savefig(fig, dpi=200)
    plt.close(fig)


def add_summary_page(pp, included, missing):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.5, 0.88, "Report Summary", ha="center", va="center", fontsize=20, fontweight="bold")

    y = 0.78
    ax.text(0.1, y, "Included figures:", fontsize=14, fontweight="bold"); y -= 0.04
    if included:
        for p in included:
            ax.text(0.12, y, f"• {p}", fontsize=12); y -= 0.03
    else:
        ax.text(0.12, y, "— none —", fontsize=12); y -= 0.03

    y -= 0.03
    ax.text(0.1, y, "Missing figures:", fontsize=14, fontweight="bold"); y -= 0.04
    if missing:
        for p in missing:
            ax.text(0.12, y, f"• {p}", fontsize=12); y -= 0.03
    else:
        ax.text(0.12, y, "— none —", fontsize=12); y -= 0.03

    pp.savefig(fig, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate an InSAR PDF report from standard project figures.")
    parser.add_argument("--project", required=True, help='Project name, e.g. "Tocantins24meses"')
    parser.add_argument("--title", required=True,type=str, help='Project title for the cover page')
    parser.add_argument("--start", required=True, help='Analysis START_DATE, e.g. "2022-12-01"')
    parser.add_argument("--end", required=True, help='Analysis END_DATE, e.g. "2024-12-01"')
    parser.add_argument("--base-dir", default="/insar-data", help='Base data directory (default: /insar-data)')
    parser.add_argument("--out", default=None, help="Output PDF path (default: ./<project>_report.pdf)")
    args = parser.parse_args()

    project = args.project
    title = args.title
    start_date = args.start
    end_date = args.end
    base_dir = Path(args.base_dir)

    if args.out:
        out_pdf = Path(args.out)
    else:
        out_pdf = Path(f"{project}_report.pdf")

    report_dir = base_dir / project / "report"

    # Expected images and human-friendly captions
    expected = [
        (f"{project}_Sections.png",
         "Sections of the bridge."),
        (f"SPlot.png",
         "S Score Breakdown by Section."),
        # (f"{project}_heatmap_by_section.png",
        #  "Base probability heatmap (all sections)."),
        # (f"{project}_heatmap_by_span.png",
        #  "Probability heatmap with equalized probabilities for sections between pillars."),
        # (f"{project}_Plot.png",
        #  "Displacement time series per section (trend and recent behavior)."),
        # (f"{project}_velocityMap.png",
        #  "Velocity pixels clipped to bridge sections (geometry overlay)."),
        # (f"{project}_velocityMap_absolute.png",
        #  "Absolute velocity pixels clipped to bridge sections (geometry overlay)."),
        # (f"{project}_Velocity.png",
        #  "Velocity of displacement time series."),
        # (f"{project}_Acceleration.png",
        #  "Acceleration of displacement time series."),
    ]

    # Create PDF
    with PdfPages(out_pdf) as pp:
        # metadata
        d = pp.infodict()
        d["Title"] = f"{title} – InSAR Report"
        d["Author"] = "InSAR Pipeline (ISCE2 / Miaplpy / MintPy)"
        d["Subject"] = f"Project {project} | {start_date} to {end_date}"
        d["Keywords"] = "InSAR, Sentinel-1, ISCE2, Miaplpy, MintPy, report"
        d["CreationDate"] = datetime.now()
        d["ModDate"] = datetime.now()

        # cover
        add_cover_page(pp, project, title, start_date, end_date)

        included, missing = [], []
        for fname, caption in expected:
            path = report_dir / fname
            add_image_page(pp, path, caption)
            (included if path.exists() else missing).append(str(path))
        #add_images_from_folder_page(pp, Path(f"/insar-data/{project}/report/variables"), title="Variables")
        #add_images_from_folder_page(pp, Path(f"/insar-data/{project}/report/scores"), title="Variables Scores")
        #add_images_from_folder_page(pp, Path(f"/insar-data/{project}/report/residuals"), title="Residuals Analysis")

        #dd_heatmap_probability_explanation_page(pp)
        
        # summary / appendix
        add_summary_page(pp, included, missing)

    print(f"Report saved to: {out_pdf.resolve()}")


if __name__ == "__main__":
    main()
