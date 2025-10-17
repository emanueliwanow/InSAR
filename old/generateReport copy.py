import os
#os.chdir("/insar-data/Tocantins/mintpy")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mintpy.utils import readfile
import h5py
import csv
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

DATA_DIR = '/insar-data'
PROJECT_NAME = "JK_DF"
ASSET_NAME = "Ponte JK - Distrito Federal"
ANALYSIS_PERIOD = "20190101-20241218"
SECTIONS = {
    "S1": {
        "points_str": "-47.83325920129325,-15.81873316452596;-47.83310162113654,-15.81891545990351;-47.83296050522731,-15.81908778092585;-47.83275200947068,-15.81932810239634;-47.83268177522332,-15.81927661558415;-47.8325877998821,-15.81919673795594;-47.83279625013162,-15.81894843551723;-47.83294763827799,-15.81876942641095;-47.83310360671356,-15.81859626423708;-47.83325417687176,-15.81842274487094;-47.83334344852211,-15.81850665406241;-47.83342486997562,-15.81856709467188;-47.83325920129325,-15.81873316452596",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10;P11;P12"
    },
    "S2": {
        "points_str": "-47.8326672236397,-15.81937676006957;-47.83246892281269,-15.81960132792186;-47.83227908218731,-15.81982973459751;-47.83206376182515,-15.82009103906436;-47.83188943321083,-15.82031191030216;-47.83177592621238,-15.82022882655367;-47.83195511496124,-15.82000058741707;-47.83216371015857,-15.81973862180347;-47.83235521408289,-15.81950779551236;-47.83255302538389,-15.81927536526509;-47.8326672236397,-15.81937676006957",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S3": {
        "points_str": "-47.83185764214208,-15.8203500504891;-47.83172687596486,-15.82053291533565;-47.83158453906811,-15.82071210046835;-47.8314257349291,-15.820921641298;-47.83125387731305,-15.82116698087253;-47.83113532338714,-15.82108134091735;-47.83131467355729,-15.8208405629889;-47.83146940024641,-15.82063190509387;-47.83161306379795,-15.82044006695032;-47.83174907814085,-15.82027049298946;-47.83185764214208,-15.8203500504891",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S4": {
        "points_str": "-47.83107126849762,-15.82137548949193;-47.83093035536606,-15.82157909775925;-47.83077705208857,-15.82181217850331;-47.83063231413211,-15.82202691857276;-47.83051900876326,-15.82195456932367;-47.83067598046635,-15.82173718254914;-47.83083677652648,-15.82150866323718;-47.83097111456184,-15.82130441721838;-47.83109872093885,-15.82112635807245;-47.83120952921161,-15.82119718631885;-47.83107126849762,-15.82137548949193",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S5": {
        "points_str": "-47.83047541974339,-15.8220155235981;-47.83058806146458,-15.82209419957912;-47.83046004503718,-15.82228883389638;-47.83034151067179,-15.82246905060452;-47.83020875218824,-15.82267089372371;-47.83007599361577,-15.82287273726939;-47.82996268811126,-15.82280038739293;-47.83008637383492,-15.82259638553115;-47.83022321015063,-15.8223936599049;-47.83034515905878,-15.82221888736191;-47.83047541974339,-15.8220155235981",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S6": {
        "points_str": "-47.82992359230782,-15.8228657677668;-47.8300307534986,-15.82293793003942;-47.82991753450873,-15.82313193309088;-47.82977968198649,-15.82335245424636;-47.82964670413501,-15.82356556417557;-47.82952074667659,-15.82378986802595;-47.82939749741048,-15.82371483026975;-47.8295237156868,-15.82350106363944;-47.82966225038064,-15.82327403855943;-47.82979171771454,-15.82305533170299;-47.82992359230782,-15.8228657677668",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S7": {
        "points_str": "-47.82936389583777,-15.8237877871526;-47.82948971283641,-15.82386438866344;-47.82934168872514,-15.82411677373986;-47.82921641234575,-15.82433457307763;-47.82908002283686,-15.82458020339647;-47.82896706377802,-15.8247847435484;-47.82884596036142,-15.8247283098787;-47.8289486490461,-15.82451751646374;-47.82909735589877,-15.82425862781283;-47.82922519980632,-15.82404239163055;-47.82936389583777,-15.8237877871526",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S8": {
        "points_str": "-47.82893456606981,-15.82483415533988;-47.82885556814597,-15.82501439724622;-47.82874286864687,-15.82522947520063;-47.82862393015132,-15.8254649725241;-47.82851295550655,-15.82571569773552;-47.82838414867148,-15.82565457366285;-47.82849889453507,-15.82541998364309;-47.82863109295212,-15.82517526108279;-47.82874379233223,-15.82496018311962;-47.82882210835661,-15.82478644547682;-47.82893456606981,-15.82483415533988",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S9": {
        "points_str": "-47.82848328601585,-15.82577720637559;-47.82838777763918,-15.82597161463029;-47.82826071399944,-15.82621946421936;-47.8281551341592,-15.82648385492484;-47.82805228208491,-15.82672223061091;-47.82792509937764,-15.82665863569089;-47.82802076978186,-15.82643664586351;-47.82814893772353,-15.82616524995067;-47.82826992321948,-15.82591023985576;-47.82837193150067,-15.82570595020626;-47.82848328601585,-15.82577720637559",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S10": {
        "points_str": "-47.82793097496905,-15.82703239849842;-47.82785412163066,-15.82723124549549;-47.82776657565883,-15.8274408816437;-47.82767672343033,-15.827659491879;-47.82755074357756,-15.82761046835479;-47.82765128884142,-15.82738106937395;-47.8277271996951,-15.82717818833195;-47.82781406328792,-15.82697505737997;-47.82791922171422,-15.82672771041543;-47.82803262290204,-15.82677945415761;-47.82793097496905,-15.82703239849842",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S11": {
        "points_str": "-47.82766476648463,-15.82772140800117;-47.82758408057355,-15.82796981689404;-47.82750995505528,-15.82814264701838;-47.827437795553,-15.82837167998552;-47.82737647547927,-15.82859144968543;-47.82721462762637,-15.82853775175081;-47.82729342980304,-15.82830065593911;-47.8273727615215,-15.82808333688659;-47.82744311615369,-15.82789437202638;-47.82752689100624,-15.82766860212261;-47.82766476648463,-15.82772140800117",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
    "S12": {
        "points_str": "-47.82729209689649,-15.82886014413399;-47.82725679933368,-15.82901816376404;-47.82718494060932,-15.82922620608968;-47.82710919237554,-15.82941754986078;-47.82692798668936,-15.82934379259008;-47.82701536945798,-15.82911750941419;-47.82707226446078,-15.82894861617872;-47.82713597080719,-15.82878168640401;-47.82720686170036,-15.82860784027175;-47.82733289324984,-15.82864985790754;-47.82729209689649,-15.82886014413399",
        "labels": "P1;P2;P3;P4;P5;P6;P7;P8;P9;P10"
    },
}



os.chdir(f"{DATA_DIR}/{PROJECT_NAME}/miaplpy2")

# --- helpers ---
def parse_points(s):
    pts = []
    for item in s.split(";"):
        lon_str, lat_str = item.strip().split(",")
        pts.append((float(lat_str), float(lon_str)))
    return pts

def latlon_to_rc(meta, lat, lon):
    x0 = float(meta["X_FIRST"])
    y0 = float(meta["Y_FIRST"])
    dx = float(meta["X_STEP"])
    dy = float(meta["Y_STEP"])    # often negative
    width  = int(meta["WIDTH"])
    length = int(meta["LENGTH"])
    col = int(round((lon - x0) / dx))
    row = int(round((lat - y0) / dy))
    if not (0 <= row < length and 0 <= col < width):
        raise IndexError(f"Point ({lat:.6f},{lon:.6f}) -> (row={row}, col={col}) outside data [{length}x{width}]")
    return row, col

def _to_str_array(x):
    """Normalize various 'date list' representations to an array of 'YYYYMMDD' strings."""
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("S", "U", "O"):
            arr = [xi.decode() if isinstance(xi, (bytes, bytearray)) else str(xi) for xi in x.tolist()]
        else:  # e.g., int array like [20171005, ...]
            arr = [f"{int(xi):08d}" for xi in x.tolist()]
    elif isinstance(x, (bytes, bytearray, str)):
        s = x.decode() if isinstance(x, (bytes, bytearray)) else x
        arr = [t for t in s.replace(",", " ").split() if t]
    else:
        arr = [str(x)]
    return np.array(arr)

def get_dates(h5_file):
    """Robust date reader: try '/date' dataset; fall back to DATE_LIST or similar attrs (root or '/timeseries')."""
    try:
        dates_raw, _ = readfile.read(h5_file, datasetName="date")
        return np.array([d.decode() if isinstance(d, (bytes, bytearray)) else str(d) for d in dates_raw])
    except Exception:
        pass

    with h5py.File(h5_file, "r") as f:
        if "date" in f:
            return _to_str_array(f["date"][:])
        root_keys = ["DATE_LIST", "DATE", "DATES"]
        ts_keys   = ["DATE_LIST", "DATE", "DATES", "date"]
        for k in root_keys:
            if k in f.attrs:
                return _to_str_array(f.attrs[k])
        if "timeseries" in f:
            for k in ts_keys:
                if k in f["timeseries"].attrs:
                    return _to_str_array(f["timeseries"].attrs[k])
        try:
            _, meta = readfile.read(h5_file)
            for k in ["DATE_LIST", "DATE", "DATES"]:
                if k in meta:
                    return _to_str_array(meta[k])
        except Exception:
            pass

    raise RuntimeError("Could not find dates in the file (no '/date' dataset or date attributes like 'DATE_LIST').")




h5_file = f'{DATA_DIR}/{PROJECT_NAME}/miaplpy2/network_delaunay_4/geo/geo_timeseries_ERA5_demErr.h5'


# ---------- section-level anomaly config ----------
SEC_ROLL_MED_WIN = 5   # window (in acquisitions) for rolling median baseline (odd numbers: 5, 7, 9...)
SEC_CM_THRESH    = 0.2 # cm threshold on residual from baseline
SEC_Z_THRESH     = 2.0 # robust z-score threshold on residual
SEC_MIN_CONSEC   = 2   # require at least this many consecutive anomalous dates

# ---------- small utilities ----------
def _max_run_length(bool_arr):
    """Max consecutive True run length in a 1D boolean array."""
    m = 0
    cur = 0
    for b in bool_arr:
        cur = cur + 1 if b else 0
        if cur > m:
            m = cur
    return m

def _enforce_min_consecutive(mask, k):
    """Keep only runs of True with length >= k."""
    if k <= 1:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    n = mask.size
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out

def generate_pdf_report(pdf_filename, project_name, asset_name, analysis_period, 
                       fig_files, anomaly_data, section_summary_rows):
    """Generate a comprehensive PDF report with title page, plots, and anomaly table."""
    
    with PdfPages(pdf_filename) as pdf:
        # Title Page
        fig_title = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        # Main title
        plt.text(0.5, 0.8, f'InSAR Analysis Report', 
                ha='center', va='center', fontsize=24, fontweight='bold')
        
        # Project and asset info
        plt.text(0.5, 0.7, f'Project: {project_name}', 
                ha='center', va='center', fontsize=18)
        plt.text(0.5, 0.65, f'Asset: {asset_name}', 
                ha='center', va='center', fontsize=18)
        
        # Analysis period
        plt.text(0.5, 0.55, f'Analysis Period', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        plt.text(0.5, 0.5, f'{analysis_period}', 
                ha='center', va='center', fontsize=14)
        
        # Report generation date
        plt.text(0.5, 0.3, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                ha='center', va='center', fontsize=12)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)
        
        # Add the three plots
        for fig_file in fig_files:
            if os.path.exists(fig_file):
                # Read and display the saved figure
                fig = plt.figure(figsize=(11, 8.5))
                img = plt.imread(fig_file)
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # Anomaly Summary Table
        if section_summary_rows:
            fig_table = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            
            # Table title
            plt.text(0.5, 0.95, 'Flagged Anomalies Summary', 
                    ha='center', va='top', fontsize=16, fontweight='bold')
            
            # Prepare table data
            headers = ['Section', 'Flagged Dates', 'Max Consecutive', 
                      'First Anomaly', 'Last Anomaly', 'Max |Residual| (cm)', 'Max |Z-score|']
            
            table_data = []
            for row in section_summary_rows:
                sec_name, n_flag, max_run, first_date, last_date, max_abs_resid, max_abs_z = row[:7]
                table_data.append([
                    sec_name, 
                    str(n_flag), 
                    str(max_run),
                    first_date if first_date else 'N/A',
                    last_date if last_date else 'N/A',
                    f'{max_abs_resid:.2f}',
                    f'{max_abs_z:.2f}'
                ])
            
            # Create table
            table = plt.table(cellText=table_data,
                             colLabels=headers,
                             cellLoc='center',
                             loc='center',
                             bbox=[0.05, 0.1, 0.9, 0.8])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color rows with anomalies
            for i, row in enumerate(table_data):
                if int(row[1]) > 0:  # If there are flagged dates
                    for j in range(len(headers)):
                        table[(i+1, j)].set_facecolor('#ffcccc')
            
            pdf.savefig(fig_table, bbox_inches='tight')
            plt.close(fig_table)
    
    print(f"PDF report generated: {pdf_filename}")

def _rolling_median(x, win):
    """Rolling median with edge handling (centered window; uses nanmedian)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    half = win // 2
    rm = np.full(n, np.nan)
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        rm[i] = np.nanmedian(x[i0:i1])
    return rm

# ---------- read data ----------
ts, meta = readfile.read(h5_file)  # 3-D (n_date, length, width), meters
print("timeseries shape:", ts.shape)

dates = get_dates(h5_file)
dates_dt = np.array([datetime.strptime(d, "%Y%m%d") for d in dates])

# ---------- compute per-section averages + section-level anomalies ----------
section_avg = {}             # name -> (n_dates,) cm
section_anom_mask = {}       # name -> (n_dates,) bool
section_resid = {}           # name -> residual cm (avg - rolling median)
section_flagged_idx = {}     # name -> indices of anomalous dates
section_z = {}               # name -> robust z-score array (n_dates,)
section_summary_rows = []    # rows for per-section summary

for sec_name, cfg in SECTIONS.items():
    pts = parse_points(cfg["points_str"])

    # collect each point's time-series (cm)
    series_cm = []
    for (lat, lon) in pts:
        r, c = latlon_to_rc(meta, lat, lon)
        disp_m = ts[:, r, c]
        series_cm.append(disp_m * 100.0)  # m -> cm

    arr = np.vstack(series_cm)                # (n_pts, n_dates)
    avg_cm = np.nanmean(arr, axis=0)         # (n_dates,)
    section_avg[sec_name] = avg_cm

    # residual vs. rolling-median baseline
    base = _rolling_median(avg_cm, SEC_ROLL_MED_WIN)
    resid = avg_cm - base
    section_resid[sec_name] = resid

    # robust z-score of residual
    med_r = np.nanmedian(resid)
    mad_r = np.nanmedian(np.abs(resid - med_r))
    robust_std = 1.4826 * mad_r + 1e-9
    z = (resid - med_r) / robust_std
    section_z[sec_name] = z

    # anomaly mask and persistence constraint
    raw_mask = (np.abs(resid) > SEC_CM_THRESH) & (np.abs(z) > SEC_Z_THRESH)
    mask = _enforce_min_consecutive(raw_mask, SEC_MIN_CONSEC)
    section_anom_mask[sec_name] = mask
    section_flagged_idx[sec_name] = np.where(mask)[0]

    # summary
    n_flag = int(np.nansum(mask))
    max_run = _max_run_length(mask)
    first_date = dates[np.where(mask)[0][0]] if n_flag > 0 else ""
    last_date  = dates[np.where(mask)[0][-1]] if n_flag > 0 else ""
    max_abs_resid = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else 0.0
    max_abs_z     = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.0

    section_summary_rows.append([
        sec_name, n_flag, max_run, first_date, last_date, max_abs_resid, max_abs_z,
        SEC_ROLL_MED_WIN, SEC_CM_THRESH, SEC_Z_THRESH, SEC_MIN_CONSEC
    ])

# ---------- plotting ----------
outbase = os.path.splitext(os.path.basename(h5_file))[0]
fig_name = f"{outbase}_ts_sectionAvgs.png"
csv_name = f"{outbase}_ts_sectionAvgs.csv"
sec_anom_date_csv = f"{outbase}_section_anom_dates.csv"
sec_anom_summary_csv = f"{outbase}_section_anom_summary.csv"
resid_fig_name = f"{outbase}_ts_sectionResiduals.png"
z_fig_name = f"{outbase}_ts_sectionZscores.png"

# 1) section averages with anomaly markers
fig1 = plt.figure(figsize=(12, 8))
for sec_name, avg_series_cm in section_avg.items():
    plt.plot(dates_dt, avg_series_cm, linewidth=2.2, marker="o", markersize=3, label=f"{sec_name} avg")
    idx = section_flagged_idx[sec_name]
    if idx.size > 0:
        plt.scatter(dates_dt[idx], avg_series_cm[idx], s=35, marker="x", zorder=4, label=f"{sec_name} anomaly dates")
plt.xlabel("Date")
plt.ylabel("Displacement (cm, LOS)")
plt.title("Bridge Sections — Average Displacement with Section-Level Anomalies")
plt.grid(True, alpha=0.3)
plt.legend(title="Legend")
plt.tight_layout()
plt.savefig(fig_name, dpi=300, bbox_inches='tight')
plt.show()





# 2) residual plot
fig2 = plt.figure(figsize=(12, 8))
for sec_name, resid in section_resid.items():
    plt.plot(dates_dt, resid, linewidth=1.8, marker="o", markersize=3, label=f"{sec_name} residual")
    idx = section_flagged_idx[sec_name]
    if idx.size > 0:
        plt.scatter(dates_dt[idx], resid[idx], s=35, marker="x", zorder=4, label=f"{sec_name} anomaly dates")
plt.axhline(0.0, linewidth=1.0)
plt.axhline(+SEC_CM_THRESH, linewidth=1.0, linestyle="--")
plt.axhline(-SEC_CM_THRESH, linewidth=1.0, linestyle="--")
plt.xlabel("Date")
plt.ylabel("Residual (cm)")
plt.title("Bridge Sections — Residuals (Average − Rolling Median)")
plt.grid(True, alpha=0.3)
plt.legend(title="Legend")
plt.tight_layout()
plt.savefig(resid_fig_name, dpi=300, bbox_inches='tight')
plt.show()

# 3) z-score plot
fig3 = plt.figure(figsize=(12, 8))
for sec_name, z in section_z.items():
    plt.plot(dates_dt, z, linewidth=1.8, marker="o", markersize=3, label=f"{sec_name} z-score")
    idx = section_flagged_idx[sec_name]
    if idx.size > 0:
        plt.scatter(dates_dt[idx], z[idx], s=35, marker="x", zorder=4, label=f"{sec_name} anomaly dates")
plt.axhline(0.0, linewidth=1.0)
plt.axhline(+SEC_Z_THRESH, linewidth=1.0, linestyle="--")
plt.axhline(-SEC_Z_THRESH, linewidth=1.0, linestyle="--")
plt.xlabel("Date")
plt.ylabel("Robust z-score")
plt.title("Bridge Sections — Z-score of Residuals")
plt.grid(True, alpha=0.3)
plt.legend(title="Legend")
plt.tight_layout()
plt.savefig(z_fig_name, dpi=300, bbox_inches='tight')
plt.show()

# ---------- smoothing filter configuration ----------
SMOOTH_FILTER_WIN = 21  # window size for smoothing filter (odd number)

def apply_smoothing_filter(series, window=5, method='gaussian'):
    """
    Apply smoothing filter to make time series smoother.
    
    Parameters:
    - series: input time series (1D array)
    - window: window size for smoothing
    - method: smoothing method ('moving_avg', 'gaussian', 'savgol')
    
    Returns:
    - smoothed_series: smoothed time series
    """
    from scipy import ndimage
    from scipy.signal import savgol_filter
    
    series = np.array(series, dtype=float)
    
    if method == 'moving_avg':
        # Simple moving average
        return _rolling_median(series, window)  # Reuse existing rolling function but with mean
    
    elif method == 'gaussian':
        # Gaussian smoothing - good for preserving overall trends while reducing noise
        sigma = window / 3.0  # Standard deviation for Gaussian kernel
        # Handle NaN values
        mask = ~np.isnan(series)
        if np.any(mask):
            smoothed = series.copy()
            smoothed[mask] = ndimage.gaussian_filter1d(series[mask], sigma=sigma)
            return smoothed
        else:
            return series
    
    elif method == 'savgol':
        # Savitzky-Golay filter - good for preserving peaks while smoothing
        if len(series) > window and window >= 3:
            # Handle NaN values by interpolating first
            mask = ~np.isnan(series)
            if np.sum(mask) > window:
                from scipy.interpolate import interp1d
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 2:
                    f = interp1d(valid_indices, series[valid_indices], 
                               kind='linear', fill_value='extrapolate')
                    series_interp = f(np.arange(len(series)))
                    return savgol_filter(series_interp, window, polyorder=2)
        return series
    
    return series

def _rolling_mean(x, win):
    """Rolling mean with edge handling."""
    x = np.asarray(x, dtype=float)
    n = x.size
    half = win // 2
    rm = np.full(n, np.nan)
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        rm[i] = np.nanmean(x[i0:i1])
    return rm

# 4) smoothed averages plot
section_smoothed = {}  # name -> (n_dates,) cm, smoothed

fig4 = plt.figure(figsize=(12, 8))
for sec_name, avg_series_cm in section_avg.items():
    # Apply smoothing filter
    smoothed_series = apply_smoothing_filter(
        avg_series_cm, 
        window=SMOOTH_FILTER_WIN, 
        method='gaussian'
    )
    section_smoothed[sec_name] = smoothed_series
    
    # Plot original series (lighter, thinner)
    #plt.plot(dates_dt, avg_series_cm, linewidth=1.2, alpha=0.5, linestyle='--',
    #         label=f"{sec_name} original")
    
    # Plot smoothed series (main line, thicker and smoother)
    #plt.plot(dates_dt, smoothed_series, linewidth=2.5, marker="o", markersize=4, 
    #         label=f"{sec_name} smoothed")
    plt.plot(dates_dt, smoothed_series, linewidth=2.5, marker="o", markersize=2.2, 
             label=f"{sec_name} smoothed")


    
plt.xlabel("Date")
plt.ylabel("Displacement (cm, LOS)")
plt.title(f"Bridge Sections — Smoothed Average Displacement (Gaussian, {SMOOTH_FILTER_WIN}-pt window)")
plt.grid(True, alpha=0.3)
plt.legend(title="Legend")
plt.tight_layout()

# Save the smoothed figure
smoothed_fig_name = f"{outbase}_ts_sectionAvgs_smoothed.png"
plt.savefig(smoothed_fig_name, dpi=300, bbox_inches='tight')
plt.show()

# ---------- CSV exports ----------
# per-date section averages
# with open(csv_name, "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["date"] + list(section_avg.keys()))
#     for i, d in enumerate(dates):
#         w.writerow([d] + [float(section_avg[name][i]) for name in section_avg.keys()])

# # per-date section-level anomaly flags (1/0)
# with open(sec_anom_date_csv, "w", newline="") as f:
#     w = csv.writer(f)
#     header = ["date"] + [f"{name}_flag" for name in section_avg.keys()]
#     w.writerow(header)
#     for i, d in enumerate(dates):
#         w.writerow([d] + [int(section_anom_mask[name][i]) for name in section_avg.keys()])

# # per-section summary
# with open(sec_anom_summary_csv, "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow([
#         "section", "n_flagged_dates", "max_consecutive_flagged",
#         "first_flagged_date", "last_flagged_date",
#         "max_abs_residual_cm", "max_abs_robust_z",
#         "cfg_roll_win", "cfg_cm_thresh", "cfg_z_thresh", "cfg_min_consec"
#     ])
#     w.writerows(section_summary_rows)

# console summary
for row in section_summary_rows:
    sec_name, n_flag, max_run, first_date, last_date, max_abs_resid, max_abs_z, *_ = row
    if n_flag == 0:
        print(f"[{sec_name}] No section-level anomalies.")
    else:
        print(f"[{sec_name}] {n_flag} anomalous dates (max run={max_run}). "
              f"First: {first_date}, Last: {last_date}. "
              f"Max |residual|={max_abs_resid:.2f} cm, Max |z|={max_abs_z:.2f}.")

# ---------- Generate PDF Report ----------
pdf_filename = f"{outbase}_InSAR_Report.pdf"
figure_files = [fig_name,smoothed_fig_name, resid_fig_name, z_fig_name]

generate_pdf_report(
    pdf_filename=pdf_filename,
    project_name=PROJECT_NAME,
    asset_name=ASSET_NAME,
    analysis_period=ANALYSIS_PERIOD,
    fig_files=figure_files,
    anomaly_data=section_anom_mask,
    section_summary_rows=section_summary_rows
)