import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import warnings
import os
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
CNT_FILE  = "/Users/ianchen/Desktop/eegdata/EEGproject/Data3/Function-041.cnt"
OUT_DIR   = "/Users/ianchen/Desktop/eegdata/EEGproject/preprocessing_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    "raw"    : "#C0392B",   # deep red
    "clean"  : "#1A7A6E",   # dark teal
    "accent" : "#D35400",   # burnt orange
    "bg"     : "#FFFFFF",   # white
    "panel"  : "#F7F8FA",   # very light gray
    "text"   : "#1A1A2E",   # near black
    "grid"   : "#D0D4DE",   # light gray
    "annot"  : "#B7770D",   # amber
}

def apply_style(fig, axes):
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in np.array(axes).flatten():
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        ax.spines[:].set_color(COLORS["grid"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.9)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ═════════════════════════════════════════════════════════════════════════════
print("Loading CNT file …")
raw = mne.io.read_raw_cnt(CNT_FILE, preload=True, verbose=False)

# Drop the Trigger channel (non-EEG)
raw.drop_channels(["Trigger"])

# Store a *clean copy* of raw before any processing (for comparison)
raw_orig = raw.copy()

sfreq   = raw.info["sfreq"]           # 1000 Hz
n_ch    = len(raw.ch_names)           # 66 EEG channels
dur_sec = raw.times[-1]

# Representative channels for plots
EEG_CH   = "FZ"        # frontal midline
EOG_CH   = "FP1"       # near eye, shows blinks
OCC_CH   = "OZ"        # occipital
TEMP_CH  = "T7"        # temporal (often EMG)
PLOT_CHS = [EEG_CH, EOG_CH, OCC_CH, TEMP_CH]

# Segment used for time-series plots (1 second zoom)
T0, T1 = 2, 3   # 2–3 s: captures blink & clean signal region

def get_seg(raw_obj, ch, t0=T0, t1=T1):
    idx = raw_obj.ch_names.index(ch)
    start = int(t0 * sfreq)
    stop  = int(t1 * sfreq)
    sig = raw_obj._data[idx, start:stop] * 1e6   # → µV
    sig = sig - sig.mean()   # demean for visualization (removes DC offset)
    return raw_obj.times[start:stop], sig

# ═════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
print("Applying preprocessing pipeline …")

# Step 1 – Average reference (before filter so ICA works well)
raw.set_eeg_reference("average", projection=False, verbose=False)

# Step 2 – Bandpass 1–40 Hz  (removes DC drift & high-freq EMG)
raw.filter(l_freq=1.0, h_freq=40.0,
           method="fir", fir_design="firwin", verbose=False)

# Step 3 – Notch filter at 60 Hz (powerline)
raw.notch_filter(freqs=[60], verbose=False)

# Step 4 – ICA for eye-blink removal
print("  Running ICA (this may take a moment) …")
ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                             max_iter=400, random_state=42)
ica.fit(raw, verbose=False)

# Auto-detect EOG components using FP1 as proxy channel
eog_indices, eog_scores = ica.find_bads_eog(
    raw, ch_name="FP1", threshold=2.5, verbose=False
)
print(f"  ICA components flagged as EOG artifacts: {eog_indices}")
ica.exclude = eog_indices[:2] if len(eog_indices) >= 2 else eog_indices

raw_clean = raw.copy()
ica.apply(raw_clean, verbose=False)

print("Pipeline complete.\n")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PSD helper
# ═════════════════════════════════════════════════════════════════════════════
def compute_psd(raw_obj, ch, fmax=80):
    psd = raw_obj.copy().pick([ch]).compute_psd(
        method="welch", fmin=0.5, fmax=fmax,
        n_fft=2048, n_overlap=1024,
        verbose=False
    )
    freqs = psd.freqs
    power = psd.get_data()[0]
    power_db = 10 * np.log10(power + 1e-30)
    return freqs, power_db

# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — TIME-SERIES: BEFORE vs AFTER
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Time-Series Comparison …")

fig1, axes1 = plt.subplots(len(PLOT_CHS), 2,
                            figsize=(16, 10), sharey=False)
fig1.suptitle("EEG Time-Series  ·  Before vs After Preprocessing  (2 – 3 s, demeaned)",
              color=COLORS["text"], fontsize=14, fontweight="bold", y=0.98)
apply_style(fig1, axes1)

artifact_labels = {
    "FP1": "Eye-blink artifact (EOG)",
    "FZ" : "Baseline drift / low-freq noise",
    "OZ" : "Occipital (alpha)",
    "T7" : "Muscle/EMG burst",
}

for row, ch in enumerate(PLOT_CHS):
    t_r, sig_r = get_seg(raw_orig,  ch)
    t_c, sig_c = get_seg(raw_clean, ch)

    for col, (t, sig, label, color, raw_flag) in enumerate([
        (t_r, sig_r, "RAW",   COLORS["raw"],   True),
        (t_c, sig_c, "CLEAN", COLORS["clean"], False),
    ]):
        ax = axes1[row, col]
        ax.plot(t, sig, color=color, lw=0.8, alpha=0.9)

        # Annotate artifact region on raw panel
        if raw_flag and ch in artifact_labels:
            ymin, ymax = ax.get_ylim() if ax.get_ylim() != (0, 1) else (sig.min(), sig.max())
            ax.set_title(f"{ch}  —  {label}  |  ⚠ {artifact_labels[ch]}",
                         fontsize=8, color=COLORS["annot"], pad=3)
        else:
            ax.set_title(f"{ch}  —  {label}", fontsize=8,
                         color=COLORS["text"], pad=3)

        ax.set_ylabel("Amplitude (µV)", fontsize=7)
        if row == len(PLOT_CHS) - 1:
            ax.set_xlabel("Time (s)", fontsize=8)

    # Shade a detected blink-like peak on FP1 raw
    if ch == "FP1":
        t_r2, sig_r2 = get_seg(raw_orig, ch)
        peak_idx = np.argmax(np.abs(sig_r2))
        peak_t   = t_r2[peak_idx]
        axes1[row, 0].axvspan(max(T0, peak_t - 0.08), min(T1, peak_t + 0.08),
                              alpha=0.25, color=COLORS["annot"], label="Blink peak")
        axes1[row, 0].legend(fontsize=7, facecolor=COLORS["panel"],
                             labelcolor=COLORS["text"], loc="upper right")

# Column headers
for ax, title in zip(axes1[0], ["◀  RAW  (unprocessed)", "▶  CLEAN  (after pipeline)"]):
    ax.annotate(title, xy=(0.5, 1.12), xycoords="axes fraction",
                ha="center", fontsize=11, fontweight="bold",
                color=COLORS["raw"] if "RAW" in title else COLORS["clean"])

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(f"{OUT_DIR}/fig1_timeseries_before_after.png",
             dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print("  Saved fig1_timeseries_before_after.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — PSD: BEFORE vs AFTER
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Power Spectral Density Comparison …")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.suptitle("Power Spectral Density  ·  Before vs After Preprocessing",
              color=COLORS["text"], fontsize=14, fontweight="bold")
apply_style(fig2, axes2)

psd_channels = [EEG_CH, EOG_CH, OCC_CH, TEMP_CH]
band_labels = {
    "Delta\n(1–4 Hz)"  : (1,  4),
    "Theta\n(4–8 Hz)"  : (4,  8),
    "Alpha\n(8–13 Hz)" : (8,  13),
    "Beta\n(13–30 Hz)" : (13, 30),
}
band_colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261"]

for ax, ch in zip(axes2.flatten(), psd_channels):
    f_r, p_r = compute_psd(raw_orig,  ch)
    f_c, p_c = compute_psd(raw_clean, ch)

    ax.plot(f_r, p_r, color=COLORS["raw"],   lw=1.2, label="Raw",   alpha=0.85)
    ax.plot(f_c, p_c, color=COLORS["clean"], lw=1.5, label="Clean", alpha=0.95)

    # Shade EEG bands
    for (band_name, (flo, fhi)), bc in zip(band_labels.items(), band_colors):
        ax.axvspan(flo, fhi, alpha=0.08, color=bc)

    # Mark 60 Hz notch effect
    ax.axvline(60, color=COLORS["annot"], lw=1, ls="--", alpha=0.7,
               label="60 Hz notch")

    ax.set_title(f"Channel: {ch}", fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Power (dB)", fontsize=8)
    ax.set_xlim(0.5, 80)
    ax.legend(fontsize=7, facecolor=COLORS["panel"], labelcolor=COLORS["text"])

    # Annotate band names on first subplot only
    if ch == EEG_CH:
        for (bname, (flo, fhi)), bc in zip(band_labels.items(), band_colors):
            ax.text((flo + fhi) / 2, ax.get_ylim()[0] + 1,
                    bname, ha="center", va="bottom",
                    fontsize=6, color=COLORS["text"], alpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(f"{OUT_DIR}/fig2_psd_before_after.png",
             dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print("  Saved fig2_psd_before_after.png")

# ═════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 3 — ARTIFACT IDENTIFICATION PANEL
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Artifact Identification Panel …")

fig3 = plt.figure(figsize=(16, 11))
fig3.patch.set_facecolor(COLORS["bg"])
fig3.suptitle("Visual Inspection & Artifact Identification  ·  Raw EEG",
              color=COLORS["text"], fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig3, hspace=0.55, wspace=0.35)
apply_style(fig3, [fig3.add_subplot(gs[i, j]) for i in range(3) for j in range(2)])
fig3.clf()
fig3.patch.set_facecolor(COLORS["bg"])

artifact_info = [
    ("Eye Blink (EOG)",      "FP1",  (0, 10),  COLORS["annot"],
     "Large slow deflections\n(>100 µV) at frontal electrodes"),
    ("Muscle/EMG Burst",     "T7",   (0, 10),  "#FF6B6B",
     "High-freq broadband bursts\n(>30 Hz) at temporal sites"),
    ("Baseline Drift",       "FZ",   (0, 30),  "#A8DADC",
     "Slow DC offset / low-freq\nwandering (<1 Hz)"),
    ("Powerline 60 Hz",      "CZ",   (0, 10),  "#C77DFF",
     "Sinusoidal 60 Hz\nriddling raw spectrum"),
    ("Before ICA (FP1)",     "FP1",  (0, 10),  COLORS["raw"],  "Raw FP1"),
    ("After ICA (FP1)",      "FP1",  (0, 10),  COLORS["clean"],"ICA-cleaned FP1"),
]

raw_objects = [raw_orig, raw_orig, raw_orig, raw_orig, raw_orig, raw_clean]

axes3 = []
for idx, ((title, ch, (t0, t1), color, desc), raw_obj) in \
        enumerate(zip(artifact_info, raw_objects)):
    row, col = divmod(idx, 2)
    ax = fig3.add_subplot(gs[row, col])
    ax.set_facecolor(COLORS["panel"])
    apply_style(fig3, [ax])

    t, sig = get_seg(raw_obj, ch, t0, t1)

    # Highlight artifact window
    ax.plot(t, sig, color=color, lw=0.9, alpha=0.9)

    # Specific annotations per artifact
    if "Eye Blink" in title:
        peaks = np.where(np.abs(sig) > np.percentile(np.abs(sig), 90))[0]
        if len(peaks):
            ax.axvspan(t[peaks[0]] - 0.2, t[peaks[0]] + 0.4,
                       alpha=0.3, color=COLORS["annot"])
            ax.annotate("Blink\npeak", xy=(t[peaks[0]], sig[peaks[0]]),
                        xytext=(t[peaks[0]] + 0.5, sig[peaks[0]] * 0.7),
                        arrowprops=dict(arrowstyle="->", color=COLORS["annot"]),
                        color=COLORS["annot"], fontsize=7)

    if "EMG" in title:
        # high-freq RMS burst detection
        rms = np.array([np.sqrt(np.mean(sig[i:i+50]**2))
                        for i in range(0, len(sig)-50, 10)])
        burst_t = t[np.argmax(rms) * 10 + 25]
        ax.axvspan(burst_t - 0.1, burst_t + 0.3, alpha=0.3, color="#FF6B6B")
        ax.annotate("EMG\nburst", xy=(burst_t, sig[int(burst_t * sfreq - t[0]*sfreq)]),
                    color="#FF6B6B", fontsize=7,
                    xytext=(burst_t + 0.4, sig.max() * 0.7),
                    arrowprops=dict(arrowstyle="->", color="#FF6B6B"))

    ax.set_title(f"⚠  {title}", color=color, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("µV", fontsize=7)

    # Description box
    ax.text(0.98, 0.97, desc, transform=ax.transAxes,
            va="top", ha="right", fontsize=6.5,
            color=COLORS["text"], alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EFEFEF",
                      edgecolor=color, alpha=0.7))
    axes3.append(ax)

fig3.savefig(f"{OUT_DIR}/fig3_artifact_identification.png",
             dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print("  Saved fig3_artifact_identification.png")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 4 — PIPELINE SUMMARY DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Pipeline Summary Dashboard …")

fig4, axes4 = plt.subplots(2, 3, figsize=(18, 10))
fig4.suptitle("Preprocessing Pipeline Summary Dashboard  ·  Function-041.cnt",
              color=COLORS["text"], fontsize=14, fontweight="bold")
apply_style(fig4, axes4)

# (A) Full raw vs clean time-series — FZ, 60 s
def get_long_seg(raw_obj, ch, t0=0, t1=60):
    idx   = raw_obj.ch_names.index(ch)
    start = int(t0 * sfreq)
    stop  = int(t1 * sfreq)
    return raw_obj.times[start:stop], raw_obj._data[idx, start:stop] * 1e6

ax = axes4[0, 0]
t_r, s_r = get_long_seg(raw_orig,  EEG_CH, 0, 60)
t_c, s_c = get_long_seg(raw_clean, EEG_CH, 0, 60)
ax.plot(t_r, s_r, color=COLORS["raw"],   lw=0.6, alpha=0.8, label="Raw")
ax.plot(t_c, s_c, color=COLORS["clean"], lw=0.6, alpha=0.9, label="Clean")
ax.set_title(f"{EEG_CH} — Raw vs Clean (0–60 s)", fontsize=9)
ax.set_xlabel("Time (s)"); ax.set_ylabel("µV")
ax.legend(fontsize=7, facecolor=COLORS["panel"], labelcolor=COLORS["text"])

# (B) PSD overlay — FZ
ax = axes4[0, 1]
f_r, p_r = compute_psd(raw_orig,  EEG_CH)
f_c, p_c = compute_psd(raw_clean, EEG_CH)
ax.plot(f_r, p_r, color=COLORS["raw"],   lw=1.2, label="Raw",   alpha=0.8)
ax.plot(f_c, p_c, color=COLORS["clean"], lw=1.5, label="Clean", alpha=0.95)
ax.axvline(60, ls="--", color=COLORS["annot"], lw=1, label="60 Hz notch")
ax.fill_between(f_r, p_r, p_c, where=(p_r > p_c),
                alpha=0.15, color=COLORS["raw"],   label="Removed power")
ax.fill_between(f_c, p_r, p_c, where=(p_c > p_r),
                alpha=0.15, color=COLORS["clean"], label="Preserved power")
ax.set_xlim(0.5, 80); ax.set_title(f"{EEG_CH} — PSD", fontsize=9)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (dB)")
ax.legend(fontsize=6, facecolor=COLORS["panel"], labelcolor=COLORS["text"])

# (C) ICA scores
ax = axes4[0, 2]
if len(eog_scores) > 0:
    scores_plot = eog_scores[0] if eog_scores[0].ndim == 1 else eog_scores[0].flatten()
    n_comp = len(scores_plot)
    bar_colors = [COLORS["raw"] if i in ica.exclude else COLORS["clean"]
                  for i in range(n_comp)]
    ax.bar(range(n_comp), np.abs(scores_plot), color=bar_colors, width=0.7)
    ax.set_title("ICA Component Scores (EOG correlation)", fontsize=9)
    ax.set_xlabel("ICA Component"); ax.set_ylabel("|Correlation|")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=COLORS["raw"],   label=f"Excluded ({len(ica.exclude)})"),
        Patch(color=COLORS["clean"], label="Retained"),
    ], fontsize=7, facecolor=COLORS["panel"], labelcolor=COLORS["text"])
else:
    ax.text(0.5, 0.5, "ICA scores\nnot available", ha="center", va="center",
            color=COLORS["text"], transform=ax.transAxes)

# (D) Before/after amplitude distribution — FZ
ax = axes4[1, 0]
_, raw_data_full  = get_long_seg(raw_orig,  EEG_CH, 0, 120)
_, clean_data_full= get_long_seg(raw_clean, EEG_CH, 0, 120)
bins = np.linspace(-150, 150, 80)
ax.hist(raw_data_full,   bins=bins, color=COLORS["raw"],   alpha=0.55,
        label=f"Raw   σ={raw_data_full.std():.1f} µV",  density=True)
ax.hist(clean_data_full, bins=bins, color=COLORS["clean"], alpha=0.55,
        label=f"Clean σ={clean_data_full.std():.1f} µV", density=True)
ax.set_title(f"{EEG_CH} — Amplitude Distribution", fontsize=9)
ax.set_xlabel("Amplitude (µV)"); ax.set_ylabel("Density")
ax.legend(fontsize=7, facecolor=COLORS["panel"], labelcolor=COLORS["text"])

# (E) Multi-channel summary: std before vs after
ax = axes4[1, 1]
sel_chs = [EEG_CH, EOG_CH, OCC_CH, TEMP_CH, "CZ", "PZ", "O1", "FC1"]
std_raw   = [raw_orig._data[raw_orig.ch_names.index(c)].std() * 1e6  for c in sel_chs]
std_clean = [raw_clean._data[raw_clean.ch_names.index(c)].std() * 1e6 for c in sel_chs]
x = np.arange(len(sel_chs))
ax.bar(x - 0.2, std_raw,   width=0.38, color=COLORS["raw"],   label="Raw",   alpha=0.85)
ax.bar(x + 0.2, std_clean, width=0.38, color=COLORS["clean"], label="Clean", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(sel_chs, rotation=30, fontsize=7)
ax.set_title("Signal Std (µV) — Key Channels", fontsize=9)
ax.set_ylabel("Std (µV)")
ax.legend(fontsize=7, facecolor=COLORS["panel"], labelcolor=COLORS["text"])

# (F) Pipeline steps text box
ax = axes4[1, 2]
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")
pipeline_text = (
    "PREPROCESSING PIPELINE\n"
    "══════════════════════\n\n"
    "  Raw EEG (67 ch, 1000 Hz)\n"
    "        ↓\n"
    "  ① Drop Trigger channel\n"
    "        ↓\n"
    "  ② Average re-reference\n"
    "        ↓\n"
    "  ③ Bandpass filter\n"
    "        1 – 40 Hz  (FIR/firwin)\n"
    "        Removes DC drift & EMG\n"
    "        ↓\n"
    "  ④ Notch filter @ 60 Hz\n"
    "        Removes powerline noise\n"
    "        ↓\n"
    "  ⑤ ICA (FastICA, 20 comp.)\n"
    f"        {len(ica.exclude)} EOG component(s) removed\n"
    "        ↓\n"
    "  ✓ Clean EEG (66 ch)\n"
)
ax.text(0.05, 0.95, pipeline_text,
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        color=COLORS["text"], family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["panel"],
                  edgecolor=COLORS["clean"], lw=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig4.savefig(f"{OUT_DIR}/fig4_pipeline_summary.png",
             dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
print("  Saved fig4_pipeline_summary.png")

print("\n✅ All figures saved to /mnt/user-data/outputs/")
print("   fig1_timeseries_before_after.png")
print("   fig2_psd_before_after.png")
print("   fig3_artifact_identification.png")
print("   fig4_pipeline_summary.png")