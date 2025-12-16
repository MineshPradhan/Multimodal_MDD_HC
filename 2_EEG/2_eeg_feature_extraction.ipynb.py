# Cell: Imports & settings
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import welch, detrend
import mne
import warnings
warnings.filterwarnings("ignore")

# Paths
ROOT_3CH = "EEG_3channels_resting_lanzhou"
ROOT_128_REST = "EEG_128channels_resting_lanzhou"
ROOT_128_ERP = "EEG_128channels_ERP_lanzhou"
OUTPUT_DIR = "data_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default sampling rates if not available in files (adjust if you know real values)
DEFAULT_SR_3CH = 250
DEFAULT_SR_128 = 500

# PSD band definitions
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Cell: helper funcs
def bandpower_from_psd(freqs, psd, band):
    fmin, fmax = band
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    bp = np.trapz(psd[idx], freqs[idx])
    return bp

def psd_welch(x, fs, nperseg=None):
    # x: 1D signal
    if nperseg is None:
        nperseg = min(4*fs, len(x))
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    return freqs, psd

# Cell: process 3-channel .txt files
rows_3ch = []
txt_files = sorted(glob.glob(os.path.join(ROOT_3CH, "*.txt")))
for f in txt_files:
    try:
        data = np.loadtxt(f)
    except Exception:
        # try reading with pandas if irregular
        data = pd.read_csv(f, sep=None, engine='python').values
    # Heuristics: detect shape
    if data.ndim == 1:
        # maybe columns separated by spaces in a single row
        continue
    # If rows are channels x time, or time x channels - try to detect
    if data.shape[0] == 3:
        ch_data = data  # (3, T)
    elif data.shape[1] == 3:
        ch_data = data.T
    else:
        # fallback: assume columns after first time col are channels
        if data.shape[1] >= 3:
            ch_data = data[:, -3:].T
        else:
            print("Unexpected shape for", f, data.shape)
            continue
    # Determine sampling rate if present in filename/neighboring file - default to DEFAULT_SR_3CH
    sr = DEFAULT_SR_3CH
    # compute PSD per channel and band powers
    subj_id = Path(f).stem
    feat = {'subject_id': subj_id}
    for i in range(ch_data.shape[0]):
        sig = detrend(ch_data[i])
        freqs, psd = psd_welch(sig, fs=sr)
        for band_name, band_range in BANDS.items():
            bp = bandpower_from_psd(freqs, psd, band_range)
            feat[f'ch{i+1}_{band_name}_power'] = float(bp)
        # relative alpha/beta etc
        # compute total power across 1-45 Hz
        total = bandpower_from_psd(freqs, psd, (1,45))
        feat[f'ch{i+1}_total_1_45'] = float(total)
    # example frontal asymmetry between ch1 and ch2 alpha (if ch1=left, ch2=right — user should adjust mapping)
    try:
        a_left = feat['ch1_alpha_power']
        a_right = feat['ch2_alpha_power']
        feat['alpha_asymmetry_ch1_ch2'] = float(a_left - a_right)
    except Exception:
        feat['alpha_asymmetry_ch1_ch2'] = np.nan
    rows_3ch.append(feat)

df_3ch = pd.DataFrame(rows_3ch)
df_3ch.to_csv(os.path.join(OUTPUT_DIR, "eeg_3ch_features.csv"), index=False)
print("Saved eeg_3ch_features.csv")

# Cell: process 128-channel resting files (.mat)
rows_128rest = []
mat_files = sorted(glob.glob(os.path.join(ROOT_128_REST, "*.mat")))
for f in mat_files:
    try:
        mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print("Failed to load mat", f, e)
        continue
    # Heuristic: try common variable names
    eeg_candidates = []
    for k in mat.keys():
        if k.startswith('__'):
            continue
        v = mat[k]
        if isinstance(v, np.ndarray):
            if v.ndim == 2 and (v.shape[0] in (128, 129) or v.shape[1] in (128, 129)):
                eeg_candidates.append((k, v))
    if not eeg_candidates:
        # try keys 'data', 'EEG', 'eeg'
        if 'data' in mat:
            v = mat['data']
            eeg_candidates.append(('data', v))
    if not eeg_candidates:
        print("No obvious EEG array in", f)
        continue
    key, arr = eeg_candidates[0]
    # normalize shape -> (n_channels, n_times)
    if arr.shape[0] == 128:
        data = arr
    elif arr.shape[1] == 128:
        data = arr.T
    else:
        # try guessing
        data = arr
    sr = mat.get('fs', mat.get('srate', DEFAULT_SR_128))
    if isinstance(sr, np.ndarray):
        sr = float(sr)
    if sr is None:
        sr = DEFAULT_SR_128
    subj_id = Path(f).stem
    feat = {'subject_id': subj_id}
    # compute band power per region: mean across channels to reduce dimension
    for band_name, band_range in BANDS.items():
        band_powers = []
        for ch in range(min(data.shape[0], 128)):
            sig = detrend(data[ch])
            freqs, psd = psd_welch(sig, fs=sr)
            bp = bandpower_from_psd(freqs, psd, band_range)
            band_powers.append(bp)
        band_powers = np.array(band_powers)
        feat[f'{band_name}_power_mean'] = float(np.nanmean(band_powers))
        feat[f'{band_name}_power_median'] = float(np.nanmedian(band_powers))
        feat[f'{band_name}_power_std'] = float(np.nanstd(band_powers))
    # simple connectivity measure: pairwise coherence between a few canonical channels (first 5)
    try:
        import itertools
        from scipy.signal import coherence
        coh_vals = []
        for (i,j) in itertools.combinations(range(min(8, data.shape[0])), 2):
            f_coh, Cxy = coherence(data[i], data[j], fs=sr, nperseg=min(1024, len(data[i])))
            # take mean coherence in alpha band
            idx = np.logical_and(f_coh >= BANDS['alpha'][0], f_coh <= BANDS['alpha'][1])
            coh_vals.append(np.mean(Cxy[idx]))
        feat['alpha_coherence_mean'] = float(np.nanmean(coh_vals)) if coh_vals else np.nan
    except Exception:
        feat['alpha_coherence_mean'] = np.nan
    rows_128rest.append(feat)

df_128rest = pd.DataFrame(rows_128rest)
df_128rest.to_csv(os.path.join(OUTPUT_DIR, "eeg_resting_128ch_features.csv"), index=False)
print("Saved eeg_resting_128ch_features.csv")

# Cell: process 128-channel ERP .raw files using MNE (attempt several readers)
rows_erpeeg = []
raw_files = sorted(glob.glob(os.path.join(ROOT_128_ERP, "*.raw")))
for f in raw_files:
    subj_id = Path(f).stem
    loaded = None
    # Try common readers
    try:
        # EGI .raw
        raw = mne.io.read_raw_egi(f, preload=True, verbose='ERROR')
        loaded = raw
    except Exception:
        try:
            # BrainVision often comes with .vhdr/.vmrk, but try read_raw_brainvision if possible
            raw = mne.io.read_raw_brainvision(f, preload=True, verbose='ERROR')
            loaded = raw
        except Exception:
            try:
                raw = mne.io.read_raw_fif(f, preload=True, verbose='ERROR')
                loaded = raw
            except Exception:
                loaded = None
    if loaded is None:
        print("Could not load", f, "- skipping")
        continue
    # Preprocessing
    raw.load_data()
    # set montage if possible, else leave
    try:
        raw.filter(1., 45., fir_design='firwin')
    except Exception:
        pass
    # If annotations/events exist, try to extract ERPs; else compute PSD on whole file
    try:
        events, event_id = mne.events_from_annotations(loaded)
        if len(events) > 0:
            epochs = mne.Epochs(loaded, events=events, event_id=event_id, tmin=-0.2, tmax=0.8, preload=True, baseline=(None, 0))
            evoked = {k: epochs[k].average() for k in event_id.keys()}
        else:
            epochs = mne.make_fixed_length_epochs(loaded, duration=1.0)
            evoked = {'all': epochs.average()}
    except Exception:
        epochs = mne.make_fixed_length_epochs(loaded, duration=1.0)
        evoked = {'all': epochs.average()}

    # Extract simple ERP features: peak amplitude and latency in typical windows for P300 (250-500ms) on channel with max absolute
    feat = {'subject_id': subj_id}
    for key, ev in evoked.items():
        data = ev.data  # shape (n_channels, n_times)
        times = ev.times
        # find channel with max absolute mean in P300 window (0.25-0.5s)
        idx_window = np.where((times >= 0.25) & (times <= 0.5))[0]
        if idx_window.size == 0:
            idx_window = np.arange(len(times))
        # mean across window per channel
        meanvals = np.mean(data[:, idx_window], axis=1)
        ch_max = np.argmax(np.abs(meanvals))
        waveform = data[ch_max, :]
        # P300 peak amplitude
        try:
            t_idx = idx_window[np.argmax(np.abs(waveform[idx_window]))]
            feat[f'erp_{key}_p300_amp'] = float(waveform[t_idx])
            feat[f'erp_{key}_p300_latency'] = float(times[t_idx])
        except Exception:
            feat[f'erp_{key}_p300_amp'] = np.nan
            feat[f'erp_{key}_p300_latency'] = np.nan
    rows_erpeeg.append(feat)

df_erpeeg = pd.DataFrame(rows_erpeeg)
df_erpeeg.to_csv(os.path.join(OUTPUT_DIR, "eeg_erp_features.csv"), index=False)
print("Saved eeg_erp_features.csv")