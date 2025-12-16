import os
import glob
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_ROOT = PROJECT_ROOT / "audio_lanzhou"
OUTPUT_DIR = PROJECT_ROOT / "data_processed"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Resolved AUDIO_ROOT:", AUDIO_ROOT)
print("Resolved OUTPUT_DIR:", OUTPUT_DIR)

# Feature extraction
SR_TARGET = None   # preserve original sample rate
MFCC_N = 13

# Helper functions
def list_subject_folders(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found")

    subs = [d for d in root.iterdir() if d.is_dir()]
    # fallback: if wavs directly in root
    if not subs:
        return [root]
    return subs

def load_audio(path, sr=None):
    data, sr_file = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # mono
    if sr is not None and sr_file != sr:
        data = librosa.resample(data.astype(float), orig_sr=sr_file, target_sr=sr)
        sr_file = sr
    return data, sr_file

def extract_basic_audio_features(y, sr):
    feats = {}
    feats['duration'] = float(len(y) / sr)

    rms = librosa.feature.rms(y=y)
    feats['rms_mean'] = float(np.mean(rms))
    feats['rms_std'] = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)
    feats['zcr_mean'] = float(np.mean(zcr))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats['spec_centroid_mean'] = float(np.mean(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats['spec_rolloff_mean'] = float(np.mean(rolloff))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N)
    for i in range(MFCC_N):
        feats[f'mfcc_{i+1}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_{i+1}_std'] = float(np.std(mfcc[i]))

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        f0_voiced = f0[~np.isnan(f0)] if f0 is not None else []
        feats['f0_mean'] = float(np.mean(f0_voiced)) if len(f0_voiced) else np.nan
        feats['f0_std'] = float(np.std(f0_voiced)) if len(f0_voiced) else np.nan
    except Exception:
        feats['f0_mean'] = np.nan
        feats['f0_std'] = np.nan

    return feats

# Praat (jitter / shimmer)
USE_PRAAT = False
try:
    import parselmouth
    USE_PRAAT = True
except Exception:
    USE_PRAAT = False

def extract_voicequality_praat(y, sr):
    feats = {}
    if not USE_PRAAT:
        feats['jitter_local'] = np.nan
        feats['shimmer_local'] = np.nan
        feats['hnr'] = np.nan
        return feats

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)

    try:
        snd = parselmouth.Sound(tmp.name)
        point_process = parselmouth.praat.call(
            snd, "To PointProcess (periodic, cc)", 75, 600
        )
        feats['jitter_local'] = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        feats['shimmer_local'] = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)",
            0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        hnr = parselmouth.praat.call(
            snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
        )
        feats['hnr'] = parselmouth.praat.call(hnr, "Get mean", 0, 0)
    except Exception:
        feats['jitter_local'] = np.nan
        feats['shimmer_local'] = np.nan
        feats['hnr'] = np.nan
    finally:
        os.unlink(tmp.name)

    return feats

# MAIN EXTRACTION LOOP
subject_folders = list_subject_folders(AUDIO_ROOT)
rows_subject = []
rows_perfile = []

for sdir in subject_folders:
    subj_id = sdir.name
    wavs = sorted(sdir.glob("*.wav"))

    if not wavs:
        print(f"No wavs found for {subj_id}, skipping.")
        continue

    aggregate = {"subject_id": subj_id, "n_files": len(wavs)}
    file_feats = []

    for wav in wavs:
        try:
            y, sr = load_audio(wav, SR_TARGET)
        except Exception as e:
            print("Failed to load", wav, e)
            continue

        basic = extract_basic_audio_features(y, sr)
        praat = extract_voicequality_praat(y, sr)

        fdict = {
            "subject_id": subj_id,
            "file": wav.name,
            "sr": sr
        }
        fdict.update(basic)
        fdict.update(praat)

        rows_perfile.append(fdict)
        file_feats.append(fdict)

    df_files = pd.DataFrame(file_feats)
    if df_files.empty:
        continue

    numeric_cols = df_files.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "sr":
            continue
        aggregate[f"{col}_mean"] = df_files[col].mean()
        aggregate[f"{col}_std"] = df_files[col].std()

    rows_subject.append(aggregate)

# SAVE OUTPUTS
df_subject = pd.DataFrame(rows_subject)
df_perfile = pd.DataFrame(rows_perfile)

df_subject.to_csv(OUTPUT_DIR / "audio_features.csv", index=False)
df_perfile.to_csv(OUTPUT_DIR / "audio_perfile_features.csv", index=False)

print("✅ Saved audio_features.csv and audio_perfile_features.csv to", OUTPUT_DIR)