import os
import random
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DATASET_PATH  = "."       # root folder containing word subfolders
SR_TARGET     = 8000
FRAME_LEN     = 256
HOP           = 128
MAX_DURATION  = 1.0
EPS           = 1e-9
N_MFCC        = 13
N_FILTERS     = 26
MAX_PER_WORD  = 150        # cap samples per word (keeps DTW fast at inference)
RANDOM_SEED   = 42

# Words to train on — set to None to use ALL folders found
TARGET_WORDS  = [
    "up", "down", "left", "right",
    "on", "off", "stop", "close",
    "start",  "open"  
]

random.seed(RANDOM_SEED)

# ---------------------------------------------------
# MEL FILTERBANK  (pure numpy)
# ---------------------------------------------------
def _hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _build_filterbank(n_filters, frame_len, sr):
    n_fft   = frame_len // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_filters + 2)
    bins    = np.clip(
        np.floor((frame_len + 1) * _mel_to_hz(mel_pts) / sr).astype(int),
        0, n_fft - 1
    )
    fb = np.zeros((n_filters, n_fft))
    for m in range(1, n_filters + 1):
        fl, fc, fr = bins[m-1], bins[m], bins[m+1]
        for k in range(fl, fc):
            if fc != fl: fb[m-1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr != fc: fb[m-1, k] = (fr - k) / (fr - fc)
    return fb

_FB  = _build_filterbank(N_FILTERS, FRAME_LEN, SR_TARGET)
_DCT = np.array([
    [np.cos(np.pi * n * (k + 0.5) / N_FILTERS) for k in range(N_FILTERS)]
    for n in range(N_MFCC)
], dtype=np.float32)

# ---------------------------------------------------
# AUDIO LOADING  (handles 8 kHz and 16 kHz)
# ---------------------------------------------------
def load_wav(path):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y[:, 0]                          # stereo → mono
    y = y / (np.max(np.abs(y)) + EPS)       # normalise
    if sr != SR_TARGET:                      # resample (e.g. 16k → 8k)
        new_len = int(len(y) * SR_TARGET / sr)
        y = np.interp(
            np.linspace(0, len(y) - 1, new_len),
            np.arange(len(y)), y
        )
    target = int(SR_TARGET * MAX_DURATION)
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    return y.astype(np.float32)

# ---------------------------------------------------
# FEATURE EXTRACTION  → (n_frames, 26)
# ---------------------------------------------------
def extract_mfcc_seq(y):
    y = y.flatten() / (np.max(np.abs(y)) + EPS)
    frames = [y[i:i+FRAME_LEN] for i in range(0, len(y) - FRAME_LEN + 1, HOP)]

    seq = []
    for f in frames:
        mag   = np.abs(np.fft.rfft(f))
        mel_e = _FB @ (mag ** 2)
        seq.append(_DCT @ np.log(mel_e + EPS))
    seq = np.array(seq, dtype=np.float32)          # (n_frames, 13)

    # Cepstral Mean Normalisation — removes static mic/channel bias
    seq -= np.mean(seq, axis=0, keepdims=True)

    # Delta (velocity) features
    n     = seq.shape[0]
    delta = np.zeros_like(seq)
    for t in range(n):
        delta[t] = (seq[min(t+1, n-1)] - seq[max(t-1, 0)]) / 2.0

    return np.concatenate([seq, delta], axis=1)    # (n_frames, 26)

# ---------------------------------------------------
# COLLECT WAV FILES
# Supports two layouts:
#   Flat   (Google SCv2): word/speaker_hash.wav
#   Nested (your own):   word/subfolder/file.wav
# ---------------------------------------------------
def collect_wavs(word_dir):
    """Return list of all .wav paths under word_dir (flat or nested)."""
    wavs = []
    for entry in os.listdir(word_dir):
        full = os.path.join(word_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(".wav"):
            wavs.append(full)                      # flat layout
        elif os.path.isdir(full):
            for f in os.listdir(full):             # nested layout
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(full, f))
    return wavs

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
all_seqs   = []
all_labels = []

folders = sorted(os.listdir(DATASET_PATH))

for folder in folders:
    word_dir = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(word_dir):
        continue

    # Match folder name to target words (strip suffixes like "-recordings")
    folder_base = folder.replace("-recordings", "").replace("_recordings", "")
    if TARGET_WORDS is not None:
        if folder_base not in TARGET_WORDS and folder not in TARGET_WORDS:
            continue

    wavs = collect_wavs(word_dir)
    if not wavs:
        continue

    # Randomly cap to MAX_PER_WORD
    random.shuffle(wavs)
    wavs = wavs[:MAX_PER_WORD]

    count = 0
    for path in wavs:
        try:
            seq = extract_mfcc_seq(load_wav(path))
            all_seqs.append(seq)
            all_labels.append(folder)             # keep original folder name as label
            count += 1
        except Exception as e:
            print(f"  Skipped {path}: {e}")

    if count:
        print(f"Trained: {folder:30s}  samples={count}")

if not all_seqs:
    print("\nERROR: No audio files found. Check DATASET_PATH and TARGET_WORDS.")
else:
    templates = np.empty(len(all_seqs), dtype=object)
    for i, s in enumerate(all_seqs):
        templates[i] = s
    labels = np.array(all_labels)

    np.save("templates_dtw.npy", templates)
    np.save("labels_dtw.npy",    labels)
    print(templates)
    print(f"\nDONE — {len(templates)} sequences across {len(set(all_labels))} words")
    print("Labels found:", sorted(set(all_labels)))
    print("Sequence shape example:", all_seqs[0].shape)