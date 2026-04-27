import os
import random
import numpy as np
from scipy.io import wavfile

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DATASET_PATH  = "."
SR_TARGET     = 8000
FRAME_LEN     = 256
HOP           = 128
MAX_DURATION  = 1.0
EPS           = 1e-9
N_MFCC        = 13
N_FILTERS     = 26
MAX_PER_WORD  = 75
RANDOM_SEED   = 42

GOERTZEL_FREQS = [
    250,  350,  450,
    550,  700,  850,
    1000, 1200, 1500,
    1800, 2100, 2500,
    3000, 4000,
    5500, 7000,
]
GOERTZEL_W = 3.0

TARGET_WORDS = [
    "up", "down", "left", "right",
    "on", "off", "stop", "close",
    "start", "open"
]

random.seed(RANDOM_SEED)

# ---------------------------------------------------
# MEL FILTERBANK
# ---------------------------------------------------
def _hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
def _mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _build_filterbank(n_filters, frame_len, sr):
    n_fft   = frame_len // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(0), _hz_to_mel(sr / 2), n_filters + 2)
    bins    = np.clip(np.floor((frame_len + 1) * _mel_to_hz(mel_pts) / sr).astype(int), 0, n_fft - 1)
    fb      = np.zeros((n_filters, n_fft))
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
# GOERTZEL
# ---------------------------------------------------
def goertzel(frame, freq, sr):
    N      = len(frame)
    k      = min(int(round(freq * N / sr)), N - 1)
    coeff  = 2.0 * np.cos(2.0 * np.pi * k / N)
    s1 = s2 = 0.0
    for x in frame:
        s = x + coeff * s1 - s2
        s2, s1 = s1, s
    return max(s2**2 + s1**2 - coeff * s1 * s2, 0.0)

def goertzel_frame(frame, freqs, sr):
    energies = np.array([goertzel(frame, f, sr) for f in freqs], dtype=np.float32)
    return np.log(energies + EPS)

# ---------------------------------------------------
# FEATURE EXTRACTION — (n_frames, 42)
#   0..25  : MFCC + delta (weight 1.0)
#   26..41 : Goertzel log-energy × GOERTZEL_W
# ---------------------------------------------------
def extract_seq(y):
    y = y.flatten() / (np.max(np.abs(y)) + EPS)
    frames = [y[i:i+FRAME_LEN] for i in range(0, len(y) - FRAME_LEN + 1, HOP)]

    mfcc_seq, goer_seq = [], []
    for f in frames:
        mag   = np.abs(np.fft.rfft(f))
        mel_e = _FB @ (mag ** 2)
        mfcc_seq.append(_DCT @ np.log(mel_e + EPS))
        goer_seq.append(goertzel_frame(f, GOERTZEL_FREQS, SR_TARGET))

    mfcc_arr = np.array(mfcc_seq, dtype=np.float32)
    goer_arr = np.array(goer_seq, dtype=np.float32)

    mfcc_arr -= np.mean(mfcc_arr, axis=0, keepdims=True)   # CMN

    n     = mfcc_arr.shape[0]
    delta = np.zeros_like(mfcc_arr)
    for t in range(n):
        delta[t] = (mfcc_arr[min(t+1,n-1)] - mfcc_arr[max(t-1,0)]) / 2.0

    goer_arr -= np.mean(goer_arr, axis=0, keepdims=True)    # channel normalise

    return np.concatenate([
        np.concatenate([mfcc_arr, delta], axis=1),          # (n_frames, 26)
        goer_arr * GOERTZEL_W                               # (n_frames, 16)
    ], axis=1).astype(np.float32)                           # (n_frames, 42)

# ---------------------------------------------------
# AUDIO LOADING
# ---------------------------------------------------
def load_wav(path):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32)
    if y.ndim > 1: y = y[:, 0]
    y = y / (np.max(np.abs(y)) + EPS)
    if sr != SR_TARGET:
        new_len = int(len(y) * SR_TARGET / sr)
        y = np.interp(np.linspace(0, len(y)-1, new_len), np.arange(len(y)), y)
    target = int(SR_TARGET * MAX_DURATION)
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    return y.astype(np.float32)

def collect_wavs(word_dir):
    wavs = []
    for entry in os.listdir(word_dir):
        full = os.path.join(word_dir, entry)
        if os.path.isfile(full) and entry.lower().endswith(".wav"):
            wavs.append(full)
        elif os.path.isdir(full):
            for f in os.listdir(full):
                if f.lower().endswith(".wav"):
                    wavs.append(os.path.join(full, f))
    return wavs

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
all_seqs   = []
all_labels = []

for folder in sorted(os.listdir(DATASET_PATH)):
    word_dir = os.path.join(DATASET_PATH, folder)
    if not os.path.isdir(word_dir): continue

    folder_base = folder.replace("-recordings","").replace("_recordings","")
    if TARGET_WORDS and folder_base not in TARGET_WORDS and folder not in TARGET_WORDS:
        continue

    wavs = collect_wavs(word_dir)
    if not wavs: continue

    random.shuffle(wavs)
    wavs = wavs[:MAX_PER_WORD]

    count = 0
    for path in wavs:
        try:
            seq = extract_seq(load_wav(path))
            all_seqs.append(seq)
            all_labels.append(folder)
            count += 1
        except Exception as e:
            print(f"  Skipped {path}: {e}")

    if count:
        print(f"Trained: {folder:30s}  samples={count}  seq_shape={all_seqs[-1].shape}")

if not all_seqs:
    print("ERROR: No audio found. Check DATASET_PATH and TARGET_WORDS.")
else:
    templates = np.empty(len(all_seqs), dtype=object)
    for i, s in enumerate(all_seqs):
        templates[i] = s
    labels = np.array(all_labels)

    np.save("templates_dtw.npy", templates)
    np.save("labels_dtw.npy",    labels)

    print(f"\nDONE — {len(templates)} sequences, {len(set(all_labels))} words")
    print("Labels:", sorted(set(all_labels)))
    print("Sequence shape:", all_seqs[0].shape)   # should be (61, 42)