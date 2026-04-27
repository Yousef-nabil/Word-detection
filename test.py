import numpy as np
import sounddevice as sd
from collections import defaultdict

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
SR        = 8000
SECONDS   = 1
FRAME_LEN = 256
HOP       = 128
EPS       = 1e-9
N_MFCC    = 13
N_FILTERS = 26
K         = 7

# Goertzel probe frequencies (Hz) — chosen for your 10 words
# Covers F1/F2 formants, nasal band, fricative band, plosive burst
GOERTZEL_FREQS = [
    250,  350,  450,   # F1 low  (on/off/stop/close/open)
    550,  700,  850,   # F1 mid  (up/down/start)
    1000, 1200, 1500,  # F1-F2 transition
    1800, 2100, 2500,  # F2 front vowels (left/right)
    3000, 4000,        # F3 + consonant resonance
    5500, 7000,        # fricative energy (off/stop)
]
GOERTZEL_W = 3.0   # weight multiplier vs MFCCs — tune this

templates = np.load("templates_dtw.npy", allow_pickle=True)
labels    = np.load("labels_dtw.npy",    allow_pickle=True)

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

_FB  = _build_filterbank(N_FILTERS, FRAME_LEN, SR)
_DCT = np.array([
    [np.cos(np.pi * n * (k + 0.5) / N_FILTERS) for k in range(N_FILTERS)]
    for n in range(N_MFCC)
], dtype=np.float32)

# ---------------------------------------------------
# GOERTZEL ALGORITHM
# Single-frequency DFT — efficient energy detector
# at exact target frequencies.
# ---------------------------------------------------
def goertzel(frame, freq, sr):
    """Return energy at `freq` Hz for a single frame."""
    N      = len(frame)
    k      = int(round(freq * N / sr))
    k      = min(k, N - 1)
    w      = 2.0 * np.pi * k / N
    coeff  = 2.0 * np.cos(w)
    s_prev = 0.0
    s_prev2= 0.0
    for x in frame:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev  = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return max(power, 0.0)

def goertzel_frame(frame, freqs, sr):
    """Return log-energy vector for all probe frequencies."""
    energies = np.array([goertzel(frame, f, sr) for f in freqs], dtype=np.float32)
    return np.log(energies + EPS)

# ---------------------------------------------------
# FEATURE EXTRACTION
# Returns (n_frames, 26 + n_goertzel) sequence
#   cols 0..25          : MFCC + delta  (weight 1.0)
#   cols 26..           : Goertzel log-energies × GOERTZEL_W
# ---------------------------------------------------
def extract_seq(y):
    y = y.flatten() / (np.max(np.abs(y)) + EPS)
    frames = [y[i:i+FRAME_LEN] for i in range(0, len(y) - FRAME_LEN + 1, HOP)]

    mfcc_seq = []
    goertzel_seq = []

    for f in frames:
        # --- MFCC ---
        mag   = np.abs(np.fft.rfft(f))
        mel_e = _FB @ (mag ** 2)
        mfcc_seq.append(_DCT @ np.log(mel_e + EPS))

        # --- Goertzel ---
        goertzel_seq.append(goertzel_frame(f, GOERTZEL_FREQS, SR))

    mfcc_arr = np.array(mfcc_seq, dtype=np.float32)   # (n_frames, 13)
    goer_arr = np.array(goertzel_seq, dtype=np.float32) # (n_frames, 16)

    # CMN on MFCCs only
    mfcc_arr -= np.mean(mfcc_arr, axis=0, keepdims=True)

    # Delta MFCCs
    n     = mfcc_arr.shape[0]
    delta = np.zeros_like(mfcc_arr)
    for t in range(n):
        delta[t] = (mfcc_arr[min(t+1,n-1)] - mfcc_arr[max(t-1,0)]) / 2.0

    # Normalise Goertzel to zero-mean per utterance (same CMN idea)
    goer_arr -= np.mean(goer_arr, axis=0, keepdims=True)

    mfcc_feat = np.concatenate([mfcc_arr, delta], axis=1)  # (n_frames, 26)
    goer_feat = goer_arr * GOERTZEL_W                       # (n_frames, 16)

    return np.concatenate([mfcc_feat, goer_feat], axis=1)  # (n_frames, 42)

# ---------------------------------------------------
# DTW — Sakoe-Chiba band
# ---------------------------------------------------
def dtw_distance(a, b):
    n, m = len(a), len(b)
    band = max(5, int(0.3 * max(n, m)))
    INF  = float('inf')
    dp   = np.full((n, m), INF, dtype=np.float64)
    dp[0, 0] = np.linalg.norm(a[0] - b[0])
    for i in range(n):
        for j in range(max(0, i - band), min(m, i + band + 1)):
            cost = np.linalg.norm(a[i] - b[j])
            if i == 0 and j == 0:
                continue
            prev = INF
            if i > 0 and j > 0: prev = min(prev, dp[i-1, j-1])
            if i > 0:            prev = min(prev, dp[i-1, j  ])
            if j > 0:            prev = min(prev, dp[i,   j-1])
            dp[i, j] = cost + prev
    return dp[n-1, m-1] / (n + m)

# ---------------------------------------------------
# CLASSIFIERS
# ---------------------------------------------------
def classify_1nn(dists, labels):
    return labels[np.argmin(dists)]

def classify_class_mean_top3(dists, labels):
    cd = defaultdict(list)
    for d, lbl in zip(dists, labels): cd[lbl].append(d)
    cs = {lbl: np.mean(sorted(ds)[:3]) for lbl, ds in cd.items()}
    return min(cs, key=cs.get), cs

def classify_weighted_knn_normalized(dists, labels, k=K):
    top_idx = np.argsort(dists)[:k]
    cc = defaultdict(int)
    for lbl in labels: cc[lbl] += 1
    rv = defaultdict(float)
    for i in top_idx: rv[labels[i]] += 1.0 / (dists[i] + 1e-9)
    nv = {lbl: score / cc[lbl] for lbl, score in rv.items()}
    return max(nv, key=nv.get), nv

def classify_ensemble(dists, labels, k=K):
    p1           = classify_1nn(dists, labels)
    p2, scores2  = classify_class_mean_top3(dists, labels)
    p3, _        = classify_weighted_knn_normalized(dists, labels, k)
    vc = defaultdict(int)
    for p in [p1, p2, p3]: vc[p] += 1
    mx      = max(vc.values())
    winners = [lbl for lbl, v in vc.items() if v == mx]
    if len(winners) == 1: return winners[0], vc, [p1, p2, p3]
    tb = min(winners, key=lambda lbl: scores2.get(lbl, float('inf')))
    return tb, vc, [p1, p2, p3]

# ---------------------------------------------------
# RECORD
# ---------------------------------------------------
print("Speak now...")
audio = sd.rec(int(SR * SECONDS), samplerate=SR, channels=1, dtype='float32')
sd.wait()
print("Processing...\n")

# ---------------------------------------------------
# EXTRACT + CLASSIFY
# ---------------------------------------------------
query = extract_seq(audio)
dists = np.array([dtw_distance(query, t) for t in templates])

p1                   = classify_1nn(dists, labels)
p2, class_scores     = classify_class_mean_top3(dists, labels)
p3, norm_votes       = classify_weighted_knn_normalized(dists, labels, 10)
final, vote_count, _ = classify_ensemble(dists, labels, K)

top_k = np.argsort(dists)[:K]
print(f"Top {K} nearest neighbours:")
for i in top_k:
    print(f"  {labels[i]:25s}  dist={dists[i]:.4f}")

print("\nPer-class mean of top-3:")
for lbl, s in sorted(class_scores.items(), key=lambda x: x[1]):
    print(f"  {lbl:25s}  score={s:.4f}")

print("\nNormalised weighted k-NN:")
for lbl, s in sorted(norm_votes.items(), key=lambda x: -x[1]):
    print(f"  {lbl:25s}  score={s:.6f}")

print("\n" + "="*50)
print(f"  1-NN                 : {p1}")
print(f"  Class mean top-3     : {p2}")
print(f"  Normalised k-NN      : {p3}")
print(f"  ENSEMBLE (final)     : {final}   [{vote_count[final]}/3 votes]")
print("="*50)