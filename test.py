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
K         = 7    # neighbours to consider (odd number avoids ties)

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
# FEATURE EXTRACTION
# ---------------------------------------------------
def extract_mfcc_seq(y):
    y = y.flatten() / (np.max(np.abs(y)) + EPS)
    frames = [y[i:i+FRAME_LEN] for i in range(0, len(y) - FRAME_LEN + 1, HOP)]
    seq = []
    for f in frames:
        mag   = np.abs(np.fft.rfft(f))
        mel_e = _FB @ (mag ** 2)
        seq.append(_DCT @ np.log(mel_e + EPS))
    seq = np.array(seq, dtype=np.float32)
    seq -= np.mean(seq, axis=0, keepdims=True)      # CMN
    n     = seq.shape[0]
    delta = np.zeros_like(seq)
    for t in range(n):
        delta[t] = (seq[min(t+1, n-1)] - seq[max(t-1, 0)]) / 2.0
    return np.concatenate([seq, delta], axis=1)     # (n_frames, 26)

# ---------------------------------------------------
# DTW  — Sakoe-Chiba band
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
    """Baseline: single nearest neighbour."""
    return labels[np.argmin(dists)]


def classify_class_mean_top3(dists, labels):
    """
    Per-class: average the 3 closest templates of each class.
    More stable than 1-NN — one outlier recording can't win alone.
    """
    class_dists = defaultdict(list)
    for d, lbl in zip(dists, labels):
        class_dists[lbl].append(d)

    class_score = {}
    for lbl, ds in class_dists.items():
        top3 = sorted(ds)[:3]
        class_score[lbl] = np.mean(top3)

    return min(class_score, key=class_score.get), class_score


def classify_weighted_knn_normalized(dists, labels, k=K):
    """
    Weighted k-NN where each class's total weight is divided by
    the number of templates it has — removes the sample-count bias.
    """
    top_idx    = np.argsort(dists)[:k]
    top_labels = labels[top_idx]
    top_dists  = dists[top_idx]

    # count templates per class (for normalisation)
    class_counts = defaultdict(int)
    for lbl in labels:
        class_counts[lbl] += 1

    # accumulate inverse-distance weights, then normalise by class size
    raw_votes = defaultdict(float)
    for lbl, d in zip(top_labels, top_dists):
        raw_votes[lbl] += 1.0 / (d + 1e-9)

    norm_votes = {lbl: score / class_counts[lbl]
                  for lbl, score in raw_votes.items()}

    return max(norm_votes, key=norm_votes.get), norm_votes


def classify_ensemble(dists, labels, k=K):
    """
    Run all three classifiers and take majority vote.
    Ties broken by lowest per-class mean distance.
    """
    p1 = classify_1nn(dists, labels)
    p2, scores2 = classify_class_mean_top3(dists, labels)
    p3, scores3 = classify_weighted_knn_normalized(dists, labels, k)

    predictions = [p1, p2, p3]
    vote_count  = defaultdict(int)
    for p in predictions:
        vote_count[p] += 1

    max_votes = max(vote_count.values())
    winners   = [lbl for lbl, v in vote_count.items() if v == max_votes]

    if len(winners) == 1:
        return winners[0], vote_count, predictions

    # tie-break: pick winner with best class_mean_top3 score
    tiebreak = min(winners, key=lambda lbl: scores2.get(lbl, float('inf')))
    return tiebreak, vote_count, predictions

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
query = extract_mfcc_seq(audio)
dists = np.array([dtw_distance(query, t) for t in templates])

# Run all classifiers
p1                  = classify_1nn(dists, labels)
p2, class_scores    = classify_class_mean_top3(dists, labels)
p3, norm_votes      = classify_weighted_knn_normalized(dists, labels, K)
final, vote_count, preds = classify_ensemble(dists, labels, K)

# ---------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------
top_k = np.argsort(dists)[:K]
print(f"Top {K} nearest neighbours:")
for i in top_k:
    print(f"  {labels[i]:25s}  dist={dists[i]:.4f}")

print("\nPer-class mean of top-3 distances:")
for lbl, s in sorted(class_scores.items(), key=lambda x: x[1]):
    print(f"  {lbl:25s}  score={s:.4f}")

print("\nNormalised weighted k-NN scores:")
for lbl, s in sorted(norm_votes.items(), key=lambda x: -x[1]):
    print(f"  {lbl:25s}  score={s:.4f}")

print("\n" + "="*50)
print(f"  1-NN result          : {p1}")
print(f"  Class mean top-3     : {p2}")
print(f"  Normalised k-NN      : {p3}")
print(f"  ENSEMBLE (final)     : {final}   [{vote_count[final]}/3 votes]")
print("="*50)