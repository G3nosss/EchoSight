#!/usr/bin/env python3
"""
EchoSight — Acoustic Preprocessing Engine
==========================================
Ingests raw sonar data (.wav or .csv), applies STFT-based spectrogram
extraction, noise floor normalization, and emits a JSON-serialized 2D
array to stdout for Node.js consumption.

Mobile-safe: uses scipy only (avoids librosa's heavy numba dependency
which fails to JIT-compile on ARM Termux without LLVM toolchain).

Usage (from Node.js subprocess or CLI):
  python3 preprocess.py --input mock_sonar.wav --mode stft
  python3 preprocess.py --input mock_sonar.csv --mode csv
  python3 preprocess.py --mock --mode stft     # pure synthetic data

Output (stdout): JSON { "spectrogram": [[...]], "meta": {...} }
Errors go to stderr only — stdout is clean for IPC.
"""

import sys
import json
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # suppress scipy deprecation noise on ARM

# ── Conditional imports (graceful fallback) ──────────────────────────────────
try:
    from scipy.io import wavfile
    from scipy.signal import stft, medfilt2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print(json.dumps({"error": "scipy not found. Run: pip install scipy"}), file=sys.stderr)
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "n_fft":        512,    # FFT window — keep low for mobile RAM
    "hop_length":   128,    # Stride between frames
    "win_length":   512,    # Must be ≤ n_fft
    "freq_bins":    128,    # Output height after mel-compression
    "time_frames":  256,    # Output width (truncate/pad to this)
    "db_ref":       1.0,    # Reference power for dB conversion
    "noise_percentile": 15, # Bottom N% treated as noise floor
    "dynamic_range_db": 80, # Clamp range above noise floor
}


# ── Synthetic Sonar Data Generator ───────────────────────────────────────────
def generate_mock_sonar(duration_s: float = 3.0, sample_rate: int = 22050) -> tuple:
    """
    Generates physically plausible mock forward-looking sonar data.
    Simulates:
      - Broadband reverberation decay envelope (exponential)
      - 3 discrete target echoes at distinct ranges (frequencies)
      - Sea-floor harmonic interference
      - Gaussian ambient noise floor
    """
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    
    # Reverberation envelope — sonar returns decay as 1/r²
    reverb_envelope = np.exp(-2.5 * t / duration_s)
    
    # Target echoes (strong reflectors at distinct "ranges")
    echo1 = 0.8 * np.sin(2 * np.pi * 440  * t) * np.exp(-10 * (t - 0.3)**2)
    echo2 = 0.6 * np.sin(2 * np.pi * 880  * t) * np.exp(-10 * (t - 0.9)**2)
    echo3 = 0.4 * np.sin(2 * np.pi * 1320 * t) * np.exp(-10 * (t - 1.7)**2)
    
    # Sea-floor broadband harmonic interference
    seafloor = 0.2 * (
        np.sin(2 * np.pi * 200 * t) +
        np.sin(2 * np.pi * 400 * t) +
        np.sin(2 * np.pi * 600 * t)
    ) * reverb_envelope
    
    # Gaussian noise floor (−40 dBFS equivalent)
    noise = np.random.normal(0, 0.01, len(t))
    
    # Superimpose all components
    signal = (echo1 + echo2 + echo3 + seafloor + noise).astype(np.float32)
    
    # Clip to float32 range
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal, sample_rate


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_wav(path: str) -> tuple:
    """Load .wav file, mono-fold if stereo, normalize to float32."""
    sr, data = wavfile.read(path)
    
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    
    # Stereo → mono via average
    if data.ndim == 2:
        data = data.mean(axis=1)
    
    return data, sr


def load_csv(path: str) -> tuple:
    """
    Load CSV sonar data. Expected format:
      Col 0: optional timestamp (ignored)
      Col 1: amplitude values
    OR single-column amplitude.
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    
    if raw.ndim == 2:
        # Take last column as signal (most CSVs have [time, amplitude])
        data = raw[:, -1].astype(np.float32)
    else:
        data = raw.astype(np.float32)
    
    # Normalize to [-1, 1]
    peak = np.abs(data).max()
    if peak > 0:
        data /= peak
    
    # Assume standard sonar sampling rate if not embedded
    sr = 22050
    return data, sr


# ── DSP Core ──────────────────────────────────────────────────────────────────
def compute_stft_spectrogram(signal: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    """
    Computes Short-Time Fourier Transform magnitude spectrogram.
    
    Returns log-power spectrogram with shape (freq_bins, time_frames)
    normalized to [0.0, 1.0] after noise floor subtraction.
    """
    n_fft      = cfg["n_fft"]
    hop_length = cfg["hop_length"]
    win_length = cfg["win_length"]
    
    # Hann window — optimal for sonar (low sidelobe energy)
    window = np.hanning(win_length)
    
    # scipy STFT: output shape (freq_bins, time_frames)
    freqs, times, Zxx = stft(
        signal,
        fs=sr,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        return_onesided=True,
        padded=True,
    )
    
    # Magnitude spectrum
    magnitude = np.abs(Zxx)   # shape: (n_fft//2 + 1, T)
    
    # Convert to dB scale (power spectrum)
    power = magnitude ** 2
    db = 10.0 * np.log10(np.maximum(power, 1e-10) / cfg["db_ref"])
    
    return db, freqs, times


def compress_frequency_axis(db: np.ndarray, target_bins: int) -> np.ndarray:
    """
    Mel-inspired frequency compression: bin the linear freq axis into
    `target_bins` non-uniform bands (more resolution at low freqs).
    
    Avoids librosa's mel filterbank while approximating its perceptual weighting.
    Mobile-safe pure numpy.
    """
    n_freqs, n_times = db.shape
    
    if n_freqs == target_bins:
        return db
    
    # Logarithmic bin edges (mel-like spacing)
    log_edges = np.logspace(np.log10(1), np.log10(n_freqs), target_bins + 1)
    log_edges = np.clip(np.round(log_edges).astype(int), 0, n_freqs - 1)
    
    compressed = np.zeros((target_bins, n_times), dtype=np.float32)
    for i in range(target_bins):
        lo = log_edges[i]
        hi = max(log_edges[i + 1], lo + 1)
        compressed[i] = db[lo:hi].mean(axis=0)
    
    return compressed


def noise_floor_normalize(spec: np.ndarray, cfg: dict) -> np.ndarray:
    """
    CFAR-inspired normalization:
      1. Estimate noise floor as Nth percentile across the entire spectrogram
      2. Subtract noise floor (lifts weak targets above zero)
      3. Clamp to [0, dynamic_range_db]
      4. Scale to [0.0, 1.0]
    
    This is analogous to Constant False Alarm Rate (CFAR) detection in
    real sonar systems, without the sliding window overhead.
    """
    noise_floor = np.percentile(spec, cfg["noise_percentile"])
    
    # Lift above noise
    spec_lifted = spec - noise_floor
    
    # Clamp to dynamic range
    spec_clamped = np.clip(spec_lifted, 0.0, cfg["dynamic_range_db"])
    
    # Apply 2D median filter to suppress salt-and-pepper noise
    # kernel must be odd; 3x3 is mobile-safe
    try:
        spec_filtered = medfilt2d(spec_clamped, kernel_size=3)
    except Exception:
        spec_filtered = spec_clamped  # fallback if medfilt2d unavailable
    
    # Min-max normalize to [0, 1]
    vmin = spec_filtered.min()
    vmax = spec_filtered.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(spec_filtered)
    
    normalized = (spec_filtered - vmin) / (vmax - vmin)
    return normalized.astype(np.float32)


def resize_time_axis(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Pad or truncate spectrogram to fixed time axis width.
    Padding uses edge replication (preserve boundary energy profile).
    """
    n_freqs, n_times = spec.shape
    
    if n_times >= target_frames:
        return spec[:, :target_frames]
    
    # Pad with zeros (silence / noise floor equivalent)
    pad_width = target_frames - n_times
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(signal: np.ndarray, sr: int, cfg: dict) -> dict:
    """
    Full preprocessing pipeline:
    WAV/CSV → STFT → dB → Freq-Compression → Noise Floor → Fixed Shape
    
    Returns dict with:
      spectrogram: List[List[float]]  shape (freq_bins × time_frames)
      meta: pipeline metadata for frontend overlay
    """
    # Step 1: STFT
    db_spec, freqs, times = compute_stft_spectrogram(signal, sr, cfg)
    
    # Step 2: Mel-like frequency compression
    compressed = compress_frequency_axis(db_spec, cfg["freq_bins"])
    
    # Step 3: Noise floor CFAR normalization
    normalized = noise_floor_normalize(compressed, cfg)
    
    # Step 4: Temporal alignment to fixed width
    aligned = resize_time_axis(normalized, cfg["time_frames"])
    
    # Shape verification
    assert aligned.shape == (cfg["freq_bins"], cfg["time_frames"]), \
        f"Shape mismatch: expected ({cfg['freq_bins']}, {cfg['time_frames']}), got {aligned.shape}"
    
    # Compute stats for frontend HUD
    active_mask = aligned > 0.3  # "significant return" threshold
    meta = {
        "shape":       list(aligned.shape),
        "sample_rate": sr,
        "duration_s":  round(len(signal) / sr, 3),
        "peak_db":     round(float(db_spec.max()), 2),
        "noise_db":    round(float(np.percentile(db_spec, cfg["noise_percentile"])), 2),
        "target_count": int(np.sum(medfilt2d((aligned > 0.6).astype(float), 5) > 0.5)),
        "snr_estimate": round(float(
            np.mean(aligned[active_mask]) / (np.mean(aligned[~active_mask]) + 1e-9)
        ), 2),
        "n_fft":       cfg["n_fft"],
        "freq_bins":   cfg["freq_bins"],
        "time_frames": cfg["time_frames"],
    }
    
    return {
        "spectrogram": aligned.tolist(),   # 2D list for JSON serialization
        "meta": meta,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="EchoSight Acoustic Preprocessor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input",  type=str, help="Path to .wav or .csv file")
    parser.add_argument("--mode",   type=str, default="stft",
                        choices=["stft", "csv"], help="Input mode")
    parser.add_argument("--mock",   action="store_true",
                        help="Use synthetic sonar data (ignores --input)")
    parser.add_argument("--n_fft",  type=int, default=DEFAULT_CONFIG["n_fft"])
    parser.add_argument("--bins",   type=int, default=DEFAULT_CONFIG["freq_bins"])
    parser.add_argument("--frames", type=int, default=DEFAULT_CONFIG["time_frames"])
    
    args = parser.parse_args()
    
    # Build runtime config
    cfg = DEFAULT_CONFIG.copy()
    cfg["n_fft"]       = args.n_fft
    cfg["freq_bins"]   = args.bins
    cfg["time_frames"] = args.frames
    cfg["win_length"]  = args.n_fft  # keep consistent
    
    try:
        # Load signal
        if args.mock or not args.input:
            print("[EchoSight] Using synthetic sonar data", file=sys.stderr)
            signal, sr = generate_mock_sonar()
        elif args.mode == "csv":
            signal, sr = load_csv(args.input)
        else:
            signal, sr = load_wav(args.input)
        
        print(f"[EchoSight] Loaded signal: {len(signal)} samples @ {sr}Hz", file=sys.stderr)
        
        # Run pipeline
        result = run_pipeline(signal, sr, cfg)
        
        print(f"[EchoSight] Spectrogram shape: {result['meta']['shape']}", file=sys.stderr)
        print(f"[EchoSight] SNR estimate: {result['meta']['snr_estimate']} dB", file=sys.stderr)
        
        # Emit clean JSON to stdout (Node.js reads this)
        print(json.dumps(result, separators=(",", ":")))
        sys.stdout.flush()
        
    except FileNotFoundError as e:
        print(json.dumps({"error": f"File not found: {e}"}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "trace": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
