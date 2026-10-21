"""RevSense: simple FFT harmonic-comb RPM estimator for single-cylinder, four-stroke audio."""
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks, windows

MIN_RPM, MAX_RPM = 900.0, 6500.0
DEFAULT_WINDOW_SEC, DEFAULT_HOP_SEC, MAX_HARMONICS = 1.5, 0.5, 8


def load_audio(path):
    audio, sr = sf.read(str(path), dtype='float64')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or len(audio) == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f'Empty, invalid, or non-finite audio: {path}')
    return audio - np.mean(audio), int(sr)


def _spectrum(segment, sr):
    n = len(segment)
    x = (segment - np.mean(segment)) * windows.hann(n, sym=False)
    mag = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(n, d=1/sr)
    keep = (freq >= 5) & (freq <= min(2000, sr/2))
    freq, mag = freq[keep], mag[keep]
    if not len(mag) or np.max(mag) <= 1e-12:
        return freq, np.zeros_like(mag), []
    mag = mag / (np.max(mag) + 1e-12)
    bin_hz = sr / n
    inds, _ = find_peaks(
        mag, prominence=.025, distance=max(1, int(round(2/bin_hz))
    ))
    inds = sorted(inds, key=lambda i: mag[i], reverse=True)[:20]
    return freq, mag, [(float(freq[i]), float(mag[i])) for i in inds]


def estimate_window(segment, sr, min_rpm=MIN_RPM, max_rpm=MAX_RPM):
    segment = np.asarray(segment, dtype=float)
    if segment.ndim != 1 or len(segment) < 16:
        raise ValueError('Window must be a 1-D audio segment of at least 16 samples.')
    rms = float(np.sqrt(np.mean(segment**2)))
    freqs, mag, peaks = _spectrum(segment, sr)
    peak_text = '; '.join(f'{f:.2f}:{a:.2f}' for f, a in peaks[:12])
    if not len(mag) or np.max(mag) <= 1e-12:
        return dict(rpm=np.nan, f0_hz=np.nan, score=0., matched_harmonics=0,
                    rms=rms, half_frequency_score=np.nan, score_margin=np.nan,
                    quality='REVIEW: no usable spectrum', peaks_hz_rel=peak_text)

    bin_hz = sr / len(segment)
    candidates = np.arange(min_rpm/120, max_rpm/120 + .001, max(.10, bin_hz/4))
    rows = []
    for f0 in candidates:
        weighted = weights = 0.
        matched = 0
        for h in range(1, MAX_HARMONICS+1):
            target = h*f0
            if target > freqs[-1]:
                break
            tol = max(1.25*bin_hz, target*.008)
            region = np.abs(freqs-target) <= tol
            strength = float(np.max(mag[region])) if np.any(region) else 0.
            weight = 1/(h**.65)
            weighted += weight*strength
            weights += weight
            if strength >= .10:
                matched += 1
        score = weighted/weights if weights else 0.
        adjusted = score + .018*min(matched, 6)
        rows.append(dict(f0_hz=float(f0), score=float(score),
                         adjusted_score=float(adjusted), matched_harmonics=matched))

    rows.sort(key=lambda r: r['adjusted_score'], reverse=True)
    best = rows[0]
    f0, score, matched = best['f0_hz'], best['score'], best['matched_harmonics']
    half_rows = [r for r in rows if abs(r['f0_hz']-f0/2) <= max(.1, bin_hz/4)*1.1]
    half_score = max((r['score'] for r in half_rows), default=np.nan)
    alternatives = [r for r in rows if abs(r['f0_hz']-f0) > max(.35, bin_hz/2)]
    score_margin = best['adjusted_score']-alternatives[0]['adjusted_score'] if alternatives else np.nan
    flags = []
    if rms < .003:
        flags.append('low signal')
    if matched < 3:
        flags.append('fewer than 3 matched harmonics')
    if score < .12:
        flags.append('weak harmonic evidence')
    if np.isfinite(half_score) and half_score >= score*.92:
        flags.append('possible octave ambiguity')
    quality = 'REVIEW: '+'; '.join(flags) if flags else 'PASS'
    return dict(rpm=float(f0*120), f0_hz=float(f0), score=float(score),
                matched_harmonics=int(matched), rms=rms,
                half_frequency_score=float(half_score) if np.isfinite(half_score) else np.nan,
                score_margin=float(score_margin) if np.isfinite(score_margin) else np.nan,
                quality=quality, peaks_hz_rel=peak_text)


def _flag_isolated_jumps(rows):
    for i, row in enumerate(rows):
        if not np.isfinite(row.get('rpm', np.nan)):
            continue
        neighbours = [
            rows[j]['rpm'] for j in range(max(0, i-2), min(len(rows), i+3))
            if j != i and np.isfinite(rows[j].get('rpm', np.nan))
            and rows[j].get('quality') == 'PASS'
        ]
        if len(neighbours) < 2:
            continue
        local = float(np.median(neighbours))
        spread = (max(neighbours)-min(neighbours))/local if local > 0 else np.inf
        jump = abs(row['rpm']-local)/local if local > 0 else np.inf
        if spread <= .12 and jump >= .30:
            row['quality'] = 'REVIEW: isolated RPM jump'


def analyse_recording(path, window_sec=DEFAULT_WINDOW_SEC, hop_sec=DEFAULT_HOP_SEC,
                      min_rpm=MIN_RPM, max_rpm=MAX_RPM):
    # Only the revving recording is routed to the dedicated tracker.
    # All other recordings keep the original steady-recording path unchanged.
    if 'revving' in Path(path).stem.lower():
        rev_df, rev_summary = analyse_revving_recording(
            path, window_sec=0.50, hop_sec=0.15,
            min_rpm=min_rpm, max_rpm=max(12000.0, max_rpm)
        )
        # Keep a consistent schema across steady and revving recordings.
        # f0_hz always means the frequency used for the displayed RPM estimate;
        # raw_f0_hz and corrected_f0_hz preserve the diagnostic distinction.
        if 'f0_hz' not in rev_df.columns:
            rev_df['f0_hz'] = rev_df['corrected_f0_hz']
        # A corrected octave candidate may still be usable; preserve the correction
        # in its own column and only mark the frame REVIEW if continuity checks fail.
        rev_df['quality'] = rev_df['quality_original'].fillna('').astype(str)
        corrected_mask = rev_df['correction'].astype(str) != 'none'
        # Accept an octave correction only when octave ambiguity is the sole
        # reason for review; do not override low-signal/weak-evidence warnings.
        for corrected_idx in rev_df.index[corrected_mask]:
            original_quality = str(rev_df.at[corrected_idx, 'quality_original'])
            if original_quality == 'PASS' or original_quality == 'REVIEW: possible octave ambiguity':
                rev_df.at[corrected_idx, 'quality'] = 'PASS: octave corrected'
        continuity_mask = rev_df['continuity_flag'].astype(str).str.startswith('REVIEW')
        rev_df.loc[continuity_mask, 'quality'] = rev_df.loc[continuity_mask, 'continuity_flag']
        valid = rev_df[np.isfinite(rev_df['rpm'])]
        accepted = valid[~valid['continuity_flag'].astype(str).str.startswith('REVIEW')]
        rev_summary.update({
            'total_windows': len(rev_df),
            'accepted_windows': len(accepted),
            'accepted_percent': round(100*len(accepted)/max(len(rev_df), 1), 1),
            'mean_rpm': float(accepted['rpm'].mean()) if len(accepted) else np.nan,
            'median_rpm': float(accepted['rpm'].median()) if len(accepted) else np.nan,
            'p10_rpm': float(accepted['rpm'].quantile(.10)) if len(accepted) else np.nan,
            'p90_rpm': float(accepted['rpm'].quantile(.90)) if len(accepted) else np.nan,
        })
        return rev_df, rev_summary

    audio, sr = load_audio(path)
    win_n, hop_n = int(round(window_sec*sr)), int(round(hop_sec*sr))
    if win_n <= 0 or hop_n <= 0:
        raise ValueError('Window and hop durations must be positive.')
    if len(audio) < win_n:
        raise ValueError(f'{Path(path).name} is shorter than {window_sec:.2f} s.')
    rows = []
    for start in range(0, len(audio)-win_n+1, hop_n):
        result = estimate_window(audio[start:start+win_n], sr, min_rpm, max_rpm)
        rows.append({'file': Path(path).name, 'start_s': start/sr,
                     'end_s': (start+win_n)/sr, **result})
    _flag_isolated_jumps(rows)
    df = pd.DataFrame(rows)
    accepted = df[df['quality'] == 'PASS']
    if len(accepted):
        median, mean = float(accepted.rpm.median()), float(accepted.rpm.mean())
        p10, p90 = float(accepted.rpm.quantile(.10)), float(accepted.rpm.quantile(.90))
    else:
        median = mean = p10 = p90 = np.nan
    summary = {
        'file': Path(path).name, 'sample_rate_hz': sr, 'duration_s': len(audio)/sr,
        'window_sec': window_sec, 'hop_sec': hop_sec, 'total_windows': len(df),
        'accepted_windows': len(accepted),
        'accepted_percent': round(100*len(accepted)/max(len(df), 1), 1),
        'median_rpm': median, 'mean_rpm': mean, 'p10_rpm': p10, 'p90_rpm': p90
    }
    return df, summary


# ---------------------------------------------------------------------------
# Revving-only estimator.
# The original steady-recording functions above are intentionally unchanged.
# ---------------------------------------------------------------------------
def analyse_revving_recording(path, window_sec=0.50, hop_sec=0.15,
                              min_rpm=900.0, max_rpm=12000.0,
                              octave_score_ratio=0.92):
    """Track RPM through a revving recording and correct likely octave errors.

    The harmonic-comb can select 2*f0 instead of f0 when the second harmonic
    is louder than the fundamental. If the half-frequency candidate has nearly
    the same score, the higher estimate is likely an octave error. This function
    preserves raw estimates and logs the correction in separate diagnostic columns.
    """
    audio, sr = load_audio(path)
    win_n, hop_n = int(round(window_sec*sr)), int(round(hop_sec*sr))
    if win_n <= 0 or hop_n <= 0:
        raise ValueError('Window and hop durations must be positive.')
    if len(audio) < win_n:
        raise ValueError(f'{Path(path).name} is shorter than {window_sec:.2f} s.')

    rows = []
    for start in range(0, len(audio)-win_n+1, hop_n):
        result = estimate_window(audio[start:start+win_n], sr, min_rpm, max_rpm)
        raw_rpm = float(result['rpm']) if np.isfinite(result['rpm']) else np.nan
        half_score = result.get('half_frequency_score', np.nan)
        score = result.get('score', np.nan)
        corrected_rpm = raw_rpm
        correction = 'none'
        if (np.isfinite(raw_rpm) and np.isfinite(half_score) and np.isfinite(score)
                and raw_rpm >= 2*min_rpm
                and score > 0
                and half_score >= score*octave_score_ratio):
            corrected_rpm = raw_rpm/2.0
            correction = 'halved: strong half-frequency candidate'
        row = {
            'file': Path(path).name,
            'start_s': start/sr,
            'end_s': (start+win_n)/sr,
            'raw_rpm': raw_rpm,
            'rpm': corrected_rpm,
            'raw_f0_hz': result.get('f0_hz', np.nan),
            'corrected_f0_hz': corrected_rpm/120 if np.isfinite(corrected_rpm) else np.nan,
            'f0_hz': corrected_rpm/120 if np.isfinite(corrected_rpm) else np.nan,
            'adjusted_score': np.nan,
            'score': score,
            'half_frequency_score': half_score,
            'half_score_ratio': (half_score/score if np.isfinite(half_score) and np.isfinite(score) and score > 0 else np.nan),
            'matched_harmonics': result.get('matched_harmonics', np.nan),
            'rms': result.get('rms', np.nan),
            'score_margin': result.get('score_margin', np.nan),
            'correction': correction,
            'quality_original': result.get('quality', ''),
            'quality': result.get('quality', ''),
            'peaks_hz_rel': result.get('peaks_hz_rel', '')
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Flag abrupt remaining changes for inspection; do not silently flatten genuine revs.
    df['delta_rpm_from_previous'] = df['rpm'].diff()
    df['delta_rpm_per_sec'] = df['delta_rpm_from_previous'] / hop_sec
    df['continuity_flag'] = ''
    for i in range(1, len(df)):
        prev_rpm, rpm = df.loc[i-1, 'rpm'], df.loc[i, 'rpm']
        if np.isfinite(prev_rpm) and np.isfinite(rpm):
            change = abs(rpm-prev_rpm)
            # Deliberately generous threshold: flags suspicious changes without
            # assuming the engine's real acceleration profile.
            if change > 2500 and df.loc[i, 'correction'] == 'none':
                df.loc[i, 'continuity_flag'] = 'REVIEW: >2500 RPM change between adjacent windows'

    valid = df['rpm'].dropna()
    summary = {
        'file': Path(path).name,
        'sample_rate_hz': sr,
        'duration_s': len(audio)/sr,
        'window_sec': window_sec,
        'hop_sec': hop_sec,
        'total_windows': len(df),
        'octave_corrections': int((df['correction'] != 'none').sum()),
        'continuity_flags': int((df['continuity_flag'] != '').sum()),
        'median_rpm': float(valid.median()) if len(valid) else np.nan,
        'p10_rpm': float(valid.quantile(.10)) if len(valid) else np.nan,
        'p90_rpm': float(valid.quantile(.90)) if len(valid) else np.nan
    }
    return df, summary
