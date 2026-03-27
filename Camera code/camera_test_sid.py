"""
autofocus_scorer.py
====================
Real-time preview + snapshot-based focus scoring for the AmScope MD500A
with the Reichert 4/.10 Oil Objective.

How it works:
--------------------------
1. CALIBRATION  — On startup you capture a set of unfocused snapshots then a
   set of focused snapshots. These become permanent anchors for normalisation,
   so the score is absolute and does not drift as more snapshots accumulate.

2. PRE-BLUR — A small Gaussian blur is applied to the grayscale ROI before any
   metric runs. Sensor noise is spatially random (each pixel independent) so it
   has very high spatial frequency and looks like sharp "edges" to Laplacian /
   Tenengrad. A gentle blur collapses that noise while leaving real structural
   edges (which are correlated across many adjacent pixels) intact.

Focus metrics (FFT removed for performance):
  1. Laplacian Variance  — second-order edge sharpness
  2. Tenengrad           — first-order Sobel gradient energy
  3. Normalized Variance — contrast spread relative to mean brightness

Controls during normal operation:
  [SPACE] — Snapshot and score current frame
  [C]     — Re-run calibration from scratch
  [Q]     — Quit

Dependencies:
  pip install opencv-python numpy
"""

import cv2
import numpy as np
import threading
import time
from collections import deque


# ==============================================================================
# CONFIGURATION — adjust to taste
# ==============================================================================

CAMERA_INDEX      = 0               # Change if MD500A is not device 0
CAMERA_BACKEND    = cv2.CAP_DSHOW   # DirectShow; best for Windows USB cameras
FRAME_WIDTH       = 2592            # MD500A native resolution
FRAME_HEIGHT      = 1944

SCORE_SCALE       = 0.25            # Fraction of native res used for scoring math

# Pre-blur kernel size — must be odd. 5 works well for the MD500A sensor noise
# floor. Increase to 7 if random-pixel frames still score unexpectedly high.
BLUR_KERNEL       = 5

COMPOSITE_WEIGHTS = {               # Must sum to 1.0
    "laplacian": 0.40,
    "tenengrad": 0.35,
    "variance":  0.25,
}

# Calibration: how many snapshots to collect for each anchor point.
# More = more stable baseline but slower startup.
CALIB_SNAPSHOTS   = 5

# Rolling history length (number of post-calibration snapshots kept).
# Used only for the pre-calibration fallback normaliser.
HISTORY_LEN       = 60


# ==============================================================================
# FOCUS METRIC FUNCTIONS
# ==============================================================================

def preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur before scoring.

    Sensor noise is spatially uncorrelated — each noisy pixel is independent
    of its neighbours. This looks identical to a very high-frequency texture
    to gradient-based metrics, causing out-of-focus noisy frames to score
    deceptively high. Gaussian blur averages each pixel with its neighbours,
    which collapses uncorrelated noise (neighbours average to zero) while
    preserving real edges (which are correlated across many adjacent pixels).
    """
    return cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)


def laplacian_variance(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian (second-order derivative).
    High variance = sharp, well-defined edges = good focus.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def tenengrad(gray: np.ndarray, kernel_size: int = 3) -> float:
    """
    Sum of squared Sobel gradient magnitudes.
    Squaring amplifies strong edges relative to weak ones, making this
    metric more discriminating than raw gradient magnitude for subtle
    focus differences at low magnification.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
    return float(np.mean(gx**2 + gy**2))


def normalized_variance(gray: np.ndarray) -> float:
    """
    Variance / mean — illumination-invariant contrast measure.
    Dividing by mean means a bright, flat field and a dim, flat field both
    score near zero, regardless of illumination level.
    """
    return float(np.var(gray) / (np.mean(gray) + 1e-6))


def compute_raw_metrics(gray: np.ndarray) -> dict:
    """Run all three metrics on a preprocessed grayscale image."""
    blurred = preprocess(gray)
    return {
        "laplacian": laplacian_variance(blurred),
        "tenengrad": tenengrad(blurred),
        "variance":  normalized_variance(blurred),
    }


# ==============================================================================
# CALIBRATION
# ==============================================================================

class Calibration:
    """
    Holds the unfocused and focused anchor points for each metric.

    After calibration, raw metric values are mapped to [0, 1] using:

        score = (value - unfocused_anchor) / (focused_anchor - unfocused_anchor)

    and clamped to [0, 1]. This makes the score absolute and stable — it no
    longer depends on what other snapshots happen to be in the rolling window.
    """

    def __init__(self):
        self.unfocused: dict | None = None   # Mean raw metrics across unfocused snaps
        self.focused:   dict | None = None   # Mean raw metrics across focused snaps
        self.ready = False

    def set_anchors(self, unfocused_snaps: list[dict], focused_snaps: list[dict]):
        """
        Average across N calibration snapshots for each anchor point.
        Averaging reduces the effect of any single noisy calibration frame.
        """
        def mean_metrics(snaps):
            keys = snaps[0].keys()
            return {k: float(np.mean([s[k] for s in snaps])) for k in keys}

        self.unfocused = mean_metrics(unfocused_snaps)
        self.focused   = mean_metrics(focused_snaps)
        self.ready     = True

        print("\n[Calibration] Anchors set:")
        for k in self.unfocused:
            print(f"  {k:12s}  unfocused={self.unfocused[k]:.2f}  "
                  f"focused={self.focused[k]:.2f}")

    def normalise(self, raw: dict) -> dict:
        """
        Map each raw metric value to [0, 1] using the calibrated anchors.

        If the focused and unfocused anchors for a metric are too close
        together (< 1e-6 apart), we fall back to 0.5 to avoid division
        by near-zero — this would only happen if calibration was performed
        identically for both states, which indicates a bad calibration.
        """
        result = {}
        for k, v in raw.items():
            lo = self.unfocused[k]
            hi = self.focused[k]
            if abs(hi - lo) < 1e-6:
                result[k] = 0.5  # Degenerate calibration fallback
            else:
                result[k] = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        return result


# ==============================================================================
# COMPOSITE SCORER
# ==============================================================================

class FocusScorer:
    """
    Orchestrates calibration and normalisation.

    Pre-calibration: falls back to min-max normalisation against rolling history
    so the display is not blank while the wizard runs.

    Post-calibration: uses the fixed calibration anchors for normalisation.
    """

    def __init__(self):
        self.calib           = Calibration()
        # Pre-calibration fallback history (min-max over rolling window)
        self._pre_calib_hist = {k: deque(maxlen=HISTORY_LEN) for k in COMPOSITE_WEIGHTS}

    def _fallback_normalise(self, raw: dict) -> dict:
        """Min-max normalise against rolling history (pre-calibration only)."""
        result = {}
        for k, v in raw.items():
            hist = self._pre_calib_hist[k]
            hist.append(v)
            lo, hi = min(hist), max(hist)
            if hi - lo < 1e-9:
                result[k] = 0.5
            else:
                result[k] = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        return result

    def score(self, gray: np.ndarray) -> dict:
        """
        Compute metrics, normalise (calibrated or fallback).
        Returns a dict with composite, raw, and normalised fields.
        """
        raw = compute_raw_metrics(gray)

        if self.calib.ready:
            normalised = self.calib.normalise(raw)
        else:
            normalised = self._fallback_normalise(raw)

        composite = sum(normalised[k] * COMPOSITE_WEIGHTS[k] for k in COMPOSITE_WEIGHTS)

        return {
            "composite":  composite,
            "raw":        raw,
            "normalised": normalised,
        }


# ==============================================================================
# CALIBRATION WIZARD (interactive, runs in main thread)
# ==============================================================================

def run_calibration_wizard(cap, scorer: FocusScorer, disp_scale: float):
    """
    Guides the user through capturing unfocused and focused anchor snapshots.

    Runs entirely in the main thread (no background scoring during calibration)
    to keep the logic simple and avoid race conditions on scorer state.

    Steps:
      Phase 1 — Unfocused: user moves objective away from focus, presses SPACE
                            CALIB_SNAPSHOTS times.
      Phase 2 — Focused:   user focuses the sample, presses SPACE
                            CALIB_SNAPSHOTS times.
    """
    def collect_phase(phase_name: str, instruction: str) -> list[dict]:
        """Show instruction on screen and collect N raw metric snapshots."""
        collected = []
        while len(collected) < CALIB_SNAPSHOTS:
            ret, frame = cap.read()
            if not ret:
                continue

            # Build display with calibration overlay
            if disp_scale < 1.0:
                dw = int(frame.shape[1] * disp_scale)
                dh = int(frame.shape[0] * disp_scale)
                display = cv2.resize(frame, (dw, dh))
            else:
                display = frame.copy()

            # Dark overlay panel
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

            cv2.putText(display, f"CALIBRATION — {phase_name}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            cv2.putText(display, instruction,
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            cv2.putText(display,
                        f"SPACE to capture ({len(collected)}/{CALIB_SNAPSHOTS})",
                        (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

            cv2.imshow("Autofocus Scorer — MD500A", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                return []   # Abort calibration

            elif key == ord(' '):
                small = cv2.resize(frame, (0, 0), fx=SCORE_SCALE, fy=SCORE_SCALE,
                                   interpolation=cv2.INTER_AREA)
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                raw   = compute_raw_metrics(gray)
                collected.append(raw)
                print(f"[Calib:{phase_name}] Snap {len(collected)}/{CALIB_SNAPSHOTS} — "
                      f"lap={raw['laplacian']:.1f}  ten={raw['tenengrad']:.1f}  "
                      f"var={raw['variance']:.3f}")

        return collected

    print("\n" + "="*60)
    print("  CALIBRATION WIZARD")
    print("  Collecting unfocused anchor first, then focused anchor.")
    print("="*60)

    # --- Phase 1: Unfocused ---
    unfocused_snaps = collect_phase(
        "UNFOCUSED",
        "Defocus the objective so the image is blurry, then press SPACE."
    )
    if not unfocused_snaps:
        print("[Calib] Aborted.")
        return

    # --- Phase 2: Focused ---
    focused_snaps = collect_phase(
        "FOCUSED",
        "Bring the sample into sharp focus, then press SPACE."
    )
    if not focused_snaps:
        print("[Calib] Aborted.")
        return

    scorer.calib.set_anchors(unfocused_snaps, focused_snaps)
    print("[Calib] Complete — normal scoring active.\n")


# ==============================================================================
# SHARED SCORE RESULT (thread-safe)
# ==============================================================================

class ScoreResult:
    """
    Thread-safe bridge between the background scoring thread and the display.
    `pending` is True while a score is being computed in the background.
    """
    def __init__(self):
        self._lock   = threading.Lock()
        self._data   = None
        self.pending = False

    def write(self, data: dict):
        with self._lock:
            self._data   = data
            self.pending = False

    def read(self) -> dict | None:
        with self._lock:
            return self._data


# ==============================================================================
# HUD OVERLAY
# ==============================================================================

def draw_hud(display: np.ndarray, result: ScoreResult, fps: float, calibrated: bool):
    """
    Draw a simplified overlay showing:
      - Calibration status
      - Composite focus score as a number + colour-coded bar
      - FPS
      - Key hint footer
    """
    scores  = result.read()
    pending = result.pending
    h, _    = display.shape[:2]

    # Semi-transparent dark panel — smaller than before since no metric bars
    panel_w, panel_h = 310, 115
    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # Calibration status badge
    calib_text   = "CALIBRATED" if calibrated else "NOT CALIBRATED — press C"
    calib_colour = (0, 210, 60) if calibrated else (0, 100, 220)
    cv2.putText(display, calib_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, calib_colour, 1)

    if pending:
        cv2.putText(display, "Scoring...", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 50), 2)

    elif scores is None:
        cv2.putText(display, "Press SPACE to score", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    else:
        composite = scores["composite"]

        # Colour ramp: red -> yellow -> green as score increases
        if composite < 0.4:
            colour = (0, 0, 220)
        elif composite < 0.7:
            colour = (0, 200, 220)
        else:
            colour = (0, 210, 60)

        cv2.putText(display, f"Focus: {composite:.3f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, colour, 2)

        # Filled progress bar — width proportional to composite score
        bx, by, bw, bh = 20, 73, panel_w - 20, 14
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
        cv2.rectangle(display, (bx, by), (bx + int(composite * bw), by + bh), colour, -1)

    # Footer
    cv2.putText(display, f"FPS: {fps:.1f}", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(display,
                "[SPACE] Score  [C] Recalibrate  [Q] Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)


# ==============================================================================
# BACKGROUND SCORING THREAD
# ==============================================================================

def run_score(frame: np.ndarray, scorer: FocusScorer, result: ScoreResult):
    """
    Runs in a daemon thread on SPACE press.
    Downscales, converts to gray, scores, writes result.
    """
    small = cv2.resize(frame, (0, 0), fx=SCORE_SCALE, fy=SCORE_SCALE,
                       interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    scores = scorer.score(gray)
    result.write(scores)

    print(f"[Score] Composite: {scores['composite']:.4f}  |  "
          f"Lap: {scores['raw']['laplacian']:.1f}  "
          f"Ten: {scores['raw']['tenengrad']:.1f}  "
          f"Var: {scores['raw']['variance']:.3f}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  AmScope MD500A — Autofocus Scorer v3 (calibrated)")
    print("=" * 60)
    print(f"  Opening camera {CAMERA_INDEX} via DirectShow...")

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check USB connection and index.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened at {actual_w} x {actual_h}")

    scorer     = FocusScorer()
    result     = ScoreResult()
    disp_scale = min(1.0, 1280 / actual_w)

    # Run calibration wizard immediately on startup
    run_calibration_wizard(cap, scorer, disp_scale)

    print("  SPACE = score  |  C = recalibrate  |  Q = quit\n")

    prev_time = time.time()
    fps       = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Frame grab failed, retrying...")
            continue

        now       = time.time()
        fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        if disp_scale < 1.0:
            dw      = int(actual_w * disp_scale)
            dh      = int(actual_h * disp_scale)
            display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame.copy()

        draw_hud(display, result, fps, scorer.calib.ready)

        cv2.imshow("Autofocus Scorer — MD500A", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting.")
            break

        elif key == ord(' '):
            if not result.pending:
                result.pending = True
                snapshot = frame.copy()
                t = threading.Thread(
                    target=run_score,
                    args=(snapshot, scorer, result),
                    daemon=True
                )
                t.start()
            else:
                print("[Score] Still processing, please wait.")

        elif key == ord('c'):
            result.pending = False
            run_calibration_wizard(cap, scorer, disp_scale)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()