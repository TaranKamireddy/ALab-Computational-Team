"""
autofocus_controller.py
=======================
Motor-assisted autofocus controller for the AmScope MD500A workflow.

This script intentionally keeps camera_test_sid.py unchanged and reuses its
focus-scoring logic:
  - FocusScorer / Calibration
  - compute_raw_metrics
  - scoring constants

Features in v1:
  - Keyboard jog control (Up/Down arrows) for a STEP/DIR stepper.
  - Calibration wizard with motor jog during both phases.
  - Hill-climb autofocus on key [A].
  - Stop autofocus once composite score >= 0.650.
  - Save successful focused frame to Mxene_sample_images/ with timestamp.
  - Future hooks for slide detection and optional Arduino button trigger.

Dependencies:
  pip install opencv-python numpy pyfirmata

Arduino requirement:
  Upload StandardFirmata to the board before running this script.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from camera_test_sid import (
    BLUR_KERNEL,
    CALIB_SNAPSHOTS,
    CAMERA_BACKEND,
    CAMERA_INDEX,
    COMPOSITE_WEIGHTS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SCORE_SCALE,
    FocusScorer,
    compute_raw_metrics,
)

try:
    import pyfirmata  # type: ignore
except ImportError:
    pyfirmata = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Camera/display
WINDOW_NAME = "Autofocus Controller - MD500A"

# Focus/autofocus behavior
FOCUS_THRESHOLD = 0.650
MAX_CYCLES = 120
PLATEAU_WINDOW = 12
IMPROVEMENT_EPSILON = 1e-4

# Scoring timing
SETTLE_AFTER_MOVE_SEC = 0.04

# Output
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Mxene_sample_images"

# Motor (Firmata, STEP/DIR)
MOTOR_ENABLED = True
COM_PORT = "COM3"  # Update to your Arduino COM port.
STEP_PIN = 2
DIR_PIN = 3
ENABLE_PIN: Optional[int] = None  # Example: 4, or None to disable enable control.
ENABLE_ACTIVE_LOW = True
DIR_UP_STATE = 1  # Digital value on DIR pin for "up".
MIN_POS = -200
MAX_POS = 200
JOG_STEP = 1
AUTO_STEP = 1
STEP_PULSE_SEC = 0.002
STEP_SETTLE_SEC = 0.0

# Optional autofocus trigger button (hook only, disabled by default)
ENABLE_BUTTON_TRIGGER = False
BUTTON_PIN: Optional[int] = None
BUTTON_ACTIVE_HIGH = True
BUTTON_DEBOUNCE_SEC = 0.25


# Key codes (Windows/OpenCV waitKeyEx)
KEY_UP = {2490368, 82}
KEY_DOWN = {2621440, 84}
KEY_SPACE = 32
KEY_A = ord("a")
KEY_C = ord("c")
KEY_Q = ord("q")


# ==============================================================================
# MOTOR CONTROL
# ==============================================================================

class StepperController:
    """Simple STEP/DIR stepper wrapper over Arduino Firmata."""

    def __init__(self):
        self.enabled = MOTOR_ENABLED
        self.position_steps = 0
        self.board = None
        self.button_pin_ref = None
        self._last_button_time = 0.0

        if not self.enabled:
            print("[Motor] Disabled by config. Running in mock mode.")
            return

        if pyfirmata is None:
            print("[Motor] pyfirmata not installed. Running in mock mode.")
            self.enabled = False
            return

        try:
            self.board = pyfirmata.Arduino(COM_PORT)
            self.board.digital[STEP_PIN].mode = pyfirmata.OUTPUT
            self.board.digital[DIR_PIN].mode = pyfirmata.OUTPUT
            self.board.digital[STEP_PIN].write(0)
            self.board.digital[DIR_PIN].write(DIR_UP_STATE)

            if ENABLE_PIN is not None:
                self.board.digital[ENABLE_PIN].mode = pyfirmata.OUTPUT
                self._set_enable(True)

            if ENABLE_BUTTON_TRIGGER and BUTTON_PIN is not None:
                self.board.digital[BUTTON_PIN].mode = pyfirmata.INPUT
                it = pyfirmata.util.Iterator(self.board)
                it.start()
                self.button_pin_ref = self.board.digital[BUTTON_PIN]

            print(f"[Motor] Connected to {COM_PORT} (STEP={STEP_PIN}, DIR={DIR_PIN}).")
        except Exception as exc:
            print(f"[Motor] Failed to initialize Firmata: {exc}")
            print("[Motor] Falling back to mock mode.")
            self.enabled = False
            self.board = None

    def _set_enable(self, active: bool) -> None:
        if not self.enabled or self.board is None or ENABLE_PIN is None:
            return
        if ENABLE_ACTIVE_LOW:
            self.board.digital[ENABLE_PIN].write(0 if active else 1)
        else:
            self.board.digital[ENABLE_PIN].write(1 if active else 0)

    def move_relative(self, steps: int) -> bool:
        """
        Move relative step count. Positive steps = up, negative = down.
        Returns True on success, False if blocked by bounds or motor disabled error.
        """
        if steps == 0:
            return True

        target = self.position_steps + steps
        if target < MIN_POS or target > MAX_POS:
            print(
                f"[Motor] Move blocked by bounds. pos={self.position_steps}, "
                f"requested={steps}, limits=[{MIN_POS}, {MAX_POS}]"
            )
            return False

        if not self.enabled or self.board is None:
            self.position_steps = target
            print(f"[Motor-Mock] position -> {self.position_steps}")
            return True

        try:
            direction_up = steps > 0
            self.board.digital[DIR_PIN].write(DIR_UP_STATE if direction_up else 1 - DIR_UP_STATE)

            for _ in range(abs(steps)):
                self.board.digital[STEP_PIN].write(1)
                time.sleep(STEP_PULSE_SEC)
                self.board.digital[STEP_PIN].write(0)
                time.sleep(STEP_PULSE_SEC)

            if STEP_SETTLE_SEC > 0:
                time.sleep(STEP_SETTLE_SEC)

            self.position_steps = target
            return True
        except Exception as exc:
            print(f"[Motor] Move failed: {exc}")
            return False

    def read_button_pressed(self) -> bool:
        """Future hook for optional hardware autofocus trigger button."""
        if not ENABLE_BUTTON_TRIGGER or self.button_pin_ref is None:
            return False

        now = time.time()
        if now - self._last_button_time < BUTTON_DEBOUNCE_SEC:
            return False

        value = self.button_pin_ref.read()
        if value is None:
            return False

        pressed = bool(value) if BUTTON_ACTIVE_HIGH else not bool(value)
        if pressed:
            self._last_button_time = now
        return pressed

    def close(self) -> None:
        if self.board is not None:
            try:
                if ENABLE_PIN is not None:
                    self._set_enable(False)
                self.board.exit()
            except Exception:
                pass
        print("[Motor] Closed.")


# ==============================================================================
# HELPERS
# ==============================================================================

def is_new_slide(frame_prev: np.ndarray, frame_curr: np.ndarray) -> bool:
    """
    Future hook for automatic slide-change detection.
    TODO: Replace stub with frame-difference/content-based heuristic.
    """
    _ = frame_prev
    _ = frame_curr
    return False


def compute_score_from_frame(frame: np.ndarray, scorer: FocusScorer) -> dict:
    small = cv2.resize(frame, (0, 0), fx=SCORE_SCALE, fy=SCORE_SCALE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return scorer.score(gray)


def save_focused_image(frame: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = OUTPUT_DIR / f"sample_{stamp}.png"
    cv2.imwrite(str(path), frame)
    return path


def draw_status_hud(display: np.ndarray, score_value: Optional[float], calibrated: bool, motor_pos: int, status: str) -> None:
    h, _ = display.shape[:2]
    panel_h = 155
    panel_w = 520

    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    calib_text = "CALIBRATED" if calibrated else "NOT CALIBRATED"
    calib_color = (0, 210, 60) if calibrated else (0, 140, 240)
    cv2.putText(display, calib_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 1)
    cv2.putText(display, f"Motor pos: {motor_pos} steps", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if score_value is None:
        cv2.putText(display, "Focus: ---.---", (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (170, 170, 170), 2)
    else:
        if score_value < 0.4:
            color = (0, 0, 220)
        elif score_value < 0.7:
            color = (0, 200, 220)
        else:
            color = (0, 210, 60)
        cv2.putText(display, f"Focus: {score_value:.3f}", (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        bx, by, bw, bh = 20, 95, panel_w - 40, 15
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (55, 55, 55), -1)
        cv2.rectangle(display, (bx, by), (bx + int(max(0.0, min(1.0, score_value)) * bw), by + bh), color, -1)

    cv2.putText(display, status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(
        display,
        "[UP/DOWN] Jog  [A] Autofocus  [SPACE] Score/Capture  [C] Calibrate  [Q] Quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
    )


# ==============================================================================
# CALIBRATION UI
# ==============================================================================

class CalibrationUI:
    """Calibration wizard with keyboard motor jog in both phases."""

    def __init__(self, cap: cv2.VideoCapture, scorer: FocusScorer, motor: StepperController, disp_scale: float):
        self.cap = cap
        self.scorer = scorer
        self.motor = motor
        self.disp_scale = disp_scale

    def _draw_phase(self, frame: np.ndarray, phase_name: str, instruction: str, captured: int) -> np.ndarray:
        if self.disp_scale < 1.0:
            dw = int(frame.shape[1] * self.disp_scale)
            dh = int(frame.shape[0] * self.disp_scale)
            display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame.copy()

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)

        cv2.putText(display, f"CALIBRATION - {phase_name}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
        cv2.putText(display, instruction, (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(
            display,
            f"UP/DOWN jog, SPACE capture ({captured}/{CALIB_SNAPSHOTS}), Q abort",
            (15, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 100),
            1,
        )
        cv2.putText(
            display,
            f"Motor pos: {self.motor.position_steps}  |  Blur kernel: {BLUR_KERNEL}  |  Weights: {COMPOSITE_WEIGHTS}",
            (15, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )
        return display

    def _collect_phase(self, phase_name: str, instruction: str) -> list[dict]:
        collected: list[dict] = []
        while len(collected) < CALIB_SNAPSHOTS:
            ret, frame = self.cap.read()
            if not ret:
                continue

            display = self._draw_phase(frame, phase_name, instruction, len(collected))
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKeyEx(1)
            key_ascii = key & 0xFF

            if key_ascii == KEY_Q:
                return []
            if key in KEY_UP:
                self.motor.move_relative(JOG_STEP)
                continue
            if key in KEY_DOWN:
                self.motor.move_relative(-JOG_STEP)
                continue

            if key_ascii == KEY_SPACE:
                small = cv2.resize(frame, (0, 0), fx=SCORE_SCALE, fy=SCORE_SCALE, interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                raw = compute_raw_metrics(gray)
                collected.append(raw)
                print(
                    f"[Calib:{phase_name}] Snap {len(collected)}/{CALIB_SNAPSHOTS} - "
                    f"lap={raw['laplacian']:.1f} ten={raw['tenengrad']:.1f} var={raw['variance']:.3f}"
                )

        return collected

    def run(self) -> bool:
        print("\n" + "=" * 60)
        print("  CALIBRATION WIZARD")
        print("  Use UP/DOWN arrows to move motor. SPACE captures each snapshot.")
        print("=" * 60)

        unfocused = self._collect_phase(
            "UNFOCUSED",
            "Defocus sample (blurred), then capture anchor snapshots.",
        )
        if not unfocused:
            print("[Calib] Aborted.")
            return False

        focused = self._collect_phase(
            "FOCUSED",
            "Bring sample into sharp focus, then capture anchor snapshots.",
        )
        if not focused:
            print("[Calib] Aborted.")
            return False

        self.scorer.calib.set_anchors(unfocused, focused)
        print("[Calib] Complete.")
        return True


# ==============================================================================
# AUTOFOCUS CONTROLLER
# ==============================================================================

class AutofocusController:
    def __init__(self, cap: cv2.VideoCapture, scorer: FocusScorer, motor: StepperController):
        self.cap = cap
        self.scorer = scorer
        self.motor = motor
        self.last_score: Optional[float] = None
        self.status = "Ready"

    def _grab_scored_frame(self) -> tuple[Optional[np.ndarray], Optional[dict]]:
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        score = compute_score_from_frame(frame, self.scorer)
        self.last_score = score["composite"]
        return frame, score

    def _measure_after_move(self, steps: int) -> tuple[bool, Optional[np.ndarray], Optional[dict]]:
        ok = self.motor.move_relative(steps)
        if not ok:
            return False, None, None
        time.sleep(SETTLE_AFTER_MOVE_SEC)
        frame, score = self._grab_scored_frame()
        return frame is not None and score is not None, frame, score

    def run_hill_climb(self) -> bool:
        if not self.scorer.calib.ready:
            self.status = "Autofocus blocked: run calibration first."
            print("[AF] Calibration required.")
            return False

        start_frame, start_score = self._grab_scored_frame()
        if start_frame is None or start_score is None:
            self.status = "Autofocus failed: no frame."
            return False

        best_score = start_score["composite"]
        best_frame = start_frame
        print(f"[AF] Start score: {best_score:.4f}")

        if best_score >= FOCUS_THRESHOLD:
            path = save_focused_image(best_frame)
            self.status = f"Already focused ({best_score:.3f}). Saved: {path.name}"
            print(f"[AF] Success without movement. Saved: {path}")
            return True

        # Probe +AUTO_STEP and return.
        ok_up, frame_up, score_up = self._measure_after_move(AUTO_STEP)
        if ok_up:
            self.motor.move_relative(-AUTO_STEP)
            time.sleep(SETTLE_AFTER_MOVE_SEC)

        # Probe -AUTO_STEP and return.
        ok_down, frame_down, score_down = self._measure_after_move(-AUTO_STEP)
        if ok_down:
            self.motor.move_relative(AUTO_STEP)
            time.sleep(SETTLE_AFTER_MOVE_SEC)

        up_val = score_up["composite"] if score_up else -1.0
        down_val = score_down["composite"] if score_down else -1.0

        if up_val < 0 and down_val < 0:
            self.status = "Autofocus failed: cannot probe either direction."
            print("[AF] Probe failed in both directions.")
            return False

        direction = AUTO_STEP if up_val >= down_val else -AUTO_STEP
        print(f"[AF] Probe up={up_val:.4f}, down={down_val:.4f}, direction={'UP' if direction > 0 else 'DOWN'}")

        plateau_count = 0
        for cycle in range(1, MAX_CYCLES + 1):
            ok, frame, score = self._measure_after_move(direction)
            if not ok or frame is None or score is None:
                self.status = "Autofocus stopped by motor bounds or frame error."
                print(f"[AF] Stop at cycle {cycle}: move blocked or frame unavailable.")
                return False

            composite = score["composite"]
            print(f"[AF] Cycle {cycle:03d}: score={composite:.4f}, pos={self.motor.position_steps}")

            if composite >= FOCUS_THRESHOLD:
                path = save_focused_image(frame)
                self.last_score = composite
                self.status = f"Focused ({composite:.3f}). Saved: {path.name}"
                print(f"[AF] Success at cycle {cycle}. Saved: {path}")
                return True

            if composite > best_score + IMPROVEMENT_EPSILON:
                best_score = composite
                best_frame = frame
                plateau_count = 0
            else:
                plateau_count += 1

            if plateau_count >= PLATEAU_WINDOW:
                self.last_score = best_score
                self.status = f"Autofocus plateau at {best_score:.3f} (threshold {FOCUS_THRESHOLD:.3f})."
                print(f"[AF] Plateau reached after {cycle} cycles.")
                return False

        self.last_score = best_score
        self.status = f"Autofocus max cycles reached ({MAX_CYCLES}), best {best_score:.3f}."
        print(f"[AF] Max cycles reached. Best score={best_score:.4f}")
        if best_score >= FOCUS_THRESHOLD and best_frame is not None:
            path = save_focused_image(best_frame)
            self.status = f"Focused near cycle limit ({best_score:.3f}). Saved: {path.name}"
            print(f"[AF] Saved near-limit frame: {path}")
            return True
        return False


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    print("=" * 60)
    print("  MD500A Autofocus Controller (Motor + Focus Scoring)")
    print("=" * 60)
    print(f"  Opening camera {CAMERA_INDEX}...")

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera opened at {actual_w} x {actual_h}")

    scorer = FocusScorer()
    motor = StepperController()
    controller = AutofocusController(cap, scorer, motor)
    disp_scale = min(1.0, 1280 / max(actual_w, 1))

    calib_ui = CalibrationUI(cap, scorer, motor, disp_scale)
    calib_ui.run()

    print("Controls: UP/DOWN jog | A autofocus | SPACE score | C recalibrate | Q quit")
    prev_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if disp_scale < 1.0:
                dw = int(actual_w * disp_scale)
                dh = int(actual_h * disp_scale)
                display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
            else:
                display = frame.copy()

            draw_status_hud(display, controller.last_score, scorer.calib.ready, motor.position_steps, controller.status)
            cv2.imshow(WINDOW_NAME, display)

            # Future hook; currently disabled by stub.
            if prev_frame is not None and is_new_slide(prev_frame, frame):
                controller.status = "New slide detected (hook)."

            if motor.read_button_pressed():
                controller.status = "Hardware button pressed: running autofocus."
                controller.run_hill_climb()

            key = cv2.waitKeyEx(1)
            key_ascii = key & 0xFF

            if key_ascii == KEY_Q:
                print("Quitting.")
                break

            if key in KEY_UP:
                moved = motor.move_relative(JOG_STEP)
                controller.status = "Jog up." if moved else "Jog up blocked by bounds."
            elif key in KEY_DOWN:
                moved = motor.move_relative(-JOG_STEP)
                controller.status = "Jog down." if moved else "Jog down blocked by bounds."
            elif key_ascii == KEY_SPACE:
                scores = compute_score_from_frame(frame, scorer)
                controller.last_score = scores["composite"]
                controller.status = f"Manual score: {scores['composite']:.3f}"
                print(
                    f"[Score] Composite={scores['composite']:.4f} | "
                    f"Lap={scores['raw']['laplacian']:.1f} Ten={scores['raw']['tenengrad']:.1f} "
                    f"Var={scores['raw']['variance']:.3f}"
                )
            elif key_ascii == KEY_A:
                controller.status = "Running autofocus..."
                controller.run_hill_climb()
            elif key_ascii == KEY_C:
                controller.status = "Recalibrating..."
                calib_ui.run()

            prev_frame = frame
    finally:
        cap.release()
        cv2.destroyAllWindows()
        motor.close()


if __name__ == "__main__":
    main()
