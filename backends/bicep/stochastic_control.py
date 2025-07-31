import os
import numpy as _np
import logging
from functools import lru_cache

# ——— Logging setup ———
LOG_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,  # backends/
    os.pardir,  # project root
    "results", "logs", "stochastic_control.log"
)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ——— Tunable thresholds ———
HIGH_THRESHOLD = 10
LOW_THRESHOLD = 2
DECAY_RATE = 0.1
FEEDBACK_WARNING_THRESHOLD = 0.9
ADAPTIVE_ADJUSTMENT = 0.05
MAX_HIGH_THRESHOLD = 100

def _log(msg):
    logging.info(msg)

def adjust_variance(base_variance, adjustment_factor=1.0):
    adj = base_variance * adjustment_factor
    _log(f"Adjusted variance {base_variance}→{adj} (factor={adjustment_factor})")
    return adj

def adaptive_randomness_control(brownian_increment, feedback_value, target_range=(0.2, 1.0)):
    if feedback_value >= FEEDBACK_WARNING_THRESHOLD:
        logging.warning(f"High feedback: {feedback_value}")
    scale = _np.clip(0.5 + feedback_value * 0.5, *target_range)
    adj = brownian_increment * scale
    _log(f"Feedback adjustment: {brownian_increment}→{adj} (scale={scale})")
    return adj

def control_randomness_by_state(state_visit_count, total_steps,
                                high_threshold=HIGH_THRESHOLD,
                                low_threshold=LOW_THRESHOLD):
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    norm = state_visit_count / total_steps
    global HIGH_THRESHOLD
    if HIGH_THRESHOLD < MAX_HIGH_THRESHOLD and norm > 0.8 * high_threshold:
        HIGH_THRESHOLD = min(MAX_HIGH_THRESHOLD, HIGH_THRESHOLD + ADAPTIVE_ADJUSTMENT)
        _log(f"Bumped HIGH_THRESHOLD → {HIGH_THRESHOLD}")
    if norm < low_threshold:
        factor = 1.5
    elif norm > high_threshold:
        factor = 0.5
    else:
        factor = 1.0
    _log(f"State control: count={state_visit_count}/{total_steps} → factor={factor}")
    return factor

def combined_variance_control(state_visit_count, total_steps, t,
                              base_variance=1.0,
                              decay_rate=DECAY_RATE):
    sf = control_randomness_by_state(state_visit_count, total_steps)
    tf = _np.exp(-decay_rate * t)
    comb = base_variance * sf * tf
    _log(f"Combined variance @ t={t}: {comb}")
    return comb

def apply_stochastic_controls(brownian_increment,
                              state_visit_count,
                              feedback_value,
                              t,
                              total_steps,
                              base_variance=1.0):
    vf = combined_variance_control(state_visit_count, total_steps, t, base_variance)
    ff = adaptive_randomness_control(brownian_increment, feedback_value)
    final = ff * vf
    _log(f"Final increment: {final}")
    return final
