"""Physical Foot Contact (PFC) metric from EDGE joint-position PKLs."""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


def physical_score(joint3d, fps=30.0):
    joint3d = np.asarray(joint3d, dtype=np.float64)
    if joint3d.ndim != 3 or joint3d.shape[1:] != (24, 3):
        raise ValueError("full_pose must have shape [frames, 24, 3]")
    if joint3d.shape[0] < 3:
        raise ValueError("PFC requires at least three frames")
    dt = 1.0 / fps
    root_velocity = np.diff(joint3d[:, 0], axis=0) / dt
    root_acceleration = np.diff(root_velocity, axis=0) / dt
    root_acceleration[:, 2] = np.maximum(root_acceleration[:, 2], 0.0)
    root_acceleration = np.linalg.norm(root_acceleration, axis=-1)
    scale = root_acceleration.max(initial=0.0)
    if scale > 0:
        root_acceleration /= scale

    feet = joint3d[:, [7, 10, 8, 11]]
    horizontal_velocity = np.linalg.norm(
        feet[2:, :, :2] - feet[1:-1, :, :2], axis=-1
    )
    left = np.minimum(horizontal_velocity[:, 0], horizontal_velocity[:, 1])
    right = np.minimum(horizontal_velocity[:, 2], horizontal_velocity[:, 3])
    return float((left * right * root_acceleration).mean() * 10000.0)


def calculate_pfc(
    motion_dir,
    max_samples=1000,
    seed=0,
    return_per_sequence=False,
    include_names=None,
):
    paths = sorted(Path(motion_dir).glob("*.pkl"))
    if include_names is not None:
        requested = set(include_names)
        paths = [path for path in paths if path.stem in requested]
    if not paths:
        raise FileNotFoundError("no motion PKLs found under {}".format(motion_dir))
    if max_samples is not None and len(paths) > max_samples:
        paths = random.Random(seed).sample(paths, max_samples)
        paths.sort()
    scores = {}
    for path in tqdm(paths, desc="PFC", unit="motion"):
        with open(str(path), "rb") as handle:
            data = pickle.load(handle)
        if "full_pose" not in data:
            raise KeyError("{} does not contain full_pose".format(path))
        scores[path.stem] = physical_score(data["full_pose"])
    mean = float(np.mean(list(scores.values())))
    return (mean, scores) if return_per_sequence else mean


def calc_physical_score(dir):
    """Backward-compatible wrapper used by the original EDGE CLI."""
    score = calculate_pfc(dir)
    print("{} has a mean PFC of {}".format(dir, score))
    return score


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_path", default="motions/")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_eval_opt()
    print(calculate_pfc(options.motion_path, options.max_samples, options.seed))
