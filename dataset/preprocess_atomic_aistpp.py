"""Build frame-aligned atomic training data from raw AIST++ files."""

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/edge-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/edge-matplotlib-cache")

import librosa
import numpy as np
import torch
from pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from tqdm import tqdm

from data.audio_extraction.baseline_features import FPS, SR, extract_audio
from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton


MOTION_FPS = 30
RAW_MOTION_FPS = 60
SLICE_FRAMES = 150
SLICE_SECONDS = 5
SLICE_STRIDE_SECONDS = 0.5
MOTION_DIM = 151


def load_class_assignments(classification_root):
    """Map classified I3D segments to labels 1..100."""
    root = Path(classification_root)
    assignments = {}
    class_dirs = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    expected = list(range(100))
    observed = [int(path.name) for path in class_dirs]
    if observed != expected:
        raise ValueError("genre_llm must contain exactly the top-level classes 0..99")

    for class_dir in class_dirs:
        label = int(class_dir.name) + 1
        metadata_files = sorted(class_dir.glob("*.json"))
        if not metadata_files:
            raise FileNotFoundError("missing top-level metadata in {}".format(class_dir))
        for metadata_path in metadata_files:
            for row in json.load(open(str(metadata_path), "r")):
                key = (
                    row["category"],
                    row["basename"],
                    int(row["motion_segment"]),
                    int(row["start_frame"]),
                    int(row["end_frame"]),
                )
                if key in assignments:
                    raise ValueError("segment assigned to multiple atomic classes: {}".format(key))
                assignments[key] = label
    return assignments


def labels_for_slice(split, slice_name, segments, assignments):
    labels = np.zeros(SLICE_FRAMES, dtype=np.int64)
    previous_end = 0
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start != previous_end or not start < end <= SLICE_FRAMES:
            raise ValueError("invalid segmentation for {}".format(slice_name))
        key = (split, slice_name, int(segment["motion"]), start, end)
        label = assignments.get(key, 0)
        labels[start:end] = label
        previous_end = end
    if previous_end != SLICE_FRAMES:
        raise ValueError("segmentation does not cover {} frames: {}".format(SLICE_FRAMES, slice_name))
    return labels


def encode_motion(raw_motion, skeleton=None):
    """Convert raw 60 FPS SMPL parameters to EDGE's 30 FPS, 151-D representation."""
    skeleton = skeleton or SMPLSkeleton()
    scale = float(np.asarray(raw_motion["smpl_scaling"]).reshape(-1)[0])
    if scale == 0:
        raise ValueError("SMPL scaling must be non-zero")
    root_pos = torch.as_tensor(raw_motion["smpl_trans"][::2], dtype=torch.float32) / scale
    local_q = torch.as_tensor(raw_motion["smpl_poses"][::2], dtype=torch.float32).reshape(-1, 24, 3)

    root_quaternion = axis_angle_to_quaternion(local_q[:, :1])
    rotation = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0], dtype=torch.float32)
    local_q[:, :1] = quaternion_to_axis_angle(quaternion_multiply(rotation, root_quaternion))
    root_pos = RotateAxisAngle(90, axis="X", degrees=True).transform_points(root_pos)

    positions = skeleton.forward(local_q.unsqueeze(0), root_pos.unsqueeze(0)).squeeze(0)
    feet = positions[:, (7, 8, 10, 11)]
    foot_velocity = torch.zeros(feet.shape[:2], dtype=torch.float32)
    foot_velocity[:-1] = (feet[1:] - feet[:-1]).norm(dim=-1)
    contacts = (foot_velocity < 0.01).float()
    rotations = ax_to_6v(local_q).reshape(local_q.shape[0], -1)
    encoded = torch.cat((contacts, root_pos, rotations), dim=-1)
    if encoded.shape[1] != MOTION_DIM or not torch.isfinite(encoded).all():
        raise ValueError("invalid encoded motion")
    return encoded


def fit_motion_range(sequence_names, raw_root):
    data_min = None
    data_max = None
    skeleton = SMPLSkeleton()
    for name in tqdm(sequence_names, desc="Fitting motion normalization"):
        with open(str(Path(raw_root) / "motions" / (name + ".pkl")), "rb") as handle:
            motion = encode_motion(pickle.load(handle), skeleton)
        current_min = motion.min(dim=0).values
        current_max = motion.max(dim=0).values
        data_min = current_min if data_min is None else torch.minimum(data_min, current_min)
        data_max = current_max if data_max is None else torch.maximum(data_max, current_max)
    if data_min is None:
        raise ValueError("cannot fit normalization without training sequences")
    return data_min, data_max


def normalize_motion(motion, data_min, data_max):
    data_range = data_max - data_min
    safe_range = torch.where(
        data_range < 10 * torch.finfo(data_range.dtype).eps,
        torch.ones_like(data_range),
        data_range,
    )
    return ((motion - data_min) * (2.0 / safe_range) - 1.0).clamp(-1.0, 1.0)


def group_segmentation_files(segmentation_root, limit=None):
    grouped = {}
    pattern = re.compile(r"^(.*)_slice(\d+)$")
    for split in ("train", "test"):
        by_sequence = defaultdict(list)
        files = sorted((Path(segmentation_root) / split).glob("*.json"))
        if limit is not None:
            files = files[:limit]
        for path in files:
            match = pattern.match(path.stem)
            if not match:
                raise ValueError("invalid slice filename: {}".format(path.name))
            by_sequence[match.group(1)].append((int(match.group(2)), path))
        grouped[split] = by_sequence
    return grouped


def preprocess(args):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers > 1:
        torch.set_num_threads(1)
    assignments = load_class_assignments(args.classification_root)
    grouped = group_segmentation_files(args.segmentation_root, args.limit)
    training_names = sorted(grouped["train"])
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    normalizer_path = output_root / "normalizer.pt"
    normalizer = None
    if normalizer_path.is_file():
        candidate = torch.load(str(normalizer_path), map_location="cpu")
        if candidate.get("training_names") == training_names:
            normalizer = candidate
    if normalizer is None:
        data_min, data_max = fit_motion_range(training_names, args.raw_root)
        normalizer = {
            "data_min": data_min,
            "data_max": data_max,
            "training_names": training_names,
        }
        torch.save(normalizer, str(normalizer_path))
    data_min = normalizer["data_min"]
    data_max = normalizer["data_max"]
    manifest = {"num_classes": 100, "motion_dim": MOTION_DIM, "music_dim": 35, "splits": {}}

    music_representatives = {}
    music_sizes = defaultdict(set)
    for sequences in grouped.values():
        for sequence_name in sequences:
            fields = sequence_name.split("_")
            music_key = (fields[1], fields[4])
            audio_path = Path(args.raw_root) / "wavs" / (sequence_name + ".wav")
            if not audio_path.is_file():
                raise FileNotFoundError("missing raw audio for {}".format(sequence_name))
            music_representatives.setdefault(music_key, audio_path)
            music_sizes[music_key].add(audio_path.stat().st_size)
    inconsistent = [key for key, sizes in music_sizes.items() if len(sizes) != 1]
    if inconsistent:
        raise ValueError("audio files differ within music cache keys: {}".format(inconsistent[:5]))

    def load_music(item):
        music_key, audio_path = item
        audio, _ = librosa.load(str(audio_path), sr=SR)
        return music_key, extract_audio(audio, audio_path.stem, max_frames=None).astype(np.float32)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        music_cache = dict(
            tqdm(
                executor.map(load_music, sorted(music_representatives.items())),
                total=len(music_representatives),
                desc="Extracting music",
            )
        )
    manifest["music_cache_entries"] = len(music_cache)

    for split, sequences in grouped.items():
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        items = []
        sample_count = 0
        for sequence_name, slices in sorted(sequences.items()):
            items.append((sample_count, sequence_name, sorted(slices)))
            sample_count += len(slices)
        motion_output = np.lib.format.open_memmap(
            str(split_root / "motion.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, SLICE_FRAMES, MOTION_DIM),
        )
        music_output = np.lib.format.open_memmap(
            str(split_root / "music.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, SLICE_FRAMES, 35),
        )
        label_output = np.lib.format.open_memmap(
            str(split_root / "labels.npy"),
            mode="w+",
            dtype=np.uint8,
            shape=(sample_count, SLICE_FRAMES),
        )
        sample_names = [None] * sample_count
        classified_frames = 0
        total_frames = 0

        def write_sequence(item):
            offset, sequence_name, slices = item
            motion_path = Path(args.raw_root) / "motions" / (sequence_name + ".pkl")
            if not motion_path.is_file():
                raise FileNotFoundError("missing raw motion for {}".format(sequence_name))
            with open(str(motion_path), "rb") as handle:
                full_motion = normalize_motion(
                    encode_motion(pickle.load(handle), SMPLSkeleton()), data_min, data_max
                )
            fields = sequence_name.split("_")
            full_music = music_cache[(fields[1], fields[4])]
            motions = []
            music_features = []
            plans = []
            names = []
            for slice_index, segmentation_path in sorted(slices):
                name = segmentation_path.stem
                motion_start = int(slice_index * SLICE_STRIDE_SECONDS * MOTION_FPS)
                motion = full_motion[motion_start : motion_start + SLICE_FRAMES]
                music_start = int(slice_index * SLICE_STRIDE_SECONDS * FPS)
                music = full_music[music_start : music_start + SLICE_FRAMES]
                if motion.shape[0] != SLICE_FRAMES or music.shape[0] != SLICE_FRAMES:
                    raise ValueError("raw sequence is too short for {}".format(name))
                segments = json.load(open(str(segmentation_path), "r"))
                labels = labels_for_slice(split, name, segments, assignments)
                motions.append(motion.numpy())
                music_features.append(music)
                plans.append(labels.astype(np.uint8))
                names.append(name)
            return (
                offset,
                np.stack(motions),
                np.stack(music_features),
                np.stack(plans),
                names,
            )

        def store_result(result):
            nonlocal classified_frames, total_frames
            offset, motions, music_features, plans, names = result
            end = offset + len(names)
            motion_output[offset:end] = motions
            music_output[offset:end] = music_features
            label_output[offset:end] = plans
            sample_names[offset:end] = names
            classified_frames += int((plans != 0).sum())
            total_frames += plans.size

        if args.workers == 1:
            results = map(write_sequence, items)
            for result in tqdm(results, total=len(items), desc="Writing {}".format(split)):
                store_result(result)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(write_sequence, items)
                for result in tqdm(results, total=len(items), desc="Writing {}".format(split)):
                    store_result(result)
        motion_output.flush()
        music_output.flush()
        label_output.flush()
        with open(str(split_root / "names.json"), "w") as handle:
            json.dump(sample_names, handle)
        manifest["splits"][split] = {
            "samples": sample_count,
            "classified_frame_fraction": classified_frames / total_frames if total_frames else 0.0,
        }
    with open(str(output_root / "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/edge_aistpp")
    parser.add_argument("--segmentation-root", default="genre/aist/i3d_18_segmentation")
    parser.add_argument("--classification-root", default="genre/aist/LLM_split_kmeans/genre_llm")
    parser.add_argument("--output-root", default="data/atomic_aistpp")
    parser.add_argument("--limit", type=int, default=None, help="maximum slices per split for a smoke test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(preprocess(parse_args()), indent=2))
