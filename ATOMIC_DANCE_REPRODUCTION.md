# Atomic Movement Dance Reproduction

This branch extends the original EDGE implementation with the two-stage method
described in *Music-to-Dance Generation via Atomic Movements*.

## Paper-to-code mapping

| Paper component | Implementation |
| --- | --- |
| Frame labels `0..K` (`0` is transition) | `dataset/atomic.py` |
| Uniform categorical forward diffusion, Eqs. (1)-(3) | `model/atomic_planner.py::UniformD3PM` |
| Full-music-aware Transformer planner | `model/atomic_planner.py::AtomicPlannerTransformer` |
| Majority vote and minimum-duration merging | `dataset/atomic.py::refine_plan` |
| Duration-nearest prototype retrieval | `dataset/atomic.py::AtomicMotionLibrary` |
| Coarse draft `M0` and mask `w` | `dataset/atomic.py::AtomicMotionLibrary.build_draft` |
| Algorithm 1 and TMR-embedding base clustering | `dataset/atomic_discovery.py` |
| Full-sequence frame-aligned data loader | `dataset/atomic_dataset.py` |
| Music/draft/mask-conditioned completion model | `model/atomic_completion.py::AtomicCompletionDecoder` |
| DDPM reconstruction and transition losses, Eq. (4) | `model/atomic_completion.py::AtomicCompletionDiffusion` |

## Motion and label contract

- Motion uses the original EDGE representation at 30 FPS: four foot contacts,
  root translation, and 24 joint rotations in 6D (`151` channels total).
- Music features must be frame-aligned with motion. Existing EDGE Jukebox
  features use `4800` channels and baseline features use `35` channels.
- Atomic labels are integer tensors with one value per motion frame. Label `0`
  is reserved for transitions; labels `1..K` identify atomic categories.
- A retrieved draft has the same normalized representation as the target
  motion. Transition ranges are zero-filled and have a zero mask.
- On atomic ranges, the mask value is the standard deviation of the noise added
  to the retrieved draft. It is therefore both a support mask and the paper's
  noise-ratio condition `w`.

## Defaults that are not specified in the paper

The manuscript does not report the D3PM step count, planner depth, smoothing
window, minimum segment duration, draft noise ratio, or transition-loss weight.
They are configurable in code; initial reproduction defaults are:

| Parameter | Default |
| --- | ---: |
| D3PM steps | 100 |
| Planner latent/layers/heads | 512 / 8 / 8 |
| Majority-vote window | 5 frames |
| Minimum planned segment | 6 frames (0.2 s at 30 FPS) |
| Completion DDPM steps | 1000 |
| Draft atomic noise ratio | 0.25 |
| Transition loss weight | 1.0 |

These values must be ablated or replaced with author settings before claiming
numerical reproduction of the paper.

## Verification status

- Static Python compilation and whitespace checks pass in the EDGE environment.
- Unit tests cover plan segmentation, smoothing, duration-nearest retrieval,
  draft masks, indexed dataset loading, D3PM tensor shapes, and completion
  losses/sampling.

Run them in the EDGE training environment with:

```bash
python -m unittest tests/test_atomic.py -v
```

## AIST++ preprocessing

Top-level `genre_llm` categories `0..99` map to planner labels `1..100`.
Unclassified I3D segments use the transition label `0`. Generate the indexed,
frame-aligned baseline-feature dataset with:

```bash
python -m dataset.preprocess_atomic_aistpp --workers 16
```

The output under `data/atomic_aistpp/{train,test}` uses memory-mapped NumPy
arrays for motion, music, and labels, plus `names.json`. The loader retains
compatibility with the original directory-per-sample layout.

## Training

Both stages support bounded debug runs, validation sampling, checkpointing,
and resume through `train_atomic.py`. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python train_atomic.py \
  --stage planner --device cuda --batch-size 16 --max-steps 10

CUDA_VISIBLE_DEVICES=0 python train_atomic.py \
  --stage completion --device cuda --batch-size 8 --max-steps 10
```

Remove `--max-steps` and set `--epochs` for a full run. Checkpoints and
validation metrics are written under `runs/atomic_debug` by default. Training
uses one dynamic progress bar, reports mean loss every 5 epochs, and saves a
resumable checkpoint every 20 epochs. Override these intervals with
`--log-every-epochs` and `--save-every-epochs`.

## Evaluation

Run inference and evaluation directly from AIST++ ground-truth PKLs and WAV
files. If `--prediction-motions` is omitted, the atomic planner and completion
checkpoints generate motions under `eval/generated_motions`; features are then
cached under `eval/cache`:

Planner + completion:

```bash
CUDA_VISIBLE_DEVICES=1 python -m eval.evaluate \
  --ground-truth-motions data/edge_aistpp/motions \
  --audio-dir data/edge_aistpp/wavs \
  --plan-source planner \
  --planner-checkpoint runs/atomic_planner/planner_epoch20_step22180.pt \
  --completion-checkpoint runs/atomic_completion/completion_epoch20_step88680.pt \
  --sequence-list data/splits/test.txt \
  --smpl-model smpl/SMPL_MALE.pkl \
  --device cuda:0 \
  --max-inference-frames 150 \
  --inference-batch-size 4 \
  --workers 4 \
  --inference-output eval/generated_planner \
  --cache-dir eval/cache_planner \
  --output eval/results_planner.json
```

GT plan + completion (planner is not loaded or called):

```bash
CUDA_VISIBLE_DEVICES=1 python -m eval.evaluate \
  --ground-truth-motions data/edge_aistpp/motions \
  --audio-dir data/edge_aistpp/wavs \
  --plan-source ground-truth \
  --completion-checkpoint runs/atomic_completion/completion_epoch20_step88680.pt \
  --sequence-list data/splits/test.txt \
  --smpl-model smpl/SMPL_MALE.pkl \
  --max-inference-frames 150 \
  --device cuda:0 \
  --inference-batch-size 4 \
  --workers 4 \
  --inference-output eval/generated_gt_plan \
  --cache-dir eval/cache_gt_plan \
  --output eval/results_gt_plan.json
```

`--ground-truth-labels` remains as a compatibility alias for
`--plan-source ground-truth`.

Planner inference samples each categorical D3PM reverse step by default, as in
the paper. `--deterministic-planner` switches to per-step argmax only for
debugging; repeated argmax can collapse the plan after post-processing.

## Data split

`dataset/preprocess_atomic_aistpp.py` preserves the directory split under
`genre/aist/i3d_18_segmentation/{train,test}`. The resulting indexed dataset is:

| split | 150-frame slices | source sequences |
|---|---:|---:|
| train | 17,733 | 952 |
| test | 372 | 40 |

There is no sequence or slice overlap between these splits. Each sample is 150
frames (5 seconds at 30 FPS), with source slices spaced by 0.5 seconds. Training
uses `data/atomic_aistpp/train`; in-training validation uses all 372 samples in
`data/atomic_aistpp/test`. Final cross-modal evaluation uses the 20-sequence
subset in `data/splits/crossmodal_test.txt`; all 20 are contained in the atomic
test split.

The unified output contains kinetic/manual FID and diversity, predicted/GT
BAS, and predicted PFC. Generated PKLs must contain EDGE's `full_pose` field;
raw AIST++ PKLs may contain `smpl_poses`/`smpl_trans` or `q`/`pos`. Use
`--force-extract` after changing source files in place. Existing feature roots
can still be supplied with `--prediction-features` and
`--ground-truth-features`. Add `--ground-truth-pfc` only when GT PKLs also
contain `full_pose`. Use `--max-inference-samples` and `--max-inference-frames`
for a bounded smoke test, and `--overwrite-inference` to regenerate existing
motion PKLs.

Following the supplied starter evaluation, prediction and ground-truth features
are standardized independently with their own per-dimension mean and standard
deviation before FID and diversity are computed.

## Remaining work

1. Reproduce the paper's R-precision definition.
