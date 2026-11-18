"""Measure what each input image size costs: decode, transform, batch memory, and forward pass.

Input:  data/raw/images/ (from fetch_data.py) + data/processed/split.csv (from make_split.py)
Output: a table on stdout + logs/runs/image_size_benchmark.csv, one row per image size and device

Samples --n-images train images (seeded) and, for each image size:
    - decode: JPEG bytes -> uint8 tensor (torchvision decode_image, as the models' datasets do).
      Files are read into memory once up front, so this is pure decode time, not disk reads.
    - transform: build_transforms('val', size, fit='squash') on the decoded tensor.
    - batch memory: bytes of one float32 batch [batch_size, 3, size, size].
    - forward: a throwaway torchvision resnet18 (random weights, 2 classes) in eval mode under
      torch.no_grad, on CPU and also on CUDA if available. The first --warmup batches are run
      but not timed (one-time setup costs), then --n-batches are timed.
Decode doesn't depend on the output size, so it varies only by noise across rows. This is a
stand-in for speed/memory only: the resnet18 is never trained and says nothing about accuracy.

Usage:
    uv run scripts/benchmark_image_size.py
    uv run scripts/benchmark_image_size.py --sizes 128 224 --n-images 100 --seed 1
"""

import argparse  # CLI flags
import sys  # put the repo root on the import path
import time  # perf_counter timings
from pathlib import Path  # filesystem paths

import pandas as pd  # read the split csv, write the results csv
import torch  # tensors, no_grad, cuda sync
from torchvision.io import decode_image, read_file  # same decoder the models' datasets use
from torchvision.models import resnet18  # throwaway network for forward-pass timing

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.transforms import build_transforms  # noqa: E402

IMAGES_DIR = REPO_ROOT / "data" / "raw" / "images"
SPLIT_CSV = REPO_ROOT / "data" / "processed" / "split.csv"
OUT_CSV = REPO_ROOT / "logs" / "runs" / "image_size_benchmark.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments: sizes, n_images, seed, batch_size, warmup, n_batches.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sizes', type=int, nargs='+', default=[96, 128, 160, 224, 320],
                        help='image sizes to compare (default: 96 128 160 224 320)')
    parser.add_argument('--n-images', type=int, default=200, help='train images to sample (default: 200)')
    parser.add_argument('--seed', type=int, default=0, help='seed for the image sample (default: 0)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for memory and forward pass (default: 32)')
    parser.add_argument('--warmup', type=int, default=3, help='untimed forward batches per size and device (default: 3)')
    parser.add_argument('--n-batches', type=int, default=10, help='timed forward batches per size and device (default: 10)')
    return parser.parse_args()


def time_preprocessing(raw_bytes: list[torch.Tensor], image_size: int) -> tuple[float, float, list[torch.Tensor]]:
    """Decode and transform every image, timing the two steps separately.

    Args:
        raw_bytes: Encoded JPEG files, one uint8 tensor each (from read_file).
        image_size: Output height and width for build_transforms.

    Returns:
        (mean decode ms per image, mean transform ms per image, transformed tensors).
    """
    transform = build_transforms('val', image_size=image_size, fit='squash')
    decode_s = transform_s = 0.0
    tensors = []
    for data in raw_bytes:
        start = time.perf_counter()
        image = decode_image(data, mode='RGB')
        mid = time.perf_counter()
        tensors.append(transform(image))
        end = time.perf_counter()
        decode_s += mid - start
        transform_s += end - mid
    n = len(raw_bytes)
    return 1000 * decode_s / n, 1000 * transform_s / n, tensors


def time_forward(net: torch.nn.Module, batch: torch.Tensor, device: torch.device, warmup: int, n_batches: int) -> float:
    """Time resnet18 forward passes on one batch, after untimed warmup passes.

    Reuses the same batch each pass: forward time depends on the tensor shape, not the pixels.

    Args:
        net: Model already on device, in eval mode.
        batch: Float tensor [batch_size, 3, size, size].
        device: Where to run.
        warmup: Untimed passes first.
        n_batches: Timed passes.

    Returns:
        Mean ms per batch over the timed passes.
    """
    batch = batch.to(device)
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)  # GPU calls return before finishing
    with torch.no_grad():
        for _ in range(warmup):
            net(batch)
        sync()
        start = time.perf_counter()
        for _ in range(n_batches):
            net(batch)
        sync()
    return 1000 * (time.perf_counter() - start) / n_batches


def main() -> None:
    """Run the benchmark for every size and device, print the table, and save the csv."""
    args = parse_args()

    df = pd.read_csv(SPLIT_CSV)
    train = df[df['split'] == 'train']
    sample = train.sample(n=min(args.n_images, len(train)), random_state=args.seed)
    raw_bytes = [read_file(str(IMAGES_DIR / name)) for name in sample['filename']]
    if len(raw_bytes) < args.batch_size:
        raise ValueError(f"--n-images ({len(raw_bytes)}) must be at least --batch-size ({args.batch_size})")

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    torch.manual_seed(args.seed)  # resnet18's random init; doesn't affect timing, but keeps runs identical
    net = resnet18(weights=None, num_classes=2).eval()

    rows = []
    for size in args.sizes:
        decode_ms, transform_ms, tensors = time_preprocessing(raw_bytes=raw_bytes, image_size=size)
        batch = torch.stack(tensors[:args.batch_size])
        batch_mb = batch.element_size() * batch.nelement() / 2**20
        for device in devices:
            forward_ms = time_forward(net=net.to(device), batch=batch, device=device,
                                      warmup=args.warmup, n_batches=args.n_batches)
            rows.append({
                'image_size': size,
                'device': device.type,
                'decode_ms_per_img': decode_ms,
                'transform_ms_per_img': transform_ms,
                'decode_share': decode_ms / (decode_ms + transform_ms + forward_ms / args.batch_size),
                'batch_mb': batch_mb,
                'forward_ms_per_batch': forward_ms,
                'forward_ms_per_img': forward_ms / args.batch_size,
            })

    results = pd.DataFrame(rows)
    print(f"{len(raw_bytes)} train images (seed {args.seed}), batch {args.batch_size}, "
          f"{args.warmup} warmup + {args.n_batches} timed forward batches, torch threads {torch.get_num_threads()}")
    print("decode_share = decode / (decode + transform + forward per image)")
    print(results.round(3).to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")


if __name__ == '__main__':
    main()
