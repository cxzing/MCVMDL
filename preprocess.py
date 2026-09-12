import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


"""
Generate compressed absorption data using block averaging, small-value
truncation, log10 transformation, and normalization to [0, 1].
Absorption.npz -> Absorption_AvgCompressed.npz
"""


def pad_numpy_array(array, target_shape):
    """Pad an array to the target shape."""
    pad_width = [(0, max(0, target - size)) for size, target in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, mode="constant", constant_values=0)


def compress_absorption_file_optimized(
    input_file,
    output_file,
    tissue_size=512,
    smooth_sigma=-1,
    noise_threshold=5e-9,
):
    """Process one absorption file."""
    try:
        absorption = np.load(input_file)["arr_0"]

        if smooth_sigma > 0:
            absorption = gaussian_filter(absorption, sigma=smooth_sigma)

        if tissue_size == 512:
            absorption_padded = pad_numpy_array(absorption, (tissue_size, tissue_size, tissue_size))
            absorption_compressed = absorption_padded.reshape(
                128, 4, 128, 4, 128, 4
            ).mean(axis=(1, 3, 5))
            absorption_compressed[absorption_compressed < noise_threshold] = 0
        else:
            absorption_compressed = absorption

        absorption_compressed = (
            np.log10(np.maximum(absorption_compressed, 1e-10)) + 10.0
        ) / 10

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez_compressed(output_file, absorption_compressed)

        return True, f"成功: {os.path.basename(input_file)}"
    except Exception as exc:
        return False, f"失败 ({input_file}): {exc}"


def process_wrapper(args):
    """Unpack one process-pool task."""
    input_path, output_path, smooth_sigma, noise_threshold = args
    return compress_absorption_file_optimized(
        input_path,
        output_path,
        tissue_size=512,
        smooth_sigma=smooth_sigma,
        noise_threshold=noise_threshold,
    )


def batch_process_multiprocess(
    root_dir,
    max_workers=None,
    smooth_sigma=-1,
    noise_threshold=5e-9,
):
    """Process every immediate sample directory with multiple processes."""
    if not os.path.exists(root_dir):
        print(f"错误: 目录不存在 - {root_dir}")
        return False

    subdirs = sorted(
        directory
        for directory in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, directory))
    )

    tasks = []
    print(f"正在扫描文件夹: {root_dir} ...")

    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)
        input_path = os.path.join(folder_path, "Absorption.npz")
        output_path = os.path.join(folder_path, "Absorption_AvgCompressed.npz")
        if os.path.exists(input_path):
            tasks.append((input_path, output_path, smooth_sigma, noise_threshold))

    total_tasks = len(tasks)
    print(f"找到 {total_tasks} 个待处理任务。")
    if total_tasks == 0:
        return True

    if max_workers is None:
        max_workers = 8
    max_workers = max(1, min(int(max_workers), multiprocessing.cpu_count()))

    print(f"启动多进程池 (核心数: {max_workers})...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_wrapper, tasks),
                total=total_tasks,
                unit="file",
                desc="Processing",
            )
        )

    success_count = sum(1 for success, _ in results if success)
    print(f"\n处理完成! 成功: {success_count}/{total_tasks}")
    for success, message in results:
        if not success:
            print(message)
    return success_count == total_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average-compress Absorption.npz volumes to 128 x 128 x 128."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    parser.add_argument("--noise-threshold", type=float, default=5e-9)
    return parser.parse_args()


def main():
    args = parse_args()
    success = batch_process_multiprocess(
        args.data_dir,
        max_workers=args.workers,
        smooth_sigma=args.smooth_sigma,
        noise_threshold=args.noise_threshold,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
