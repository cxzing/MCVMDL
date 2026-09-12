import datetime
import logging
import torch

import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mcvmdl import UNet3D_SourceMod
from utils.data_loader import PhotonSimulationDataset_GPU_Optimized,prepare_batch_on_gpu
import time

import math


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
    return logger


def calculate_psnr_ssim(pred, target):
    """
    Calculate PSNR and slice-averaged SSIM used by the paper.
    """
    import numpy as np
    import torch
    import math
    from skimage.metrics import structural_similarity as ssim

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    pred = to_numpy(pred)
    target = to_numpy(target)

    # Remove an optional singleton channel dimension.
    if pred.ndim == 5 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]

    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: pred={pred.shape}, target={target.shape}")

    flat_pred = pred.reshape(-1)
    flat_target = target.reshape(-1)

    # 1) MSE -> PSNR
    mse = float(np.mean((flat_pred - flat_target) ** 2))
    if mse < 1e-12:
        psnr = 100.0
    else:
        psnr = 20.0 * math.log10(1.0 / math.sqrt(mse))

    # 2) SSIM
    try:
        if pred.ndim == 1:
            ssim_val = 0.0
        elif pred.ndim == 3:
            ssim_list = [ssim(target[i], pred[i], data_range=1.0) for i in range(pred.shape[0])]
            ssim_val = float(np.mean(ssim_list))
        elif pred.ndim == 4:
            batch_ssim = []
            for b in range(pred.shape[0]):
                ssim_list = [ssim(target[b, i], pred[b, i], data_range=1.0) for i in range(pred.shape[1])]
                batch_ssim.append(float(np.mean(ssim_list)))
            ssim_val = float(np.mean(batch_ssim))
        else:
            ssim_val = 0.0
    except Exception:
        ssim_val = 0.0

    return psnr, ssim_val


def evaluate_model(model, test_loader, criterion, device, num_channels, output_dir="./evaluation_results"):
    """
    Evaluate the model with a configurable tissue channel count.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_samples = 0
    total_psnr = 0.0
    total_ssim = 0.0
    batch_times = []
    total_samples = 0
    progress_bar = tqdm(test_loader, desc="Inference Progress", unit="batch")
    print("Starting evaluation...")

    with torch.no_grad():
        # Unpack tissue, mask, and label.
        for batch_idx, (tissue, mask, label) in enumerate(progress_bar):
            # Move tensors to the selected device.
            tissue = tissue.to(device)
            mask = mask.to(device)
            label = label.to(device)

            # Validate tissue labels.
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                print(f"Warning: Batch {batch_idx} mask contains NaN/Inf, skipping.")
                continue

            start_time = time.time()

            # Construct the one-hot tissue and mask input on the device.
            model_input = prepare_batch_on_gpu(tissue, mask, num_channels)

            # Run model inference.
            output = model(model_input).squeeze(1)  # Output shape: [batch_size, 64, 64, 64]

            end_time = time.time()
            batch_time = end_time - start_time
            batch_times.append(batch_time)

            label = label.float()


            # Calculate metrics.
            mse = torch.mean((output - label) ** 2).item()
            mae = torch.mean(torch.abs(output - label)).item()

            total_mse += mse
            total_mae += mae

            psnr, ssim_val = calculate_psnr_ssim(output, label)
            total_psnr += psnr
            total_ssim += ssim_val

            num_samples += 1
            total_samples += tissue.size(0)

    avg_mse = total_mse / num_samples
    avg_mae = total_mae / num_samples
    avg_re = 100.0 * math.sqrt(avg_mse)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    avg_sample_time = sum(batch_times) / total_samples

    return avg_mse, avg_mae, avg_re, avg_psnr, avg_ssim, batch_times, avg_sample_time


if __name__ == "__main__":
    LOG_DIR = os.environ.get(
        "MCS_EVAL_LOG_DIR",
        os.path.join(SCRIPT_DIR, "outputs", "evaluation"),
    )
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    logger = setup_logger(os.path.join(LOG_DIR, "evaluation_results.log"))

    TEST_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "ScatterBrains-Subject01-Full", "Test")
    MODEL_PATH = os.path.join(SCRIPT_DIR, "weight", "best_model.pth")
    NUM_CHANNELS = 17
    SAME_TISSUE = True

    TEST_DATA_DIR = os.environ.get("MCS_TEST_DATA_DIR", TEST_DATA_DIR)
    MODEL_PATH = os.environ.get("MCS_MODEL_PATH", MODEL_PATH)
    NUM_CHANNELS = int(os.environ.get("MCS_NUM_CHANNELS", str(NUM_CHANNELS)))
    SAME_TISSUE = os.environ.get("MCS_SAME_TISSUE", str(SAME_TISSUE)).lower() in {"1", "true", "yes", "y"}

    BATCH_SIZE = int(os.environ.get("MCS_EVAL_BATCH_SIZE", "2"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{current_time}")
    logger.info("=" * 30 + " New Experiment " + "=" * 30)
    logger.info("Mode: ScatterBrains")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Data Dir: {TEST_DATA_DIR}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")

    OUTPUT_DIR = os.environ.get(
        "MCS_EVAL_OUTPUT_DIR",
        os.path.join(SCRIPT_DIR, "outputs", "evaluation"),
    )


    try:

        test_dataset = PhotonSimulationDataset_GPU_Optimized(TEST_DATA_DIR, same_tissue=SAME_TISSUE)


        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"Dataset loaded from {TEST_DATA_DIR}, found {len(test_dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()


    model = UNet3D_SourceMod(in_channels=NUM_CHANNELS + 1, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()


    criterion = torch.nn.MSELoss()


    results = evaluate_model(
        model, test_loader, criterion, DEVICE, NUM_CHANNELS, output_dir=OUTPUT_DIR
    )

    avg_mse, avg_mae, avg_re, avg_psnr, avg_ssim, batch_times, avg_sample_time = results


    print("\n=== Evaluation Report ===")
    print(f"Test MSE: {avg_mse:.8f}")
    print(f"Test MAE: {avg_mae:.8f}")
    print(f"Test RE: {avg_re:.4f}%")
    print(f"Test PSNR: {avg_psnr:.8f}")
    print(f"Test SSIM: {avg_ssim:.8f}")
    print("-" * 30)
    print(f"Avg Inference Time: {avg_sample_time:.6f} s/sample")

    logger.info(f"Test MSE: {avg_mse:.8f}")
    logger.info(f"Test MAE: {avg_mae:.8f}")
    logger.info(f"Test RE: {avg_re:.4f}%")
    logger.info(f"Test PSNR: {avg_psnr:.4f}")
    logger.info(f"Test SSIM: {avg_ssim:.4f}")
    logger.info(f"Avg Inference Time: {avg_sample_time:.6f} s/sample")
    logger.info("-" * 75)
    logger.info("\n")
