from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR

import sys

sys.path.insert(0, str(REPO_DIR))
from models.mcvmdl import UNet3D_SourceMod  # noqa: E402
from utils.data_loader import PhotonSimulationDataset_GPU_Optimized, prepare_batch_on_gpu  # noqa: E402
from utils.losses import build_loss  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_dir: Path) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_filename = log_dir / "train_log.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()],
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, num_channels):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training Progress", unit="batch")

    for tissue, mask, label in progress_bar:
        tissue = tissue.to(device)
        mask = mask.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        model_input = prepare_batch_on_gpu(tissue, mask, num_channels)
        output = model(model_input).squeeze(1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def validate(model, dataloader, criterion, device, num_channels):
    model.eval()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validation Progress", unit="batch")
    with torch.no_grad():
        for tissue, mask, label in progress_bar:
            tissue = tissue.to(device)
            mask = mask.to(device)
            label = label.to(device)
            model_input = prepare_batch_on_gpu(tissue, mask, num_channels)
            output = model(model_input).squeeze(1)
            loss = criterion(output, label)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one tuning run for ScatterBrains Subject01.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--num-channels", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--same-tissue", action="store_true")
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--sched-factor", type=float, default=0.5)
    parser.add_argument("--sched-patience", type=int, default=5)
    parser.add_argument("--sched-min-lr", type=float, default=1e-6)
    parser.add_argument("--loss-name", default="mse")
    parser.add_argument("--loss-alpha", type=float, default=0.7)
    parser.add_argument("--loss-beta", type=float, default=0.3)
    parser.add_argument("--peak-lambda", type=float, default=4.0)
    parser.add_argument("--peak-gamma", type=float, default=2.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    weight_dir = output_dir / "weights"
    log_dir = output_dir / "logs"
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(args.seed)
    setup_logger(log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dir = data_dir / "Train"
    test_dir = data_dir / "Test"

    logging.info("=" * 30)
    logging.info(
        f"RUN={args.run_name} epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
        f"patience={args.patience} scheduler={args.use_scheduler}"
    )
    logging.info(f"data_dir={data_dir}")
    logging.info(
        f"loss={args.loss_name} alpha={args.loss_alpha} beta={args.loss_beta} "
        f"peak_lambda={args.peak_lambda} peak_gamma={args.peak_gamma}"
    )
    logging.info("=" * 30)

    train_dataset = PhotonSimulationDataset_GPU_Optimized(str(train_dir), same_tissue=args.same_tissue)
    test_dataset = PhotonSimulationDataset_GPU_Optimized(str(test_dir), same_tissue=args.same_tissue)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet3D_SourceMod(in_channels=args.num_channels + 1, out_channels=1).to(device)
    criterion = build_loss(
        loss_name=args.loss_name,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        peak_lambda=args.peak_lambda,
        peak_gamma=args.peak_gamma,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.sched_factor,
            patience=args.sched_patience,
            min_lr=args.sched_min_lr,
        )

    best_val_loss = float("inf")
    best_epoch = -1
    counter = 0
    best_model_path = weight_dir / "best_model.pth"
    total_start = time.time()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.num_channels)
        val_loss = validate(model, test_loader, criterion, device, args.num_channels)
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logging.info(
            f"Epoch {epoch + 1}/{args.epochs} - LR={current_lr:.2e} Train={train_loss:.8f} Val={val_loss:.8f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Updated best model at epoch {best_epoch}: {best_val_loss:.8f}")
        else:
            counter += 1
            logging.info(f"EarlyStopping: {counter}/{args.patience}")
            if counter >= args.patience:
                logging.warning("Training stopped.")
                break

    logging.info(f"Finished. Total Duration: {time.time() - total_start:.2f}s")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
