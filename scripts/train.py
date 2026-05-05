"""
Training script for the COCO Latent Diffusion Model.

Usage::

    # Train the VAE first
    python scripts/train.py --config configs/autoencoder.yaml \\
                            --name kl_vae --base_lr 4.5e-6

    # Train the LDM (requires a VAE checkpoint)
    python scripts/train.py --config configs/coco_ldm.yaml \\
                            --name coco_ldm \\
                            --first_stage_ckpt checkpoints/autoencoder/autoencoder.ckpt
"""

import argparse
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a COCO LDM model")
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to the YAML config file")
    parser.add_argument("--name", "-n", type=str, default="",
                        help="Experiment name (appended to the log directory)")
    parser.add_argument("--base_lr", type=float, default=None,
                        help="Override base learning rate from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--resume", "-r", type=str, default="",
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--first_stage_ckpt", type=str, default="",
                        help="Path to pre-trained VAE checkpoint (LDM training only)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # Load config
    config = OmegaConf.load(args.config)

    # Override hyperparameters from CLI
    if args.base_lr is not None:
        config.optimizer.params.lr = args.base_lr
    if args.batch_size is not None:
        config.data.params.batch_size = args.batch_size
    if args.gpus != 1:
        config.lightning.trainer.devices = args.gpus

    # Instantiate model
    model = instantiate_from_config(config.model)
    model.learning_rate = config.optimizer.params.lr

    # Load first-stage (VAE) checkpoint for LDM training
    if args.first_stage_ckpt:
        print(f"Loading first-stage checkpoint: {args.first_stage_ckpt}")
        sd = torch.load(args.first_stage_ckpt, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        # Strip any "first_stage_model." prefix if present
        sd = {k.replace("first_stage_model.", ""): v for k, v in sd.items()
              if k.startswith("first_stage_model.") or "first_stage_model." not in k}
        missing, unexpected = model.first_stage_model.load_state_dict(sd, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Instantiate data
    data_module = instantiate_from_config(config.data)

    # Build callbacks
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    exp_name = f"{args.name}_{timestamp}" if args.name else timestamp

    ckpt_cfg = config.lightning.callbacks.model_checkpoint.params
    ckpt_callback = ModelCheckpoint(**ckpt_cfg)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Logger
    logger_cfg = config.lightning.get("logger", None)
    if logger_cfg is not None:
        logger = TensorBoardLogger(
            save_dir=logger_cfg.params.save_dir,
            name=exp_name,
        )
    else:
        logger = True

    # Trainer
    trainer_kwargs = OmegaConf.to_container(config.lightning.trainer, resolve=True)
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[ckpt_callback, lr_monitor],
        logger=logger,
    )

    # Train
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
        ckpt_path=args.resume or None,
    )


if __name__ == "__main__":
    main()
