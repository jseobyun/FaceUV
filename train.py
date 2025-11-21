import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pathlib import Path
import yaml
import json

from src.models import FaceUVModel
from src.data import FaceUVDataModule


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Face UV Coordinate Estimation Model')

    # Data arguments - now supports config file or command line
    parser.add_argument('--config', type=str, default="/home/jseob/Desktop/yjs/codes/FaceUV/configs/default_config.yaml", help='Path to configuration YAML file')

    # Other arguments
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')

    return parser.parse_args()



def setup_callbacks(callback_config):
    """Setup training callbacks."""
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(callback_config["checkpoint"]["checkpoint_dir"]) / callback_config["checkpoint"]["experiment_name"],
        filename='{epoch:02d}-{val_loss:.4f}',
        monitor=callback_config["checkpoint"]["monitor"],
        mode=callback_config["checkpoint"]["mode"],
        save_top_k=callback_config["checkpoint"]["save_top_k"],
        save_last=callback_config["checkpoint"]["save_last"],
        every_n_epochs=callback_config["checkpoint"]["every_n_epochs"],
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=callback_config["early_stopping"]["monitor"],
        patience=callback_config["early_stopping"]["patience"],
        mode=callback_config["early_stopping"]["mode"],
        verbose=True
    )
    callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    return callbacks


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset configuration
    training_config = config["training"]
    dataset_config = config["dataset"]
    hardware_config = config["hardware"]
    callback_config = config["callbacks"]
    logging_config =config["logging"]

    pl.seed_everything(training_config["seed"])

    print("Dataset Configuration:")
    print(json.dumps(dataset_config, indent=2))

    # Create data module
    data_module = FaceUVDataModule(
        data_dir=dataset_config["data_dir"],
        bg_dir=dataset_config["bg_dir"],
        img_size=training_config["img_size"],
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        augmentation=True
    )

    # Create model
    model = FaceUVModel(
        output_channels=3,  # UV (2 channels) + mask (1 channel)
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        scheduler_config={
            'T_max': training_config["max_epochs"],
            'eta_min': 1e-6
        }
    )

    # Setup callbacks
    callbacks = setup_callbacks(callback_config)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=logging_config["log_dir"],
        name=callback_config["checkpoint"]["experiment_name"],
        default_hp_metric=False
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"],
        accelerator=hardware_config["accelerator"],
        devices= hardware_config["gpus"] if hardware_config["accelerator"] == 'gpu' else 'auto',
        callbacks=callbacks,
        logger=logger,
        precision=32,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=logging_config["log_every_n_steps"],
        val_check_interval=logging_config["val_check_interval"],
        check_val_every_n_epoch=logging_config["check_val_every_n_epoch"],
        num_sanity_val_steps=0,
        deterministic=False,
        strategy='auto' if len(hardware_config["gpus"]) == 1 else 'ddp'
    )

    # Train model
    if args.resume_from:
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, data_module)

    # Test model with best checkpoint
    trainer.test(model, data_module, ckpt_path='best')


if __name__ == '__main__':
    main()
