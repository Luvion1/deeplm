import argparse
import os

import torch

from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.training.trainer import Trainer, TrainingArgs


def main():
    parser = argparse.ArgumentParser(description="Train Deeplm model")
    parser.add_argument("--config", type=str, default="deeplm_config.yaml", help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default="./deeplm_output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6.0e-4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = DeeplmConfig.from_yaml(args.config)
    else:
        config = DeeplmConfig()

    model = DeeplmModel(config)
    print(f"Model parameters: {model.num_parameters():,}")

    training_args = TrainingArgs(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    trainer = Trainer(model, config, args=training_args)

    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
