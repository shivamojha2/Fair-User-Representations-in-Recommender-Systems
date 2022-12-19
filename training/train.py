"""
Training module
"""
import argparse
import logging
import os
import random

import numpy as np
import torch
from config import Config
from datagen_zoo import DataCreator
from loss_zoo import LossZoo
from metrics_zoo import ModelMetricsZoo
from model_zoo import ModelZoo, OptimizerZoo, SchedulerZoo
from torch.utils.tensorboard import SummaryWriter
from utils.torch_utils import add_scalars, move_to_device, save_model
from utils.utility_funcs import save_file, set_logging


def training(config: Config):
    """
    Run training

    Args:
        config (Config): Instance of config file
    """
    # logging
    exp_out_dir = os.path.join(config.save_dir, config.exp_name)
    logging_dir = os.path.join(exp_out_dir, "logs")
    set_logging(save_dir=logging_dir)
    save_file(src=config.json_path, dest=exp_out_dir, prefix=config.exp_name)

    # GPU settings
    device = torch.device(f"cuda:{config.gpu}")

    # Reproducibility settings
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load data
    train_dataloader, val_dataloader = DataCreator.load_train_val(config)

    # Initialize loss function
    line_id_weights = torch.ones(10)
    loss_fct = LossZoo(
        config.model_kwargs["loss"], line_id_weights=line_id_weights, device=device
    )

    # Initialize model
    model = ModelZoo(config.model_name, config.model_kwargs, loss_function=loss_fct).to(
        device
    )

    # Optimizer
    optimizer = OptimizerZoo(
        config.optimizer_name, model.parameters(), **config.optimizer_kwargs
    )

    # Learning Rate Scheduler
    num_training_steps = len(train_dataloader) * config.num_epoch
    num_warmup_steps = int(len(train_dataloader) * config.scheduler_warmup_ratio)
    scheduler = SchedulerZoo(
        config.scheduler_name,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        **config.scheduler_kwargs,
    )

    # Metrics
    metrics = ModelMetricsZoo(config.model_name)

    # Initialize tensor-board
    train_writer = SummaryWriter(log_dir=os.path.join(config.save_dir, "train"))
    val_writer = SummaryWriter(log_dir=os.path.join(config.save_dir, "val"))

    # Log
    logging.info("Starting training.....")
    logging.info(
        "Training samples: {}; Effective batch size: {}; Num batch per epoch: {}".format(
            len(train_dataloader.dataset),
            train_dataloader.batch_size,
            len(train_dataloader),
        )
    )
    logging.info(
        "Validation samples: {}; Effective batch size: {}; Num batch per epoch: {}".format(
            len(val_dataloader.dataset), val_dataloader.batch_size, len(val_dataloader)
        )
    )

    running_loss_avg = float("inf")
    for epoch in range(config.num_epoch):
        train_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            metrics=metrics,
            optimizer=optimizer,
            scheduler=scheduler,
            writer=train_writer,
            log_every=config.log_every,
            device=device,
        )

        loss_avg = val_epoch(
            epoch=epoch,
            model=model,
            dataloader=val_dataloader,
            metrics=metrics,
            writer=val_writer,
            device=device,
        )

        if loss_avg < running_loss_avg:
            running_loss_avg = loss_avg
            save_model(epoch, model, optimizer, scheduler, exp_out_dir)


def train_epoch(
    epoch,
    model,
    dataloader,
    metrics,
    optimizer,
    scheduler,
    writer,
    log_every,
    device,
):
    """
    Training for epoch
    """
    losses = []
    model.train()
    for i, batch in enumerate(dataloader):
        current_iteration = epoch * len(dataloader) + i
        current_lr = scheduler.get_last_lr()[0]

        # Model input
        batch = move_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss.mean()

        # Back-propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Metric
        losses.append(loss.item())
        metrics.add_batch(outputs, batch)

        # Loss
        if current_iteration % log_every == 0:
            logging.info(
                f"[Train] Writing to tensorboard; Current iteration: {current_iteration}"
            )
            logging.info(
                f"[Train] Epoch: {epoch}; Iteration: {current_iteration}; Total Loss {loss.item(): .4f}"
            )
            add_scalars(
                writer=writer,
                iteration=current_iteration,
                prefix="batch",
                learning_rate=current_lr,
                loss=loss.item(),
            )

        # Compute epoch average
        loss_avg = np.mean(losses)
        metric_dict = metrics.compute_metrics()

        # Logging
        logging.info(f"[Train] Writing epoch {epoch} averages to tensorboard")
        logging.info(f"[Train] Epoch: {epoch}; Total Loss {loss_avg:.4f}")
        add_scalars(
            writer=writer, iteration=epoch, prefix="epoch", loss=loss_avg, **metric_dict
        )

        # Save model


def val_epoch(
    epoch,
    model,
    dataloader,
    metrics,
    writer,
    device,
):
    """
    Validation for epoch
    """
    losses = []
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            # Model input
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss.mean()

            # Metric
            losses.append(loss.item())
            metrics.add_batch(outputs, batch)

        # Compute epoch average
        loss_avg = np.mean(losses)
        metric_dict = metrics.compute_metrics()

        # Logging
        logging.info(f"[Validation] Writing epoch {epoch} averages to tensorboard")
        logging.info(f"[Validation] Epoch: {epoch}; Total Loss {loss_avg:.4f}")
        add_scalars(
            writer=writer, iteration=epoch, prefix="epoch", loss=loss_avg, **metric_dict
        )
    return loss_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=str, dest="json_path", help="Path to json config")
    args = parser.parse_args()

    config = Config(json_path=args.json_path)
    training(config)
