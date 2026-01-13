# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
import time
from datetime import datetime
from functools import partial

import torch
import torch.distributed as dist
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.utils.helper import (
    Task_Mode,
    get_autocast_ctx,
    get_grad_scaler,
    get_local_rank,
    get_node_rank,
    get_op_verifier_ctx,
    get_world_size,
    init_qaic_profiling,
    is_rank_zero,
    save_to_json,
    stop_qaic_profiling,
)
from QEfficient.finetune.utils.logging_utils import logger

try:
    import torch_qaic  # noqa: F401
except ImportError as e:
    logger.log_rank_zero(
        f"Unable to import 'torch_qaic' package due to exception: {e}. Moving ahead without the torch_qaic extension.",
        logging.WARNING,
    )


def train(
    model,
    tokenizer,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    train_config: TrainConfig,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        tokenizer: tokenizer used in the eval for decoding the predictions
        train_dataloader: The dataloader containing the training data
        eval_dataloader: The dataloader containing the eval data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        train_config: The training configuration

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    device = train_config.device
    device_type = torch.device(device).type

    node_rank = get_node_rank()
    local_rank = get_local_rank()

    # Update output_dir to include the node rank suffix
    train_config.output_dir = f"{train_config.output_dir}_node_rank_{node_rank}"

    train_metric = []
    train_loss = []
    eval_metric = []
    eval_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_node_{node_rank}_rank_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_metric = []
        train_step_loss = []
        eval_step_loss = []
        eval_step_metric = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_eval_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached

    tensorboard_updates = None
    if is_rank_zero():
        tensorboard_log_dir = train_config.output_dir + "/runs/" + f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        tensorboard_updates = SummaryWriter(log_dir=tensorboard_log_dir)

    scaler = get_grad_scaler(train_config.grad_scaler, device_type)

    loss_0_counter = torch.tensor([0]).to(device)

    if train_config.enable_ddp:
        dist.broadcast(loss_0_counter, src=0)

    acc_helper = None
    if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
        if train_config.enable_ddp:
            num_classes = model.module.classifier.out_features
        else:
            num_classes = model.classifier.out_features
        acc_helper = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)

    autocast_ctx = get_autocast_ctx(train_config.use_autocast, device_type, dtype=torch.float16)
    op_verifier_ctx = partial(get_op_verifier_ctx, train_config.opByOpVerifier, device_type, train_config.output_dir)

    # Start the training loop
    for epoch in range(train_config.num_epochs):
        if loss_0_counter.item() == train_config.convergence_counter:
            logger.log_rank_zero(
                f"Skipping epoch {epoch + 1} since loss value has been <= {train_config.convergence_loss} for last {loss_0_counter.item()} steps."
            )
            break

        if train_config.use_peft and train_config.from_peft_checkpoint:
            path = train_config.from_peft_checkpoint.rstrip("/")
            try:
                intermediate_epoch = int(path.split("/")[-2].split("_")[-1]) - 1
                intermediate_step = int(path.split("/")[-1].split("_")[-1])
            except (IndexError, ValueError):
                intermediate_epoch = int(path.split("/")[-1].split("_")[-1]) - 1
                intermediate_step = 0

            if epoch < intermediate_epoch:
                logger.log_rank_zero(f"Skipping epoch {epoch + 1} since fine tuning has already completed for it.")
                continue
            if intermediate_step == 0 and epoch == intermediate_epoch:
                logger.log_rank_zero(f"Skipping epoch {epoch + 1}, since fine tuning has already completed for it.")
                continue

        logger.log_rank_zero(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        model.train()

        total_loss = 0.0
        total_length = len(train_dataloader) // train_config.gradient_accumulation_steps
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch + 1}",
            total=total_length,
            dynamic_ncols=True,
        )

        init_qaic_profiling(train_config.use_profiler, device_type)

        num_dummy_samples = 0
        for step, batch in enumerate(train_dataloader):
            # total_train_steps indicates the cumulative number of training steps completed across all epochs.
            # When resuming fine-tuning from previously saved checkpoints, total_train_steps indicates the total number of steps trained across the earlier session and the ongoing one.
            total_train_steps = (epoch) * len(train_dataloader) + step
            # resume training from a particular checkpoint, assuming the dataset is not shuffled
            if train_config.use_peft and train_config.from_peft_checkpoint:
                # to bring the count of train_step in sync with where it left off

                if epoch == intermediate_epoch and step == 0:
                    logger.log_rank_zero(
                        f"Skipping first {intermediate_step} steps for epoch {epoch + 1}, since fine tuning has already completed for it."
                    )
                if epoch == intermediate_epoch and step < intermediate_step:
                    continue

            if train_config.max_train_step > 0 and total_train_steps >= train_config.max_train_step:
                max_steps_reached = True
                logger.log_rank_zero(
                    "Maximum training steps reached "
                    f"(max_train_step={train_config.max_train_step}). Stopping "
                    "the training process."
                )
                break
            batch = {k: v.to(device) for k, v in batch.items()}  # move the batch elements to qaic device

            is_optimizer_step = (step + 1) % train_config.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1
            if train_config.enable_ddp:
                # Below block derived from : https://github.com/karpathy/nanoGPT/blob/93a43d9a5c22450bbf06e78da2cb6eeef084b717/train.py#L293
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # using too many context managers may bloat the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = is_optimizer_step

            with autocast_ctx, op_verifier_ctx(step) as verifier:
                model_outputs = model(**batch)
                loss = model_outputs.loss  # Forward call
                if (batch["labels"] != -100).sum() == 0:
                    loss = loss.nan_to_num(nan=0.0)
                    num_dummy_samples += train_config.train_batch_size
                else:
                    num_dummy_samples_per_batch = (
                        (torch.sum(batch["labels"] == -100, dim=1) == batch["labels"].shape[1]).sum().item()
                    )
                    if num_dummy_samples_per_batch > 0:
                        num_dummy_samples += num_dummy_samples_per_batch
                        loss = loss * train_config.train_batch_size / num_dummy_samples_per_batch

                if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
                    logits = model_outputs.logits
                    labels = batch["labels"][:, 0]
                    preds = torch.nn.functional.softmax(logits, dim=-1)
                    acc_helper.forward(preds, labels)
            if train_config.opByOpVerifier:
                logger.info("Mismatches detected:", verifier.get_perop_mismatch_count())

            total_loss += loss.detach().float()

            if is_rank_zero():
                tensorboard_updates.add_scalars("loss", {"train": loss}, total_train_steps)
                if loss <= train_config.convergence_loss:
                    loss_0_counter += 1
                else:
                    loss_0_counter = torch.tensor([0]).to(device)
            if train_config.enable_ddp:
                dist.broadcast(loss_0_counter, src=0)

            if train_config.save_metrics:
                train_step_loss.append(loss.detach().float().item())
                if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
                    step_metric_value = float(acc_helper.compute())
                else:
                    step_metric_value = float(torch.exp(loss.detach().float()))
                train_step_metric.append(step_metric_value)

            # Accumulate gradients
            complete_accum_steps = (
                len(train_dataloader) - len(train_dataloader) % train_config.gradient_accumulation_steps
            )
            if step < complete_accum_steps:
                num_samples_in_cur_update = train_config.gradient_accumulation_steps
            else:
                num_samples_in_cur_update = len(train_dataloader) % train_config.gradient_accumulation_steps

            normalized_loss = loss / num_samples_in_cur_update

            if train_config.grad_scaler:
                scaler.scale(normalized_loss).backward()  # backward pass
            else:
                normalized_loss.backward()  # backward pass

            if is_optimizer_step:
                if train_config.grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            # Save the trained checkpoints for every given steps
            if (step + 1) % train_config.intermediate_step_save == 0:
                stop_qaic_profiling(train_config.use_profiler, device_type)
                if train_config.enable_ddp:
                    if dist.get_rank() == 0:
                        model.module.save_pretrained(
                            train_config.output_dir + f"/trained_weights/epoch_{epoch + 1}/step_{step + 1}"
                        )
                else:
                    model.save_pretrained(
                        train_config.output_dir + f"/trained_weights/epoch_{epoch + 1}/step_{step + 1}"
                    )

            pbar.set_description(
                f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, step {step + 1}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
            )
            if train_config.save_metrics:
                save_to_json(
                    metrics_filename,
                    train_step_loss,
                    train_loss,
                    train_step_metric,
                    train_metric,
                    eval_step_loss,
                    eval_loss,
                    eval_step_metric,
                    eval_metric,
                )
            if loss_0_counter.item() == train_config.convergence_counter:
                logger.log_rank_zero(
                    f"Loss value has been <= {train_config.convergence_loss} for last {loss_0_counter.item()} steps.Hence,stopping the fine tuning."
                )
                break

        pbar.close()
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        # corrects the step count if fine-tuning is resumed through saved checkpoint
        step_correction = (
            -intermediate_step
            if (train_config.use_peft and train_config.from_peft_checkpoint and epoch == intermediate_epoch)
            else 1
        )

        denominator = step + step_correction - (num_dummy_samples / train_config.train_batch_size)
        train_epoch_loss = total_loss / denominator if total_loss != 0.0 else torch.tensor(0.0).to(device)

        if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
            train_epoch_metric = acc_helper.compute()
            acc_helper.reset()
        else:
            train_epoch_metric = torch.exp(train_epoch_loss)

        train_metric.append(float(train_epoch_metric))
        train_loss.append(float(train_epoch_loss))

        if train_config.enable_ddp:
            dist.all_reduce(train_epoch_loss, op=dist.ReduceOp.SUM)
            train_epoch_loss /= get_world_size()
            dist.all_reduce(train_epoch_metric, op=dist.ReduceOp.SUM)
            train_epoch_metric /= get_world_size()

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation:
            eval_epoch_loss, eval_epoch_metric, step_loss, step_metric = evaluation(
                model, train_config, eval_dataloader, device
            )

            if is_rank_zero():
                tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)
            if train_config.save_metrics:
                eval_step_loss.extend(step_loss)
                eval_step_metric.extend(step_metric)
                eval_loss.append(float(eval_epoch_loss))
                eval_metric.append(float(eval_epoch_metric))

            if train_config.enable_ddp:
                dist.all_reduce(eval_epoch_loss, op=dist.ReduceOp.SUM)
                eval_epoch_loss /= get_world_size()
                dist.all_reduce(eval_epoch_metric, op=dist.ReduceOp.SUM)
                eval_epoch_metric /= get_world_size()

            if eval_epoch_loss < best_eval_loss:
                best_eval_loss = eval_epoch_loss
                logger.log_rank_zero(f"Best eval loss on epoch {epoch + 1} is {best_eval_loss:.4f}")

            logger.log_rank_zero(
                f"Epoch {epoch + 1}: Eval Loss: {eval_epoch_loss.detach().cpu():.4f}, Eval metric: {eval_epoch_metric.detach().cpu():.4f}"
            )

        # saving the adapters after completion of each epoch
        if train_config.save_model:
            if train_config.enable_ddp:
                if dist.get_rank() == 0:
                    model.module.save_pretrained(train_config.output_dir + f"/complete_epoch_{epoch + 1}")
            else:
                model.save_pretrained(train_config.output_dir + f"/complete_epoch_{epoch + 1}")

        logger.log_rank_zero(
            f"Epoch {epoch + 1}: Train epoch loss: {train_epoch_loss:.4f}, Train metric: {train_epoch_metric:.4f}, Epoch time {epoch_end_time:.2f} sec"
        )
        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_metric,
                train_metric,
                eval_step_loss,
                eval_loss,
                eval_step_metric,
                eval_metric,
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0

    results["last_epoch_train_loss"] = train_epoch_loss.cpu()
    results["last_epoch_train_metric"] = train_epoch_metric.cpu()
    results["train_step_loss"] = train_step_loss
    results["train_step_metric"] = train_step_metric

    if train_config.run_validation:
        results["last_epoch_eval_loss"] = eval_epoch_loss.cpu()
        results["last_epoch_eval_metric"] = eval_epoch_metric.cpu()
        results["eval_step_loss"] = eval_step_loss
        results["eval_step_metric"] = eval_step_metric
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    return results


def evaluation(model, train_config, eval_dataloader, device):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data

    Returns: eval_epoch_loss, eval_epoch_metric, eval_step_loss, eval_step_metric
    """
    if train_config.enable_ddp:
        dist.barrier()

    model.eval()

    if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
        if train_config.enable_ddp:
            num_classes = model.module.classifier.out_features
        else:
            num_classes = model.classifier.out_features
        acc_helper = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)

    # special handling for qaic device and dtype
    # model.to(device)

    eval_step_loss = []
    eval_step_metric = []

    eval_loss = torch.tensor(0.0, dtype=torch.float32, device=device)  # Initialize evaluation loss
    device_type = torch.device(device).type

    num_dummy_samples = 0
    autocast_ctx = get_autocast_ctx(train_config.use_autocast, device_type, dtype=torch.float16)
    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        #  stop when the maximum number of eval steps is reached
        if train_config.max_eval_step > 0 and step >= train_config.max_eval_step:
            break
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            with autocast_ctx:
                outputs = model(**batch)
            loss = outputs.loss

            if (batch["labels"] != -100).sum() == 0:
                loss = loss.nan_to_num(nan=0.0)
                num_dummy_samples += 1
            else:
                num_dummy_samples_per_batch = (
                    (torch.sum(batch["labels"] == -100, dim=1) == batch["labels"].shape[1]).sum().item()
                )
                if num_dummy_samples_per_batch > 0:
                    num_dummy_samples += num_dummy_samples_per_batch
                    loss = loss * train_config.val_batch_size / num_dummy_samples_per_batch

            if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
                logits = outputs.logits
                labels = batch["labels"][:, 0]
                preds = torch.nn.functional.softmax(logits, dim=-1)
                eval_acc = acc_helper.forward(preds, labels)
                metric_value = eval_acc.detach().float().item()
            else:
                metric_value = float(torch.exp(loss.detach().float()))

            if train_config.save_metrics:
                eval_step_loss.append(loss.detach().float().item())
                eval_step_metric.append(metric_value)

            eval_loss += loss.detach().float()

    # Compute average loss and metric
    eval_epoch_loss = (
        torch.tensor(0.0).to(device)
        if eval_loss == 0.0
        else eval_loss / (step + 1 - num_dummy_samples / train_config.val_batch_size)
    )
    if train_config.task_mode == Task_Mode.SEQ_CLASSIFICATION:
        eval_epoch_metric = acc_helper.compute()
    else:
        eval_epoch_metric = torch.exp(eval_epoch_loss)

    return eval_epoch_loss, eval_epoch_metric, eval_step_loss, eval_step_metric


def print_model_size(model) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.
    Args:
        model: PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_rank_zero(f"Model has {total_params / 1e6} Million params.")


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters, all params and percentage of trainablke params.
    Args:
        model: The PyTorch model.
    """
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.log_rank_zero(
        f"Trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )