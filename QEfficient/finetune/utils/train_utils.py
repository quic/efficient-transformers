# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from QEfficient.finetune.configs.training import TrainConfig

try:
    import torch_qaic  # noqa: F401
    import torch_qaic.debug as qaic_debug  # noqa: F401
    import torch_qaic.profile as qaic_profile  # noqa: F401
    import torch_qaic.utils as qaic_utils  # noqa: F401
    from torch.qaic.amp import GradScaler as QAicGradScaler
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")

from torch.amp import GradScaler


def train(
    model,
    tokenizer,
    train_dataloader,
    eval_dataloader,
    optimizer,
    lr_scheduler,
    train_config: TrainConfig,
    local_rank=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        tokenizer: tokenizer used in the eval for decoding the predicitons
        train_dataloader: The dataloader containing the training data
        eval_dataloader: The dataloader containing the eval data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        train_config: The training configuration
        local_rank: The rank of the current node in a distributed setting

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    device = train_config.device

    train_metric = []
    train_loss = []
    val_metric = []
    val_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = (
            f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        train_step_metric = []
        train_step_loss = []
        val_step_loss = []
        val_step_metric = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    device_type = device.split(":")[0]

    tensorboard_updates = None
    if train_config.enable_ddp:
        if local_rank == 0:
            tensorboard_updates = SummaryWriter()
    else:
        tensorboard_updates = SummaryWriter()

    if train_config.grad_scaler:
        if device.startswith("qaic"):
            scaler = QAicGradScaler()
        else:
            scaler = GradScaler(device_type)

    loss_0_counter = torch.tensor([0]).to(device)

    if train_config.enable_ddp:
        dist.broadcast(loss_0_counter, src=0)

    acc_helper = None
    if train_config.task_type == "seq_classification":
        if train_config.enable_ddp:
            num_classes = model.module.classifier.out_features
        else:
            num_classes = model.classifier.out_features
        acc_helper = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)

    # Start the training loop
    for epoch in range(train_config.num_epochs):
        if loss_0_counter.item() == train_config.convergence_counter:
            if train_config.enable_ddp:
                print(
                    f"Not proceeding with epoch {epoch + 1} on device {local_rank} since loss value has been <= {train_config.convergence_loss} for last {loss_0_counter.item()} steps."
                )
                break
            else:
                print(
                    f"Not proceeding with epoch {epoch + 1} since loss value has been <= {train_config.convergence_loss}  for last {loss_0_counter.item()} steps."
                )
                break

        if train_config.use_peft and train_config.from_peft_checkpoint:
            intermediate_epoch = int(train_config.from_peft_checkpoint.split("/")[-2].split("_")[-1]) - 1
            if epoch < intermediate_epoch:
                print(f"Skipping epoch {epoch + 1} since fine tuning has already completed for it.")
                # to bring the count of train_step in sync with where it left off
                total_train_steps += len(train_dataloader)
                continue

        print(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
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

        # enable profile for qaic
        qaic_profile.start_profiling(device, 1) if train_config.use_profiler else None

        for step, batch in enumerate(train_dataloader):
            # resume training from a particular checkpoint, assuming the dataset is not shuffled
            if train_config.use_peft and train_config.from_peft_checkpoint:
                intermediate_step = int(train_config.from_peft_checkpoint.split("/")[-1].split("_")[-1])
                intermediate_epoch = int(train_config.from_peft_checkpoint.split("/")[-2].split("_")[-1]) - 1
                # to bring the count of train_step in sync with where it left off
                if epoch == intermediate_epoch and step == 0:
                    total_train_steps += intermediate_step
                    print(
                        f"skipping first {intermediate_step} steps for epoch {epoch + 1}, since fine tuning has already completed for them."
                    )
                if epoch == intermediate_epoch and step < intermediate_step:
                    total_train_steps += 1
                    continue
            total_train_steps += 1

            #  stop when the maximum number of training steps is reached
            if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                max_steps_reached = True
                break
            batch = {k: v.to(device) for k, v in batch.items()}  # move the batch elements to qaic device

            with (
                torch.autocast(device_type=device, dtype=torch.float16) if train_config.use_autocast else nullcontext()
            ):
                # an additional condition can be put here to avoid opByOpVerifier getting triggered for each step
                if train_config.opByOpVerifier:
                    with qaic_debug.OpByOpVerifierMode(
                        ref_device="cpu",
                        ref_dtype=torch.float32,
                        # adjust atol & rtol this as required
                        atol=1e-1,
                        use_ref_output_on_mismatch=True,
                        filter_config=qaic_debug.DispatchFilterConfig.default(device),
                        dump_root_dir=train_config.dump_root_dir + str(step),
                    ) as verifier:
                        model_outputs = model(**batch)
                        loss = model_outputs.loss  # Forward call
                        if train_config.task_type == "seq_classification":
                            logits = model_outputs.logits
                            labels = batch["labels"][:, 0]
                            preds = torch.nn.functional.softmax(logits, dim=-1)
                            acc_helper.forward(preds, labels)
                    print("Mismatches detected:", verifier.get_perop_mismatch_count())
                else:
                    model_outputs = model(**batch)
                    loss = model_outputs.loss  # Forward call
                    if train_config.task_type == "seq_classification":
                        logits = model_outputs.logits
                        labels = batch["labels"][:, 0]
                        preds = torch.nn.functional.softmax(logits, dim=-1)
                        acc_helper.forward(preds, labels)

            total_loss += loss.detach().float()
            # Accumalate graidents
            loss = loss / train_config.gradient_accumulation_steps
            if train_config.enable_ddp:
                if local_rank == 0:
                    if loss <= train_config.convergence_loss:
                        loss_0_counter += 1
                    else:
                        loss_0_counter = torch.tensor([0]).to(device)
                dist.broadcast(loss_0_counter, src=0)
            else:
                if loss <= train_config.convergence_loss:
                    loss_0_counter += 1
                else:
                    loss_0_counter = torch.tensor([0]).to(device)

            if train_config.enable_ddp:
                if local_rank == 0:
                    tensorboard_updates.add_scalars("loss", {"train": loss}, total_train_steps)
            else:
                tensorboard_updates.add_scalars("loss", {"train": loss}, total_train_steps)

            if train_config.save_metrics:
                train_step_loss.append(loss.detach().float().item())
                if train_config.task_type == "seq_classification":
                    step_metric_val = float(acc_helper.compute())
                else:
                    step_metric_val = float(torch.exp(loss.detach().float()))
                train_step_metric.append(step_metric_val)

            if train_config.grad_scaler:
                scaler.scale(loss).backward()  # backward pass
            else:
                loss.backward()  # backward pass

            if (step + 1) % train_config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if train_config.grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            # Save the trained checkpoints for every given steps
            if step % train_config.intermediate_step_save == 0:
                qaic_profile.stop_profiling(device) if train_config.use_profiler else None
                if train_config.enable_ddp:
                    if dist.get_rank() == 0:
                        model.module.save_pretrained(
                            train_config.output_dir + f"/trained_weights/epoch_{epoch + 1}/step_{step}"
                        )
                else:
                    model.save_pretrained(train_config.output_dir + f"/trained_weights/epoch_{epoch + 1}/step_{step}")

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
                    val_step_loss,
                    val_loss,
                    val_step_metric,
                    val_metric,
                )
            if train_config.enable_ddp:
                if loss_0_counter.item() == train_config.convergence_counter:
                    print(
                        f"Loss value has been <= {train_config.convergence_loss} for last {loss_0_counter.item()} steps. Hence, stopping the fine tuning on device {local_rank}."
                    )
                    break
            else:
                if loss_0_counter.item() == train_config.convergence_counter:
                    print(
                        f"Loss value has been  <= {train_config.convergence_loss}  for last {loss_0_counter.item()} steps. Hence, stopping the fine tuning."
                    )
                    break

        pbar.close()
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        if loss_0_counter.item() == train_config.convergence_counter:
            if train_config.use_peft and train_config.from_peft_checkpoint and epoch == intermediate_epoch:
                train_epoch_loss = total_loss / (step - intermediate_step)
            else:
                train_epoch_loss = total_loss / step
        else:
            if train_config.use_peft and train_config.from_peft_checkpoint and epoch == intermediate_epoch:
                train_epoch_loss = total_loss / (len(train_dataloader) - intermediate_step)
            else:
                train_epoch_loss = total_loss / len(train_dataloader)

        if train_config.task_type == "seq_classification":
            metric_val = acc_helper.compute()
            acc_helper.reset()
        else:
            metric_val = torch.exp(train_epoch_loss)

        train_metric.append(float(metric_val))
        train_loss.append(float(train_epoch_loss))

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation:
            if train_config.enable_ddp:
                dist.barrier()
                eval_epoch_loss, eval_metric, temp_val_loss, temp_step_metric = evaluation_helper(
                    model, train_config, eval_dataloader, device
                )
                if local_rank == 0:
                    tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)

            else:
                eval_epoch_loss, eval_metric, temp_val_loss, temp_step_metric = evaluation_helper(
                    model, train_config, eval_dataloader, device
                )
                tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)

            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_metric.extend(temp_step_metric)

        # saving the adapters after completion of each epoch
        if train_config.save_model:
            if train_config.enable_ddp:
                if dist.get_rank() == 0:
                    model.module.save_pretrained(train_config.output_dir + f"/complete_epoch_{epoch + 1}")
            else:
                model.save_pretrained(train_config.output_dir + f"/complete_epoch_{epoch + 1}")

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_metric.append(float(eval_metric))
        if train_config.task_type == "seq_classification":
            print(
                f"Epoch {epoch + 1}: train_acc={metric_val:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )
        else:
            print(
                f"Epoch {epoch + 1}: train_metric={metric_val:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_metric,
                train_metric,
                val_step_loss,
                val_loss,
                val_step_metric,
                val_metric,
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_metric = sum(train_metric) / len(train_metric)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_metric = sum(val_metric) / len(val_metric)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_metric"] = avg_train_metric
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_metric"] = avg_eval_metric
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    return results


def evaluation_helper(model, train_config, eval_dataloader, device):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data

    Returns: eval_epoch_loss, eval_metric, eval_step_loss, eval_step_metric
    """
    model.eval()

    if train_config.task_type == "seq_classification":
        if train_config.enable_ddp:
            num_classes = model.module.classifier.out_features
        else:
            num_classes = model.classifier.out_features
        acc_helper = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)

    # special handling for qaic device and dtype
    # model.to(device)

    val_step_loss = []
    val_step_metric = []

    eval_loss = 0.0  # Initialize evaluation loss

    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        #  stop when the maximum number of eval steps is reached
        if train_config.max_eval_step > 0 and step > train_config.max_eval_step:
            break
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            with (
                torch.autocast(device_type=device, dtype=torch.float16) if train_config.use_autocast else nullcontext()
            ):
                outputs = model(**batch)
            loss = outputs.loss

            if train_config.task_type == "seq_classification":
                logits = outputs.logits
                labels = batch["labels"][:, 0]
                preds = torch.nn.functional.softmax(logits, dim=-1)
                val_acc = acc_helper.forward(preds, labels)
                metric_val = val_acc.detach().float().item()
            else:
                metric_val = float(torch.exp(loss.detach().float()))

            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_metric.append(metric_val)

            eval_loss += loss.detach().float()

    # Compute average loss and metric
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.task_type == "seq_classification":
        eval_metric = acc_helper.compute()
    else:
        eval_metric = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    print(f" {eval_metric.detach().cpu()=} {eval_epoch_loss.detach().cpu()=}")

    return eval_epoch_loss, eval_metric, val_step_loss, val_step_metric


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
    """

    print(f"--> Model {config.model_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def save_to_json(
    output_filename,
    train_step_loss,
    train_epoch_loss,
    train_step_metric,
    train_epoch_metric,
    val_step_loss,
    val_epoch_loss,
    val_step_metric,
    val_epoch_metric,
):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_metric": train_step_metric,
        "train_epoch_metric": train_epoch_metric,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_metric": val_step_metric,
        "val_epoch_metric": val_epoch_metric,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
