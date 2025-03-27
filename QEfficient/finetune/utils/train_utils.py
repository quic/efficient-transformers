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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from QEfficient.finetune.configs.training import train_config as TRAIN_CONFIG

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
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config: TRAIN_CONFIG,
    device,
    local_rank=None,
    rank=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = (
            f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

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
                        filter_config=qaic_debug.DispatchFilterConfig.default(device, opwise_limit=10),
                        dump_root_dir=train_config.dump_root_dir + str(step),
                    ) as verifier:
                        loss = model(**batch).loss  # Forward call
                    print("Mismatches detected:", verifier.get_perop_mismatch_count())
                else:
                    loss = model(**batch).loss  # Forward call

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
                train_step_perplexity.append(float(torch.exp(loss.detach().float())))

            if train_config.gradient_checkpointing:
                # Enforce that the loss retains its gradient tracking.
                loss.requires_grad = True

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
                    train_step_perplexity,
                    train_prep,
                    val_step_loss,
                    val_loss,
                    val_step_perplexity,
                    val_prep,
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

        # Get the correct train loss from all the nodes.
        dist.barrier()
        dist.all_reduce(train_epoch_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss /= dist.get_world_size()

        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation:
            if train_config.enable_ddp:
                dist.barrier()
                eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                    model, train_config, eval_dataloader, local_rank, tokenizer, device
                )
                dist.barrier()
                dist.all_reduce(eval_epoch_loss, op=dist.ReduceOp.SUM)
                eval_epoch_loss /= dist.get_world_size()
                if local_rank == 0:
                    tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)

            else:
                eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                    model, train_config, eval_dataloader, local_rank, tokenizer, device
                )
                tensorboard_updates.add_scalars("loss", {"eval": eval_epoch_loss}, total_train_steps)

            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

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
            val_prep.append(float(eval_ppl))
        print(
            f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
        )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep,
            )
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, device):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()

    # special handling for qaic device and dtype
    # model.to(device)

    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []

    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    # max_steps_reached = False  # Flag to indicate max eval steps reached

    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        total_eval_steps += 1
        #  stop when the maximum number of eval steps is reached
        if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
            # max_steps_reached = True
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

            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            eval_loss += loss.detach().float()
        # Decode predictions and add to evaluation predictions list
        preds = torch.argmax(outputs.logits, -1)
        eval_preds.extend(tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True))

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    print(f" {eval_ppl.detach().cpu()=} {eval_epoch_loss.detach().cpu()=}")

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


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
    train_step_ppl,
    train_epoch_ppl,
    val_step_loss,
    val_epoch_loss,
    val_step_ppl,
    val_epoch_ppl,
):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
