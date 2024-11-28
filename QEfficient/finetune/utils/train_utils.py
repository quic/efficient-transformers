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

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import torch_qaic  # noqa: F401
    import torch_qaic.debug as qaic_debug  # noqa: F401
    import torch_qaic.profile as qaic_profile  # noqa: F401
    import torch_qaic.utils as qaic_utils  # noqa: F401
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
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
    tensorboard_updates = SummaryWriter()

    # Start the training loop
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
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
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
        )

        # enable profile for qaic
        qaic_profile.start_profiling(device, 1) if train_config.use_profiler else None

        for step, batch in enumerate(train_dataloader):
            total_train_steps += 1
            #  stop when the maximum number of training steps is reached
            if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                max_steps_reached = True
            batch = {k: v.to(device) for k, v in batch.items()}  # move the batch elements to qaic device

            with torch.autocast(
                device_type=device, dtype=torch.float16
            ) if train_config.use_autocast else nullcontext():
                loss = model(**batch).loss  # Forward call
            tensorboard_updates.add_scalars("loss", {"train": loss}, step)
            total_loss += loss.detach().float()
            # Accumalate graidents
            loss = loss / train_config.gradient_accumulation_steps

            if train_config.save_metrics:
                train_step_loss.append(loss.detach().float().item())
                train_step_perplexity.append(float(torch.exp(loss.detach().float())))

            loss.backward()  # backward pass
            if (step + 1) % train_config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

            # Save the trained checkpoints for every given steps
            if step % train_config.intermediate_step_save == 0:
                qaic_profile.stop_profiling(device) if train_config.use_profiler else None
                model.save_pretrained(train_config.output_dir + f"/trained_weights/step_{step}")

            pbar.set_description(
                f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
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

        pbar.close()
        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                model, train_config, eval_dataloader, local_rank, tokenizer, device
            )
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss

        if should_save_model:
            model.save_pretrained(train_config.output_dir)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        print(
            f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
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
    # total_eval_steps = 0
    for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        # Ensure no gradients are computed for this scope to save memory
        with torch.no_grad():
            # Forward pass and compute loss
            with torch.autocast(
                device_type=device, dtype=torch.float16
            ) if train_config.use_autocast else nullcontext():
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
