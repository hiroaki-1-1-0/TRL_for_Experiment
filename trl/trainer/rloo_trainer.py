# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_rich_available

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from .rloo_config import RLOOConfig
from .utils import empty_cache, generate_model_card, get_comet_experiment_url, log_table_to_comet_experiment


if is_wandb_available():
    import wandb
import pandas as pd
import math
import time
import gc
import torch
import torch.nn as nn

INVALID_LOGPROB = 1.0


class RLOOTrainer(Trainer):
    _tag_names = ["trl", "rloo"]

    def _get_model_device(self, model):
        """
        Get the device of a model safely, handling wrapped models
        """
        try:
            # Handle different types of wrapped models
            if hasattr(model, 'module'):  # DistributedDataParallel or similar
                return next(model.module.parameters()).device
            elif hasattr(model, '_modules') and len(model._modules) > 0:  # Regular module
                return next(model.parameters()).device
            else:  # Fallback for special cases
                return next(model.parameters()).device
        except (StopIteration, AttributeError):
            # Fallback to accelerator device if model has no parameters or other issues
            if hasattr(self, 'accelerator') and hasattr(self.accelerator, 'device'):
                return self.accelerator.device
            return torch.device('cuda:0')  # final fallback

    def __init__(
        self,
        config: RLOOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: Union[nn.Module, Callable[[list[str]], list[float]]],
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy

        # Multi-GPU device mapping setup - auto-detect from models
        self.policy_device = self._get_model_device(policy)
        self.ref_policy_device = self._get_model_device(ref_policy) 
        if isinstance(reward_model, nn.Module):
            self.reward_model_device = self._get_model_device(reward_model)
        else:
            self.reward_model_device = self.policy_device

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again
        
        # Update policy device info after accelerator.prepare
        self.policy_device = self._get_model_device(self.model)

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
                # Update device info after DeepSpeed wrapping
                self.reward_model_device = self._get_model_device(self.reward_model)
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            # Update device info after DeepSpeed wrapping
            self.ref_policy_device = self._get_model_device(self.ref_policy)
            self.deepspeed = self.model

    def _move_tensors_to_device(self, tensors, target_device):
        """
        Safely move tensors to target device, handling various tensor types
        """
        if tensors is None:
            return tensors
        
        if torch.is_tensor(tensors):
            return tensors.to(target_device, non_blocking=True)
        elif isinstance(tensors, dict):
            return {k: self._move_tensors_to_device(v, target_device) for k, v in tensors.items()}
        elif isinstance(tensors, (list, tuple)):
            return type(tensors)(self._move_tensors_to_device(item, target_device) for item in tensors)
        else:
            return tensors

    def _safe_forward_on_device(self, model, input_data, target_device, pad_token_id):
        """
        Safely perform forward pass on specific device
        """
        # Ensure model is on correct device first
        if isinstance(model, nn.Module):
            model_device = self._get_model_device(model)
            target_device = model_device  # Always use model's device
        
        # Move input data to model's device
        input_data_on_device = self._move_tensors_to_device(input_data, target_device)
        
        # Perform forward pass
        with torch.cuda.device(target_device):
            if torch.is_tensor(input_data_on_device):
                attention_mask = (input_data_on_device != pad_token_id).long().to(target_device)
                output = model(input_ids=input_data_on_device, attention_mask=attention_mask)
            else:
                # Ensure all components of input_data are on the same device
                input_data_on_device = {k: v.to(target_device) if torch.is_tensor(v) else v 
                                      for k, v in input_data_on_device.items()}
                output = model(**input_data_on_device)
        
        return output

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                queries = queries.repeat(args.rloo_k, 1)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []

                # Generate responses and compute logprobs
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # Process responses in batches
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    empty_cache()

                    # Reference policy forward pass with device management
                    ref_policy_device = self._get_model_device(ref_policy)
                    query_response_for_ref = self._move_tensors_to_device(query_response, ref_policy_device)
                    
                    ref_output = self._safe_forward_on_device(
                        ref_policy, query_response_for_ref, ref_policy_device, processing_class.pad_token_id
                    )
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response.to(ref_logits.device))
                    
                    # Move results back to main device if needed
                    ref_logprob = ref_logprob.to(device)
                    
                    del ref_output, ref_logits
                    empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses with device management
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

                    if isinstance(reward_model, nn.Module):
                        reward_model_device = self._get_model_device(reward_model)
                        postprocessed_query_response_for_reward = self._move_tensors_to_device(
                            postprocessed_query_response, reward_model_device
                        )
                        
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response_for_reward, 
                            processing_class.pad_token_id, context_length
                        )
                        
                        # Move score back to main device
                        score = score.to(device)
                    else:
                        score = torch.tensor(
                            reward_model(
                                processing_class.batch_decode(postprocessed_query_response, skip_special_tokens=True)
                            ),
                            dtype=torch.float,
                        ).to(device)

                    # Store batch results (ensure all tensors are on main device)
                    responses.append(response.to(device))
                    postprocessed_responses.append(postprocessed_response.to(device))
                    logprobs.append(logprob.to(device))
                    ref_logprobs.append(ref_logprob.to(device))
                    sequence_lengths.append(sequence_length.to(device))
                    scores.append(score.to(device))

                # Concatenate all batched results
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, score)
                empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                # Compute KL divergence
                kl = logprobs - ref_logprobs

                # Normalize rewards
                if args.normalize_reward:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                    scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                # Compute total reward with KL penalty
                if args.token_level_kl:
                    # Token-level KL penalty: apply KL penalty per token
                    kl_reward = -args.kl_coef * kl

                    # Get the index of the last non-padded token for each sequence
                    eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    last_reward = torch.zeros_like(kl)
                    # Ensure scores has correct shape and type
                    scores_shaped = scores.reshape(-1, 1).to(kl.dtype)
                    last_reward.scatter_(dim=1, index=eos_indices, src=scores_shaped)

                    # Combine KL reward and last reward
                    non_score_reward = kl_reward.sum(1)  # Keep this for logging
                    reward = last_reward + kl_reward
                    rlhf_reward = reward.sum(1)  # Sum across sequence length
                else:
                    # Sequence-level KL penalty: sum KL across tokens first
                    sequence_kl = kl.sum(1)
                    non_score_reward = -args.kl_coef * sequence_kl
                    rlhf_reward = non_score_reward + scores

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()

                # Normalize advantages
                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # Get batch data and ensure they are on the correct device
                            mb_advantage = advantages[micro_batch_inds].to(device)
                            mb_responses = responses[micro_batch_inds].to(device)
                            mb_query_responses = query_responses[micro_batch_inds].to(device)
                            mb_logprobs = logprobs[micro_batch_inds].to(device)

                            # Forward pass with device management
                            policy_device = self._get_model_device(model)
                            mb_query_responses_for_policy = self._move_tensors_to_device(mb_query_responses, policy_device)
                            
                            output = self._safe_forward_on_device(
                                model, mb_query_responses_for_policy, policy_device, processing_class.pad_token_id
                            )
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7

                            # Compute new logprobs with device consistency
                            mb_responses_for_logits = mb_responses.to(logits.device)
                            new_logprobs = selective_log_softmax(logits, mb_responses_for_logits)
                            
                            # Ensure padding mask is on correct device
                            padding_mask_micro = padding_mask[micro_batch_inds].to(new_logprobs.device)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask_micro, INVALID_LOGPROB
                            )

                            # Move results back to main device for computation
                            new_logprobs = new_logprobs.to(device)
                            logits = logits.to(device)

                            # Compute probability ratios
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            # PPO clipped loss
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()

                            # Final loss
                            loss = pg_loss

                            # Optimization step
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1

                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    empty_cache()

            # Compute metrics
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                try:
                    self.generate_completions(sampling=True)
                except Exception as e:
                    # Log the error but continue training
                    if self.accelerator.is_main_process:
                        print(f"⚠️ Sample generation failed: {e}")
                        print("Continuing with training...")

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)

                    if isinstance(self.reward_model, nn.Module):
                        # Move input to reward model device for proper computation
                        reward_model_device = self._get_model_device(self.reward_model)
                        
                        # Ensure input is on the same device as reward model
                        postprocessed_query_response_on_reward_device = postprocessed_query_response.to(reward_model_device)
                        
                        _, score, _ = get_reward(
                            self.reward_model,
                            postprocessed_query_response_on_reward_device,
                            processing_class.pad_token_id,
                            context_length,
                        )
                        # Move score back to the original device for consistency
                        score = score.to(postprocessed_query_response.device)
                    else:
                        score = torch.tensor(
                            self.reward_model(
                                processing_class.batch_decode(postprocessed_query_response, skip_special_tokens=True)
                            ),
                            dtype=torch.float,
                        ).to(postprocessed_query_response.device)
                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            if is_rich_available():
                print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        
        # Handle shared tensor issue with safetensors
        try:
            super()._save_checkpoint(model, trial)
        except RuntimeError as e:
            if "Some tensors share memory" in str(e):
                print(f"⚠️ Safetensors memory sharing issue detected. Attempting tensor unsharing...")
                
                # Try to fix the shared tensor issue by cloning shared weights
                unwrapped_model = self._get_unwrapped_model()
                
                # Check if lm_head and embeddings share memory (common in transformer models)
                if (hasattr(unwrapped_model, 'lm_head') and 
                    hasattr(unwrapped_model, 'model') and 
                    hasattr(unwrapped_model.model, 'embed_tokens')):
                    
                    # Check if they share memory
                    lm_head_ptr = unwrapped_model.lm_head.weight.data_ptr()
                    embed_ptr = unwrapped_model.model.embed_tokens.weight.data_ptr()
                    
                    if lm_head_ptr == embed_ptr:
                        print("🔧 Fixing shared memory between lm_head and embed_tokens...")
                        # Clone the lm_head weight to break the sharing
                        unwrapped_model.lm_head.weight = nn.Parameter(
                            unwrapped_model.lm_head.weight.clone()
                        )
                        print("✅ Tensor sharing resolved")
                
                # Try saving again with fixed tensors
                try:
                    super()._save_checkpoint(model, trial)
                    print("✅ Successfully saved with tensor unsharing")
                except RuntimeError as e2:
                    # Fallback to PyTorch format if still failing
                    print(f"⚠️ Still having issues, using PyTorch format fallback...")
                    
                    if self.is_world_process_zero():
                        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.state.global_step}")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save with safe_serialization=False to avoid safetensors issues
                        unwrapped_model.save_pretrained(
                            output_dir,
                            safe_serialization=False,  # Use PyTorch format instead of safetensors
                            max_shard_size="5GB"
                        )
                        
                        # Save tokenizer if available
                        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                            self.tokenizer.save_pretrained(output_dir)
                        elif hasattr(self, 'processing_class') and self.processing_class is not None:
                            self.processing_class.save_pretrained(output_dir)
                        
                        print(f"✅ Model saved to {output_dir} using PyTorch format")
            else:
                raise e

    def _get_unwrapped_model(self):
        """
        Get the unwrapped model for accessing config and other attributes
        """
        if hasattr(self.model, 'module'):
            # DistributedDataParallel case
            return self.model.module
        else:
            # Regular model case
            return self.model

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with handling for shared tensor issues.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        try:
            # Try the normal save first
            super().save_model(output_dir, _internal_call)
        except RuntimeError as e:
            if "Some tensors share memory" in str(e):
                print(f"⚠️ Safetensors memory sharing issue detected during save_model. Attempting tensor unsharing...")
                
                # Try to fix the shared tensor issue by cloning shared weights
                unwrapped_model = self._get_unwrapped_model()
                
                # Check if lm_head and embeddings share memory (common in transformer models)
                if (hasattr(unwrapped_model, 'lm_head') and 
                    hasattr(unwrapped_model, 'model') and 
                    hasattr(unwrapped_model.model, 'embed_tokens')):
                    
                    # Check if they share memory
                    lm_head_ptr = unwrapped_model.lm_head.weight.data_ptr()
                    embed_ptr = unwrapped_model.model.embed_tokens.weight.data_ptr()
                    
                    if lm_head_ptr == embed_ptr:
                        print("🔧 Fixing shared memory between lm_head and embed_tokens...")
                        # Clone the lm_head weight to break the sharing
                        unwrapped_model.lm_head.weight = nn.Parameter(
                            unwrapped_model.lm_head.weight.clone()
                        )
                        print("✅ Tensor sharing resolved")
                
                # Try saving again with fixed tensors
                try:
                    super().save_model(output_dir, _internal_call)
                    print("✅ Successfully saved with tensor unsharing")
                except RuntimeError as e2:
                    # Fallback to PyTorch format if still failing
                    print(f"⚠️ Still having issues, using PyTorch format fallback...")
                    
                    if self.is_world_process_zero():
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save with PyTorch format instead of safetensors
                        unwrapped_model.save_pretrained(
                            output_dir,
                            safe_serialization=False,  # Use PyTorch format
                            max_shard_size="5GB"
                        )
                        
                        # Save tokenizer if available
                        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                            self.tokenizer.save_pretrained(output_dir)
                        elif hasattr(self, 'processing_class') and self.processing_class is not None:
                            self.processing_class.save_pretrained(output_dir)
                        
                        print(f"✅ Model saved to {output_dir} using PyTorch format (shared tensor workaround)")
            else:
                raise e

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        # Get unwrapped model for config access
        unwrapped_model = self._get_unwrapped_model()

        if hasattr(unwrapped_model.config, "_name_or_path") and not os.path.isdir(unwrapped_model.config._name_or_path):
            base_model = unwrapped_model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(unwrapped_model.config, "unsloth_version"):
            tags.add("unsloth")

        tags.update(self._tag_names)

        citation = textwrap.dedent("""\
        @inproceedings{ahmadian2024back,
            title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
            author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{\'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {\"{U}}st{\"{u}}n and Sara Hooker},
            year         = 2024,
            booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
            publisher    = {Association for Computational Linguistics},
            pages        = {12248--12267},
            editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="RLOO",
            trainer_citation=citation,
            paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
            paper_id="2402.14740",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
