import copy
import functools
import math
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import io
import torch 

from diffuvqa.utils import dist_util, logger
from diffuvqa.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from diffuvqa.utils.nn import update_ema
from diffuvqa.step_sample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class _SingleGPUDDP(th.nn.Module):
    """Minimal DDP-compatible wrapper for single-GPU / CPU runs.

    gaussian_diffusion.py calls model.model.module.* assuming DDP wrapping.
    This wrapper exposes a .module attribute without requiring a process group.
    """
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            learning_steps=0,
            checkpoint_path='',
            gradient_clipping=-1.,
            eval_data=None,
            eval_interval=-1,
            warmup_steps=2000,
            lr_min=5e-6,
            use_bf16=False,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.use_bf16 = use_bf16
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping

        self.warmup_steps = warmup_steps
        self.lr_min = lr_min
        # BF16 AMP: A100 has native BF16 Tensor Cores, 2x faster than FP32.
        # BF16 rarely needs loss scaling (wider dynamic range than FP16).
        self.bf16_scaler = (
            th.cuda.amp.GradScaler(enabled=False)  # no scaling for BF16
            if use_bf16 else None
        )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self.checkpoint_path = checkpoint_path  # DEBUG **

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            # _anneal_lr will set the correct LR each step; start with base lr.
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            print(dist_util.dev())
            # gaussian_diffusion.py accesses model.model.module.* (DDP convention).
            # On single-GPU Colab dist is not initialized, so wrap with DDP(device_ids=None)
            # which gives a .module attribute without requiring a process group.
            if dist.is_initialized() and dist.get_world_size() > 1:
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
            else:
                # Single GPU: wrap in a lightweight shell so .module resolves correctly
                self.ddp_model = _SingleGPUDDP(self.model)
        else:
            self.use_ddp = False
            self.ddp_model = _SingleGPUDDP(self.model)

    def _load_and_sync_parameters(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        _is_real = main_checkpoint and str(main_checkpoint).lower() not in ('', 'none', 'false')
        if _is_real:
            # bf.exists() can silently fail on Colab Drive FUSE paths; fall back to os.path.exists
            _found = bf.exists(main_checkpoint) or os.path.exists(main_checkpoint)
            if not _found:
                raise FileNotFoundError(
                    f"resume_checkpoint specified but file not found: {main_checkpoint}\n"
                    "Check the path on Drive or clear RESUME_CHECKPOINT to train from scratch."
                )
            self.resume_step = parse_resume_step_from_filename(main_checkpoint)
            logger.log(f"loading model from checkpoint: {main_checkpoint} (resume_step={self.resume_step})")
            print(f"### Resuming from step {self.resume_step}: {main_checkpoint}")
            state_dict = dist_util.load_state_dict(main_checkpoint, map_location=dist_util.dev())
            self.model.load_state_dict(state_dict, strict=False)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    actual_model_path(ema_checkpoint), map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        if dist.is_initialized():
            dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        opt_filename = f"opt_{self.resume_step:06d}.pt"
        opt_path = bf.join(self.checkpoint_path, opt_filename)
        exists = bf.exists(opt_path) or os.path.exists(opt_path)
        if exists:
            logger.log(f"loading optimizer state from: {opt_path}")
            try:
                state_dict = dist_util.load_state_dict(opt_path, map_location=dist_util.dev())
                self.opt.load_state_dict(state_dict)
            except Exception as e:
                logger.log(f"optimizer state corrupt or unreadable ({e}), starting fresh")
        else:
            logger.log(f"no optimizer state found at {opt_path}, starting fresh")

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self, config):
        # while (
        #     not self.learning_steps
        #     or self.step + self.resume_step < self.learning_steps
        # ):
        #     image, cond = next(self.data)
        #     self.run_step(image, cond)
        #     if self.step % self.log_interval == 0:
        #         logger.dumpkvs()
        #     if self.eval_data is not None and self.step % self.eval_interval == 0:
        #         batch_eval, cond_eval = next(self.eval_data)
        #         self.forward_only(batch_eval, cond_eval)
        #         print('eval on validation set')
        #         logger.dumpkvs()
        #     if self.step > 0 and self.step % self.save_interval == 0:
        #         self.save()
        #         # Run for a finite amount of time in integration tests.
        #         if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
        #             return
        #     self.step += 1
        #     print(self.step)
        from tqdm import tqdm
        pbar = tqdm(total=self.learning_steps, initial=self.step + self.resume_step,
                    desc="Training", unit="step", dynamic_ncols=True)
        while self.step + self.resume_step < self.learning_steps:
            for image, cond in self.data:
                if self.step + self.resume_step >= self.learning_steps:
                    break
                self.run_step(image, cond)
                global_step = self.step + self.resume_step
                if self.step > 0 and global_step % self.log_interval == 0:
                    logger.dumpkvs()
                if self.eval_data is not None and self.step > 0 and global_step % self.eval_interval == 0:
                    batch_eval, cond_eval = next(self.eval_data)
                    self.forward_only(batch_eval, cond_eval)
                    logger.dumpkvs()
                if self.step > 0 and global_step % self.save_interval == 0:
                    self.save()
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        pbar.close()
                        return
                self.step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{self._last_loss:.4f}"} if hasattr(self, "_last_loss") else {})
        pbar.close()
        # Save the last checkpoint if it wasn't already saved.
        if (self.step + self.resume_step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, image, cond):
        self.forward_backward(image, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_only(self, image, cond):
        with th.no_grad():
            zero_grad(self.model_params)
            cond.pop('image_name', None)  # remove once before microbatch loop
            for i in range(0, image.shape[0], self.microbatch):
                micro_image = image[i: i + self.microbatch].to(dist_util.dev())
                micro_cond = {
                    k: v[i: i + self.microbatch].to(dist_util.dev())
                    for k, v in cond.items()
                }
                t, weights = self.schedule_sampler.sample(micro_image.shape[0], dist_util.dev())
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro_image,
                    t,
                    model_kwargs=micro_cond,
                )

                losses = compute_losses()
                loss = (losses["loss"] * weights).mean()
                log_loss_dict(
                    self.diffusion, t, {f"eval_{k}": v * weights for k, v in losses.items()}
                )
                logger.logkv("eval_loss", loss.item())

    def forward_backward(self, image, cond):
        zero_grad(self.model_params)
        cond.pop('image_name', None)  # remove once before microbatch loop
        # Cosine decay gate for pre_answer_loss: smooth 1→0 over first 150k steps.
        # Zero slope at both endpoints avoids gradient cliffs near step 130k.
        global_step = self.step + self.resume_step
        pre_answer_weight = 0.5 * (1 + math.cos(math.pi * min(global_step, 150000) / 150000))
        for i in range(0, image.shape[0], self.microbatch):
            micro_image = image[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            micro_cond['pre_answer_weight'] = pre_answer_weight
            t, weights = self.schedule_sampler.sample(micro_image.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_image,
                t,
                model_kwargs=micro_cond,
            )

            with th.autocast("cuda", dtype=th.bfloat16, enabled=self.use_bf16):
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self._last_loss = loss.item()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            elif self.use_bf16:
                self.bf16_scaler.scale(loss).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def grad_clip(self):
        # print('doing gradient clipping')
        max_grad_norm = self.gradient_clipping  #3.0
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            th.nn.utils.clip_grad_norm_(
                self.model.parameters(),  #amp.master_params(self.opt) if self.use_apex else
                max_grad_norm,
            )

    def optimize_normal(self):
        if self.use_bf16:
            if self.gradient_clipping > 0:
                self.bf16_scaler.unscale_(self.opt)
                self.grad_clip()
            self._log_grad_norm()
            self._anneal_lr()
            self.bf16_scaler.step(self.opt)
            self.bf16_scaler.update()
        else:
            if self.gradient_clipping > 0:
                self.grad_clip()
            self._log_grad_norm()
            self._anneal_lr()
            self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            # print(cnt, p) ## DEBUG
            # print(cnt, p.grad)
            # cnt += 1
            if p.grad != None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        global_step = self.step + self.resume_step
        if global_step < self.warmup_steps:
            # Linear warmup: 0 → lr
            lr = self.lr * global_step / max(1, self.warmup_steps)
        else:
            # Cosine decay: lr → lr_min
            decay_steps = self.learning_steps - self.warmup_steps
            progress = (global_step - self.warmup_steps) / max(1, decay_steps)
            progress = min(progress, 1.0)
            lr = self.lr_min + 0.5 * (self.lr - self.lr_min) * (1.0 + math.cos(math.pi * progress))
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    
    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            print('writing to', bf.join(get_blob_logdir(), filename))
            print('writing to', bf.join(self.checkpoint_path, filename))
            # with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
            #     th.save(state_dict, f)
            with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:  # DEBUG **
                th.save(state_dict, f)  # save locally
                # pass # save empty

        # save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        opt_filename = f"opt_{(self.step + self.resume_step):06d}.pt"
        opt_path = bf.join(self.checkpoint_path, opt_filename)
        logger.log(f"saving optimizer state to {opt_path}...")
        with bf.BlobFile(opt_path, "wb") as f:
            th.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params  # DEBUG **
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        # Fall back to current model param for keys absent in checkpoint (e.g. newly
        # added question_type_head when resuming from a pre-classifier checkpoint).
        params = [
            state_dict[name] if name in state_dict else param.data
            for name, param in self.model.named_parameters()
        ]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    if filename[-3:] == '.pt':
        return int(filename[-9:-3])
    else:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def actual_model_path(model_path):
    return model_path
