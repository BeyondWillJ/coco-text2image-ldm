"""
DDPM / Latent Diffusion Model (LDM) core.

Implements:
  - DDPM         – base denoising diffusion probabilistic model
  - LatentDiffusionModel – LDM that operates in the latent space of a
                           pre-trained VAE and conditions on text via CLIP.
"""

import itertools
import os
from contextlib import contextmanager

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from ldm.models.autoencoder import AutoencoderKL, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import (
    extract_into_tensor,
    make_beta_schedule,
    noise_like,
)
from ldm.modules.ema import LitEma
from ldm.util import default, exists, instantiate_from_config, mean_flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def disabled_train(self, mode=True):
    """Overwrite model.train() so that it does nothing (keeps eval mode)."""
    return self


# ---------------------------------------------------------------------------
# DDPM base
# ---------------------------------------------------------------------------

class DDPM(pl.LightningModule):
    """
    Denoising Diffusion Probabilistic Model base class.

    The forward model uses a fixed linear noise schedule.  The model is
    parameterised to predict the noise ε added at each timestep (ε-prediction).
    """

    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=None,
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        image_size=64,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        learn_logvar=False,
        logvar_init=0.0,
    ):
        super().__init__()
        assert parameterization in ("eps", "x0"), "parameterization must be 'eps' or 'x0'"
        self.parameterization = parameterization
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.use_ema = use_ema
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.v_posterior = v_posterior
        self.conditioning_key = conditioning_key
        self.loss_type = loss_type

        self.model = instantiate_from_config(unet_config)

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type
        self.learn_logvar = learn_logvar
        self.logvar = nn.Parameter(
            torch.full(fill_value=logvar_init, size=(self.num_timesteps,)),
            requires_grad=learn_logvar,
        )

        if use_ema:
            self.model_ema = LitEma(self.model)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys or [], only_model=load_only_unet)

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def register_schedule(
        self, given_betas=None, beta_schedule="linear", timesteps=1000,
        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps,
                linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_torch = lambda x: torch.tensor(x, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # Diffusion q(x_t | x_{t-1}) and posterior
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * torch.sqrt(to_torch(alphas_cumprod)) / (
                2.0 - to_torch(alphas_cumprod)
            )
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in (ignore_keys or []):
                if k.startswith(ik):
                    del sd[k]
        missing, unexpected = (
            self.model.load_state_dict(sd, strict=False)
            if only_model
            else self.load_state_dict(sd, strict=False)
        )
        print(f"Restored from {path}: {len(missing)} missing, {len(unexpected)} unexpected keys")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    # ------------------------------------------------------------------
    # Diffusion forward process
    # ------------------------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        """Sample x_t from q(x_t | x_0)."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def p_mean_variance(self, x, t, clip_denoised=True, model_kwargs=None):
        model_kwargs = model_kwargs or {}
        model_out = self.model(x, t, **model_kwargs)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, model_kwargs=None):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        noise = noise_like(x.shape, x.device, repeat=False)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, model_kwargs=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="DDPM sampling", total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, clip_denoised=self.clip_denoised, model_kwargs=model_kwargs)
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, model_kwargs=None):
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        return self.p_sample_loop(shape, model_kwargs=model_kwargs)

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
        elif self.loss_type == "l2":
            loss = F.mse_loss(target, pred, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return loss.mean() if mean else loss

    def p_losses(self, x_start, t, noise=None, model_kwargs=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_kwargs = model_kwargs or {}
        model_out = self.model(x_noisy, t, **model_kwargs)
        target = noise if self.parameterization == "eps" else x_start
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss = loss_simple + self.original_elbo_weight * loss_vlb
        return loss, {"loss_simple": loss_simple, "loss_vlb": loss_vlb}

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, "b h w c -> b c h w")
        return x.float().contiguous()

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            prog_bar=True, logger=True, on_step=True, on_epoch=True,
        )
        self.log("global_step", float(self.global_step), prog_bar=True, logger=True, on_step=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def validation_step(self, batch, batch_idx):
        _, loss_dict = self.shared_step(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            prog_bar=False, logger=True, on_step=False, on_epoch=True,
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


# ---------------------------------------------------------------------------
# LDM – Latent Diffusion Model
# ---------------------------------------------------------------------------

class LatentDiffusionModel(DDPM):
    """
    Latent Diffusion Model that:
      1. Encodes images to latent space via a frozen VAE (first_stage_model).
      2. Conditions the UNet on text embeddings via a frozen text encoder
         (cond_stage_model, e.g. FrozenCLIPEmbedder).
      3. Runs the diffusion process in the lower-dimensional latent space.
    """

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="caption",
        cond_stage_trainable=False,
        cond_stage_forward=None,
        conditioning_key="crossattn",
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.cond_stage_forward = cond_stage_forward

        self.first_stage_model = self.instantiate_first_stage(first_stage_config)
        self.cond_stage_model = self.instantiate_cond_stage(cond_stage_config)

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", torch.tensor(scale_factor))

    @torch.no_grad()
    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        return model

    def instantiate_cond_stage(self, config):
        if config == "__is_first_stage__":
            return self.first_stage_model
        if config == "__is_unconditional__":
            return None
        model = instantiate_from_config(config)
        if not self.cond_stage_trainable:
            model.eval()
            model.train = disabled_train
            for param in model.parameters():
                param.requires_grad = False
        return model

    # ------------------------------------------------------------------
    # Encode / decode helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
            else:
                c = self.cond_stage_model(c)
        else:
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_input(self, batch, k, return_first_stage_outputs=False, return_original_cond=False):
        x = super().get_input(batch, k)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        cond_key = self.cond_stage_key
        xc = batch[cond_key]
        if not self.cond_stage_trainable:
            with torch.no_grad():
                c = self.get_learned_conditioning(xc)
        else:
            c = self.get_learned_conditioning(xc)

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x, c)
        return loss, loss_dict

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond):
        """Run the UNet with cross-attention conditioning."""
        if isinstance(cond, dict):
            # Conditioning can be a dictionary of different modalities
            cond_txt = cond.get("crossattn", None)
        else:
            cond_txt = cond
        out = self.model(x_noisy, t, context=cond_txt)
        return out

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.apply_model(x_noisy, t, cond)
        target = noise if self.parameterization == "eps" else x_start
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        total = loss_simple + self.original_elbo_weight * loss_vlb
        return total, {"loss_simple": loss_simple, "loss_vlb": loss_vlb, "loss": total}

    # ------------------------------------------------------------------
    # Classifier-free guidance sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim=False, ddim_steps=50, guidance_scale=7.5, **kwargs):
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        if ddim:
            samples = self._ddim_sample(shape, cond, ddim_steps, guidance_scale)
        else:
            samples = self.p_sample_loop(shape, model_kwargs={"context": cond})
        return samples

    @torch.no_grad()
    def _ddim_sample(self, shape, cond, steps=50, guidance_scale=7.5):
        """Simplified DDIM sampler with classifier-free guidance."""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        # Build sub-sampled timestep sequence
        ddim_timesteps = torch.linspace(0, self.num_timesteps - 1, steps, dtype=torch.long, device=device)
        ddim_timesteps = ddim_timesteps.flip(0)

        # Unconditional embedding for CFG
        uc = self.get_learned_conditioning([""] * b)

        alphas = self.alphas_cumprod
        alphas_prev = self.alphas_cumprod_prev

        for i, step in enumerate(tqdm(ddim_timesteps, desc="DDIM")):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            # Get conditioning
            eps_cond = self.apply_model(img, t, cond)
            eps_uncond = self.apply_model(img, t, uc)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            a_t = extract_into_tensor(alphas, t, img.shape)
            a_prev = extract_into_tensor(
                alphas,
                torch.clamp(t - self.num_timesteps // steps, min=0),
                img.shape,
            )

            x0_pred = (img - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            img = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * eps

        return img

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            params += list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
