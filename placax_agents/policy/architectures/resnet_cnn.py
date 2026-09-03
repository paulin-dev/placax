"""Two-branch actor-critic mirroring MaskPlace's network: a fine local conv stack merged with a coarse ResNet-backbone branch."""
import math
import pathlib

from placax.extras.masks import lookahead_illegal_masks
from placax.types import EnvParams
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp
from flax import linen as nn


def build_untrained_resnet_backbone() -> nn.Module:
    """A randomly-initialized ResNet18 matching the pretrained one's activation-dict shape (no network call)."""
    import flaxmodels as fm  # only imported if this specific helper is called

    return fm.ResNet18(output="activations", pretrained=None)


def build_pretrained_resnet_backbone(ckpt_dir: pathlib.Path | str | None = None) -> nn.Module:
    """The real ImageNet-pretrained coarse-branch backbone; ckpt_dir optionally sets flaxmodels' weights cache dir."""
    import flaxmodels as fm  # only imported if this specific helper is called

    return fm.ResNet18(output="activations", pretrained="imagenet", ckpt_dir=str(ckpt_dir) if ckpt_dir else None)


def extract_resnet_backbone_weights(variables: dict) -> dict:
    """Pulls just the "resnet_backbone" submodule's variables out, ready for ops.checkpoint.save_checkpoint."""
    return {
        collection: variables[collection]["resnet_backbone"]
        for collection in variables
        if "resnet_backbone" in variables[collection]
    }


def load_resnet_backbone_weights(variables: dict, path: pathlib.Path) -> dict:
    """Overlays a saved backbone checkpoint onto an initialized model's "resnet_backbone" submodule only."""
    from placax_agents.ops.checkpoint import load_checkpoint

    # Build a template with just the backbone's shapes/dtypes, then load the checkpoint into it.
    backbone_template = extract_resnet_backbone_weights(variables)
    loaded_backbone = load_checkpoint(backbone_template, path)
    # Splice the loaded backbone weights back into a full copy of variables, leaving everything
    # else (fine branch, merge, critic) exactly as it was.
    return {
        collection: (
            {**variables[collection], "resnet_backbone": loaded_backbone[collection]}
            if collection in loaded_backbone
            else variables[collection]
        )
        for collection in variables
    }


def _normalize_by_shared_max(current: jax.Array, next_: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Scales (current, next) wiremasks by their *shared* max, matching MaskPlace's own
    `net_img /= max(net_img.max(), net_img_2.max())` — normalizing each independently would
    discard the relative magnitude between the two lookahead steps."""
    scale = jnp.maximum(current.max(), next_.max())
    scale = jnp.where(scale > 0, scale, 1.0)
    return current / scale, next_ / scale


def _run_backbone(module: nn.Module, x: jax.Array) -> dict:
    """A plain function wrapping the backbone call so nn.remat can wrap it.

    train=True (live batch statistics), matching MaskPlace's own PPO2.py: it never calls .eval()
    on its resnet, so BatchNorm always normalizes against each forward pass's own live mean/var
    rather than the frozen ImageNet running stats - self-correcting no matter how far the
    preceding conv weights drift during training. An eval-mode backbone was tried first (frozen
    running stats, since observations are processed one at a time) but proved fragile: as the
    conv weights fine-tune, their output no longer matches what those frozen stats expect, and
    nothing corrects the mismatch - confirmed by direct inspection to compound into a ~3.8e6-scale
    activation blowup by the backbone's last block within ~10-16 iterations, saturating the final
    softmax to exact 0/1 (2026-09-03 investigation). See ResNetCoarseFineActorCritic.apply's own
    override for how the resulting batch_stats mutation this requires gets discarded."""
    return module(x, train=True)


class _FineBranch(nn.Module):
    """MaskPlace's MyCNN: a shallow 1x1-conv stack with no receptive-field growth."""

    features: int = 8
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Each layer only looks at one cell at a time (1x1 kernel), never mixing in neighbors.
        for _ in range(self.num_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=(1, 1))(x))
        return nn.Conv(features=1, kernel_size=(1, 1))(x)


class _CoarseBranch(nn.Module):
    """Reproduces MaskPlace's MyCNNCoarse: pools backbone features to a vector, decodes a small seed, then upsamples to any (grid_x, grid_y) via learned transposed convolutions."""

    grid_x: int
    grid_y: int
    seed: int = 7
    seed_features: int = 16
    channel_schedule: tuple[int, ...] = (8, 4, 2, 1, 1)  # MaskPlace's own, past which the last count repeats

    @nn.compact
    def __call__(self, backbone_features: jax.Array) -> jax.Array:
        # 1. Global-average-pool the backbone's spatial feature map into one vector.
        pooled = backbone_features.mean(axis=(0, 1))
        # 2. Learn a small spatial seed from that pooled vector (the reference's fc(512 -> 16*7*7)).
        seed_flat = nn.Dense(features=self.seed_features * self.seed * self.seed)(pooled)
        x = seed_flat.reshape(self.seed, self.seed, self.seed_features)
        # 3. Learned transposed-conv upsampling, doubling spatial size each stage until past target.
        n_doublings = max(1, math.ceil(math.log2(max(self.grid_x, self.grid_y) / self.seed)))
        for i in range(n_doublings):
            out_features = self.channel_schedule[min(i, len(self.channel_schedule) - 1)]
            x = nn.ConvTranspose(features=out_features, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
            if i < n_doublings - 1:
                x = nn.relu(x)
        # 4. Resize to the exact target so any grid size works, not just powers of two from seed.
        return jax.image.resize(x, (self.grid_x, self.grid_y, x.shape[-1]), method="bilinear")


class ResNetCoarseFineActorCritic(nn.Module):
    """Two-branch actor-critic: a fine local branch plus a coarse ResNet-backbone branch, merged into
    one policy head. The backbone runs with live batch statistics (see _run_backbone), matching
    MaskPlace's own PPO2.py - this module's own apply() override requests and discards the resulting
    batch_stats mutation, so it's transparent to every AlgorithmFn call site (rollout/eval/training
    loops keep the plain (action_logits, value) contract, exactly as for any non-BatchNorm policy)."""

    resnet_backbone: nn.Module
    params: EnvParams
    cell_size: float
    resnet_feature_key: str = "block4_1"
    fine_features: int = 8
    fine_layers: int = 2
    coarse_seed_features: int = 16  # channel count of _CoarseBranch's learned (seed, seed) starting tensor
    critic_style: str = "canvas"  # "canvas" (default) or "step_embedding" (MaskPlace's own)
    max_episode_macros: int = 2048  # only used if critic_style == "step_embedding"

    def apply(self, variables, *args, **kwargs):
        """Overrides nn.Module.apply: the backbone's live BatchNorm (_run_backbone) needs
        mutable=["batch_stats"] to run at all, but that mutation is never read back (MaskPlace
        itself never switches its resnet to eval mode either, so its accumulated running stats
        are equally never actually used) - requesting and discarding it here, once, keeps every
        caller's (action_logits, value) contract identical to a policy with no BatchNorm at all."""
        (action_logits, value), _mutated_batch_stats = super().apply(variables, *args, mutable=["batch_stats"], **kwargs)
        return action_logits, value

    def _current_and_next_wiremasks(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        """Returns normalized (current, next) lookahead wiremasks, falling back to `current` if horizon==1."""
        wiremasks = obs["lookahead_wiremasks"]  # (horizon, grid_x, grid_y), horizon>=1
        current = wiremasks[0]
        # Reuse current as "next" when there's no real next slice (horizon==1).
        next_ = wiremasks[1] if wiremasks.shape[0] > 1 else wiremasks[0]
        return _normalize_by_shared_max(current, next_)

    def _current_and_next_posmasks(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        """Returns illegal-position masks for the current and next macro to place."""
        # Convert the next two macros' real-unit sizes to grid units, then find where each is illegal to place.
        macro_sizes_grid = to_grid_units(obs["lookahead_sizes"][:2], self.cell_size)
        illegal = lookahead_illegal_masks(obs["canvas"], self.params, macro_sizes_grid).astype(jnp.float32)
        current = illegal[0]
        next_ = illegal[1] if illegal.shape[0] > 1 else current
        return current, next_

    def _coarse_branch_features(self, canvas: jax.Array, wiremask_cur: jax.Array, wiremask_next: jax.Array) -> jax.Array:
        """Runs the live-BatchNorm ResNet backbone, then the coarse-branch projection/resize."""
        # Stack the 3 input channels and add a batch dim of 1 (the backbone expects a batch axis).
        coarse_input = jnp.stack([canvas, wiremask_cur, wiremask_next], axis=-1)[None]
        # Gradient checkpointing: trades memory for recompute on the backward pass only.
        backbone_out = nn.remat(_run_backbone)(self.resnet_backbone, coarse_input)
        backbone_features = backbone_out[self.resnet_feature_key][0]  # drop the batch dim again
        # Pool the backbone's feature map, decode a small seed, upsample to the action grid.
        return _CoarseBranch(
            grid_x=self.params.grid_x, grid_y=self.params.effective_grid_y, seed_features=self.coarse_seed_features
        )(backbone_features)

    def _critic_value(self, obs: dict, fine_logits: jax.Array, coarse_logits: jax.Array) -> jax.Array:
        """Computes the value estimate; "step_embedding" style shares zero params with the actor."""
        if self.critic_style == "step_embedding":
            # MaskPlace's own design: a small MLP over just the step index, sharing no params with the actor.
            emb = nn.Embed(num_embeddings=self.max_episode_macros, features=64, name="critic_step_embed")(obs["step"])
            hidden = nn.relu(nn.Dense(features=64, name="critic_hidden1")(emb))
            hidden = nn.relu(nn.Dense(features=64, name="critic_hidden2")(hidden))
            return nn.Dense(features=1, name="critic_value")(hidden)[0]
        # Default: read the same fine/coarse logits the actor uses, pooled to one vector.
        pooled = jnp.concatenate([fine_logits, coarse_logits], axis=-1).mean(axis=(0, 1))
        return nn.Dense(features=1, name="critic_value")(pooled)[0]

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        # 1. Gather the inputs both branches need: the canvas, and current/next wiremask + illegal-position previews.
        canvas = obs["canvas"].astype(jnp.float32)
        wiremask_cur, wiremask_next = self._current_and_next_wiremasks(obs)
        posmask_cur, posmask_next = self._current_and_next_posmasks(obs)

        # 2. Fine branch: shallow per-cell refinement over wiremask/legality channels only.
        fine_input = jnp.stack([wiremask_cur, posmask_cur, wiremask_next, posmask_next], axis=-1)
        fine_logits = _FineBranch(features=self.fine_features, num_layers=self.fine_layers)(fine_input)

        # 3. Coarse branch: wider-context features from the ResNet backbone over canvas + wiremasks.
        #    _CoarseBranch's own channel_schedule already ends at 1 feature (MyCNNCoarse's own deconv
        #    output), so unlike an earlier version of this code, there's no extra projection conv here -
        #    MaskPlace's Actor.merge takes the coarse branch's raw 1-channel output directly too.
        coarse_logits = self._coarse_branch_features(canvas, wiremask_cur, wiremask_next)

        # 4. Merge both branches' per-cell logits into the final action logits.
        merged = jnp.concatenate([fine_logits, coarse_logits], axis=-1)
        action_logits = nn.Conv(features=1, kernel_size=(1, 1))(merged)[..., 0]

        # 5. Value head, using whichever critic_style this instance was configured with.
        value = self._critic_value(obs, fine_logits, coarse_logits)
        return action_logits, value
