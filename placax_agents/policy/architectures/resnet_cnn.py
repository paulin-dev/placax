"""Two-branch actor-critic mirroring MaskPlace's own network: a fine
local conv stack over wiremask/legality channels, merged with a coarse
branch built on an injected ResNet backbone over (canvas, wiremask,
next-macro wiremask).

The backbone is injected, not baked in, so importing this module never
requires `placax[resnet]` (only the backbone-building/loading helpers do).
Needs obs to carry "canvas", "wiremask", "lookahead_wiremasks" (horizon>=2),
and "lookahead_sizes" (horizon>=2) - pair with
policy.observation.make_wiremask_observation(..., lookahead=2)."""
import pathlib

from placax.extras.masks import lookahead_illegal_masks
from placax.types import EnvParams
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp
from flax import linen as nn


def build_untrained_resnet_backbone() -> nn.Module:
    """A ResNet18 with the same activation-dict shape as the pretrained
    one, but randomly initialized - no network call. Used by this
    module's own tests; swap for build_pretrained_resnet_backbone() for
    the real thing."""
    import flaxmodels as fm  # only imported if this specific helper is called

    return fm.ResNet18(output="activations", pretrained=None)


def build_pretrained_resnet_backbone(ckpt_dir: pathlib.Path | str | None = None) -> nn.Module:
    """The real, ImageNet-pretrained coarse-branch backbone. ckpt_dir, if
    given, is where flaxmodels caches the weights file (pre-place it
    there to skip the ~45MB download); None uses flaxmodels' own temp
    cache. For non-flaxmodels weights, use build_untrained_resnet_backbone()
    + load_resnet_backbone_weights() instead."""
    import flaxmodels as fm  # only imported if this specific helper is called

    return fm.ResNet18(output="activations", pretrained="imagenet", ckpt_dir=str(ckpt_dir) if ckpt_dir else None)


def extract_resnet_backbone_weights(variables: dict) -> dict:
    """Pulls just the "resnet_backbone" submodule's variables out (one
    entry per collection), ready for ops.checkpoint.save_checkpoint.
    Inverse of load_resnet_backbone_weights."""
    return {
        collection: variables[collection]["resnet_backbone"]
        for collection in variables
        if "resnet_backbone" in variables[collection]
    }


def load_resnet_backbone_weights(variables: dict, path: pathlib.Path) -> dict:
    """Overlays a saved checkpoint (from extract_resnet_backbone_weights())
    onto just the "resnet_backbone" submodule of an already-initialized
    ResNetCoarseFineActorCritic's variables, leaving the fine branch/
    merge/critic untouched."""
    from placax_agents.ops.checkpoint import load_checkpoint

    backbone_template = extract_resnet_backbone_weights(variables)
    loaded_backbone = load_checkpoint(backbone_template, path)
    return {
        collection: (
            {**variables[collection], "resnet_backbone": loaded_backbone[collection]}
            if collection in loaded_backbone
            else variables[collection]
        )
        for collection in variables
    }


def _normalize_channel(x: jax.Array) -> jax.Array:
    """Scales a non-negative per-cell score to roughly [0, 1] - keeps
    wiremask on a similar footing to the other (already 0/1) channels."""
    return x / (x.max() + 1e-6)


class _FineBranch(nn.Module):
    """MaskPlace's MyCNN: a shallow 1x1-conv stack, no receptive-field
    growth."""

    features: int = 8
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.num_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=(1, 1))(x))
        return nn.Conv(features=1, kernel_size=(1, 1))(x)


class _CoarseBranch(nn.Module):
    """Backbone features -> one (grid_x, grid_y) map: 1x1 projection then
    resize to the action grid (generalizes to any grid/backbone
    resolution, unlike MaskPlace's fixed deconv chain)."""

    grid_x: int
    grid_y: int
    features: int = 16

    @nn.compact
    def __call__(self, backbone_features: jax.Array) -> jax.Array:
        x = nn.Conv(features=self.features, kernel_size=(1, 1))(backbone_features)
        return jax.image.resize(x, (self.grid_x, self.grid_y, self.features), method="bilinear")


class ResNetCoarseFineActorCritic(nn.Module):
    """obs dict -> (action_logits (grid_x, grid_y), value scalar).

    critic_style="step_embedding" gives a critic with zero shared
    parameters with the actor (MaskPlace's own design - pairs with
    algorithm.split_optimizer for independent per-network optimization).

    The backbone always runs in eval mode (train=False): every call site
    applies the policy to one observation at a time, so BatchNorm in
    train mode would see a batch of 1 and compute degenerate statistics.
    Eval mode (frozen running stats) is still fully differentiable -
    gradients still flow into the backbone's conv weights."""

    resnet_backbone: nn.Module
    params: EnvParams
    cell_size: float
    resnet_feature_key: str = "block4_1"
    fine_features: int = 8
    fine_layers: int = 2
    coarse_features: int = 16
    critic_style: str = "canvas"  # "canvas" (default) or "step_embedding" (MaskPlace's own)
    max_episode_macros: int = 2048  # only used if critic_style == "step_embedding"

    def _current_and_next_wiremasks(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        """Normalized (current, next) lookahead wiremasks - falls back to `current` if horizon==1."""
        wiremasks = obs["lookahead_wiremasks"]  # (horizon, grid_x, grid_y), horizon>=1
        current = _normalize_channel(wiremasks[0])
        next_ = _normalize_channel(wiremasks[1] if wiremasks.shape[0] > 1 else wiremasks[0])
        return current, next_

    def _current_and_next_posmasks(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        """Illegal-position masks for the current and next macro to place."""
        macro_sizes_grid = to_grid_units(obs["lookahead_sizes"][:2], self.cell_size)
        illegal = lookahead_illegal_masks(obs["canvas"], self.params, macro_sizes_grid).astype(jnp.float32)
        current = illegal[0]
        next_ = illegal[1] if illegal.shape[0] > 1 else current
        return current, next_

    def _coarse_branch_features(self, canvas: jax.Array, wiremask_cur: jax.Array, wiremask_next: jax.Array) -> jax.Array:
        """Runs the ResNet backbone (frozen BatchNorm stats) then the coarse-branch projection/resize."""
        coarse_input = jnp.stack([canvas, wiremask_cur, wiremask_next], axis=-1)[None]  # add batch dim
        backbone_out = self.resnet_backbone(coarse_input, train=False)
        backbone_features = backbone_out[self.resnet_feature_key][0]  # drop batch dim
        return _CoarseBranch(
            grid_x=self.params.grid_x, grid_y=self.params.effective_grid_y, features=self.coarse_features
        )(backbone_features)

    def _critic_value(self, obs: dict, fine_logits: jax.Array, coarse_features: jax.Array) -> jax.Array:
        """Value head; "step_embedding" shares zero params with the actor (MaskPlace's own design)."""
        if self.critic_style == "step_embedding":
            # "critic_" prefix lets split_optimizer.label_params_by_name_prefix isolate this group.
            emb = nn.Embed(num_embeddings=self.max_episode_macros, features=64, name="critic_step_embed")(obs["step"])
            hidden = nn.relu(nn.Dense(features=64, name="critic_hidden")(emb))
            return nn.Dense(features=1, name="critic_value")(hidden)[0]
        # value head reads a trunk shared with the actor - only its own projection is isolated.
        pooled = jnp.concatenate([fine_logits, coarse_features], axis=-1).mean(axis=(0, 1))
        return nn.Dense(features=1, name="critic_value")(pooled)[0]

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        canvas = obs["canvas"].astype(jnp.float32)
        wiremask_cur, wiremask_next = self._current_and_next_wiremasks(obs)
        posmask_cur, posmask_next = self._current_and_next_posmasks(obs)

        fine_input = jnp.stack([wiremask_cur, posmask_cur, wiremask_next, posmask_next], axis=-1)
        fine_logits = _FineBranch(features=self.fine_features, num_layers=self.fine_layers)(fine_input)

        coarse_features = self._coarse_branch_features(canvas, wiremask_cur, wiremask_next)
        coarse_logits = nn.Conv(features=1, kernel_size=(1, 1))(coarse_features)

        merged = jnp.concatenate([fine_logits, coarse_logits], axis=-1)
        action_logits = nn.Conv(features=1, kernel_size=(1, 1))(merged)[..., 0]

        value = self._critic_value(obs, fine_logits, coarse_features)
        return action_logits, value
