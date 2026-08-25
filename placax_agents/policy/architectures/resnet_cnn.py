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


def _normalize_channel(x: jax.Array) -> jax.Array:
    """Scales a non-negative per-cell score to roughly [0, 1], on a similar footing to 0/1 channels."""
    return x / (x.max() + 1e-6)


class _FineBranch(nn.Module):
    """MaskPlace's MyCNN: a shallow 1x1-conv stack with no receptive-field growth."""

    features: int = 8
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Each layer only looks at one cell at a time (1x1 kernel), so this refines per-cell
        # features without ever mixing in neighboring cells.
        for _ in range(self.num_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=(1, 1))(x))
        return nn.Conv(features=1, kernel_size=(1, 1))(x)


class _CoarseBranch(nn.Module):
    """Projects backbone features down to 1 channel, then resizes them onto the action grid."""

    grid_x: int
    grid_y: int
    features: int = 16

    @nn.compact
    def __call__(self, backbone_features: jax.Array) -> jax.Array:
        # 1x1 conv to squeeze the backbone's many channels down to `features`.
        x = nn.Conv(features=self.features, kernel_size=(1, 1))(backbone_features)
        # Backbone feature maps are lower-resolution than the action grid; resize up to match
        # (bilinear so it works for any grid/backbone resolution, unlike a fixed deconv chain).
        return jax.image.resize(x, (self.grid_x, self.grid_y, self.features), method="bilinear")


class ResNetCoarseFineActorCritic(nn.Module):
    """Two-branch actor-critic: a fine local branch plus a coarse ResNet-backbone branch, merged into one policy head.

    Note: the backbone always runs in eval mode (train=False) because the policy is applied to
    one observation at a time, and BatchNorm in train mode would see a batch of 1. Eval mode
    (frozen running stats) is still fully differentiable, so backbone weights still get gradients."""

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
        """Returns normalized (current, next) lookahead wiremasks, falling back to `current` if horizon==1."""
        wiremasks = obs["lookahead_wiremasks"]  # (horizon, grid_x, grid_y), horizon>=1
        current = _normalize_channel(wiremasks[0])
        # If there's no real "next" slice (horizon==1), just reuse current so shapes stay consistent.
        next_ = _normalize_channel(wiremasks[1] if wiremasks.shape[0] > 1 else wiremasks[0])
        return current, next_

    def _current_and_next_posmasks(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        """Returns illegal-position masks for the current and next macro to place."""
        # Convert the next two macros' real-unit sizes to grid units, then find where each is illegal to place.
        macro_sizes_grid = to_grid_units(obs["lookahead_sizes"][:2], self.cell_size)
        illegal = lookahead_illegal_masks(obs["canvas"], self.params, macro_sizes_grid).astype(jnp.float32)
        current = illegal[0]
        next_ = illegal[1] if illegal.shape[0] > 1 else current
        return current, next_

    def _coarse_branch_features(self, canvas: jax.Array, wiremask_cur: jax.Array, wiremask_next: jax.Array) -> jax.Array:
        """Runs the frozen-BatchNorm ResNet backbone, then the coarse-branch projection/resize."""
        # Stack the 3 input channels and add a batch dim of 1 (the backbone expects a batch axis).
        coarse_input = jnp.stack([canvas, wiremask_cur, wiremask_next], axis=-1)[None]
        backbone_out = self.resnet_backbone(coarse_input, train=False)
        backbone_features = backbone_out[self.resnet_feature_key][0]  # drop the batch dim again
        # Project the backbone's feature map down to the action grid's resolution.
        return _CoarseBranch(
            grid_x=self.params.grid_x, grid_y=self.params.effective_grid_y, features=self.coarse_features
        )(backbone_features)

    def _critic_value(self, obs: dict, fine_logits: jax.Array, coarse_features: jax.Array) -> jax.Array:
        """Computes the value estimate; "step_embedding" style shares zero params with the actor."""
        if self.critic_style == "step_embedding":
            # MaskPlace's own design: the critic gets its own small MLP over just the step index,
            # sharing no parameters with the actor. "critic_" prefix lets split_optimizer isolate it.
            emb = nn.Embed(num_embeddings=self.max_episode_macros, features=64, name="critic_step_embed")(obs["step"])
            hidden = nn.relu(nn.Dense(features=64, name="critic_hidden")(emb))
            return nn.Dense(features=1, name="critic_value")(hidden)[0]
        # Default: read the same fine/coarse features the actor uses, pooled to one vector.
        pooled = jnp.concatenate([fine_logits, coarse_features], axis=-1).mean(axis=(0, 1))
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
        coarse_features = self._coarse_branch_features(canvas, wiremask_cur, wiremask_next)
        coarse_logits = nn.Conv(features=1, kernel_size=(1, 1))(coarse_features)

        # 4. Merge both branches' per-cell logits into the final action logits.
        merged = jnp.concatenate([fine_logits, coarse_logits], axis=-1)
        action_logits = nn.Conv(features=1, kernel_size=(1, 1))(merged)[..., 0]

        # 5. Value head, using whichever critic_style this instance was configured with.
        value = self._critic_value(obs, fine_logits, coarse_features)
        return action_logits, value
