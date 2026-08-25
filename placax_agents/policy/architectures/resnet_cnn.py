"""Two-branch actor-critic mirroring MaskPlace's own network: a fine
local conv stack over wiremask/legality channels, merged with a coarse
branch built on an injected ResNet backbone (optionally ImageNet-
pretrained) over (canvas, wiremask, next-macro wiremask). The network-
capacity axis flagged as a real, deliberate gap in the placax-vs-
MaskPlace comparison - one more architecture option (see cnn.py,
wiremask_cnn.py), not a replacement for either.

The backbone is injected, not baked in - this module never imports
flaxmodels itself, so importing it costs nothing extra. Three ways to
get one, in increasing order of fidelity: build_untrained_resnet_backbone()
(offline, no network, used by this module's own tests), a directly-
constructed flaxmodels.ResNet18(pretrained="imagenet") (downloads
~45MB the first time), or build_pretrained_resnet_backbone() below,
which wraps that with a `ckpt_dir`/cache story. Once you have an
initialized policy, load_resnet_backbone_weights() below overlays any
locally saved backbone checkpoint (your own fine-tuned run, a converted
torchvision export, ...) onto just the backbone's variables - a fully
general "pass a weights path" that doesn't depend on flaxmodels' own
download/caching at all. All three of these need `placax[resnet]`;
importing this module itself never does.

Needs obs to carry "canvas", "wiremask", "lookahead_wiremasks"
(horizon>=2), and "lookahead_sizes" (horizon>=2) - pair with
policy.observation.make_wiremask_observation(..., lookahead=2)."""
import pathlib

from placax.extras.masks import lookahead_illegal_masks
from placax.types import EnvParams
from placax_agents.policy.scale import to_grid_units

import jax
import jax.numpy as jnp
from flax import linen as nn


def build_untrained_resnet_backbone() -> nn.Module:
    """A ResNet18 with the exact activation-dict shape a real pretrained
    one has, weights randomly initialized instead of downloaded - no
    extra dependency behavior beyond flaxmodels itself, no network call.
    What this module's own tests use; swap for
    build_pretrained_resnet_backbone() (or a directly-constructed
    flaxmodels.ResNet18(pretrained="imagenet")) for the real thing."""
    import flaxmodels as fm  # only imported if this specific helper is called

    return fm.ResNet18(output="activations", pretrained=None)


def build_pretrained_resnet_backbone(ckpt_dir: pathlib.Path | str | None = None) -> nn.Module:
    """The real, ImageNet-pretrained coarse-branch backbone. ckpt_dir, if
    given, is where flaxmodels looks for (and caches) the weights file
    (`{ckpt_dir}/flaxmodels/resnet18_weights.h5`) - pre-place that file
    there yourself to skip the ~45MB download entirely; leave it as None
    to use flaxmodels' own temp-directory cache. For weights that don't
    come from flaxmodels at all (your own checkpoint, a converted
    torchvision export), build with build_untrained_resnet_backbone()
    instead and overlay them post-init with load_resnet_backbone_weights()
    below - that path accepts an arbitrary local file, not just
    flaxmodels' one fixed filename."""
    import flaxmodels as fm  # only imported if this specific helper is called

    return fm.ResNet18(output="activations", pretrained="imagenet", ckpt_dir=str(ckpt_dir) if ckpt_dir else None)


def extract_resnet_backbone_weights(variables: dict) -> dict:
    """The inverse of load_resnet_backbone_weights: pulls just the
    "resnet_backbone" submodule's variables out (one entry per
    collection - params, batch_stats, ...), ready to hand to
    ops.checkpoint.save_checkpoint. Round-trips with
    load_resnet_backbone_weights - save this, and it loads back in."""
    return {
        collection: variables[collection]["resnet_backbone"]
        for collection in variables
        if "resnet_backbone" in variables[collection]
    }


def load_resnet_backbone_weights(variables: dict, path: pathlib.Path) -> dict:
    """Overlays a saved checkpoint (from extract_resnet_backbone_weights()
    + ops.checkpoint.save_checkpoint - any local file, any origin) onto
    just the "resnet_backbone" submodule of an already-initialized
    ResNetCoarseFineActorCritic's variables, leaving the fine branch/
    merge/critic untouched. General on purpose: works for a previously
    fine-tuned placax checkpoint, a hand-converted torchvision export, or
    anything else with the same backbone shape - not tied to flaxmodels'
    own pretrained-weights mechanism at all."""
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
    """MaskPlace's MyCNN: a shallow 1x1-conv stack - no receptive-field
    growth beyond what's already in each input channel."""

    features: int = 8
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for _ in range(self.num_layers):
            x = nn.relu(nn.Conv(features=self.features, kernel_size=(1, 1))(x))
        return nn.Conv(features=1, kernel_size=(1, 1))(x)


class _CoarseBranch(nn.Module):
    """Backbone features -> one (grid_x, grid_y) map: a 1x1 projection,
    then resize to the action grid. MaskPlace's MyCNNCoarse used a fixed
    deconv chain that only matches grid=224 (7x2^5); resize generalizes
    to any grid size and any backbone output resolution."""

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
    parameters with the actor (MaskPlace's own design - see
    algorithm.split_optimizer for genuinely independent per-network
    optimization built on that, matching MaskPlace's two separate
    optimizers without a second backward pass).

    The backbone always runs in eval mode (train=False), fixed rather
    than a config knob - not a missing feature, a correctness constraint:
    every call site in this codebase (collect_rollout, ppo_loss,
    evaluate) applies the policy to one observation at a time (even under
    vmap, which traces a per-example function - the batch dimension is
    added by the transform, never seen inside it), so BatchNorm would see
    a batch of 1 in train mode and compute degenerate statistics. Eval
    mode uses the backbone's frozen running stats instead, and is fully
    differentiable regardless - gradients still flow into the backbone's
    conv weights during training, this only fixes how its BatchNorm
    layers behave."""

    resnet_backbone: nn.Module
    params: EnvParams
    cell_size: float
    resnet_feature_key: str = "block4_1"
    fine_features: int = 8
    fine_layers: int = 2
    coarse_features: int = 16
    critic_style: str = "canvas"  # "canvas" (default) or "step_embedding" (MaskPlace's own)
    max_episode_macros: int = 2048  # only used if critic_style == "step_embedding"

    @nn.compact
    def __call__(self, obs: dict) -> tuple[jax.Array, jax.Array]:
        canvas = obs["canvas"].astype(jnp.float32)
        wiremasks = obs["lookahead_wiremasks"]  # (horizon, grid_x, grid_y), horizon>=1
        wiremask_cur = _normalize_channel(wiremasks[0])
        wiremask_next = _normalize_channel(wiremasks[1] if wiremasks.shape[0] > 1 else wiremasks[0])

        macro_sizes_grid = to_grid_units(obs["lookahead_sizes"][:2], self.cell_size)
        illegal = lookahead_illegal_masks(obs["canvas"], self.params, macro_sizes_grid).astype(jnp.float32)
        posmask_cur = illegal[0]
        posmask_next = illegal[1] if illegal.shape[0] > 1 else posmask_cur

        fine_input = jnp.stack([wiremask_cur, posmask_cur, wiremask_next, posmask_next], axis=-1)
        fine_logits = _FineBranch(features=self.fine_features, num_layers=self.fine_layers)(fine_input)

        coarse_input = jnp.stack([canvas, wiremask_cur, wiremask_next], axis=-1)[None]  # add batch dim
        backbone_out = self.resnet_backbone(coarse_input, train=False)
        backbone_features = backbone_out[self.resnet_feature_key][0]  # drop batch dim
        coarse_features = _CoarseBranch(
            grid_x=self.params.grid_x, grid_y=self.params.effective_grid_y, features=self.coarse_features
        )(backbone_features)
        coarse_logits = nn.Conv(features=1, kernel_size=(1, 1))(coarse_features)

        merged = jnp.concatenate([fine_logits, coarse_logits], axis=-1)
        action_logits = nn.Conv(features=1, kernel_size=(1, 1))(merged)[..., 0]

        if self.critic_style == "step_embedding":
            # MaskPlace's own critic: a value keyed purely on how many macros are placed, no canvas
            # input - shares zero parameters with the actor computed above. Named with a "critic_"
            # prefix so algorithm.split_optimizer.label_params_by_name_prefix can isolate this group
            # from the actor's, e.g. for genuinely independent per-network optimization.
            emb = nn.Embed(num_embeddings=self.max_episode_macros, features=64, name="critic_step_embed")(obs["step"])
            hidden = nn.relu(nn.Dense(features=64, name="critic_hidden")(emb))
            value = nn.Dense(features=1, name="critic_value")(hidden)[0]
        else:
            # "canvas" style: value head is a separate parameter, but it reads a trunk shared with
            # the actor - naming it "critic_..." only isolates its own final projection, not a
            # genuinely independent network. Use critic_style="step_embedding" for real disjointness.
            pooled = jnp.concatenate([fine_logits, coarse_features], axis=-1).mean(axis=(0, 1))
            value = nn.Dense(features=1, name="critic_value")(pooled)[0]
        return action_logits, value
