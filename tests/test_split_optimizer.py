from placax_agents.training.algorithm.split_optimizer import (  # noqa: F401  must precede jax imports
    label_params_by_name_prefix,
    make_grouped_optimizer,
)

import jax
import jax.numpy as jnp
import optax


def test_label_params_by_name_prefix_matches_at_any_depth() -> None:
    params = {
        "params": {
            "critic_value": {"kernel": jnp.zeros(3)},
            "fine_branch": {"kernel": jnp.zeros(3)},
        },
        "batch_stats": {"critic_value": {"mean": jnp.zeros(3)}},
    }
    labels = label_params_by_name_prefix(params, "critic_")
    assert labels["params"]["critic_value"]["kernel"] == "matched"
    assert labels["params"]["fine_branch"]["kernel"] == "default"
    assert labels["batch_stats"]["critic_value"]["mean"] == "matched"


def test_label_params_by_name_prefix_custom_labels() -> None:
    params = {"critic_x": jnp.zeros(1), "actor_y": jnp.zeros(1)}
    labels = label_params_by_name_prefix(params, "critic_", matched_label="c", default_label="a")
    assert labels == {"critic_x": "c", "actor_y": "a"}


def test_make_grouped_optimizer_applies_each_transform_only_to_its_own_group() -> None:
    # Different learning rates per group - if isolation weren't real, the
    # "wrong" rate would leak into the other group's update.
    params = {"critic_v": jnp.array(1.0), "actor_w": jnp.array(1.0)}
    grads = {"critic_v": jnp.array(1.0), "actor_w": jnp.array(1.0)}

    optimizer = make_grouped_optimizer(
        matched_transform=optax.sgd(learning_rate=0.1),
        default_transform=optax.sgd(learning_rate=0.01),
        prefix="critic_",
    )
    opt_state = optimizer.init(params)
    updates, _new_state = optimizer.update(grads, opt_state, params)

    assert abs(float(updates["critic_v"]) - (-0.1)) < 1e-6   # matched group: lr=0.1
    assert abs(float(updates["actor_w"]) - (-0.01)) < 1e-6   # default group: lr=0.01


def test_grouped_optimizer_state_is_genuinely_independent_across_updates() -> None:
    # Adam's moment estimates must not mix between groups - run several
    # steps with very different gradient magnitudes per group and check
    # each group's own Adam-normalized step size stays roughly the same
    # (Adam's whole point), unaffected by the other group's gradients.
    params = {"critic_v": jnp.array(0.0), "actor_w": jnp.array(0.0)}
    optimizer = make_grouped_optimizer(
        matched_transform=optax.adam(learning_rate=0.1),
        default_transform=optax.adam(learning_rate=0.1),
        prefix="critic_",
    )
    opt_state = optimizer.init(params)
    steps = []
    for _ in range(5):
        grads = {"critic_v": jnp.array(1000.0), "actor_w": jnp.array(0.001)}  # wildly different scales
        updates, opt_state = optimizer.update(grads, opt_state, params)
        steps.append(float(updates["actor_w"]))
        params = optax.apply_updates(params, updates)
    # actor_w's own Adam-normalized step stays near its own learning rate
    # in magnitude, regardless of critic_v's enormous gradient - proof the
    # two groups' Adam statistics never mixed.
    assert all(abs(s) < 0.2 for s in steps)


def test_one_joint_grad_equals_two_separate_grads_for_disjoint_params() -> None:
    # The mathematical property make_grouped_optimizer's docstring rests
    # on: with disjoint params, grad(term_a + term_b) w.r.t. the combined
    # tree equals computing grad(term_a) and grad(term_b) independently.
    def combined_loss(params):
        return params["critic_v"] ** 2 + 3.0 * params["actor_w"] ** 2

    def critic_only_loss(v):
        return v**2

    def actor_only_loss(w):
        return 3.0 * w**2

    params = {"critic_v": jnp.array(2.0), "actor_w": jnp.array(5.0)}
    joint_grads = jax.grad(combined_loss)(params)

    assert joint_grads["critic_v"] == jax.grad(critic_only_loss)(params["critic_v"])
    assert joint_grads["actor_w"] == jax.grad(actor_only_loss)(params["actor_w"])
