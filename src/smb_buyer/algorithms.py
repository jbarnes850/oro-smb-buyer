"""Small advantage/loss seam around the unchanged official Cookbook runner.

Use ``configure_algorithm`` then keep ``algorithm_context`` around the entire
awaited official run. Only one such run may own this process. This does not
choose rewards, data, a parent, dose, or a failure-to-reward policy.

MaxRL is the authors' practical grouped estimator, not a claim of an unbiased
estimator or full paper reproduction. Source: tajwarfahim/maxrl commit
7197bbb46a2ecd866da52f6b401ff20a34fe9390, verl/trainer/ppo/core_algos.py:403-441,
SHA256 9a9ba6f0dc0e649a959002390612023ddd273130d54ab62e0ba128e6f9bc9e4d.
Binary success must be declared by the experiment; economic returns are never
thresholded here. Compare against RLOO on that same binary objective.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from numbers import Real
from threading import Lock

import chz
import torch
from tinker_cookbook.rl import data_processing, train

ARMS = {
    "sr_mean": ("stable_reinforce", "mean"),
    "sr_rloo": ("stable_reinforce", "rloo"),
    "cispo_rloo": ("cispo", "rloo"),
    "cispo_maxrl_binary": ("cispo", "maxrl_binary"),
}
MAXRL_EPSILON = 1e-6
_OFFICIAL_ADVANTAGES = data_processing.compute_advantages
_OWNER = Lock()


def _finite(value):
    return isinstance(value, Real) and math.isfinite(value)


def _loss_config(loss, values):
    if not isinstance(values, dict):
        raise TypeError("Supply an explicit loss_fn_config dictionary")
    if loss == "stable_reinforce":
        if (
            set(values) != {"kl_coeff", "kl_estimator"}
            or not _finite(values["kl_coeff"])
            or values["kl_coeff"] < 0
            or values["kl_estimator"] not in {"k1", "k3"}
        ):
            raise ValueError("SR requires explicit finite kl_coeff >=0 and kl_estimator k1/k3")
    elif (
        set(values) != {"clip_low_threshold", "clip_high_threshold"}
        or not all(_finite(v) for v in values.values())
        or not 0 <= values["clip_low_threshold"] < values["clip_high_threshold"]
    ):
        raise ValueError("CISPO requires explicit finite clipping thresholds, never SR KL keys")


def configure_algorithm(
    config, *, arm, loss_fn_config, reference_kl_coef, reference_kl_discount_factor
):
    """Return official Config with explicit sync/async loss and KL settings.

    Loss configs stay separate: SR uses kl_coeff/kl_estimator; CISPO uses clipping
    thresholds. Reference KL is a separate mechanism. No regularization value is
    chosen here, and the original Config is not mutated.
    """
    loss, _ = ARMS[arm]
    _loss_config(loss, loss_fn_config)
    if not all(_finite(v) and v >= 0 for v in (reference_kl_coef, reference_kl_discount_factor)):
        raise ValueError("Reference KL settings must be explicit finite nonnegative values")
    if reference_kl_coef > 0 and config.kl_reference_config is None:
        raise ValueError("Positive reference KL requires the original explicit reference config")
    if loss == "stable_reinforce" and reference_kl_coef > 0 and reference_kl_discount_factor > 0:
        raise ValueError("Official SR does not support discounted reference KL on prompt tokens")
    async_config = config.async_config
    if async_config is not None:
        async_config = chz.replace(async_config, loss_fn=loss)
    return chz.replace(
        config,
        loss_fn=loss,
        loss_fn_config=dict(loss_fn_config),
        async_config=async_config,
        remove_constant_reward_groups=False,
        kl_penalty_coef=reference_kl_coef,
        kl_discount_factor=reference_kl_discount_factor,
    )


def compute_group_advantages(groups, *, estimator):
    """Validate original groups and return one scalar per complete trajectory.

    No trajectory, reward, prompt, token, logprob, or sequence ID is changed. Empty
    trajectories retain their actual group reward and receive no invented tokens.
    All constant groups remain present, including all-zero/all-one binary groups.
    """
    if estimator not in {"mean", "rloo", "maxrl_binary"}:
        raise ValueError("Unknown advantage estimator")
    rewards = []
    for group in groups:
        n = len(group.trajectories_G)
        if not n or len(group.final_rewards_G) != n or len(group.metrics_G) != n:
            raise ValueError("Trajectory/reward/metric counts must agree and be nonempty")
        if estimator != "mean" and n < 2:
            raise ValueError(
                "RLOO and this grouped MaxRL comparison require at least two trajectories"
            )
        values = group.get_total_rewards()
        if len(values) != n or not all(_finite(value) for value in values):
            raise ValueError("Group returns must be finite and match trajectory count")
        # Check original values before float32 conversion can round near-binary values.
        if estimator == "maxrl_binary" and any(value not in (0, 1) for value in values):
            raise ValueError("MaxRL requires an explicitly declared binary 0/1 success reward")
        for trajectory in group.trajectories_G:
            for transition in trajectory.transitions:
                action = transition.ac
                if (
                    not isinstance(action.sequence_id, str)
                    or not action.sequence_id
                    or action.maybe_logprobs is None
                    or len(action.tokens) != len(action.maybe_logprobs)
                    or not all(_finite(value) for value in action.maybe_logprobs)
                ):
                    raise ValueError(
                        "Every original sampled action requires its ID and exact logprobs"
                    )
        tensor = torch.tensor(values)
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.get_default_dtype())
        if not torch.isfinite(tensor).all():
            raise ValueError("Returns overflow the official advantage dtype")
        rewards.append(tensor)
    result = []
    for values in rewards:
        centered = values - values.mean()
        if estimator == "mean":
            advantage = centered  # Same group-mean formula as the installed Cookbook.
        elif estimator == "rloo":
            advantage = centered * (len(values) / (len(values) - 1))
        else:
            advantage = centered / (values.mean() + MAXRL_EPSILON)
        if not torch.isfinite(advantage).all():
            raise ValueError("Nonfinite grouped advantage")
        result.append(advantage)
    return result


@contextmanager
def algorithm_context(config, *, arm):
    """Temporarily bind ONLY train.compute_advantages; always restore it.

    The installed prepare_minibatch imports this symbol directly; it has no
    advantage callback Config field. Keep this context alive until all official
    training tasks finish. Concurrent arms in one process and nested contexts are
    rejected. Assembly/forward_backward/the runner are never replaced.
    """
    loss, estimator = ARMS[arm]
    _loss_config(loss, config.loss_fn_config)
    if (
        config.loss_fn != loss
        or config.training_loss_fn() != loss
        or (config.async_config is not None and config.async_config.loss_fn != loss)
        or config.remove_constant_reward_groups
    ):
        raise ValueError("Explicit sync/async loss must match the arm and retain constant groups")
    if not _OWNER.acquire(blocking=False):
        raise RuntimeError("Another algorithm context already owns this process")
    previous = train.compute_advantages
    try:
        if previous is not _OFFICIAL_ADVANTAGES:
            raise RuntimeError("Official advantage function was already replaced")
        train.compute_advantages = lambda groups: compute_group_advantages(
            groups, estimator=estimator
        )
        yield config
    finally:
        train.compute_advantages = previous
        _OWNER.release()
