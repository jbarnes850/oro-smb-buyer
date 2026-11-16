"""Local official assembly/advantage checks; no engine, service or gradient calls."""

import asyncio
import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import chz
import pytest
import tinker
import torch
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl import data_processing, train
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

from smb_buyer.algorithms import (
    ARMS,
    algorithm_context,
    compute_group_advantages,
    configure_algorithm,
)


def group(rewards, *, empty=False):
    trajectories = []
    for i, _ in enumerate(rewards):
        transitions = (
            []
            if empty
            else [
                Transition(
                    ob=tinker.ModelInput.from_ints([10, 11]),
                    ac=TokensWithLogprobs([12, 13], [-0.3, -0.4], sequence_id=f"local-{i}-a"),
                    reward=0.0,
                    episode_done=False,
                ),
                Transition(
                    ob=tinker.ModelInput.from_ints([20, 21]),
                    ac=TokensWithLogprobs([22], [-0.5], sequence_id=f"local-{i}-b"),
                    reward=0.0,
                    episode_done=True,
                ),
            ]
        )
        trajectories.append(Trajectory(transitions, tinker.ModelInput.from_ints([30])))
    return TrajectoryGroup(trajectories, list(rewards), [{} for _ in rewards])


def config(arm, *, asynchronous=False):
    original = train.Config(
        learning_rate=1e-5,
        dataset_builder=None,
        model_name="local-only-fixture",
        recipe_name="local-only",
        max_tokens=16,
        log_path="unused",
        async_config=(
            train.AsyncConfig(max_steps_off_policy=0, groups_per_batch=1) if asynchronous else None
        ),
    )
    loss = (
        {"kl_coeff": 0.0, "kl_estimator": "k1"}
        if arm.startswith("sr_")
        else {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
    )
    return configure_algorithm(
        original,
        arm=arm,
        loss_fn_config=loss,
        reference_kl_coef=0.0,
        reference_kl_discount_factor=0.0,
    )


def test_mean_is_official_and_rloo_is_leave_one_out():
    groups = [group([1.5, -2.0, 4.5]), group([5.0, 5.0])]
    actual = compute_group_advantages(groups, estimator="mean")
    expected = data_processing.compute_advantages(groups)
    for a, e in zip(actual, expected, strict=True):
        assert torch.equal(a, e)
    rloo = compute_group_advantages(groups, estimator="rloo")
    rewards = groups[0].get_total_rewards()
    torch.testing.assert_close(
        rloo[0],
        torch.tensor(
            [
                reward - sum(rewards[:i] + rewards[i + 1 :]) / (len(rewards) - 1)
                for i, reward in enumerate(rewards)
            ]
        ),
    )
    assert len(rloo) == 2 and torch.equal(rloo[1], torch.zeros(2))


@pytest.mark.parametrize("rewards", [[0, 0], [1, 1], [0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
def test_maxrl_practical_binary_formula_keeps_constant_groups(rewards):
    result = compute_group_advantages([group(rewards)], estimator="maxrl_binary")
    values = torch.tensor(rewards, dtype=torch.float32)
    assert len(result) == 1
    torch.testing.assert_close(result[0], (values - values.mean()) / (values.mean() + 1e-6))


@pytest.mark.parametrize("rewards", [[-1, 1], [0, 2], [0, 0.5], [0, 1 + 1e-9]])
def test_maxrl_never_thresholds_or_rounds_economic_rewards(rewards):
    with pytest.raises(ValueError, match="binary"):
        compute_group_advantages([group(rewards)], estimator="maxrl_binary")


@pytest.mark.parametrize("estimator", ["mean", "rloo", "maxrl_binary"])
def test_count_finiteness_and_original_sample_guards(estimator):
    for bad in [group([]), group([float("nan"), 1]), group([float("inf"), 1])]:
        with pytest.raises(ValueError):
            compute_group_advantages([bad], estimator=estimator)
    bad = group([0, 1])
    bad.final_rewards_G.pop()
    with pytest.raises(ValueError, match="counts"):
        compute_group_advantages([bad], estimator=estimator)
    for attr, value in [
        ("sequence_id", None),
        ("maybe_logprobs", [-0.3]),
        ("maybe_logprobs", [float("nan"), -0.3]),
    ]:
        bad = group([0, 1])
        setattr(bad.trajectories_G[0].transitions[0].ac, attr, value)
        with pytest.raises(ValueError, match="original sampled"):
            compute_group_advantages([bad], estimator=estimator)


@pytest.mark.parametrize("estimator", ["rloo", "maxrl_binary"])
def test_grouped_comparison_rejects_singleton(estimator):
    with pytest.raises(ValueError, match="at least two"):
        compute_group_advantages([group([1])], estimator=estimator)


def test_empty_trajectories_retain_return_without_fake_tokens():
    original = group([0, 1], empty=True)
    advantages = compute_group_advantages([original], estimator="rloo")
    assert advantages[0].tolist() == [-1.0, 1.0]
    data, metadata = data_processing.assemble_training_data([original], advantages)
    assert data == metadata == [] and original.final_rewards_G == [0, 1]


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_config_sets_both_loss_fields_and_retains_constant_groups(arm, asynchronous):
    cfg = config(arm, asynchronous=asynchronous)
    assert cfg.loss_fn == cfg.training_loss_fn() == ARMS[arm][0]
    assert cfg.remove_constant_reward_groups is False
    assert cfg.kl_penalty_coef == cfg.kl_discount_factor == 0.0
    if asynchronous:
        assert cfg.async_config.loss_fn == cfg.loss_fn


def test_cispo_rejects_sr_config_and_context_rejects_async_override():
    cfg = config("cispo_rloo", asynchronous=True)
    with pytest.raises(ValueError, match="CISPO"):
        configure_algorithm(
            cfg,
            arm="cispo_rloo",
            loss_fn_config={"kl_coeff": 0.0},
            reference_kl_coef=0.0,
            reference_kl_discount_factor=0.0,
        )
    bad = chz.replace(cfg, async_config=chz.replace(cfg.async_config, loss_fn="stable_reinforce"))
    with pytest.raises(ValueError, match="sync/async"), algorithm_context(bad, arm="cispo_rloo"):
        pass


def test_context_restores_exact_symbol_after_failure_and_rejects_nested_owner():
    before = train.compute_advantages
    cfg = config("sr_rloo")
    with pytest.raises(RuntimeError, match="deliberate"), algorithm_context(cfg, arm="sr_rloo"):
        assert train.compute_advantages is not before
        assert data_processing.compute_advantages is before
        with (
            pytest.raises(RuntimeError, match="already owns"),
            algorithm_context(cfg, arm="sr_rloo"),
        ):
            pass
        raise RuntimeError("deliberate local failure")
    assert train.compute_advantages is before


def test_official_prepare_minibatch_preserves_all_arrays_and_provenance_except_advantage():
    original = group([0.0, 1.0, 1.0, 0.0])
    original_copy = copy.deepcopy(original)
    tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
    builder = SimpleNamespace(logging_tags=list)
    fingerprints = []
    runner, assemble, fb = train.do_sync_training, train.assemble_training_data, train.train_step
    for arm in ARMS:
        with algorithm_context(config(arm), arm=arm):
            data, _metrics, _usage = asyncio.run(
                train.prepare_minibatch(
                    [builder],
                    [original],
                    tokenizer,
                    None,
                    0.0,
                    0.0,
                )
            )
        assert len(data) == 8
        fingerprint = []
        scale = {
            "sr_mean": 0.5,
            "sr_rloo": 2 / 3,
            "cispo_rloo": 2 / 3,
            "cispo_maxrl_binary": 0.5 / (0.5 + 1e-6),
        }[arm]
        for i, datum in enumerate(data):
            assert datum.model_input_spans is not None and datum.loss_fn_input_spans is not None
            mask = datum.loss_fn_inputs["mask"].to_torch()
            advantage = datum.loss_fn_inputs["advantages"].to_torch()
            assert torch.all(advantage[mask == 0] == 0)
            expected = scale * [-1, 1, 1, -1][i // 2]
            torch.testing.assert_close(
                advantage[mask == 1], torch.full_like(advantage[mask == 1], expected)
            )
            serialized = asdict(datum)
            del serialized["loss_fn_inputs"]["advantages"]
            fingerprint.append(
                json.dumps(
                    serialized,
                    sort_keys=True,
                    default=lambda a: (
                        a.model_dump(mode="json") if hasattr(a, "model_dump") else a.tolist()
                    ),
                )
            )
        fingerprints.append(fingerprint)
    assert all(value == fingerprints[0] for value in fingerprints)
    assert asdict(original) == asdict(original_copy)
    assert (train.do_sync_training, train.assemble_training_data, train.train_step) == (
        runner,
        assemble,
        fb,
    )
