"""Verify task reduction, runner updates, evaluation, and checkpointing.

Run with the project's PyTorch interpreter: python scripts/test_runner.py.
Only unittest and the project's existing dependencies are required.
"""

from __future__ import annotations

import copy
import sys
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from DataPipeline.batch import TextBatch
from ModernLM.config import ModelConfig, RuntimeConfig, TrainConfig
from ModernLM.initialization import initialize_parameters
from ModernLM.model import CausalLM
from ModernLM.outputs import TaskOutput
from Tasks.pretrained import pretrained_task, pretrain_normalizers
from TrainingSystems.runner.base import BaseRunner
from TrainingSystems.runner.lm_runner import LMRunner


@dataclass(frozen=True)
class ToyBatch:
    x: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, *, lr=0.03):
        super().__init__(parameters, lr=lr)
        self.step_calls = 0

    def step(self, closure=None):
        result = super().step(closure)
        self.step_calls += 1
        return result


class CountingScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.observed_optimizer_steps = []

    def step(self):
        self.observed_optimizer_steps.append(self.optimizer.step_calls)


class FixtureRunner(BaseRunner):
    """Supply already initialized models while exercising setup hooks."""

    def __init__(
        self,
        model,
        task_fn,
        *,
        accumulation_steps=1,
        normalizer_fn=None,
        runtime_cfg=None,
        optimizer_factory=None,
        with_scheduler=False,
        prepared_model=None,
    ):
        self.source_model = model
        self.prepared_model = prepared_model
        self.optimizer_factory = optimizer_factory or (
            lambda parameters: CountingSGD(parameters)
        )
        self.with_scheduler = with_scheduler
        self.setup_events = []
        super().__init__(
            TrainConfig(gradient_accumulation_steps=accumulation_steps),
            runtime_cfg or RuntimeConfig(),
            task_fn,
            normalizer_fn=normalizer_fn,
        )

    def build_model(self) -> None:
        self.setup_events.append("model")
        self.model = self.source_model

    def prepare_model(self) -> None:
        self.setup_events.append("prepare")
        model = self.prepared_model if self.prepared_model is not None else self.model
        self.model = model.to(
            device=self.runtime_cfg.device, dtype=self.runtime_cfg.dtype
        )

    def build_optimizer(self) -> None:
        self.setup_events.append("optimizer")
        self.optimizer = self.optimizer_factory(self.model.parameters())

    def build_scheduler(self) -> None:
        self.setup_events.append("scheduler")
        self.scheduler = CountingScheduler(self.optimizer) if self.with_scheduler else None


class ConfigRunner(BaseRunner):
    """Exercise shared builders with name/params configuration on a model."""

    def __init__(
        self, model, train_cfg, task_fn, *, runtime_cfg=None, normalizer_fn=None
    ):
        self.source_model = model
        super().__init__(
            train_cfg, runtime_cfg or RuntimeConfig(), task_fn,
            normalizer_fn=normalizer_fn,
        )

    def build_model(self) -> None:
        self.model = self.source_model

    def prepare_model(self) -> None:
        self.model = self.model.to(
            device=self.runtime_cfg.device, dtype=self.runtime_cfg.dtype
        )


class QuarterGradientRunner(FixtureRunner):
    """Exercise custom clipping through the runner's subclass hook."""

    def __init__(self, *args, **kwargs):
        self.clip_calls = 0
        super().__init__(*args, **kwargs)

    def _clip_gradients(self):
        self.clip_calls += 1
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.mul_(0.25)
        return None


def make_toy_model():
    model = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Identity())
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.25, -0.4]]))
    return model


def toy_batches():
    return [
        ToyBatch(
            torch.tensor([[1.0, 2.0], [-1.0, 0.5]]),
            torch.tensor([0.2, -0.3]),
            torch.tensor([True, False]),
        ),
        ToyBatch(
            torch.tensor([[0.4, -0.7], [2.0, 1.0], [-0.5, -1.0]]),
            torch.tensor([-0.4, 0.8, 0.1]),
            torch.tensor([True, True, True]),
        ),
    ]


def join_toy_batches(batches):
    return ToyBatch(
        torch.cat([batch.x for batch in batches]),
        torch.cat([batch.targets for batch in batches]),
        torch.cat([batch.mask for batch in batches]),
    )


def two_mean_normalizers(batch):
    return {"selected": int(batch.mask.sum()), "samples": batch.targets.numel()}


def two_mean_task(model, batch, *, normalizers):
    prediction = model(batch.x).squeeze(-1)
    squared = (prediction[batch.mask] - batch.targets[batch.mask]).square().sum()
    absolute = prediction.abs().sum()
    counts = two_mean_normalizers(batch)
    loss = squared / normalizers["selected"]
    loss = loss + 0.35 * absolute / normalizers["samples"]
    return TaskOutput(
        loss=loss,
        statistics={
            "mse": (squared.detach(), counts["selected"]),
            "absolute": (absolute.detach(), counts["samples"]),
        },
    )


def sum_task(model, batch, *, normalizers):
    prediction = model(batch.x).squeeze(-1)
    loss = (prediction - batch.targets).square().sum()
    return TaskOutput(loss=loss)


def make_lm(seed=71):
    cfg = ModelConfig(
        vocab_size=17,
        hidden_size=8,
        num_layers=2,
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=4,
        value_head_dim=3,
        intermediate_size=16,
        max_seq_len=8,
    )
    model = CausalLM(cfg, device="cpu", dtype=torch.float32)
    initialize_parameters(model, generator=torch.Generator().manual_seed(seed))
    return model


def lm_batches():
    return [
        TextBatch(
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[1, -100, 3, -100]]),
            torch.tensor([[True, False, True, False]]),
        ),
        TextBatch(
            torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]]),
            torch.tensor([[5, 6, 7, 8], [9, -100, 11, 12]]),
            torch.tensor([[True, True, True, True], [True, False, True, True]]),
        ),
    ]


def reference_ce(model, batch, *, reduction="mean"):
    logits = model(input_ids=batch.input_ids).logits
    mask = batch.loss_mask
    if mask is None:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = batch.targets.reshape(-1)
    else:
        logits = logits[mask]
        targets = batch.targets[mask]
    return F.cross_entropy(logits, targets, reduction=reduction)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def assert_models_close(self, actual, expected, *, atol=2e-6, rtol=2e-5):
        actual_parameters = dict(actual.named_parameters())
        expected_parameters = dict(expected.named_parameters())
        self.assertEqual(actual_parameters.keys(), expected_parameters.keys())
        for name, parameter in actual_parameters.items():
            with self.subTest(parameter=name):
                torch.testing.assert_close(
                    parameter, expected_parameters[name], atol=atol, rtol=rtol
                )

    def test_pretrain_mask_reduction_and_normalizers(self):
        model = make_lm()
        batch = lm_batches()[0]
        batch_normalizers = pretrain_normalizers(batch)
        self.assertEqual(batch_normalizers, {"tokens": 2})
        expected_sum = reference_ce(model, batch, reduction="sum")
        output = pretrained_task(model, batch, normalizers=batch_normalizers)
        torch.testing.assert_close(output.loss, expected_sum / 2)
        self.assertFalse(output.statistics["ce"][0].requires_grad)
        self.assertEqual(output.statistics["ce"][1], 2)
        normalizers = {"tokens": 9}
        output = pretrained_task(model, batch, normalizers=normalizers)
        torch.testing.assert_close(output.loss, expected_sum / 9)
        summed = pretrained_task(
            model, batch, normalizers=normalizers, reduction="sum"
        )
        torch.testing.assert_close(summed.loss, expected_sum)
        with torch.no_grad():
            output = pretrained_task(model, batch, normalizers=batch_normalizers)
        self.assertFalse(output.loss.requires_grad)

    def test_plain_batch_and_empty_mask(self):
        model = make_lm()
        plain = TextBatch(torch.tensor([[0]]), torch.tensor([[1]]))
        plain_normalizers = pretrain_normalizers(plain)
        self.assertEqual(plain_normalizers, {"tokens": 1})
        torch.testing.assert_close(
            pretrained_task(model, plain, normalizers=plain_normalizers).loss,
            reference_ce(model, plain),
        )
        empty = TextBatch(
            torch.tensor([[0, 1]]),
            torch.tensor([[-100, -100]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        self.assertEqual(pretrain_normalizers(empty), {"tokens": 0})
        self.assertIsNone(
            pretrained_task(model, empty, normalizers={"tokens": 0}).loss
        )

    def test_manual_adamw_update_matches_tiny_lm(self):
        model = make_lm()
        expected = copy.deepcopy(model)
        batch = lm_batches()[1]
        options = dict(lr=0.003, weight_decay=0.05, betas=(0.9, 0.95))
        runner = FixtureRunner(
            model,
            pretrained_task,
            normalizer_fn=pretrain_normalizers,
            optimizer_factory=lambda parameters: torch.optim.AdamW(parameters, **options),
        )
        reference_optimizer = torch.optim.AdamW(expected.parameters(), **options)
        reference_ce(expected, batch).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in expected.parameters()))
        reference_optimizer.step()
        history = runner.fit([batch], num_steps=1)
        self.assertEqual(set(history), {"1"})
        self.assertEqual(runner.global_step, 1)
        self.assertEqual(runner.micro_step, 1)
        self.assert_models_close(runner.model, expected)

    def test_uneven_supervision_accumulation_matches_whole_batch(self):
        batches = lm_batches()
        combined = TextBatch(
            torch.cat([batch.input_ids for batch in batches]),
            torch.cat([batch.targets for batch in batches]),
            torch.cat([batch.loss_mask for batch in batches]),
        )
        model = make_lm()
        expected = copy.deepcopy(model)
        reference_optimizer = CountingSGD(expected.parameters())
        expected_loss = reference_ce(expected, combined)
        expected_loss.backward()
        reference_optimizer.step()
        runner = FixtureRunner(
            model, pretrained_task, accumulation_steps=2,
            normalizer_fn=pretrain_normalizers,
        )
        history = runner.fit(batches, num_steps=1)
        self.assert_models_close(runner.model, expected)
        self.assertEqual(runner.micro_step, 2)
        self.assertEqual(set(history["1"]), {1, 2, "window_loss"})
        ce_statistics = [history["1"][step]["ce"] for step in (1, 2)]
        self.assertEqual([count for _, count in ce_statistics], [2, 7])
        self.assertAlmostEqual(
            sum(total for total, _ in ce_statistics) / 9,
            expected_loss.item(), places=5,
        )
        self.assertAlmostEqual(
            history["1"]["window_loss"], expected_loss.item(), places=5
        )

    def test_two_distinct_denominators(self):
        batches = toy_batches()
        model = make_toy_model()
        expected = copy.deepcopy(model)
        optimizer = CountingSGD(expected.parameters())
        combined = join_toy_batches(batches)
        expected_output = two_mean_task(
            expected, combined, normalizers=two_mean_normalizers(combined)
        )
        expected_output.loss.backward()
        optimizer.step()
        runner = FixtureRunner(
            model, two_mean_task, accumulation_steps=2,
            normalizer_fn=two_mean_normalizers,
        )
        history = runner.fit(batches, num_steps=1)
        self.assert_models_close(runner.model, expected)
        self.assertAlmostEqual(
            history["1"]["window_loss"],
            expected_output.loss.item(), places=6,
        )
        for name, (numerator, count) in expected_output.statistics.items():
            observations = [history["1"][step][name] for step in (1, 2)]
            self.assertAlmostEqual(
                sum(total for total, _ in observations), numerator.item(), places=6
            )
            self.assertEqual(sum(size for _, size in observations), count)

    def test_sum_objective_is_not_divided_by_window_size(self):
        batches = toy_batches()
        model = make_toy_model()
        expected = copy.deepcopy(model)
        optimizer = CountingSGD(expected.parameters())
        expected_loss = sum_task(
            expected, join_toy_batches(batches), normalizers={}
        ).loss
        expected_loss.backward()
        optimizer.step()
        runner = FixtureRunner(model, sum_task, accumulation_steps=2)
        history = runner.fit(batches, num_steps=1)
        self.assert_models_close(runner.model, expected)
        self.assertAlmostEqual(
            history["1"]["window_loss"], expected_loss.item(), places=6
        )
        self.assertEqual(history["1"][1], {})
        self.assertEqual(history["1"][2], {})

    def test_train_step_accumulates_without_updating(self):
        runner = FixtureRunner(make_toy_model(), sum_task, with_scheduler=True)
        runner.setup()
        before = copy.deepcopy(runner.model)
        first = runner.train_step(toy_batches()[0], normalizers={})
        gradients = [parameter.grad.clone() for parameter in runner.model.parameters()]
        self.assertEqual(first["micro_step"], 1)
        self.assertTrue(first["has_loss"])
        self.assertIsInstance(first["loss"], float)
        self.assertEqual(runner.global_step, 0)
        self.assertEqual(runner.optimizer.step_calls, 0)
        self.assert_models_close(runner.model, before)
        second = runner.train_step(toy_batches()[0], normalizers={})
        self.assertEqual(second["micro_step"], 2)
        for parameter, gradient in zip(runner.model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient * 2)
        result = runner.optimizer_step()
        self.assertIsNone(result)
        self.assertEqual(runner.global_step, 1)
        self.assertEqual(runner.scheduler.observed_optimizer_steps, [1])
        self.assertTrue(all(parameter.grad is None for parameter in runner.model.parameters()))

    def test_train_step_requires_explicit_normalizers(self):
        runner = FixtureRunner(make_toy_model(), sum_task)
        with self.assertRaises(TypeError):
            runner.train_step(toy_batches()[0])
        self.assertEqual(runner.setup_events, [])

    def test_history_contains_cpu_scalars_and_absolute_micro_steps(self):
        model = make_toy_model()
        batches = toy_batches()
        expected = [
            two_mean_task(model, batch, normalizers=two_mean_normalizers(batch))
            for batch in batches
        ]
        runner = FixtureRunner(
            model, two_mean_task, accumulation_steps=2,
            normalizer_fn=two_mean_normalizers,
            optimizer_factory=lambda parameters: CountingSGD(parameters, lr=0.0),
        )
        history = runner.fit(batches, num_steps=1)
        self.assertEqual(set(history), {"1"})
        self.assertEqual(set(history["1"]), {1, 2, "window_loss"})
        self.assertIsInstance(history["1"]["window_loss"], float)
        for step, output in enumerate(expected, start=1):
            self.assertEqual(set(history["1"][step]), set(output.statistics))
            for name, (total, count) in output.statistics.items():
                saved_total, saved_count = history["1"][step][name]
                self.assertIsInstance(saved_total, float)
                self.assertIsInstance(saved_count, (int, float))
                self.assertAlmostEqual(saved_total, total.item(), places=6)
                self.assertEqual(saved_count, count)
        repeated = runner.fit([batches[0]], num_steps=1)
        self.assertEqual(set(repeated), {"1"})
        self.assertEqual(set(repeated["1"]), {3, "window_loss"})
        self.assertEqual(runner.micro_step, 3)
        self.assertEqual(runner.global_step, 2)

    def test_clipping_strategies_match_manual_accumulated_gradient(self):
        batches = toy_batches()
        strategies = [
            (RuntimeConfig(grad_max_norm=0.12, grad_norm_type=1.0), "norm"),
            (RuntimeConfig(grad_clip_type="value", grad_clip_value=0.07), "value"),
            (RuntimeConfig(grad_clip_type=None), "none"),
            (RuntimeConfig(), "default_none"),
            (RuntimeConfig(grad_clip_type="value"), "value_none"),
            (RuntimeConfig(grad_max_norm=0.001), "custom"),
        ]
        for config, strategy in strategies:
            with self.subTest(strategy=strategy):
                model = make_toy_model()
                expected = copy.deepcopy(model)
                optimizer = CountingSGD(expected.parameters())
                sum_task(
                    expected, join_toy_batches(batches), normalizers={}
                ).loss.backward()
                if strategy == "norm":
                    nn.utils.clip_grad_norm_(expected.parameters(), 0.12, norm_type=1.0)
                elif strategy == "value":
                    nn.utils.clip_grad_value_(expected.parameters(), 0.07)
                elif strategy == "custom":
                    for parameter in expected.parameters():
                        parameter.grad.mul_(0.25)
                optimizer.step()
                runner_class = (
                    QuarterGradientRunner if strategy == "custom" else FixtureRunner
                )
                runner = runner_class(
                    model, sum_task, accumulation_steps=2, runtime_cfg=config,
                )
                runner.fit(batches, num_steps=1)
                self.assert_models_close(runner.model, expected)
                if strategy == "custom":
                    self.assertEqual(runner.clip_calls, 1)

    def test_configured_clipping_accepts_no_gradients(self):
        for config in (
            RuntimeConfig(grad_max_norm=0.12),
            RuntimeConfig(grad_clip_type="value", grad_clip_value=0.07),
        ):
            with self.subTest(config=config):
                runner = FixtureRunner(make_toy_model(), sum_task, runtime_cfg=config)
                runner.setup()
                before = copy.deepcopy(runner.model)
                runner._clip_gradients()
                self.assert_models_close(runner.model, before)
                self.assertTrue(
                    all(parameter.grad is None for parameter in runner.model.parameters())
                )

    def test_microbatch_mean_uses_named_counts_in_partial_tail(self):
        batches = [toy_batches()[index % 2] for index in range(5)]
        model = make_toy_model()
        expected = copy.deepcopy(model)
        optimizer = CountingSGD(expected.parameters())
        reference_objectives = []
        for window in (batches[:3], batches[3:]):
            optimizer.zero_grad(set_to_none=True)
            loss = sum(
                sum_task(expected, batch, normalizers={}).loss for batch in window
            ) / len(window)
            reference_objectives.append(loss.item())
            loss.backward()
            optimizer.step()

        observed_counts = []

        def microbatch_mean_task(model, batch, *, normalizers):
            observed_counts.append(normalizers["microbatches"])
            return TaskOutput(
                loss=sum_task(model, batch, normalizers={}).loss / normalizers["microbatches"]
            )

        runner = FixtureRunner(
            model, microbatch_mean_task, accumulation_steps=3,
            normalizer_fn=lambda batch: {"microbatches": 1},
        )
        history = runner.fit(batches, num_steps=4)
        self.assert_models_close(runner.model, expected)
        self.assertEqual(observed_counts, [3, 3, 3, 2, 2])
        self.assertEqual(runner.global_step, 2)
        self.assertEqual(runner.micro_step, 5)
        self.assertEqual(set(history), {"1", "2"})
        for index, objective in enumerate(reference_objectives, start=1):
            self.assertAlmostEqual(history[str(index)]["window_loss"], objective, places=6)

    def test_partial_tail_and_relative_fit_budget(self):
        batches = [toy_batches()[index % 2] for index in range(5)]
        model = make_toy_model()
        expected = copy.deepcopy(model)
        optimizer = CountingSGD(expected.parameters())
        reference_objectives = []
        for window in (batches[:3], batches[3:]):
            optimizer.zero_grad(set_to_none=True)
            combined = join_toy_batches(window)
            loss = two_mean_task(
                expected, combined, normalizers=two_mean_normalizers(combined)
            ).loss
            reference_objectives.append(loss.item())
            loss.backward()
            optimizer.step()
        runner = FixtureRunner(
            model, two_mean_task, accumulation_steps=3,
            normalizer_fn=two_mean_normalizers, with_scheduler=True,
        )
        history = runner.fit(batches, num_steps=4)
        self.assert_models_close(runner.model, expected)
        self.assertEqual(runner.global_step, 2)
        self.assertEqual(runner.micro_step, 5)
        self.assertEqual(set(history["1"]), {1, 2, 3, "window_loss"})
        self.assertEqual(set(history["2"]), {4, 5, "window_loss"})
        self.assertEqual(runner.scheduler.observed_optimizer_steps, [1, 2])
        for index, objective in enumerate(reference_objectives, start=1):
            self.assertAlmostEqual(history[str(index)]["window_loss"], objective, places=6)
        identity = id(runner.model)
        optimizer_identity = id(runner.optimizer)
        repeated = runner.fit([batches[0]], num_steps=1)
        self.assertEqual(set(repeated), {"1"})
        self.assertEqual(set(repeated["1"]), {6, "window_loss"})
        self.assertEqual(runner.global_step, 3)
        self.assertEqual(runner.micro_step, 6)
        self.assertEqual(id(runner.model), identity)
        self.assertEqual(id(runner.optimizer), optimizer_identity)
        self.assertEqual(runner.setup_events, ["model", "prepare", "optimizer", "scheduler"])

    def test_setup_is_explicit_idempotent_and_binds_prepared_model(self):
        source = make_toy_model()
        prepared = make_toy_model()
        runner = FixtureRunner(source, sum_task, prepared_model=prepared)
        self.assertEqual(runner.setup_events, [])
        runner.setup()
        runner.setup()
        self.assertIs(runner.model, prepared)
        self.assertEqual(runner.setup_events, ["model", "prepare", "optimizer", "scheduler"])
        optimizer_parameters = [
            parameter for group in runner.optimizer.param_groups for parameter in group["params"]
        ]
        self.assertEqual([id(parameter) for parameter in optimizer_parameters],
                         [id(parameter) for parameter in prepared.parameters()])

    def test_shared_optimizer_builders_match_torch_and_preserve_config(self):
        cases = [
            ("sGd", torch.optim.SGD, {"lr": 0.04, "momentum": 0.3, "weight_decay": 0.01}),
            ("aDaMw", torch.optim.AdamW, {
                "lr": 0.003, "betas": (0.8, 0.9), "eps": 1e-7, "weight_decay": 0.05,
            }),
            ("aDaM", torch.optim.Adam, {
                "lr": 0.002, "betas": (0.85, 0.95), "eps": 1e-9, "weight_decay": 0.02,
            }),
            ("rMsPrOp", torch.optim.RMSprop, {
                "lr": 0.008, "alpha": 0.85, "eps": 1e-7,
                "momentum": 0.2, "weight_decay": 0.01,
            }),
            ("aDaGrAd", torch.optim.Adagrad, {
                "lr": 0.02, "lr_decay": 0.02,
                "initial_accumulator_value": 0.1, "eps": 1e-9, "weight_decay": 0.01,
            }),
        ]
        for name, optimizer_class, options in cases:
            with self.subTest(optimizer=name):
                model = make_toy_model()
                expected = copy.deepcopy(model)
                train_cfg = TrainConfig(
                    optimizer={"name": name, "params": copy.deepcopy(options)}
                )
                saved_config = copy.deepcopy(train_cfg.optimizer)
                runner = ConfigRunner(model, train_cfg, sum_task)
                reference_optimizer = optimizer_class(expected.parameters(), **options)
                for batch in toy_batches():
                    reference_optimizer.zero_grad(set_to_none=True)
                    sum_task(expected, batch, normalizers={}).loss.backward()
                    reference_optimizer.step()
                    runner.fit([batch], num_steps=1)
                    self.assert_models_close(runner.model, expected)
                self.assertIsInstance(runner.optimizer, optimizer_class)
                self.assertIsNone(runner.scheduler)
                self.assertEqual(runner.global_step, 2)
                self.assertEqual(train_cfg.optimizer, saved_config)

    def test_default_optimizer_binds_only_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        model[0].requires_grad_(False)
        train_cfg = TrainConfig()
        runner = ConfigRunner(model, train_cfg, sum_task)
        runner.setup()
        self.assertIs(ConfigRunner.build_optimizer, BaseRunner.build_optimizer)
        self.assertIs(ConfigRunner.build_scheduler, BaseRunner.build_scheduler)
        self.assertIsInstance(runner.optimizer, torch.optim.AdamW)
        self.assertIsNone(runner.scheduler)
        self.assertEqual(len(runner.optimizer.param_groups), 1)
        parameters = runner.optimizer.param_groups[0]["params"]
        self.assertEqual(
            [id(parameter) for parameter in parameters],
            [id(parameter) for parameter in model.parameters() if parameter.requires_grad],
        )
        self.assertEqual(train_cfg.optimizer, {})
        self.assertIsNone(train_cfg.scheduler)

    def test_shared_schedulers_match_torch_across_repeated_fit(self):
        cases = [
            ("sTePlR", torch.optim.lr_scheduler.StepLR, {"step_size": 2, "gamma": 0.5}),
            ("mUlTiStEpLr", torch.optim.lr_scheduler.MultiStepLR, {
                "milestones": [1, 3], "gamma": 0.6,
            }),
            ("eXpOnEnTiAlLr", torch.optim.lr_scheduler.ExponentialLR, {"gamma": 0.7}),
            ("cOsInEaNnEaLiNgLr", torch.optim.lr_scheduler.CosineAnnealingLR, {
                "T_max": 4, "eta_min": 0.002,
            }),
            ("lInEaRlR", torch.optim.lr_scheduler.LinearLR, {
                "start_factor": 0.25, "end_factor": 1.0, "total_iters": 3,
            }),
            ("cOnStAnTlR", torch.optim.lr_scheduler.ConstantLR, {
                "factor": 0.5, "total_iters": 2,
            }),
        ]
        for name, scheduler_class, options in cases:
            with self.subTest(scheduler=name):
                model = make_toy_model()
                expected = copy.deepcopy(model)
                train_cfg = TrainConfig(
                    optimizer={"name": "SGD", "params": {"lr": 0.04, "momentum": 0.2}},
                    scheduler={"name": name, "params": copy.deepcopy(options)},
                )
                saved_optimizer_config = copy.deepcopy(train_cfg.optimizer)
                saved_scheduler_config = copy.deepcopy(train_cfg.scheduler)
                runner = ConfigRunner(model, train_cfg, sum_task)
                runner.setup()
                reference_optimizer = torch.optim.SGD(
                    expected.parameters(), lr=0.04, momentum=0.2
                )
                reference_scheduler = scheduler_class(
                    reference_optimizer, **copy.deepcopy(options)
                )
                self.assertIsInstance(runner.scheduler, scheduler_class)
                self.assertEqual(runner.scheduler.last_epoch, reference_scheduler.last_epoch)
                self.assertEqual(runner.scheduler.get_last_lr(), reference_scheduler.get_last_lr())
                scheduler_identity = id(runner.scheduler)
                for index in range(3):
                    batch = toy_batches()[index % 2]
                    reference_optimizer.zero_grad(set_to_none=True)
                    sum_task(expected, batch, normalizers={}).loss.backward()
                    reference_optimizer.step()
                    reference_scheduler.step()
                    runner.fit([batch], num_steps=1)
                    self.assert_models_close(runner.model, expected)
                    self.assertEqual(runner.global_step, index + 1)
                    self.assertEqual(id(runner.scheduler), scheduler_identity)
                    self.assertEqual(runner.scheduler.last_epoch, reference_scheduler.last_epoch)
                    self.assertEqual(
                        runner.scheduler.get_last_lr(), reference_scheduler.get_last_lr()
                    )
                self.assertEqual(train_cfg.optimizer, saved_optimizer_config)
                self.assertEqual(train_cfg.scheduler, saved_scheduler_config)

    def test_all_masked_window_does_not_advance_shared_scheduler(self):
        runner = ConfigRunner(
            make_lm(),
            TrainConfig(
                optimizer={"name": "AdamW", "params": {"lr": 0.003, "weight_decay": 0.2}},
                scheduler={"name": "StepLR", "params": {"step_size": 1, "gamma": 0.5}},
            ),
            pretrained_task, normalizer_fn=pretrain_normalizers,
        )
        runner.fit([lm_batches()[0]], num_steps=1)
        before = copy.deepcopy(runner.model)
        scheduler_state = copy.deepcopy(runner.scheduler.state_dict())
        optimizer_state = copy.deepcopy(runner.optimizer.state_dict())
        masked = TextBatch(
            torch.tensor([[0, 1]]), torch.tensor([[-100, -100]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        history = runner.fit([masked, masked], num_steps=1)
        self.assertEqual(history, {"1": {2: {}}, "2": {3: {}}})
        self.assertEqual(runner.global_step, 1)
        self.assertEqual(runner.micro_step, 3)
        self.assert_models_close(runner.model, before)
        self.assertEqual(runner.scheduler.state_dict(), scheduler_state)
        after_state = runner.optimizer.state_dict()
        self.assertEqual(after_state["param_groups"], optimizer_state["param_groups"])
        for parameter_index, state in optimizer_state["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, after_state["state"][parameter_index][name])

    def test_runtime_initialization_mode_is_dataclass_field(self):
        runtime_cfg = RuntimeConfig(init_mode="standard")
        self.assertEqual(runtime_cfg.init_mode, "standard")
        self.assertEqual(asdict(runtime_cfg)["init_mode"], "standard")

    def test_lm_runner_initializes_once_and_uses_runtime_placement(self):
        model_cfg = ModelConfig(
            vocab_size=17, hidden_size=8, num_layers=2,
            num_attention_heads=2, num_kv_heads=2, head_dim=4,
            value_head_dim=3, intermediate_size=16, max_seq_len=8,
        )
        train_cfg = TrainConfig(
            optimizer={"name": "AdamW", "params": {"lr": 0.003, "weight_decay": 0.05}},
            scheduler={"name": "StepLR", "params": {"step_size": 2, "gamma": 0.5}},
        )
        runtime_cfg = RuntimeConfig(device="cpu", dtype=torch.float64, init_mode="standard")
        with patch(
            "TrainingSystems.runner.lm_runner.initialize_parameters",
            wraps=initialize_parameters,
        ) as initialization:
            runner = LMRunner(
                model_cfg, train_cfg, runtime_cfg, pretrained_task,
                normalizer_fn=pretrain_normalizers,
            )
            self.assertIsNone(runner.model)
            self.assertIsNone(runner.optimizer)
            self.assertIsNone(runner.scheduler)
            self.assertEqual(initialization.call_count, 0)
            runner.setup()
            initialization.assert_called_once()
            self.assertIs(initialization.call_args.args[0], runner.model)
            self.assertEqual(initialization.call_args.kwargs["mode"], runtime_cfg.init_mode)
            self.assertIsInstance(runner.model, CausalLM)
            self.assertIsInstance(runner.optimizer, torch.optim.AdamW)
            self.assertIsInstance(runner.scheduler, torch.optim.lr_scheduler.StepLR)
            self.assertIs(LMRunner.build_optimizer, BaseRunner.build_optimizer)
            self.assertIs(LMRunner.build_scheduler, BaseRunner.build_scheduler)
            for parameter in runner.model.parameters():
                self.assertEqual(parameter.device, torch.device("cpu"))
                self.assertEqual(parameter.dtype, torch.float64)
                self.assertTrue(torch.isfinite(parameter).all())
            self.assertEqual(len(runner.optimizer.param_groups), 1)
            self.assertEqual(
                [id(parameter) for parameter in runner.optimizer.param_groups[0]["params"]],
                [id(parameter) for parameter in runner.model.parameters() if parameter.requires_grad],
            )
            before = copy.deepcopy(runner.model)
            identities = (id(runner.model), id(runner.optimizer), id(runner.scheduler))
            runner.setup()
            self.assertEqual(initialization.call_count, 1)
            self.assertEqual(
                (id(runner.model), id(runner.optimizer), id(runner.scheduler)), identities
            )
            self.assert_models_close(runner.model, before)
            self.assertEqual(runner.micro_step, 0)
            self.assertEqual(runner.global_step, 0)
            with torch.no_grad():
                logits = runner.model(input_ids=lm_batches()[0].input_ids).logits
            self.assertEqual(logits.shape, (1, 4, 17))
            self.assertEqual(logits.dtype, runtime_cfg.dtype)
            self.assertTrue(torch.isfinite(logits).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_lm_runner_uses_requested_cuda_device(self):
        model_cfg = ModelConfig(
            vocab_size=17, hidden_size=8, num_layers=2,
            num_attention_heads=2, num_kv_heads=2, head_dim=4,
            value_head_dim=3, intermediate_size=16, max_seq_len=8,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        runner = LMRunner(
            model_cfg, TrainConfig(), RuntimeConfig(device=device), pretrained_task,
            normalizer_fn=pretrain_normalizers,
        )
        runner.setup()
        self.assertTrue(all(parameter.device == device for parameter in runner.model.parameters()))
        with torch.no_grad():
            logits = runner.model(input_ids=lm_batches()[0].input_ids.to(device)).logits
        self.assertEqual(logits.device, device)
        self.assertTrue(torch.isfinite(logits).all())

    def test_all_masked_window_does_not_decay_existing_optimizer_state(self):
        model = make_lm()
        runner = FixtureRunner(
            model, pretrained_task, normalizer_fn=pretrain_normalizers,
            optimizer_factory=lambda parameters: torch.optim.AdamW(
                parameters, lr=0.003, weight_decay=0.2
            ),
        )
        runner.fit([lm_batches()[0]], num_steps=1)
        before = copy.deepcopy(model)
        optimizer_state = copy.deepcopy(runner.optimizer.state_dict())
        masked = TextBatch(
            torch.tensor([[0, 1]]), torch.tensor([[-100, -100]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        result = runner.fit([masked, masked], num_steps=1)
        self.assertEqual(result, {"1": {2: {}}, "2": {3: {}}})
        self.assertEqual(runner.global_step, 1)
        self.assertEqual(runner.micro_step, 3)
        self.assert_models_close(model, before)
        after_state = runner.optimizer.state_dict()
        self.assertEqual(optimizer_state["param_groups"], after_state["param_groups"])
        for parameter_index, state in optimizer_state["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, after_state["state"][parameter_index][name])

    def test_evaluation_weighting_preserves_modes_gradients_and_counters(self):
        model = make_toy_model()
        observed_states = []

        def checked_task(model, batch, *, normalizers):
            observed_states.append((
                all(not module.training for module in model.modules()),
                torch.is_grad_enabled(),
            ))
            return two_mean_task(model, batch, normalizers=normalizers)

        runner = FixtureRunner(
            model, checked_task, normalizer_fn=two_mean_normalizers,
            with_scheduler=True,
        )
        runner.setup()
        model.train()
        model[0].eval()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.31)
        modes = [module.training for module in model.modules()]
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        before = copy.deepcopy(model)
        runner.micro_step = 7
        runner.global_step = 4
        batches = toy_batches()
        combined = join_toy_batches(batches)
        expected = two_mean_task(
            model, combined, normalizers=two_mean_normalizers(combined)
        )
        output = runner.eval(iter(batches))
        self.assertEqual(set(output), {"mse", "absolute"})
        self.assertEqual(observed_states, [(True, False), (True, False)])
        self.assertEqual(runner.global_step, 4)
        self.assertEqual(runner.micro_step, 7)
        self.assertEqual(runner.optimizer.step_calls, 0)
        self.assertEqual(runner.scheduler.observed_optimizer_steps, [])
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assert_models_close(model, before)
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient)
        for name, (numerator, count) in expected.statistics.items():
            self.assertIsInstance(output[name], float)
            self.assertAlmostEqual(output[name], numerator.item() / count, places=6)

    def test_evaluation_consumes_finite_source_with_larger_limit(self):
        model = make_toy_model()
        batches = toy_batches()
        combined = join_toy_batches(batches)
        expected = two_mean_task(
            model, combined, normalizers=two_mean_normalizers(combined)
        )
        runner = FixtureRunner(
            model, two_mean_task, normalizer_fn=two_mean_normalizers
        )
        consumed = []

        def finite_source():
            for batch in batches:
                consumed.append(batch)
                yield batch

        output = runner.eval(finite_source(), max_batches=5)
        self.assertEqual(len(consumed), 2)
        for name, (total, count) in expected.statistics.items():
            self.assertAlmostEqual(output[name], total.item() / count, places=6)

    def test_evaluation_empty_source_and_zero_count_statistics(self):
        def zero_count_task(model, batch, *, normalizers):
            return TaskOutput(
                loss=None, statistics={"empty": (torch.tensor(0.0), 0)}
            )

        runner = FixtureRunner(make_toy_model(), zero_count_task)
        self.assertEqual(runner.eval([]), {})
        self.assertEqual(runner.eval(toy_batches()), {})
        self.assertEqual(runner.micro_step, 0)
        self.assertEqual(runner.global_step, 0)
        self.assertEqual(runner.optimizer.step_calls, 0)

        lm_runner = FixtureRunner(
            make_lm(), pretrained_task, normalizer_fn=pretrain_normalizers
        )
        masked = TextBatch(
            torch.tensor([[0, 1]]), torch.tensor([[-100, -100]]),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        self.assertEqual(lm_runner.eval([masked]), {})
        self.assertTrue(
            all(parameter.grad is None for parameter in lm_runner.model.parameters())
        )

    def test_evaluation_restores_modes_and_gradients_after_exception(self):
        def failing_task(model, batch, *, normalizers):
            self.assertTrue(all(not module.training for module in model.modules()))
            self.assertFalse(torch.is_grad_enabled())
            raise RuntimeError("evaluation task failed")

        model = make_toy_model()
        runner = FixtureRunner(model, failing_task, with_scheduler=True)
        runner.setup()
        model.train()
        model[0].eval()
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.31)
        modes = [module.training for module in model.modules()]
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        before = copy.deepcopy(model)
        with self.assertRaisesRegex(RuntimeError, "evaluation task failed"):
            runner.eval(toy_batches())
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.is_grad_enabled())
        self.assert_models_close(model, before)
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient)
        self.assertEqual(runner.micro_step, 0)
        self.assertEqual(runner.global_step, 0)
        self.assertEqual(runner.optimizer.step_calls, 0)
        self.assertEqual(runner.scheduler.observed_optimizer_steps, [])

    def test_iteration_limits_do_not_consume_extra_batches(self):
        runner = FixtureRunner(make_toy_model(), sum_task, accumulation_steps=2)
        consumed = []

        def stream():
            while True:
                consumed.append(1)
                yield toy_batches()[0]

        history = runner.fit(stream(), num_steps=1)
        self.assertEqual(set(history["1"]), {1, 2, "window_loss"})
        self.assertEqual(runner.micro_step, 2)
        self.assertEqual(len(consumed), 2)
        consumed.clear()
        self.assertEqual(runner.eval(stream(), max_batches=3), {})
        self.assertEqual(len(consumed), 3)
        consumed.clear()
        self.assertEqual(runner.eval(stream(), max_batches=0), {})
        self.assertEqual(consumed, [])

    def test_checkpointing_matches_forward_and_every_parameter_gradient(self):
        plain = make_lm()
        checkpointed = copy.deepcopy(plain)
        checkpointed.gradient_checkpointing_enable()
        self.assertTrue(checkpointed.backbone.gradient_checkpointing)
        calls = [0] * len(checkpointed.backbone.layers)
        handles = []
        for index, layer in enumerate(checkpointed.backbone.layers):
            def count_forward(module, inputs, *, layer_index=index):
                calls[layer_index] += 1
            handles.append(layer.register_forward_pre_hook(count_forward))
        batch = lm_batches()[1]
        plain_output = plain(input_ids=batch.input_ids)
        checkpointed_output = checkpointed(input_ids=batch.input_ids)
        torch.testing.assert_close(plain_output.logits, checkpointed_output.logits)
        plain_output.logits.square().mean().backward()
        checkpointed_output.logits.square().mean().backward()
        for name, parameter in plain.named_parameters():
            with self.subTest(parameter=name):
                checkpointed_parameter = dict(checkpointed.named_parameters())[name]
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(checkpointed_parameter.grad)
                torch.testing.assert_close(parameter.grad, checkpointed_parameter.grad,
                                           atol=2e-6, rtol=2e-5)
        self.assertTrue(all(count >= 2 for count in calls))
        for handle in handles:
            handle.remove()
        checkpointed.gradient_checkpointing_disable()
        self.assertFalse(checkpointed.backbone.gradient_checkpointing)

    def test_runtime_checkpointing_is_enabled_at_setup_and_skipped_in_eval(self):
        model = make_lm()
        runner = FixtureRunner(
            model, pretrained_task, normalizer_fn=pretrain_normalizers,
            runtime_cfg=RuntimeConfig(gradient_checkpointing=True),
        )
        runner.setup()
        self.assertTrue(model.backbone.gradient_checkpointing)
        calls = [0] * len(model.backbone.layers)
        handles = []
        for index, layer in enumerate(model.backbone.layers):
            def count_forward(module, inputs, *, layer_index=index):
                calls[layer_index] += 1
            handles.append(layer.register_forward_pre_hook(count_forward))
        output = runner.eval([lm_batches()[0]])
        self.assertEqual(set(output), {"ce"})
        self.assertIsInstance(output["ce"], float)
        self.assertEqual(calls, [1] * len(calls))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    unittest.main(verbosity=2)
