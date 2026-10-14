"""Resource and end-to-end checks for the local CausalLM experiment."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataPipeline.sampling import get_batch
from ModernLM.config import ModelConfig, RuntimeConfig, TrainConfig
from scripts.common import ExperimentConfig, iter_batches, load_bpe_tokenizer, prepare_token_streams
from scripts.run_pretrained import run_pretrained


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output_root = (PROJECT_ROOT / "outputs").resolve()
        output_root.mkdir(exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="experiment-check-", dir=output_root)
        cls.directory = Path(cls.temporary.name).resolve()
        assert cls.directory.is_relative_to(output_root)
        cls.train_path = cls.directory / "train.txt"
        cls.val_path = cls.directory / "val.txt"
        cls.train_text = "The fox reads a small book. The book is blue.\n" * 40
        cls.val_text = "A blue book is on the table. A small fox reads.\n" * 20
        cls.train_path.write_bytes(cls.train_text.encode("utf-8"))
        cls.val_path.write_bytes(cls.val_text.encode("utf-8"))
        cls.tokenizer = Tokenizer(models.BPE())
        cls.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        cls.tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<|endoftext|>"], show_progress=False,
        )
        cls.tokenizer.train_from_iterator([cls.train_text], trainer=trainer)
        cls.tokenizer_path = cls.directory / "tokenizer.json"
        cls.tokenizer.save(str(cls.tokenizer_path))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_loads_native_tokenizer_and_preserves_unicode(self):
        tokenizer = load_bpe_tokenizer(self.tokenizer_path)
        self.assertIsInstance(tokenizer, Tokenizer)
        sample = "hello \u4f60\u597d \U0001f680<|endoftext|>\n"
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample).ids, skip_special_tokens=False), sample)
        self.assertEqual(tokenizer.get_vocab_size(), self.tokenizer.get_vocab_size())

    def test_manual_sources_do_not_apply_ratio(self):
        arguments = dict(val_paths=[self.val_path], max_train_tokens=31, max_val_tokens=17)
        first = prepare_token_streams(self.tokenizer, [self.train_path], train_fraction=0.2, **arguments)
        second = prepare_token_streams(self.tokenizer, [self.train_path], train_fraction=0.8, **arguments)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            self.assertEqual(left.dtype, torch.long)
            self.assertEqual(left.device.type, "cpu")
            self.assertEqual(left.ndim, 1)
        self.assertEqual(first[0].numel(), 31)
        self.assertEqual(first[1].numel(), 17)

    def test_ratio_splits_raw_text_before_encoding(self):
        corpus = "abcdef ghi jkl mno pqr stu\n"
        path = self.directory / "ratio.txt"
        path.write_bytes(corpus.encode("utf-8"))
        split = int(len(corpus) * 0.6)
        train, val = prepare_token_streams(self.tokenizer, [path], train_fraction=0.6)
        self.assertEqual(train.tolist(), self.tokenizer.encode(corpus[:split], add_special_tokens=False).ids)
        self.assertEqual(val.tolist(), self.tokenizer.encode(corpus[split:], add_special_tokens=False).ids)

    def test_formats_preserve_blank_rows_and_custom_columns(self):
        rows = ["  hello\n", "", "world\n"]
        txt = self.directory / "formats.txt"
        parquet = self.directory / "formats.parquet"
        jsonl = self.directory / "formats.jsonl"
        txt.write_bytes(b"  hello\n\nworld\n")
        pq.write_table(pa.table({"body": rows + [None]}), parquet)
        jsonl.write_text("\n".join(json.dumps({"body": row}) for row in rows + [None]), encoding="utf-8")
        results = [prepare_token_streams(
            self.tokenizer, [path], val_paths=[self.val_path], text_column="body"
        )[0] for path in (txt, parquet, jsonl)]
        for result in results[1:]:
            torch.testing.assert_close(results[0], result, rtol=0, atol=0)
        expected = sum((self.tokenizer.encode(text, add_special_tokens=False).ids
                        for text in ("  hello\n", "\n", "world\n")), [])
        self.assertEqual(results[0].tolist(), expected)

    def test_cache_invalidates_when_source_changes(self):
        path = self.directory / "cache_source.txt"
        path.write_text("first text\n", encoding="utf-8")
        cache = self.directory / "cache"
        options = dict(val_paths=[self.val_path], cache_dir=cache)
        first = prepare_token_streams(self.tokenizer, [path], **options)
        cached = prepare_token_streams(self.tokenizer, [path], **options)
        torch.testing.assert_close(first[0], cached[0], rtol=0, atol=0)
        self.assertEqual(len(list(cache.glob("*.pt"))), 1)
        path.write_text("a changed and longer source\n", encoding="utf-8")
        changed = prepare_token_streams(self.tokenizer, [path], **options)
        self.assertNotEqual(first[0].tolist(), changed[0].tolist())
        self.assertEqual(len(list(cache.glob("*.pt"))), 2)

    def test_batch_iterator_delegates_to_existing_sampler(self):
        tokens = torch.arange(100, dtype=torch.long)
        source = iter_batches(tokens, batch_size=2, context_length=8,
                              generator=torch.Generator().manual_seed(10), num_batches=3)
        reference_generator = torch.Generator().manual_seed(10)
        batches = list(source)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            reference = get_batch(tokens, 2, 8, generator=reference_generator)
            torch.testing.assert_close(batch.input_ids, reference.input_ids, rtol=0, atol=0)
            torch.testing.assert_close(batch.targets, reference.targets, rtol=0, atol=0)

    def test_complete_run_and_resume_match_continuous_training(self):
        cfg = ExperimentConfig(
            model_cfg=ModelConfig(
                vocab_size=1, hidden_size=16, num_layers=1,
                num_attention_heads=2, num_kv_heads=2,
                head_dim=8, value_head_dim=8, intermediate_size=32, max_seq_len=16,
            ),
            train_cfg=TrainConfig(
                optimizer={"name": "adamw", "params": {"lr": 1e-3}},
                scheduler={"name": "StepLR", "params": {"step_size": 2, "gamma": 0.5}},
                gradient_accumulation_steps=2,
            ),
            runtime_cfg=RuntimeConfig(gradient_checkpointing=True),
            tokenizer_path=self.tokenizer_path,
            train_paths=[self.train_path], val_paths=[self.val_path],
            max_train_tokens=512, max_val_tokens=128,
            batch_size=2, context_length=8, num_updates=4,
            eval_every=2, eval_batches=2, sample_every=2,
            sample_prompts=("The fox",), max_new_tokens=4,
            output_dir=self.directory / "continuous",
        )
        continuous_records = run_pretrained(cfg)
        partial_cfg = replace(cfg, num_updates=2, output_dir=self.directory / "resumed")
        run_pretrained(partial_cfg)
        resumed_records = run_pretrained(replace(
            partial_cfg, resume_from=Path(partial_cfg.output_dir) / "last.pt"
        ))
        left = torch.load(Path(cfg.output_dir) / "last.pt", weights_only=False)
        right = torch.load(Path(partial_cfg.output_dir) / "last.pt", weights_only=False)
        self.assertEqual(left["global_step"], 4)
        self.assertEqual(right["global_step"], 4)
        self.assertEqual(left["micro_step"], right["micro_step"])
        self.assertEqual(left["configs"]["model"]["vocab_size"], self.tokenizer.get_vocab_size())
        self.assertEqual(continuous_records, resumed_records)
        self.assertEqual(left["metadata"]["samples"], right["metadata"]["samples"])
        for name in left["model"]:
            torch.testing.assert_close(left["model"][name], right["model"][name], rtol=0, atol=0)
        for states_left, states_right in zip(left["optimizer"]["state"].values(),
                                           right["optimizer"]["state"].values()):
            for name in states_left:
                torch.testing.assert_close(states_left[name], states_right[name], rtol=0, atol=0)
        torch.testing.assert_close(left["train_generator"], right["train_generator"], rtol=0, atol=0)
        self.assertTrue((Path(cfg.output_dir) / "best.pt").exists())
        self.assertTrue((Path(cfg.output_dir) / "metrics.json").exists())
        self.assertTrue((Path(cfg.output_dir) / "samples.json").exists())

        migrated_cfg = replace(
            partial_cfg, num_updates=0, output_dir=self.directory / "migrated",
            resume_from=Path(partial_cfg.output_dir) / "last.pt",
        )
        self.assertEqual(run_pretrained(migrated_cfg), resumed_records)
        self.assertEqual(
            (Path(migrated_cfg.output_dir) / "best.pt").read_bytes(),
            (Path(partial_cfg.output_dir) / "best.pt").read_bytes(),
        )
        migrated = torch.load(Path(migrated_cfg.output_dir) / "last.pt", weights_only=False)
        self.assertEqual(migrated["metadata"]["best_val_ce"], right["metadata"]["best_val_ce"])
        with self.assertRaisesRegex(ValueError, "eval_batches"):
            run_pretrained(replace(migrated_cfg, eval_batches=3))


if __name__ == "__main__":
    unittest.main()
