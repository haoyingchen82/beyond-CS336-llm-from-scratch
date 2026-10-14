"""Run document-aware pretraining, evaluation, and text generation."""

import hashlib
import sys
import traceback
from array import array
from dataclasses import asdict, replace
from functools import partial
from itertools import chain
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataPipeline.sampling import get_batch
from DataPipeline.text import iter_texts
from ModernLM.config import (
    ModelConfig,
    TrainConfig,
    RuntimeConfig
)
from ModernLM.generation.sampling import generate
from Tasks.pretrained import pretrained_task, pretrain_normalizers
from Tokenizer.common import save_params, tokenizer_fingerprint
from Tokenizer.tokenizer import RustTokenizer
from TrainingSystems.runner.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from TrainingSystems.runner.lm_runner import LMRunner

from scripts.common import (
    ExperimentConfig,
    encode_limited_tokens,
    load_bpe_tokenizer,
    write_json,
)


def run_pretrained(experiment: ExperimentConfig) -> None:
    """Save results at update intervals and at the final update.

    Fresh runs train a Rust BPE tokenizer; resumed runs reuse its exact
    saved parameters. num_updates is an additional optimizer-update budget.
    """
    if experiment.num_updates < 0:
        raise ValueError("num_updates must be nonnegative")
    if min(experiment.eval_interval, experiment.generate_interval, experiment.eval_batches) <= 0:
        raise ValueError("Evaluation/generation intervals and eval_batches must be positive")
    if not experiment.train_paths or not experiment.val_paths:
        raise ValueError("Provide separate training and validation paths")
    if not 0 < experiment.context_length <= experiment.model_cfg.max_seq_len:
        raise ValueError("context_length must be between 1 and model max_seq_len")

    output_dir = Path(experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(experiment.seed)

    # Build text iterators and train or restore the tokenizer.
    train_iter = chain.from_iterable(
        iter_texts(path, text_column=experiment.text_column)
        for path in experiment.train_paths
    )
    val_iter = chain.from_iterable(
        iter_texts(path, text_column=experiment.text_column)
        for path in experiment.val_paths
    )
    if experiment.resume_from is None:
        tokenizer = RustTokenizer.train(
            texts=train_iter,
            vocab_size=experiment.model_cfg.vocab_size,
            special_tokens=() if experiment.eos_token is None else (experiment.eos_token,),
        )
    else:
        tokenizer = load_bpe_tokenizer(
            Path(experiment.resume_from).parent / "tokenizer" / "bpe_params.json"
        )

    if len(tokenizer.vocab) > experiment.model_cfg.vocab_size:
        raise ValueError("The tokenizer vocabulary exceeds the model vocabulary")
    # BPE may stop below the requested vocabulary size on small corpora.
    model_cfg = replace(experiment.model_cfg, vocab_size=len(tokenizer.vocab))
    eos_token_id = (
        None if experiment.eos_token is None
        else tokenizer.special_tokens_map[experiment.eos_token]
    )
    tokenizer_id = tokenizer_fingerprint(tokenizer.params)
    save_params(tokenizer.params, output_dir / "tokenizer")
    write_json(output_dir / "config.json", {
        **asdict(experiment),
        "model_cfg": asdict(model_cfg),
        "tokenizer": {
            "params_path": output_dir / "tokenizer" / "bpe_params.json",
            "vocab_size": len(tokenizer.vocab),
            "fingerprint": tokenizer_id,
        },
    })

    # Rebuild the training iterator consumed by tokenizer training.
    train_iter_rebuilt = chain.from_iterable(
        iter_texts(path, text_column=experiment.text_column)
        for path in experiment.train_paths
    )
    cache_root = output_dir if experiment.token_cache_dir is None else Path(experiment.token_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    token_streams = {}
    document_boundaries = {}
    corpus_fingerprints = {}
    train_batches = val_sampler = None

    with TemporaryDirectory(prefix="token-cache-", dir=cache_root) as directory:
        try:
            # Keep both encoded corpora on disk and map their CPU token streams.
            for split, texts, max_tokens in (
                ("train", train_iter_rebuilt, experiment.max_train_tokens),
                ("val", val_iter, experiment.max_val_tokens),
            ):
                cache_path = Path(directory) / f"{split}.bin"
                offsets = [0]
                fingerprint = hashlib.sha256()
                with cache_path.open("wb") as cache:
                    for tokens in encode_limited_tokens(
                        tokenizer, max_tokens, texts, eos_token=experiment.eos_token
                    ):
                        encoded = array("q", tokens.tolist())
                        encoded.tofile(cache)
                        offsets.append(offsets[-1] + len(encoded))
                        fingerprint.update(array("q", [len(encoded)]).tobytes())
                        fingerprint.update(encoded.tobytes())
                if offsets[-1] == 0:
                    raise ValueError(f"No {split} document fits its token budget")
                token_streams[split] = torch.from_file(
                    str(cache_path), shared=False, size=offsets[-1], dtype=torch.long
                )
                document_boundaries[split] = torch.tensor(offsets, dtype=torch.long)
                corpus_fingerprints[split] = fingerprint.hexdigest()

            # Callable iterators sample lazily; each stage owns its RNG.
            train_generator = torch.Generator(device="cpu").manual_seed(experiment.seed)
            eval_generator = torch.Generator(device="cpu")
            generation_generator = torch.Generator(device=experiment.runtime_cfg.device)
            train_batches = iter(partial(
                get_batch,
                token_ids=token_streams["train"],
                batch_size=experiment.train_cfg.batch_size,
                context_length=experiment.context_length,
                device=experiment.runtime_cfg.device,
                generator=train_generator,
                document_offsets=document_boundaries["train"],
            ), None)
            val_sampler = partial(
                get_batch,
                token_ids=token_streams["val"],
                batch_size=experiment.train_cfg.batch_size,
                context_length=experiment.context_length,
                device=experiment.runtime_cfg.device,
                generator=eval_generator,
                document_offsets=document_boundaries["val"],
            )
            runner = LMRunner(
                model_cfg=model_cfg,
                train_cfg=experiment.train_cfg,
                runtime_cfg=experiment.runtime_cfg,
                task_fn=pretrained_task,
                normalizer_fn=pretrain_normalizers,
            )
            identity = {
                "tokenizer_fingerprint": tokenizer_id,
                "corpus_fingerprints": corpus_fingerprints,
                "context_length": experiment.context_length,
                "eval_batches": experiment.eval_batches,
                "seed": experiment.seed,
            }
            train_results, eval_results, generate_results = {}, [], []
            best_val_ce = None
            if experiment.resume_from is not None:
                restored = load_training_checkpoint(
                    runner, experiment.resume_from,
                    train_generator=train_generator, expected_metadata=identity,
                )["metadata"]
                train_results = restored["train"]
                eval_results = restored["eval"]
                generate_results = restored["generate"]
                best_val_ce = restored["best_val_ce"]
                previous_best = Path(experiment.resume_from).parent / "best.pt"
                if previous_best.exists() and previous_best.resolve() != (output_dir / "best.pt").resolve():
                    copyfile(previous_best, output_dir / "best.pt")
            runner.setup()
            target_step = runner.global_step + experiment.num_updates

            # Evaluate and sample at step zero, configured intervals, and the end.
            while True:
                step = runner.global_step
                improved = False
                if (step % experiment.eval_interval == 0 or step == target_step) and (
                    not eval_results or eval_results[-1]["step"] != step
                ):
                    # Reuse the same validation windows at every evaluation.
                    eval_generator.manual_seed(experiment.seed + 1)
                    metrics = runner.eval(
                        iter(val_sampler, None), max_batches=experiment.eval_batches
                    )
                    eval_results.append({"step": step, **metrics})
                    improved = best_val_ce is None or metrics["ce"] < best_val_ce
                    if improved:
                        best_val_ce = metrics["ce"]
                    print(f"step={step} val_ce={metrics['ce']:.6f}", flush=True)

                if (step % experiment.generate_interval == 0 or step == target_step) and (
                    not generate_results or generate_results[-1]["step"] != step
                ):
                    generation_generator.manual_seed(experiment.seed + 2)
                    samples = []
                    for prompt in experiment.generate_prompts:
                        prompt_ids = tokenizer.encode(prompt)
                        if not prompt_ids:
                            raise ValueError("Generation prompts must encode to at least one token")
                        generated = generate(
                            runner.model,
                            torch.tensor([prompt_ids], dtype=torch.long, device=experiment.runtime_cfg.device),
                            max_new_tokens=experiment.max_new_tokens,
                            temperature=experiment.temperature,
                            top_k=experiment.top_k,
                            eos_token_id=eos_token_id,
                            generator=generation_generator,
                        )[0].tolist()
                        samples.append({
                            "prompt": prompt,
                            "text": tokenizer.decode(generated),
                            "completion": tokenizer.decode(generated[len(prompt_ids):]),
                            "token_ids": generated,
                        })
                    generate_results.append({"step": step, "samples": samples})

                write_json(output_dir / "train.json", train_results)
                write_json(output_dir / "eval.json", eval_results)
                write_json(output_dir / "generate.json", generate_results)
                metadata = {
                    **identity, "train": train_results, "eval": eval_results,
                    "generate": generate_results, "best_val_ce": best_val_ce,
                }
                if improved:
                    save_training_checkpoint(
                        runner, output_dir / "best.pt",
                        train_generator=train_generator, metadata=metadata,
                    )
                save_training_checkpoint(
                    runner, output_dir / "last.pt",
                    train_generator=train_generator, metadata=metadata,
                )
                if step == target_step:
                    break

                # Stop at the next event without resetting the training iterator.
                next_step = min(
                    target_step,
                    (step // experiment.eval_interval + 1) * experiment.eval_interval,
                    (step // experiment.generate_interval + 1) * experiment.generate_interval,
                )
                history = runner.fit(train_batches, num_steps=next_step - step)
                train_results.update({
                    str(step + int(index)): window for index, window in history.items()
                })
                print(f"step={runner.global_step} train_loss={history[str(len(history))]['window_loss']:.6f}", flush=True)
        finally:
            # Failed sampler frames can also retain mapped tensors on Windows.
            if sys.exc_info()[2] is not None:
                traceback.clear_frames(sys.exc_info()[2])
            # Callable iterators hold their partials and mapped tensor references.
            train_batches = val_sampler = None
            token_streams.clear()


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "wikitext-103" / "wikitext-103-raw-v1"
    num_updates = 10_000
    experiment = ExperimentConfig(
        model_cfg=ModelConfig(
            vocab_size=16000, hidden_size=512, num_layers=6,
            num_attention_heads=8, num_kv_heads=8,
            head_dim=64, value_head_dim=64, intermediate_size=1536,
            max_seq_len=256,
        ),
        train_cfg=TrainConfig(
            optimizer={"name": "adamw", "params": {
                "lr": 1.5e-4, "betas": (0.9, 0.95),
                "weight_decay": 0.1, "foreach": False,
            }},
            # Advance once per optimizer update, including across fit calls.
            scheduler={"name": "CosineAnnealingLR", "params": {
                "T_max": num_updates, "eta_min": 1.5e-5,
            }},
            gradient_accumulation_steps=4,
        ),
        runtime_cfg=RuntimeConfig(
            device="cuda:0", dtype=torch.float32,
            grad_max_norm=1.0, gradient_checkpointing=True,
        ),
        tokenizer_path=PROJECT_ROOT / "outputs" / "Tokenizer" / "wikitext103_bpe_fast" / "tokenizer.json",
        train_paths=[
            data_dir / "train-00000-of-00002.parquet",
            data_dir / "train-00001-of-00002.parquet",
        ],
        val_paths=[data_dir / "validation-00000-of-00001.parquet"],
        max_train_tokens=None,
        max_val_tokens=262_144,
        context_length=256,
        num_updates=num_updates, eval_interval=500, eval_batches=64, generate_interval=1000,
        output_dir=PROJECT_ROOT / "outputs" / "wikitext_causallm_cosine",
    )

    run_pretrained(experiment)
