"""Real BGE reranker LoRA pilot on disjoint HotpotQA train questions.

No LLM API. Validation here is a tiny pilot holdout, not the ABC benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import time
import urllib.request

MODEL_ID = "BAAI/bge-reranker-v2-m3"
DATA_URL = ("https://datasets-server.huggingface.co/rows?"
            "dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=train&offset=0&length=100")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ranking_metrics(ranking, gold, k):
    gold = set(gold)
    assert gold and len(set(ranking)) == len(ranking)
    hits = len(gold.intersection(ranking[:k]))
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranking[:k]) if p in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"recall": hits / len(gold), "all_evidence": float(hits == len(gold)), "ndcg": dcg / ideal}


def prepare(out, train_n, val_n, seed, forbidden_ids):
    from rank_bm25 import BM25Okapi
    import numpy as np
    cache = out.parent / "hotpot_train_first100.json"
    if not cache.exists():
        with urllib.request.urlopen(DATA_URL, timeout=90) as response:
            raw = json.load(response)
        save(cache, raw)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    assert not any(x.get("truncated_cells") for x in raw["rows"])
    eligible = []
    for entry in raw["rows"]:
        r = entry["row"]
        titles, sentences = r["context"]["title"], r["context"]["sentences"]
        gold_titles = set(r["supporting_facts"]["title"])
        if len(titles) != 10 or len(set(titles)) != 10 or not gold_titles <= set(titles):
            continue
        if r["id"] in forbidden_ids or not 1 <= len(gold_titles) <= 6:
            continue
        eligible.append({"id": r["id"], "question": r["question"],
                         "passages": [{"title": t, "text": " ".join(s.strip() for s in ss)}
                                      for t, ss in zip(titles, sentences)],
                         "gold": [i for i, t in enumerate(titles) if t in gold_titles]})
    selected = random.Random(seed).sample(eligible, train_n + val_n)
    train, val = selected[:train_n], selected[train_n:]
    assert not {r["id"] for r in train} & {r["id"] for r in val}
    assert not {r["id"] for r in selected} & forbidden_ids
    # Group creation is AFTER question split. All known positives are excluded from negatives.
    groups = []
    for r in train:
        tokenize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
        bm25 = BM25Okapi([tokenize(p["title"] + " " + p["text"]) for p in r["passages"]])
        order = np.argsort(-bm25.get_scores(tokenize(r["question"])), kind="stable").tolist()
        negatives = [i for i in order if i not in r["gold"]][:4]
        assert len(negatives) == 4 and not set(negatives) & set(r["gold"])
        for pos in r["gold"]:
            groups.append({"question_id": r["id"], "question": r["question"],
                           "positive_id": pos, "negative_ids": negatives,
                           "texts": [r["passages"][i]["text"] for i in [pos] + negatives]})
    dataset = {"source": DATA_URL, "license": "CC-BY-SA-4.0", "seed": seed,
               "sampling": "first 100 official train rows; tiny nonrepresentative pilot",
               "train": train, "validation": val, "groups": groups}
    save(out / "dataset.json", dataset)
    return dataset


def main():
    import torch
    import numpy as np
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-questions", type=int, default=32)
    parser.add_argument("--val-questions", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-ids", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; refusing slow accidental CPU training")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This pilot uses BF16 for numerical stability; a BF16-capable GPU is required")
    amp_dtype = torch.bfloat16
    if min(args.epochs, args.train_questions, args.val_questions, args.accumulation) < 1:
        raise ValueError("Counts must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    forbidden_ids = set(json.loads(args.exclude_ids.read_text(encoding="utf-8")))
    dataset = prepare(args.out, args.train_questions, args.val_questions, args.seed, forbidden_ids)
    print(f"DATA train={len(dataset['train'])} val={len(dataset['validation'])} groups={len(dataset['groups'])}", flush=True)
    revision = HfApi().model_info(MODEL_ID).sha
    print(f"Downloading {MODEL_ID} revision={revision}", flush=True)
    model_path = snapshot_download(MODEL_ID, revision=revision,
                                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                                  "tokenizer_config.json", "special_tokens_map.json",
                                                  "sentencepiece.bpe.model"], max_workers=2)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print("Loading pretrained reranker", flush=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=amp_dtype, use_safetensors=True,
        local_files_only=True, attn_implementation="eager")
    assert base.config.num_labels == 1
    base.to("cuda")
    config = {"model": MODEL_ID, "revision": revision, "seed": args.seed,
              "train_questions": args.train_questions, "validation_questions": args.val_questions,
              "groups_per_epoch": len(dataset["groups"]), "epochs": args.epochs,
              "positive_per_group": 1, "negative_per_group": 4, "negative_mining": "BM25 top non-gold",
              "max_length": args.max_length, "microbatch_groups": 1,
              "gradient_accumulation": args.accumulation, "lr": 2e-5,
              "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
              "target_modules": ["query", "value"], "modules_to_save": ["classifier"],
              "objective": "cross entropy over one positive and four negatives",
              "precision": "bf16 frozen weights + fp32 trainable weights + BF16 autocast; no loss scaling",
              "gradient_checkpointing": True, "input": "question + passage body; no title, matching ABC",
              "gpu": torch.cuda.get_device_name(),
              "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
              "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]},
              "data_sha256": hashlib.sha256((args.out / "dataset.json").read_bytes()).hexdigest(),
              "warning": "Pilot only; different split/length/precision from earlier ABC, not directly comparable"}
    save(args.out / "config.json", config)

    def encode(question, texts):
        return tokenizer([question] * len(texts), texts, padding=True, truncation=True,
                         max_length=args.max_length, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def evaluate(model, label):
        model.eval()
        rows = []
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for r in dataset["validation"]:
            scores = []
            for first in range(0, len(r["passages"]), 5):
                inputs = encode(r["question"], [p["text"] for p in r["passages"][first:first+5]])
                with torch.autocast("cuda", dtype=amp_dtype):
                    output = model(**inputs).logits.float().view(-1)
                scores.extend(output.cpu().tolist())
            assert all(math.isfinite(x) for x in scores)
            ranking = sorted(range(len(scores)), key=lambda i: -scores[i])
            rows.append({"id": r["id"], "scores": scores, "ranking": ranking, "gold": r["gold"],
                         "metrics": {str(k): ranking_metrics(ranking, r["gold"], k) for k in [2, 3, 5]}})
        torch.cuda.synchronize()
        result = {"rows": rows, "seconds": time.perf_counter() - begin,
                  "mean": {str(k): {m: sum(r["metrics"][str(k)][m] for r in rows) / len(rows)
                                    for m in ["recall", "all_evidence", "ndcg"]} for k in [2, 3, 5]}}
        save(args.out / f"{label}.json", result)
        print(f"{label}: {json.dumps(result['mean'])}", flush=True)
        return result

    baseline = evaluate(base, "before")
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                           lora_dropout=0.05, target_modules=["query", "value"],
                                           modules_to_save=["classifier"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for p in trainable.values():
        p.data = p.data.float()
    original_trainable = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    frozen_samples = {n: p.detach().reshape(-1)[::4096].cpu().clone()
                      for n, p in model.named_parameters() if not p.requires_grad}
    config["trainable_parameters"] = sum(p.numel() for p in trainable.values())
    config["total_parameters"] = sum(p.numel() for p in model.parameters())
    save(args.out / "config.json", config)
    print(f"PARAMETERS trainable={config['trainable_parameters']:,} total={config['total_parameters']:,}", flush=True)
    optimizer = torch.optim.AdamW(trainable.values(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    groups = dataset["groups"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_begin = time.perf_counter()
    history = []
    optimizer_steps = 0
    for epoch in range(args.epochs):
        order = list(range(len(groups)))
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), args.accumulation):
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batch = order[start:start+args.accumulation]
            losses = []
            for group_index in batch:
                group = groups[group_index]
                inputs = encode(group["question"], group["texts"])
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = model(**inputs).logits.float().view(1, 5)
                    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device="cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                losses.append(float(loss.detach()))
                scaler.scale(loss / len(batch)).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Nonfinite gradient norm")
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
            torch.cuda.synchronize()
            record = {"epoch": epoch + 1, "step": optimizer_steps, "groups": len(batch),
                      "loss": sum(losses) / len(losses), "grad_norm": float(grad_norm),
                      "seconds": time.perf_counter() - step_start,
                      "allocated_gib": torch.cuda.memory_allocated() / 1024**3}
            history.append(record)
            with (args.out / "train_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f"TRAIN epoch={epoch+1} step={optimizer_steps} loss={record['loss']:.4f} seconds={record['seconds']:.2f} vram={record['allocated_gib']:.2f}GiB", flush=True)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_begin
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    updated_names = [n for n, p in trainable.items() if not torch.equal(original_trainable[n], p.detach().cpu())]
    assert any("lora_" in n for n in updated_names), "LoRA weights did not update"
    for n, p in model.named_parameters():
        if n in frozen_samples:
            assert p.grad is None
            assert torch.equal(frozen_samples[n], p.detach().reshape(-1)[::4096].cpu()), n
    model.gradient_checkpointing_disable()
    after = evaluate(model, "after")
    adapter_dir = args.out / "adapter"
    model.peft_config["default"].base_model_name_or_path = MODEL_ID
    model.peft_config["default"].revision = revision
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    # Reload in a separate fresh base instance after releasing training references.
    ref = dataset["validation"][0]
    expected = after["rows"][0]["scores"][:5]
    del optimizer, trainable, model, base, inputs, logits, loss, original_trainable, frozen_samples, p
    gc.collect()
    torch.cuda.empty_cache()
    fresh_base = AutoModelForSequenceClassification.from_pretrained(
        model_path, torch_dtype=amp_dtype, use_safetensors=True,
        local_files_only=True, attn_implementation="eager")
    restored = PeftModel.from_pretrained(fresh_base, adapter_dir)
    for n, p in restored.named_parameters():
        if "lora_" in n or "modules_to_save" in n:
            p.data = p.data.float()
    restored.to("cuda").eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
        reloaded_scores = restored(**encode(ref["question"], [p["text"] for p in ref["passages"][:5]])).logits.float().view(-1).cpu()
    max_difference = float((reloaded_scores - torch.tensor(expected)).abs().max())
    assert torch.allclose(reloaded_scores, torch.tensor(expected), atol=0.02, rtol=0.005), max_difference
    report = {"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "train_questions": len(dataset["train"]), "validation_questions": len(dataset["validation"]),
              "groups_per_epoch": len(groups), "epochs": args.epochs, "optimizer_steps": optimizer_steps,
              "train_seconds": train_seconds, "mean_optimizer_step_seconds": train_seconds / optimizer_steps,
              "peak_train_allocated_gib": peak_allocated, "peak_train_reserved_gib": peak_reserved,
              "epoch_mean_losses": {str(e): sum(x["loss"] * x["groups"] for x in history if x["epoch"] == e) /
                                    sum(x["groups"] for x in history if x["epoch"] == e) for e in range(1, args.epochs+1)},
              "trainable_parameters": config["trainable_parameters"], "total_parameters": config["total_parameters"],
              "updated_parameter_tensors": len(updated_names), "frozen_weight_samples_unchanged": True,
              "adapter_reload_verified": True, "reload_max_abs_difference": max_difference,
              "before": baseline["mean"], "after": after["mean"],
              "total_seconds_including_download_load_eval": time.perf_counter() - started,
              "adapter_path": str(adapter_dir), "limitations": config["warning"]}
    save(args.out / "summary.json", report)
    print("COMPLETED\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
