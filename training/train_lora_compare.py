"""Scale LoRA training; select on train-split dev, compare C/D on held-out validation.

All candidates are reranked (10 per question). C is therefore equivalent to
the existing hybrid+rerank stage with all 10 candidates admitted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
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

from train_lora_pilot import MODEL_ID, ranking_metrics, save

REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def normalized_question(question):
    return " ".join(question.lower().split())


def fetch_prefix(cache, split, count):
    def fetch(offset):
        path = cache / f"{split}_{offset:06d}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            url = (f"https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa"
                   f"&config=distractor&split={split}&offset={offset}&length=100")
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=90) as response:
                        raw = json.load(response)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            save(path, raw)
        assert len(raw["rows"]) == 100, (split, offset)
        assert not any(entry.get("truncated_cells") for entry in raw["rows"])
        print(f"DATA page {split} {offset} ready", flush=True)
        return [entry["row"] for entry in raw["rows"]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages = list(pool.map(fetch, range(0, count, 100)))
    return [row for page in pages for row in page]


def convert_rows(rows, excluded_ids, excluded_questions=()):
    seen_ids, seen_questions = set(excluded_ids), set(excluded_questions)
    converted = []
    for row in rows:
        titles, sentences = row["context"]["title"], row["context"]["sentences"]
        gold_titles = set(row["supporting_facts"]["title"])
        question = normalized_question(row["question"])
        if row["id"] in seen_ids or question in seen_questions:
            continue
        if len(titles) != 10 or len(set(titles)) != 10 or len(sentences) != 10:
            continue
        if not gold_titles <= set(titles) or not 1 <= len(gold_titles) <= 6:
            continue
        seen_ids.add(row["id"])
        seen_questions.add(question)
        converted.append({"id": row["id"], "question": row["question"], "type": row["type"],
                          "passages": [{"title": title, "text": " ".join(s.strip() for s in ss)}
                                       for title, ss in zip(titles, sentences)],
                          "gold": [i for i, title in enumerate(titles) if title in gold_titles]})
    return converted


def make_groups(records):
    import numpy as np
    from rank_bm25 import BM25Okapi
    groups = []
    for record_index, r in enumerate(records):
        tokens = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        bm25 = BM25Okapi([tokens(p["title"] + " " + p["text"]) for p in r["passages"]])
        ranking = np.argsort(-bm25.get_scores(tokens(r["question"])), kind="stable").tolist()
        negatives = [i for i in ranking if i not in r["gold"]][:4]
        assert len(negatives) == 4 and not set(negatives) & set(r["gold"])
        for positive in r["gold"]:
            groups.append({"record_index": record_index, "question_id": r["id"],
                           "positive": positive, "negatives": negatives})
    return groups


def prepare(args):
    cache = args.out.parent / "hotpot_pages"
    exclusions = set(json.loads(args.exclude_ids.read_text(encoding="utf-8-sig")))
    train_pool = convert_rows(fetch_prefix(cache, "train", 2000), exclusions)
    selected = random.Random(args.seed).sample(train_pool, args.train_questions + args.dev_questions)
    train, dev = selected[:args.train_questions], selected[args.train_questions:]
    all_train_ids = {r["id"] for r in selected}
    all_train_questions = {normalized_question(r["question"]) for r in selected}
    test_pool = convert_rows(fetch_prefix(cache, "validation", 1000), exclusions | all_train_ids,
                             all_train_questions)
    test = random.Random(args.seed + 1).sample(test_pool, args.test_questions)
    parts = [train, dev, test]
    for i in range(len(parts)):
        for j in range(i):
            assert not {r["id"] for r in parts[i]} & {r["id"] for r in parts[j]}
            assert not {normalized_question(r["question"]) for r in parts[i]} & {
                normalized_question(r["question"]) for r in parts[j]}
    dataset = {"train": train, "dev": dev, "test": test, "groups": make_groups(train),
               "excluded_prior_experiment_ids": sorted(exclusions),
               "source": "https://huggingface.co/datasets/hotpotqa/hotpot_qa", "license": "CC-BY-SA-4.0",
               "sampling": "seeded sample from first 2000 train and first 1000 validation rows; not full-corpus random sample"}
    save(args.out / "dataset.json", dataset)
    return dataset


def paired_comparison(before, after, seed):
    import numpy as np
    assert [r["id"] for r in before["rows"]] == [r["id"] for r in after["rows"]]
    rng = np.random.default_rng(seed)
    n = len(before["rows"])
    bootstrap_indices = rng.integers(0, n, size=(5000, n))
    report = {}
    for k in [2, 3, 5]:
        report[str(k)] = {}
        for metric in ["recall", "all_evidence", "ndcg"]:
            a = np.array([r["metrics"][str(k)][metric] for r in before["rows"]])
            b = np.array([r["metrics"][str(k)][metric] for r in after["rows"]])
            difference = b - a
            low, high = np.quantile(difference[bootstrap_indices].mean(axis=1), [0.025, 0.975])
            report[str(k)][metric] = {"delta_D_minus_C": float(difference.mean()),
                                     "paired_bootstrap_95ci": [float(low), float(high)],
                                     "improved": int((difference > 1e-10).sum()),
                                     "worsened": int((difference < -1e-10).sum()),
                                     "unchanged": int((abs(difference) <= 1e-10).sum())}
    return report


def main():
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path, required=True)
    parser.add_argument("--train-questions", type=int, default=1000)
    parser.add_argument("--dev-questions", type=int, default=100)
    parser.add_argument("--test-questions", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("BF16 CUDA GPU required")
    if min(args.train_questions, args.dev_questions, args.test_questions, args.epochs, args.accumulation) < 1:
        raise ValueError("Counts must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    data = prepare(args)
    print(f"DATA train={len(data['train'])} dev={len(data['dev'])} test={len(data['test'])} groups={len(data['groups'])}", flush=True)
    model_path = snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16

    def load_base():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=dtype, use_safetensors=True, local_files_only=True,
            attn_implementation="eager")
        assert model.config.num_labels == 1
        return model.to("cuda")

    def encode(question, texts):
        return tokenizer([question] * len(texts), texts, padding=True, truncation=True,
                         max_length=args.max_length, return_tensors="pt").to("cuda")

    @torch.no_grad()
    def evaluate(model, records, name):
        model.eval()
        # Identical small warmup for C and D; model loading and warmup excluded from timing.
        with torch.autocast("cuda", dtype=dtype):
            model(**encode(records[0]["question"], [p["text"] for p in records[0]["passages"][:5]]))
        rows = []
        for index, record in enumerate(records):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            scores = []
            for start in range(0, 10, 5):
                inputs = encode(record["question"], [p["text"] for p in record["passages"][start:start+5]])
                with torch.autocast("cuda", dtype=dtype):
                    outputs = model(**inputs).logits.float().view(-1)
                scores.extend(outputs.cpu().tolist())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            assert all(math.isfinite(s) for s in scores)
            ranking = sorted(range(10), key=lambda i: -scores[i])
            rows.append({"id": record["id"], "scores": scores, "ranking": ranking, "gold": record["gold"],
                         "seconds": elapsed, "metrics": {str(k): ranking_metrics(ranking, record["gold"], k)
                                                         for k in [2, 3, 5]}})
            if (index + 1) % 50 == 0:
                print(f"EVAL {name} {index+1}/{len(records)}", flush=True)
        result = {"rows": rows,
                  "mean": {str(k): {m: sum(r["metrics"][str(k)][m] for r in rows) / len(rows)
                                    for m in ["recall", "all_evidence", "ndcg"]} for k in [2, 3, 5]},
                  "mean_rerank_seconds": sum(r["seconds"] for r in rows) / len(rows)}
        save(args.out / f"{name}.json", result)
        print(f"METRICS {name} {json.dumps(result['mean'])}", flush=True)
        return result

    config = {"model": MODEL_ID, "revision": REVISION, "seed": args.seed,
              "train_questions": len(data["train"]), "dev_questions": len(data["dev"]),
              "test_questions": len(data["test"]), "groups_per_epoch": len(data["groups"]),
              "epochs": args.epochs, "max_length": args.max_length, "lr": 2e-5,
              "precision": "BF16 base and autocast, FP32 trainable weights", "lora_r": 8,
              "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["query", "value"],
              "modules_to_save": ["classifier"], "negative_mining": "BM25 top 4 excluding all gold",
              "microbatch_groups": 1, "accumulation": args.accumulation, "weight_decay": 0.01,
              "gradient_checkpointing": True, "warmup_fraction": 0.05, "lr_schedule": "linear warmup and decay",
              "selection": "best TRAINED epoch by dev all_evidence@3, then nDCG@3; earlier epoch wins ties",
              "test_policy": "official validation only used AFTER dev checkpoint selection; no test tuning",
              "candidate_policy": "all 10 provided paragraphs; equivalent to C candidate set; no title in model input",
              "gpu": torch.cuda.get_device_name(), "data_sha256": hashlib.sha256((args.out / "dataset.json").read_bytes()).hexdigest(),
              "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft"]},
              "limitations": ["English HotpotQA only", "subset sampled from prefixes, not entire dataset",
                              "no LLM answer quality measured", "timing reranker only, excludes BM25/dense",
                              "BF16 GPU baseline recomputed; do not compare to earlier CPU 20-question aggregate"]}
    save(args.out / "config.json", config)
    model = load_base()
    baseline_dev = evaluate(model, data["dev"], "dev_C")
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                                           lora_dropout=0.05, target_modules=["query", "value"],
                                           modules_to_save=["classifier"]))
    model.peft_config["default"].base_model_name_or_path = MODEL_ID
    model.peft_config["default"].revision = REVISION
    model.enable_input_require_grads()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for p in trainable.values():
        p.data = p.data.float()
    original = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    config["trainable_parameters"] = sum(p.numel() for p in trainable.values())
    config["total_parameters"] = sum(p.numel() for p in model.parameters())
    save(args.out / "config.json", config)
    optimizer = torch.optim.AdamW(trainable.values(), lr=config["lr"], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(data["groups"]) / args.accumulation)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, round(total_steps * 0.05))
    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    history, dev_epochs = [], []
    best_key = None
    best_epoch = None
    best_dir = args.out / "adapter"
    step = 0
    peak_allocated = peak_reserved = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        order = list(range(len(data["groups"])))
        random.Random(args.seed + epoch).shuffle(order)
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(order), args.accumulation):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            group_ids = order[start:start+args.accumulation]
            losses = []
            for group_id in group_ids:
                group = data["groups"][group_id]
                record = data["train"][group["record_index"]]
                inputs = encode(record["question"], [record["passages"][i]["text"]
                                                     for i in [group["positive"]] + group["negatives"]])
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(**inputs).logits.float().view(1, 5)
                    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device="cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                (loss / len(group_ids)).backward()
                losses.append(float(loss.detach()))
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            scheduler.step()
            step += 1
            torch.cuda.synchronize()
            row = {"epoch": epoch, "step": step, "loss": sum(losses) / len(losses),
                   "groups": len(group_ids), "seconds": time.perf_counter() - begin,
                   "grad_norm": float(grad_norm), "lr_next_step": scheduler.get_last_lr()[0]}
            history.append(row)
            with (args.out / "train_log.jsonl").open("a") as file:
                file.write(json.dumps(row) + "\n")
            if step % 25 == 0 or step == 1:
                recent = history[-25:]
                print(f"TRAIN epoch={epoch} step={step}/{total_steps} loss25={sum(r['loss'] for r in recent)/len(recent):.4f} seconds25={sum(r['seconds'] for r in recent)/len(recent):.3f}", flush=True)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 1024**3)
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 1024**3)
        model.gradient_checkpointing_disable()
        evaluated = evaluate(model, data["dev"], f"dev_epoch_{epoch}")
        dev_epochs.append(evaluated["mean"])
        key = (evaluated["mean"]["3"]["all_evidence"], evaluated["mean"]["3"]["ndcg"])
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            save(args.out / "selection.json", {"epoch": best_epoch, "dev_key": list(best_key),
                                              "rule": config["selection"], "test_not_yet_used": True})
            print(f"CHECKPOINT selected epoch={epoch} dev_key={key}", flush=True)
        # Keep every epoch checkpoint for reproducibility, never select using final test.
        model.save_pretrained(args.out / f"epoch_{epoch}_adapter", safe_serialization=True)
    changed = [n for n, p in trainable.items() if not torch.equal(original[n], p.detach().cpu())]
    assert any("lora_" in n for n in changed)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    del model, optimizer, scheduler, trainable, original, inputs, logits, loss, p
    gc.collect()
    torch.cuda.empty_cache()
    # Final test is touched only after the selected adapter is fixed.
    base = load_base()
    c = evaluate(base, data["test"], "test_C")
    del base
    gc.collect()
    torch.cuda.empty_cache()
    adapted = PeftModel.from_pretrained(load_base(), best_dir)
    for n, p in adapted.named_parameters():
        if "lora_" in n or "modules_to_save" in n:
            p.data = p.data.float()
    d = evaluate(adapted, data["test"], "test_D")
    reference = data["dev"][0]
    expected = json.loads((args.out / f"dev_epoch_{best_epoch}.json").read_text())["rows"][0]["scores"][:5]
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        actual = adapted(**encode(reference["question"], [p["text"] for p in reference["passages"][:5]])).logits.float().view(-1).cpu()
    reload_difference = float((actual - torch.tensor(expected)).abs().max())
    assert torch.allclose(actual, torch.tensor(expected), atol=0.02, rtol=0.005)
    paired = paired_comparison(c, d, args.seed)
    summary = {"status": "completed", "completed_utc": datetime.now(timezone.utc).isoformat(),
               "train_questions": len(data["train"]), "dev_questions": len(data["dev"]),
               "test_questions": len(data["test"]), "epochs": args.epochs, "optimizer_steps": step,
               "selected_epoch": best_epoch, "baseline_dev": baseline_dev["mean"], "dev_epochs": dev_epochs,
               "train_seconds": sum(r["seconds"] for r in history),
               "mean_optimizer_step_seconds": sum(r["seconds"] for r in history) / len(history),
               "epoch_mean_losses": {str(e): sum(r["loss"] * r["groups"] for r in history if r["epoch"] == e) /
                                     sum(r["groups"] for r in history if r["epoch"] == e) for e in range(1, args.epochs+1)},
               "peak_train_allocated_gib": peak_allocated, "peak_train_reserved_gib": peak_reserved,
               "C": c["mean"], "D": d["mean"], "paired_comparison": paired,
               "mean_rerank_seconds_C": c["mean_rerank_seconds"], "mean_rerank_seconds_D": d["mean_rerank_seconds"],
               "adapter_reload_verified": True, "reload_max_abs_difference": reload_difference,
               "updated_tensors": len(changed), "trainable_parameters": config["trainable_parameters"],
               "total_seconds": time.perf_counter() - started, "limitations": config["limitations"]}
    save(args.out / "summary.json", summary)
    with (args.out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "k", "recall", "all_evidence", "ndcg", "rerank_seconds"])
        writer.writeheader()
        for name, result in [("C_pretrained", c), ("D_lora", d)]:
            for k in [2, 3, 5]:
                writer.writerow({"method": name, "k": k, **result["mean"][str(k)],
                                 "rerank_seconds": result["mean_rerank_seconds"]})
    print("COMPLETED\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
