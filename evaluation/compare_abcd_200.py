"""Evaluate A/B on the frozen C/D test set; reuse audited C/D predictions."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from retrieval_eval import tokenize, rrf


def metrics(ranking, gold, k):
    gold = set(gold)
    hits = len(set(ranking[:k]) & gold)
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranking[:k]) if p in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dict(recall=hits / len(gold), all_evidence=float(hits == len(gold)), ndcg=dcg / ideal)


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data_path = args.source / 'dataset.json'
    records = json.loads(data_path.read_text())['test']
    assert len(records) == 200
    runs = {label: json.loads((args.source / name).read_text())['rows']
            for label, name in [('C', 'test_C.json'), ('D', 'test_D.json')]}
    expected = [r['id'] for r in records]
    for rows in runs.values():
        assert [r['id'] for r in rows] == expected
        for r, result in zip(records, rows):
            assert r['gold'] == result['gold']
            assert sorted(result['ranking']) == list(range(10))
            for k in [2, 3, 5]:
                for key, value in metrics(result['ranking'], r['gold'], k).items():
                    assert abs(value - result['metrics'][str(k)][key]) < 1e-12
    save(args.out / 'dataset.json', records)
    print('Frozen 200 questions and C/D metrics verified. Loading bge-m3.', flush=True)
    torch.set_num_threads(4)
    model = SentenceTransformer('BAAI/bge-m3', device='cuda')
    model.max_seq_length = 512
    model.eval()
    runs['A'], runs['B'] = [], []
    for index, record in enumerate(records):
        passages = record['passages']
        bm25 = BM25Okapi([tokenize(p['title'] + ' ' + p['text']) for p in passages])
        scores = bm25.get_scores(tokenize(record['question']))
        a = np.argsort(-scores, kind='stable').tolist()
        with torch.inference_mode():
            vectors = model.encode([record['question']] + [p['text'] for p in passages],
                                   normalize_embeddings=True, batch_size=4, show_progress_bar=False)
        dense_scores = vectors[1:] @ vectors[0]
        assert np.isfinite(dense_scores).all()
        dense = np.argsort(-dense_scores, kind='stable').tolist()
        b = rrf([a, dense])
        for label, ranking in [('A', a), ('B', b)]:
            assert sorted(ranking) == list(range(10))
            runs[label].append(dict(id=record['id'], gold=record['gold'], ranking=ranking,
                                    metrics={str(k): metrics(ranking, record['gold'], k) for k in [2, 3, 5]}))
        if (index + 1) % 20 == 0:
            print(f'A/B {index + 1}/200', flush=True)
    summary = []
    for label in ['A', 'B', 'C', 'D']:
        save(args.out / f'{label}.json', {'rows': runs[label]})
        for k in [2, 3, 5]:
            row = dict(method=label, n=200, k=k)
            row.update({metric: float(np.mean([r['metrics'][str(k)][metric] for r in runs[label]]))
                        for metric in ['recall', 'all_evidence', 'ndcg']})
            summary.append(row)
    config = dict(source_run=str(args.source), source_dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                  embed_model='BAAI/bge-m3', embed_revision=getattr(model[0].auto_model.config, '_commit_hash', None),
                  embedding_dtype=str(next(model.parameters()).dtype), device='cuda', max_length=512,
                  rrf_k=60, candidates=10, seed='frozen source test set',
                  policy='BM25 title+text; dense and reranker body only. C/D predictions reused on identical test data.',
                  limitations=['candidate ranking, not full-corpus retrieval', 'C/D rerank all ten candidates; hybrid order is irrelevant except ties',
                               'no LLM answer evaluation', '200-question exploratory subset already inspected',
                               'no latency comparison; dense FP32 and reranker BF16'])
    save(args.out / 'config.json', config)
    save(args.out / 'summary.json', summary)
    with (args.out / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
