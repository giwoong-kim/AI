# Selected LoRA adapter

This directory contains the epoch-2 adapter selected using the development set, not the full base model.

Base: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

Training: HotpotQA 1,000 questions, two epochs, query/value LoRA (r=8, alpha=16, dropout=0.05), with classifier parameters saved. See the root README and `results/training/config.json` for reproduction and limitations.

Load using PEFT on top of the pinned base model. Tokenizer files are obtained from that base revision. Scores indicate question–passage relevance, not factual truth or calibrated confidence.

The base model is distributed under Apache License 2.0; a copy is included in `LICENSE-BASE-MODEL.txt`. This adapter adds HotpotQA-trained LoRA matrices and updates the saved classifier parameters; it is not an unmodified copy of the base checkpoint. Base-model attribution: BAAI / FlagEmbedding.
