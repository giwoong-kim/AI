# Fakekiller: LoRA Reranker Experiment

SNS 사실확인 서비스 Fakekiller의 **근거 선택 모듈** 실험입니다. BGE 재정렬 모델을 HotpotQA로 LoRA 파인튜닝하고, 동일한 후보 문단에서 BM25·하이브리드 검색·재정렬을 비교했습니다.

이 저장소는 학습 코드, 선택된 LoRA 어댑터, 고정 데이터와 평가 결과를 공개합니다. 웹 서비스 전체 코드는 포함하지 않습니다.

## 결과

동일한 평가 질문 200개, 질문당 후보 문단 10개(정답 근거 2개)에서 상위 3개를 평가했습니다.

| 방식 | Recall@3 | 전체 근거 포함률@3 | nDCG@3 |
|---|---:|---:|---:|
| A: BM25 | 74.25% | 52.5% | 0.7512 |
| B: BM25 + BGE-M3 + RRF | 81.50% | 64.0% | 0.8178 |
| C: 기본 BGE Reranker | 93.75% | 87.5% | 0.9283 |
| D: LoRA 적용 BGE Reranker | 94.75% | 90.0% | 0.9391 |

전체 근거 포함률은 두 정답 근거를 모두 찾은 질문의 비율입니다. nDCG는 정답 근거가 높은 순위에 있는지를 평가합니다. 정답 근거의 관련도는 동일한 1, 나머지는 0으로 처리합니다.

**이 수치는 SNS 사실 판정 정확도가 아닙니다.** DDGS 웹 자료 수집과 최종 LLM 답변은 이 평가에 포함되지 않습니다. C/D는 후보 10개 전체를 정렬하므로 대규모 검색의 후보 축소 효과나 하이브리드와 재정렬의 결합 효과를 입증하지 않습니다. 영어 데이터셋 앞부분에서 추출한 탐색적 표본이며, 실제 SNS 일반화와 통계적 유의성을 이 표만으로 주장하지 않습니다. A/B/C/D 전체 지연시간 비교도 수행하지 않았습니다.

## 데이터 구성

- 출처: [HotpotQA](https://hotpotqa.github.io/), [Hugging Face distractor 구성](https://huggingface.co/datasets/hotpotqa/hotpot_qa).
- 공식 train 앞 2,000행에서 학습 1,000문항과 개발 100문항을 추출(seed 42).
- 공식 validation 앞 1,000행에서 평가 200문항을 추출(seed 43).
- 후보 10개인 문제만 사용하고 기존 실험 ID 및 중복 질문을 제외했습니다. 학습·개발·평가 간 ID와 정규화한 질문의 중복을 검사합니다.
- 원본 문장들을 문단으로 결합하고, supporting_facts에 나타난 문서 제목에 해당하는 문단을 정답으로 지정했습니다.
- 각 학습 묶음은 정답 문단 1개와 BM25 상위 방해 문단 4개로 구성합니다. 모든 정답 문단을 방해 문단에서 제외합니다.
- 데이터 라이선스·변환 내역은 [DATA_LICENSE.md](DATA_LICENSE.md)에 있습니다.

## 모델 및 학습

- 기반 모델: `BAAI/bge-reranker-v2-m3`
- 기반 모델 revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- LoRA: query/value, r=8, alpha=16, dropout=0.05. 분류 head도 학습·저장합니다.
- 2 epochs, lr=2e-5, 최대 512 tokens, gradient accumulation=4.
- BF16 기반 모델, FP32 학습 파라미터. 원 실험 GPU: RTX 4070 SUPER.
- 개발셋 전체 근거 포함률@3, 이어서 nDCG@3로 체크포인트 선택. 선택된 epoch는 2입니다.
- 테스트 결과를 계산하기 전에 개발셋으로 모델을 선택했습니다. `selection.json`은 해당 선택 시점의 기록입니다.

`adapter/`는 기반 모델 전체가 아니라 선택된 **추가 학습 가중치**입니다. 토크나이저는 위 기반 모델 revision에서 불러옵니다.

## 설치

Python 가상환경에서 실행합니다. 원 실험은 PyTorch 2.5.1, CUDA GPU 환경입니다.

```bash
pip install torch==2.5.1
pip install -r requirements-training.txt
```

PyTorch는 사용 환경에 맞는 CUDA 빌드를 설치하세요. 학습에는 BF16을 지원하는 CUDA GPU와 기반 모델 다운로드가 필요합니다. LLM API 키는 필요하지 않습니다.

## 학습 재실행

저장소 루트에서 실행합니다. 새 출력 디렉터리를 사용하세요.

```bash
python training/train_lora_compare.py --out runs/reproduction --exclude-ids training/prior_experiment_ids.json
```

이 명령은 데이터를 내려받아 학습·개발·평가 결과와 어댑터를 저장합니다. 공개된 실행의 고정 데이터와 결과는 `results/training/`에 있습니다. 환경·수치 정밀도·상위 데이터 변경에 따라 재실행 결과가 달라질 수 있습니다.

## A/B/C/D 비교 재실행

```bash
python evaluation/compare_abcd_200.py --source results/training --out runs/abcd_reproduction
```

이 스크립트는 고정 평가셋에서 A/B를 계산하고, 같은 데이터에 대해 저장된 C/D 예측을 재사용해 지표를 검증합니다. C/D까지 새로 계산하려면 먼저 학습 스크립트를 실행한 뒤 `--source runs/reproduction`을 사용하세요. 기존 실행 결과는 `results/abcd/`에 있습니다.

## 어댑터 사용

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

model_id = "BAAI/bge-reranker-v2-m3"
revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
base = AutoModelForSequenceClassification.from_pretrained(
    model_id, revision=revision, torch_dtype=dtype
)
model = PeftModel.from_pretrained(base, "adapter").to(device).eval()
pairs = [("Who directed the film?", "The film was directed by Jane Doe.")]
inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
with torch.inference_mode():
    scores = model(**inputs).logits.view(-1).float().cpu().tolist()
print(scores)  # 관련도 점수이며 진실 확률이 아닙니다.
```

이 예시는 사용 안내이며 이번 공개 준비 과정에서 모델 추론을 재실행하지 않았습니다. 원 학습 실행의 어댑터 재로딩 검증 기록은 `results/training/summary.json`에 포함돼 있습니다.

## 출처와 범위

- [BGE Reranker 기반 모델](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [HotpotQA](https://hotpotqa.github.io/)

기반 모델과 데이터는 각 원저작자의 이용 조건을 따릅니다. 이 저장소는 기존 모델을 사용한 학생 프로젝트 실험이며 새로운 기반 모델이나 사실확인 성능의 우위를 주장하지 않습니다.
