# 데이터 출처 및 라이선스

`results/training/dataset.json`과 `results/abcd/dataset.json`은 HotpotQA의 일부를 변환한 자료입니다.

- 원저작자: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning.
- 원 연구: HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (EMNLP 2018).
- 프로젝트: https://hotpotqa.github.io/
- 배포 원본: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- 데이터 라이선스: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
- 라이선스 전문: https://creativecommons.org/licenses/by-sa/4.0/

변경 내용: 일부 행 표본 추출, 중복·기존 실험 ID 제외, 문장들을 문단으로 결합, supporting_facts의 제목을 문단 인덱스 정답으로 변환, 학습·개발·평가 분할 및 학습용 방해 문단 인덱스 추가.

변환 데이터는 동일한 CC BY-SA 4.0 조건으로 제공합니다. 원저작자가 이 프로젝트를 보증한다는 뜻은 아닙니다. 모델·코드의 이용 조건과 데이터의 이용 조건은 구분됩니다.
