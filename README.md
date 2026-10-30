# 📩 SMS Spam Classification

### 머신러닝 기반 스팸 메시지 분류 및 모델 성능 비교

**TF-IDF와 머신러닝을 활용한 SMS 스팸 분류 및 오분류 분석 프로젝트**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-7952B3?style=flat-square)

**2026 명지대학교 MCC 코딩동아리 | 개인 프로젝트**

</div>

---

## 📋 01. 프로젝트 소개 | Overview

### 프로젝트 개요

SMS 메시지를 **스팸(Spam)**과 **정상(Ham)**으로 분류하는 머신러닝 프로젝트입니다.

문자 데이터를 전처리하고 TF-IDF로 수치화한 뒤, 세 가지 머신러닝 모델의 성능을 비교했습니다.

전체 데이터의 스팸 비율이 13.4%로 클래스 불균형이 존재하기 때문에, 단순 정확도뿐 아니라 **Precision, Recall, F1-score와 실제 오분류 사례**를 함께 분석했습니다.

| 항목 | 내용 |
|---|---|
| 프로젝트명 | SMS Spam Classification |
| 프로젝트 유형 | MCC 코딩동아리 개인 프로젝트 |
| 최초 수행 | 2026.03 ~ 2026.06 |
| 재구현 및 검증 | 2026.10 (Claude 활용) |
| 개발 분야 | Machine Learning, NLP, Text Classification |
| 주요 기술 | Python, scikit-learn, Pandas, Matplotlib |
| 최종 선택 모델 | Linear SVM |
| 주요 성과 | F1-score 0.936 |

### 프로젝트 목표

- SMS 텍스트 전처리 및 TF-IDF 특징 추출
- Naive Bayes, Logistic Regression, Linear SVM 성능 비교
- 클래스 불균형을 고려한 평가 지표 분석
- False Positive와 False Negative를 비교하여 모델 선택

---

## 🛠️ 02. 기술 스택 | Tech Stack

| 분류 | 기술 | 활용 내용 |
|---|---|---|
| Language | Python | 데이터 처리 및 모델 학습 |
| Data Processing | Pandas | SMS 데이터 로드 및 분석 |
| Machine Learning | scikit-learn | TF-IDF 벡터화, 분류 모델 학습 및 평가 |
| Visualization | Matplotlib | 성능 비교 및 분석 결과 시각화 |
| NLP | TF-IDF | SMS 텍스트를 수치형 특징으로 변환 |
| Models | Naive Bayes, Logistic Regression, Linear SVM | 분류 성능 및 오분류 비교 |

---

## 🏗️ 03. 데이터 및 처리 과정 | Pipeline

### 데이터셋

SMS Spam Collection 데이터를 사용하여 총 **5,572개의 메시지**를 분석했습니다.

| 구분 | 내용 |
|---|---|
| 전체 데이터 | 5,572개 |
| 정상 메시지(Ham) | 약 86.6% |
| 스팸 메시지(Spam) | 약 13.4% |
| 학습 데이터 | 4,457개 (80%) |
| 테스트 데이터 | 1,115개 (20%) |
| 데이터 분할 | 클래스 비율을 유지한 Stratified Split |

### 머신러닝 파이프라인

```text
SMS 데이터 수집
       │
       ▼
   텍스트 전처리
       │
       ▼
  학습·테스트 분할
       │
       ▼
    TF-IDF 변환
       │
       ▼
   분류 모델 학습
       │
       ├── Naive Bayes
       ├── Logistic Regression
       └── Linear SVM
       │
       ▼
  모델 성능 비교 및 평가
       │
       ▼
    오분류 사례 분석
       │
       ▼
      모델 선택
```

### 데이터 전처리

- 텍스트 소문자 변환
- 영문과 숫자를 제외한 문자 정제
- 영어 불용어(Stop Words) 제거
- TF-IDF를 활용한 텍스트 벡터화
- 학습 데이터와 테스트 데이터를 분리하여 평가

---

## 💻 04. 주요 구현 | Implementation

### ① TF-IDF 기반 텍스트 벡터화

머신러닝 모델에서 텍스트를 처리할 수 있도록 TF-IDF를 사용하여 SMS 메시지를 수치형 벡터로 변환했습니다.

**테스트 데이터의 정보가 학습에 유입되지 않도록 TF-IDF 변환기는 학습 데이터에만 적합(fit)**했습니다.

```python
vec = TfidfVectorizer(stop_words="english")

X_train_v = vec.fit_transform(X_train)
X_test_v = vec.transform(X_test)
```

- `fit_transform()` : 학습 데이터의 단어 정보를 학습하고 변환
- `transform()` : 학습된 변환기를 사용하여 테스트 데이터 변환

이를 통해 데이터 누출(Data Leakage)을 방지하도록 구성했습니다.

### ② 머신러닝 모델 학습

세 가지 분류 모델을 사용하여 동일한 테스트 데이터에서 성능을 비교했습니다.

| 모델 | 특징 |
|---|---|
| Naive Bayes | 확률 기반 텍스트 분류 |
| Logistic Regression | 선형 모델 기반 이진 분류 |
| Linear SVM | 클래스 구분을 위한 선형 결정 경계 학습 |

### ③ 성능 평가 및 오분류 분석

다음 평가 지표를 활용했습니다.

- **Precision:** 스팸으로 예측한 메시지 중 실제 스팸 비율
- **Recall:** 실제 스팸 메시지 중 올바르게 탐지한 비율
- **F1-score:** Precision과 Recall의 조화평균
- **False Positive:** 정상 메시지를 스팸으로 잘못 분류
- **False Negative:** 스팸 메시지를 정상으로 잘못 분류

학습 데이터에서 5회 교차검증을 수행하고, 별도의 테스트 데이터로 모델 성능을 평가했습니다.

---

## 🔍 05. 주요 문제 해결 | Problem Solving

### 클래스 불균형을 고려한 모델 평가

**Problem — 문제 발견**

전체 데이터 중 정상 메시지가 약 86.6%를 차지했습니다.

따라서 모든 메시지를 정상으로 예측하더라도 약 86.6%의 정확도가 나올 수 있었습니다.

단순 정확도만으로는 실제 스팸 탐지 성능을 충분히 평가하기 어렵다고 판단했습니다.

**Solution — 대응 방법**

- Precision, Recall, F1-score를 함께 비교
- 실제 스팸을 놓친 경우(FN) 분석
- 정상 메시지를 차단한 경우(FP) 분석
- 클래스 가중치를 적용한 Logistic Regression 추가 비교

**Result — 결과**

Linear SVM은 Naive Bayes 대비 놓친 스팸 메시지 수를 **33개에서 17개로 감소**시켰습니다.

스팸에 가중치를 적용한 Logistic Regression은 놓친 스팸이 13개로 더 적었지만, 정상 메시지 13개를 스팸으로 잘못 분류했습니다.

정상 메시지 차단을 최소화하면서 스팸 탐지 성능을 확보하기 위해 **Linear SVM을 최종 모델로 선택**했습니다.

---

## 📊 06. 모델 평가 결과 | Results

### 모델별 F1-score

| 모델 | F1-score |
|---|---:|
| Naive Bayes | 0.875 |
| Logistic Regression | 0.863 |
| **Linear SVM** | **0.936** |

※ 동일한 테스트 데이터 1,115개 기준입니다.

### 오분류 비교

| 모델 | 놓친 스팸 (FN) | 정상 메시지 오분류 (FP) |
|---|---:|---:|
| Naive Bayes | 33개 | 0개 |
| Logistic Regression | 36개 | 0개 |
| **Linear SVM** | **17개** | **1개** |
| 가중치 적용 Logistic Regression | 13개 | 13개 |

### 최종 모델 선택

**Linear SVM — F1-score 0.936**

- 세 가지 기본 모델 중 가장 높은 F1-score
- Naive Bayes 대비 놓친 스팸 메시지 16개 감소
- 정상 메시지를 스팸으로 잘못 분류한 사례 1개
- 스팸 탐지 성능과 정상 메시지 보호 사이의 균형 고려

### 실제 오분류 사례

**False Negative — 스팸을 정상으로 분류**

```text
ringtoneking 84484
```

**False Positive — 정상 메시지를 스팸으로 분류**

```text
K..u also dont msg or reply to his msg..
```

이처럼 실제 오분류 메시지를 확인하여 모델이 놓치는 패턴을 분석했습니다.

---

## 📌 07. 프로젝트 정보 및 개선 방향 | Project Information

| 항목 | 내용 |
|---|---|
| 활동 | MCC 코딩동아리 |
| 프로젝트 형태 | 개인 프로젝트 |
| 수행 기간 | 2026.03 ~ 2026.06 |
| 재구현 | 2026.10 |
| 사용 언어 | Python |
| 최종 선택 모델 | Linear SVM |
| 테스트 F1-score | 0.936 |

### 향후 개선 방향

- 문자 단위 n-gram 특징 적용
- 전화번호 및 특수 패턴 정규화 검토
- 오분류 메시지 유형별 성능 분석 확대

### AI 활용 안내

2026년 3~6월 진행한 개인 프로젝트를 바탕으로, 2026년 10월 Claude를 활용하여 코드를 재구현하고 실행 결과를 검증했습니다.
