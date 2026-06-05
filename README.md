# Fashion MNIST CNN Classification

인공지능개론 미니 프로젝트

---

## 프로젝트 개요

Fashion MNIST 데이터셋을 이용하여 의류 이미지를 분류하는 CNN(Convolutional Neural Network) 모델을 구현하였습니다.

본 프로젝트는 다음 두 가지 방식으로 진행되었습니다.

### 2.1 PyTorch CNN

- PyTorch 프레임워크를 활용하여 CNN 모델 구현
- Fashion MNIST 데이터셋 학습 및 평가
- 학습/테스트 정확도 그래프 생성

### 2.2 Scratch CNN

- 「밑바닥부터 시작하는 딥러닝」 교재의 CNN 소스를 기반으로 구현
- 딥러닝 프레임워크 API를 사용하지 않고 NumPy 기반으로 구현
- Fashion MNIST 데이터셋 학습 및 평가
- CNN 구조를 교재 예제에서 확장하여 적용

---

## Scratch CNN 구조

기존 교재 구조

text Conv - Relu - Pool - Affine - Relu - Affine - Softmax 

변경 후 구조

text Conv1 - Relu1 - Pool1 Conv2 - Relu2 - Pool2 Affine1 - Relu3 Affine2 - Softmax 

변경 사항

- 합성곱 계층 1개 → 2개로 확장
- 필터 수 증가
- Hidden Layer 크기 조정
- Adam Optimizer 적용
- Fashion MNIST 데이터셋에 맞도록 구조 개선

---

## 실험 환경

- Python 3
- NumPy
- Matplotlib
- PyTorch

---

## 최종 결과

### Scratch CNN

- Train Accuracy : 97.50%
- Test Accuracy : 92.00%

과제 요구사항

- Train Accuracy 95% 이상
- Test Accuracy 92% 이상

요구 조건을 충족하였습니다.
