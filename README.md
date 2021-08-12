# Self-Supervised_Learning (SSL)

## 정의 및 용어
- 정의 : 레이블이 없는 데이터로부터 데이터간의 관계를 통해 레이블을 자동으로 학습
- 필요성 : 지도학습은 강력하지만 많은 양의 레이블링된 데이터 필요 → 시간 및 비용 소모 多
- Pretext Task : 사용자가 정의한 문제 (ex. Downstream Task를 위해 시각적 특징을 배우는 단계)
- Downstream Task : SSL로 학습된 features들이 잘 학습되었는지 판단하며, 최종적으로 이루고자하는 task (ex. 분류)

## GAN 기반
- 단점 : 픽셀 단위로 복원 or 예측하기에 계산 복잡도 높다

### AE(Auto Encoder)
- 신경망을 통해 재건한(reconstruction) 출력 데이터의 정답 레이블 = 입력 데이터 사용 → 입력과 출력의 결과가 같도록 학습
<img src="https://user-images.githubusercontent.com/67107675/128656576-92e082f3-91b8-4767-8c16-f69fd1a73537.png" width="70%">

### SSGAN(Self-Supervised GAN)
- 판별자는 이미지의 real/fake와 rotation degree(0, 90, 180, 270) 총 2가지를 구분
- 생성자가 만든 이미지를 rotation, rotation 얼마나 했는지 cross entropy loss로 만들어서 GAN loss에 더함 → identity(회전 x) 이미지에 대해서 real/fake 구분, 나머지 이미지에 대해서는 rotation angle 학습
<img src="https://user-images.githubusercontent.com/67107675/128664866-7aa8fed7-794e-4993-a7d3-a99dbfa70939.png" width="70%">

## Pretext Task 기반
- 사용자가 새로운 문제를 정의(=pretext task) 및 학습함으로 모델이 데이터 자체에 대한 이해 ↑ → pre-train 되어진 모델을 transfer learning(이전 학습)함으로 downstream task(ex. 분류)를 더 잘 해결할 수 있음
### Context Prediction
- object 부분을 인지하여 공간적 관계를 파악하는 것을 학습
- Pros : 최초의 자기지도학습 방법으로 object 부분에 대하여 학습하게 하는 직관적인 태스크
- Cons
 > 1. 학습 이미지가 표준성이 있어야 하며, 이미지의 representation(대표성)이 목적이지만 patch를 학습한다.
 > 2. 다른 이미지의 negative가 없기 때문에 충분히 fine-grained(정밀) 하지 않다.
 > 3. 중심 patch 외 8개의 선택지 뿐이라 output space가 적다.
 <img src="https://user-images.githubusercontent.com/67107675/128665052-2a5fec95-c81a-4b04-9e73-55a6384336cb.png" width="70%">

### Jigsaw puzzles
- 이미지를 패치로 분할하고 순서를 바꾼다. 네트워크가 순서를 예측하는 것이 목표
 <img src="https://user-images.githubusercontent.com/67107675/128665491-851a9f91-dd63-4d03-84cd-7898310bc4a1.png" width="70%">

### Rotation prediction
- 원래 이미지의 canonical(규정)된 orientation(방향)을 잘 이해해야 한다. → 이미지의 전반적 특징 학습 가능
- Pros : 적용이 용이
- Cons 
 > 1. 학습 이미지가 표준성(ex.중력으로 땅이 바닥에있는)이 있어야 한다.
 > 2. 다른 이미지의 negative가 없기 때문에 충분히 fine-grained(정밀) 하지 않다.
 > 3. 중심 patch 외 8개의 선택지 뿐이라 output space가 적다.
<img src="https://user-images.githubusercontent.com/67107675/128665562-554addfd-4657-467a-9c91-e01384920f6b.png" width="70%">
<img src="https://user-images.githubusercontent.com/67107675/128665720-d08cb140-df84-44c7-bbe1-46456ff375c6.png" width="70%">

### Exemplar ConvNets
- 이미지에서 분할 추출(=seed patch)하여 transformation(변형) 한 뒤, 다른 종류의 여러 seed 중 같은 원본 seed를 고르게 학습
- Pros : N개의 exemplar에 N개의 클래스 존재 → fine-grained(세밀한) 정보 보존 가능
- Cons : N개의 exemplar에 N개의 클래스 존재 → 파라미터 수 ↑ & 소요 시간 ↑
<img src="https://user-images.githubusercontent.com/67107675/128666141-ea1ef6af-4a57-4433-adfd-c7b15430b941.png">
<img src="https://user-images.githubusercontent.com/67107675/128665947-b436b994-3768-477a-ae26-9b62cc63f2d0.png" width="70%">

## Contrastive learning (대조 학습)
- 추출된 feature값은 instance간의 유사도 정보가 있을 것이라는 가정에서 시작되며, Positive/Negative pair로 구되어 Positive pair끼리 거리를 좁히고, Negative pair끼리는 거리를 멀리 띄워놓는 것이 학습 원리
> 1. 같은 image에 서로 다른 augmentation 실행, 두 positive pair의 feature representation은 거리가 가까워(유사해)지도록 학습
> 2. 다른 image에 서로 다른 augmentation 실행, 두 negative pair의 feature representation은 거리가 멀어지도록 학습


### MoCo (Momentum contrast for unsupervised visual representation learning)
- 과정
> 1. 데이터를 인코더에 통과시켜 key와 query를 생성
> 2. query와 매칭되는 key는 유사하게, 매칭되지 않는 key는 차별화 되도록 학습
> 3. 인코딩된 key들을 queue에 삽입하여 dictionary 구성

### SimCLR (A simple framework for contrastive learning of visual representations)
- 과정
> 1. 원본에서 예제를 무작위로 추출하여 두 번 변환 (random cropping & color distortion)
> 2. representation(대표성)계산
> 3. representation(대표성)의 비선형 투영 계산 → representation 자체에서 대조 손실을 구하는 것보다 성능 좋음
<img src="https://user-images.githubusercontent.com/67107675/128668285-a1c569d8-3011-4136-8625-fcfa18574025.png" width="70%">
<img src="https://user-images.githubusercontent.com/67107675/128818464-894c7faf-f5c0-4daf-ab48-8cb77f3b89d7.png" width="50%">


### BYOL (Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning)
- 과정
> 1. 2개 network에 서로 다른 augmentation을 적용하여 feature vector(prediction) 추출
> 2. online network와 target network의 projection output에 L2 normalization
> 3. MSE를 최소화시키는 방향으로 online network를 학습(online network output = target network output 되도록 학습)
> 4. loss의 대칭화를 위해 사용한 augmentation을 바꿔서 loss 한번 더 계산하여 두 loss의 합으로 학습
- Pros
> 1. negative 쌍을 사용하지 않고, positive 쌍만 이용하여 SimCLR보다 2% 성능 향상
> 2. 기존 negative 쌍을 사용 시 배치 크기가 매우 커야 학습이 잘 된다는 문제 & 데이터 증강에 따른 성능 편차가 크다는 문제 해결
> 3. representation을 잘 배우는 것이므로 학습이 끝나면 onlline network의 encoder 제외 나머지는 사용 안함
<img src="https://user-images.githubusercontent.com/67107675/128669208-3e68df1c-7a3f-4266-85fd-dcf0fd9d98f8.png" width="70%">


## 출처
https://velog.io/@tobigs-gm1/Self-Supervised-Learning#2-%EC%B4%88%EC%B0%BD%EA%B8%B0%EC%9D%98-gan-%EA%B8%B0%EB%B0%98-%EC%9E%90%EA%B8%B0%EC%A7%80%EB%8F%84%ED%95%99%EC%8A%B5

https://daeun-computer-uneasy.tistory.com/37
