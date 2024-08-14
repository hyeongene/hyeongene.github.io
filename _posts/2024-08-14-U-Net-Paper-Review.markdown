---
layout: post
published: true
title:  "[Paper Review]U-Net: Convolutional Networks for Biomedical Image Segmentation"
date:   2024-08-14 16:10:12 +0900
categories: Computer Vision
---
![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/4877ec0b-c235-4809-a13f-d5a8d252d0a0/image.png)

## 1. Abstract

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/4d480929-3a8f-4b2d-9364-3b04558097ce/image.png)

- U-Net 에서 주목해야 할 부분 중 하나는 biomedical image에 특화된 강력한 data augmentation 기법이다.
- U-Net의 아키텍쳐는 이미지의 의미정보(context, 이미지 전체 맥락으로도 해석 가능)를 뽑아내는 contracting path와 정확한 이미지 localization을 가능하게 하는 대칭적인 구조의 expanding path로 구성되어 있다.
- 다시 말해,  contracting path는 image의 전역적인 feature를 압축적으로 뽑아내고, 이후의 expanding path는 압축된 image feature를 다시 원본 이미지 크기에 맞춰 키워줌(이로 인해 precise localization이 가능하다).  이후 각 픽셀별 레이블을 예측하는 segmentation이 진행된다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/e7884355-055c-4f98-9ee9-56ace330c89f/7d0ac75f-bcf3-4bd0-8429-191415b48a50.png)

## 1. Introduction

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/908e0127-519b-45f8-a1b8-971dac6f4929/image.png)

- 기존의 cnn은 보편적으로 이미지의 단일 레이블을 예측하는 classification taske에 많이 활용되었다.
- 하지만 biomedical image processing의 경우 이미지의 단일 레이블이 아니라 이미지를 이루는 각 픽셀별 클래스 레이블을 output으로 요구하는 경우가 많다 !!
- —> should include localization

- U-Net 이전의 연구 언급: 이미지를 Patch단위로 나누어 input으로 활용하여 1) 이미지의 local region정보 획득 2) 데이터 부족 문제를 해결하려는 시도가 있었다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/2ce35e34-5c85-46bc-bb18-34455b13b5b0/e17fef06-eafe-4983-ae87-22ca750677cf.png)

- 그러나 해당 선행연구에는 두 가지 한계점이 있었는데…

1. 느리다 !! : 이미지가 여러 개의 patch로 나뉘어 input으로 들어갔으며, patch간의 중복도 많아 학습이 매우 느렸음
2. trade-off가 존재 : larger patch의 경우 더 많은 pooling layer를 거치게 되는데 그러면 이미지의 local한 정보를 많이 잃게 됨 + small patch의 경우 이미지의 의미 정보(context)가 많이 담기지 않음

여기서  2번 trade off에 대해 자세히 짚고 갑시다.

Image Segmentation의 경우 앞서 언급 했듯이 기존의 classification과는 요구되는 output이 다르다.

- 이미지의 global한 context를 압축적으로 잘 담은 feature map을 활용해 단일 레이블만 잘 예측하면 되는 기존의 classification과 달리
- 이러한 context 정보와 함께 이미지의 정밀한 local 정보가 수반되어야 각 픽셀의 클래스를 예측하는 segmentation task 수행이 가능함

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/3baeaca6-1907-4d64-a07d-94108d7ad448/image.png)

→ 보편적으로 convolution layer의 얕은 층에서는 이미지의 local하고 detail한 정보가 담긴 feature map이, 깊은 층에서는 이미지의 global 하고 abstract한.. context 정보가 담긴 비교적 작은 size의 feature map이 생성되는 경향이 있기 때문에 이 둘 간에 trade-off가 존재한다고 표현한다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/ec8ce680-0369-45f5-9a03-a9e3e99eff84/7b530f86-2ff3-4f41-8571-ec4fd7eda366.png)

→ U-Net은 이러한 기존의 연구가 갖는 한계를 fcn의 수정 및 보완을 통해 해결하고 있다.

→ 아주 적은 이미지로도 잘 작동하며 더 정확한 segmentation을 수행하는 U-Net !!

**해당 자료에서는 fcn을 따로 다루고 있지 않다. fcn의 main idea에 대한 상세한 내용이 궁금하신 분은 따로 찾아보시는 걸 추천드립니다//  (CNN2 세션에서 다루었음)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/82e06d98-11b5-47ca-9848-9ace0a850d90/5e723e35-f068-410a-8673-2c7d47a35191.png)

important modifications are…

- upsampling part(expansive path)에서 다수의 feature channel을 두고 있음 (기존 FCN보다 더 많은 conv filter channel 사용)
- → which allow the network to propagete context information to higher resolution layers.
- → 이를 통해 고해상도 계층(이미지 사이즈가 큰 계층을 의미하는 듯하다) 에 이미지의 context information 전달이 가능하다 !!

** 앞서 trade-off와 연결해서 해석해보면 기존의 한계를 해결했음을 알 수 있음. 

고해상도 계층 = 얕은 계층 = 의미정보 잘 안담김

 → 근데 여기에 의미정보 전달 가능 → trade off 해결?

- This strategy( = X using fc layer)는 입력 이미지 크기에 제한 받지 않으며, over-tile strategy를 이용해 매끄러운 (seamless) segmentation 수행이 가능하도록 만들다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/364e7dfe-ba81-42bf-9028-c519299ac7f6/image.png)

- Overlap-tile strategy
- yellow area segmentation을 위해선 input으로 blue area를 요구함. (U-Net의 특징)
- 따라서 missing input(blue에서  yellow를 뺀 만큼의 데이터 = 논문에서는 missing context라고 언급)은 mirroring에 기반하여 추론 된다.

이에 대해 더 자세히 알아보자.

U-Net의 구조를 살펴보면 알 수 있는 부분이 input과 output size가 다르다는 것 ! (input이 더 큼.)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/5f275e2a-b231-4ee1-a016-ab3901f34723/image.png)

이를 해석해보면 실제로 segmentation이 수행되어 나오는 부분은 input image의 일부분이라고 할 수 있음.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/105f4eb2-10e4-422f-9edd-65e5b9511844/image.png)

따라서 실제 segmentation하고자 하는 patch가 있다면 mirroring을 활용한 missing context 추론(extrapolation)을 통해 실제보다 큰 사이즈의 patch를 만듦 → 이를 input으로 제공하게 된다.

결과적으로 중복 없는 패치 단위의 이미지 segmentation 이 가능함 (이걸 매끄럽다(seamless)고 표현한 듯)

→ 이러한 mirroring extrapolation은 biomedical image 의 특성에 의해 가능. 

→ overlap-tile strategy 덕분에 input 이미지가 아무리 커도 segmentation이 가능.

→ 중복이 없어 기존의 속도가 느리다는 한계점 어느 정도 해결 가능

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/0f89f2d7-1e33-4c3d-91c1-ad76b41c9179/f31dfd46-e73a-4e99-9d2c-bafb3aa0e448.png)

Segmentation 학습을 위해 매우 적은 training data가 있었기 때문에 excessive data augmentation 을 진행했다고 한다. 

- elastic deformation 외에도 여러가지 방법론을 적용

→ 논문에서 나중에 데이터 증강을 한 문단으로 따로 정리해두었기에 일단 넘어가도록 하자.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/5664c8dc-8926-4029-ab21-6fbb02be155f/ca6573ec-7ec9-4774-9334-ca8f1bf15b9a.png)

- cell segmentation에서 또 다른 어려움은 닿아있는 같은 class 세포 간의 분리이다.
- 이를 위해 weighted loss 활용, 역시 뒤에서 자세히 살펴보자.

## 2. Network Architecture

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/be173925-5baa-4cc2-b85d-06225cd72d91/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/107cc3ed-204d-4ba6-8dac-028c95ef0dff/image.png)

- U 모양의 Network Architecture를 뜯어보자.

(과제에서 여러분이 파이토치로 구현해야 하는 부분입니다.)

- Contracting path
    - Two 3x3 Conv filter ( stride = 1 with zero padding, ReLU) layer - featuremap size reduction O
    - 2x2 Max Pooling layer : featuremap size reduction to half <<
    - 이 과정을 3번 반복한다.
    - feature channel의 개수를 반복마다 2배로 늘려준다.
        - (64→128→256→512→1024 가 되는 것 확인 가능)
    
- Bottle Neck
    - 수축 경로에서 확장 경로로 전환되는 구간
    - 3×3 Convolution Layer + ReLu + BatchNorm (No Padding, Stride 1) 두 번 반복
    - Dropout Layer
        - 마지막에 적용된 Dropout Layer는 모델을 일반화하고 노이즈에 견고하게(Robust) 만드는 장치. (data augmentation 수행과 맥을 같이함)
    
- Expansive path
    - feature channel의 개수를 반으로 줄여주며 output size는 두 배로 늘려주는 2x2 de-conv filter
    - Crop and Concat
    - Two 3x3 Conv filter ( stride = 1 with zero padding, ReLU) layer
    - Final layer : 388 x 388 x 64 feature map을 output segmentation map으로 만들어 주기 위해 1x1 conv filter 이용
        - → 388x388x2 feature map 생성 (2는 classification 해야 할 class 개수임.)
        
    - 그림을 잘 보면 feature channel 개수가 de-conv를 거치면서 한 번, 3x3 conv filter를 거치면서 한 번 더 반으로 주는 것을 알 수 있음.
    - 이는 디코더(expansive path는 디코더와도 같은 역할을 함)의 일반적인 특징으로 특징맵의 수를 점진적으로 줄이면서, 모델의 복잡성과 계산 부하를 줄이고, 메모리 사용량을 관리하는 과정. 이를 통해 모델의 효율성을 증가시키고, 학습 시간을 단축시킬 수 있음.
    - 엥 de-conv 거치면서 feature channel 개수 안 줄어드는데요?? → 뒤에서 설명 드릴게요 !!

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/f328d98e-f438-491d-bed8-4b090d55ea6d/image.png)

구조 일부를 따로 떼온 모습이다.

(expansive path에서 언급했던 crop & concat 과정을 표현한 그림.)

- ResNet과 FCN 에서도 언급되었던 Skip Connection이 U-Net에서 조금 변형하여 활용됨.

**Skip Connection : 얕은 층의 fine location 정보와 깊은 층의 global semantic 정보를 결합하는 architecture

contracting path의 각 단계에서 output을 crop하여 저장해둠.

→ expansive path에서 feature map이 conv layer를 거칠 때 대칭 단계의 contracting path에서 저장되어 있던 output이 해당 feature map과 concat되어 conv layer의 input으로 활용됨.

de-conv를 거치면서 feature channel수가 줄어들지 않은 게 아니라,

반으로 줄어든 다음 

→ 앞서 저장해 둔 contracting path의 output과 concat되어 다시 feature channel 수가 두 배로 늘어나 → 줄어들지 않은 것처럼 보이는 것 ! (그림의 (2) 256+ 256을 확인해주세요)

Crop 은 왜 하나요?

→ U자형의 Symmetric한 구조임에도 각 단계에서 concat되는 contracting path의 output과 expansive path의 input size가 다름 

→ 이를 맞춰 주기 위해 output을 expansive path의 input size에 맞게 crop하여 저장

→ 이를 거치며 앞서 mirroring이 적용된 부분이 제거되는 것으로 추정

## 3. Training

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/21b39c71-d07b-428a-8eea-8acf90c2c556/85d9932b-7821-4b59-bcd9-fe7a43725251.png)

- Optimizer
    - Stochastic Gradient Descent (SGD)
    - Momentum : 0.99
- Deep learning framework : Caffe
- Batch : A single Image

1) Loss Function

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/da17a84e-d359-4183-897e-7e144b2655c3/ce8d0b34-4426-49eb-8490-449bede5a2d2.png)

위의 수식은 U-Net의 손실함수이다. 기존과 달리 각 픽셀마다의 softmax 값이 필요하다. (여기서 x는 특정 pixel 좌표 정보를 represent한다고 할 수 있음)

수식을 보면 Cross-Entropy식에 가중치 w(x)를 새로 도입하고 있는데, 개념부터 이해하고 넘어가 보자. 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/55b300fd-fee3-4a0c-ab7e-bd666efe5389/image.png)

앞서 언급 했듯이 Cell image segmentation의 어려움 중 하나는 같은 class를 갖는 인접한 cell 간의 구분이다. 

해당 논문에서는 이를 위해 경계에 해당하는 pixel을 더 잘 분류할 수 있도록 boundary에 속한 pixel들의 가중치를 더 강하게 주는 손실함수를 새로 정의하고 있다. 

→ 여기서 “가중치”가 바로 위 수식의 w(x)에 해당함.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/e66302e3-8a44-4d25-8185-080f8a2a35a7/2e6ed83c-9535-4f08-8b0d-a6eab9296345.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/0429fa9b-c140-4e3a-bb31-b1cd7741a924/c7a0d7db-d598-4787-9fd3-794c9fbc1732.png)

출처: https://joungheekim.github.io/2020/09/28/paper-review/ 

w0에 집중하여 수식을 살펴보자…

가장 가까운 세포 두 개까지의 거리가 가까운 pixel 일수록 

→ 거리합 제곱의 값이 작고

→ 여기에 - 가 붙기 때문에 exp() 함수의 input값은 커진다

→ 지수함수값은 커짐 → 최종적으로 해당 pixel은 큰 가중치 w(x) 를 갖게 된다

>> 손실함수 계산 시에 큰 가중치를 두게 된다 !!

2) Weight Initialization

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/0beb1da3-cc7a-4ae1-9cbe-557cdc0e1f3f/597e16f5-76c4-4956-b34a-1f1cbac08721.png)

- Gaussian distribution initialization (=standard deviation distribution)
    - Standard deviation :
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/63774ccc-3e43-4503-9f85-37a5acfe5c80/image.png)
        
    - N = conv filter size x conv filter channel

## 3.1 Data Augmentation

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/c6d9962c-ac4d-4fd9-8d26-8dc23bb5b9a2/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7362d1f8-12cf-44ce-9c42-854ee11e25c9/ce478698-39a0-4e78-ac22-e21e92b467b0/image.png)

- Data Augmentation
    - Shift
    - Rotation
    - Gray Value
    - Random elastic deformation

## Summary

U-Net은요…

1) 이미지 픽셀로부터 의미정보를 추출(contract), 이를 기반으로 각 픽셀마다 객체를 분류하는 (expand) 두 path로 이루어진 U 모양의 아키텍쳐 제시

2) 서로 근접한 객체 경계를 잘 구분하도록 학습의 weighted loss 제시

3)  biomedical 분야에 적합한 증강 기법을 제시하여 소규모 데이터셋이 갖는 한계 극복

해당 논문 리뷰를 잘 참고하여 과제를 수행해보세요.

수고하셨습니다 !!

References: 

논문 링크 : https://arxiv.org/abs/1505.04597 

https://89douner.tistory.com/297
