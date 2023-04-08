# Gaussian filter and Hybrid image
## Introduction
aligned images를 각각 low frequency와 high frequency로 filtering하여 두 이미지를 hybrid시켜 본다.

만들어진 hybrid Image는 크게 보았을 때 high frequency 이미지가 강조되고, 작게 보았을 때 low frequency로 만든 이미지가 강조된다.

<p align="center">
    <img src="./result_images/hybrid_image.png?raw=true" />
    <img src="./result_images/hybrid_image.png?raw=true" width="10%" />
</p>
<p align="center">최종 hybrid 이미지(좌우 같은 이미지)</p>

## Contents
1. Gaussian Filtering 구현
    1. Box filter 구현
    2. 1D – Gaussian filter 구현
    3. 2D – Gaussian filter 구현
    4. Gaussian filter를 사용하여 이미지에 convolution 적용
2. Hybrid Image 생성
    1. low frequency image 생성
    2. high frequency image 생성
    3. 두 이미지 합치기





<!--
## process
1. chipmunk_head.png  
    Image.crop()을 사용하여 머리 부분 좌표로 이미지 잘라내기
2. chipmunk_head_bright.png  
    50 더하여 모든 픽셀 밝게(픽셀의 intensity의 범위는 0-255로 최대값 255)  
    python
    for x in range(0,150):
        for y in range(0,150):
            im3_array[y,x] = min(im3_array[y,x] + 50, 255)
    
3. chipmunk_head_dark.png  
    0.5 곱하여 모든 픽셀 어둡게(image의 타입은 uint8로 맞춰야 함)  
    python
    im4_array = im4_array * 0.5
    im4_array = im4_array.astype('uint8')
-->
