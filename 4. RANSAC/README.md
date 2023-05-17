# SIFT Keypoint Matching with RANSAC
## Introduction
이미지 매칭에서 RANSAC 알고리즘을 직접 구현하여 적용해 본다.

RANSAC은 RANdom SAmple Comsesus의 약자로, 여러 개에서 일부분의 샘플을 골라 하나의
가정(assumption)을 만들고, 가정이 다른 데이터에도 잘 맞아 떨어지면 채택한다.


<p align="center">
    <img src="./result_images/library_matchRANSAC.png?raw=true"/>
</p>
<p align="center">RANSAC image matching</p>

## Execution
```sh
(.venv) $ python main_match.py
```

## Contents
1. Image descriptors Matching
2. Image descriptors RANSAC Matching

## Process
1. Image descriptors Matching  
    각 이미지의 keypoints와 descriptors들은 이미지와 함께 개별 파일로 존재한다.
    - 이미지 파일 : data/some_image.pgm
    - keypoints와 descriptors 파일 : data/some_image.key
        - keypoints: (row, column, scale, orientation)
        - descriptors: 128 Diemsions SIFT descriptors

    descriptor matching 판단을 위해 ratio distance를 사용한다.

    $\text{ratio distance} = ||f_1 - f_2|| / ||f_1 - f_2'|| $
    
    - $f_1$ : 기준 이미지에서의 feature(descriptor) 위치
    - $f_2$ : 비교 이미지에서의 best match의 위치
    - $f_2'$ : 비교 이미지에서의 second best match의 위치

    > 두 번째 best match의 거리 계산하여 첫번째 best match의 거리에 나눈다.  
    > 비슷한 특징점이 많은 경우를 제외 가능하다.

    <p align="center">
        <img src="./result_images/scene-book_match.png?raw=true"/>
    </p>
    <p align="center">Image descriptors Matching Result</p>

    RANSAC을 거치지 않아 잘못된 지점을 매칭한 것을 확인할 수 있다.

2. Image descriptors RANSAC Matching  
    RANSAC 매칭을 추가한다. 이번 lab에서는 homography를 사용하지 않고 간단하게 방향과 크기를 통하여 match를 판별한다.

    

    <p align="center">
        <img src="./result_images/library_match.png?raw=true" />
    </p>
    <p align="center">non-RANSAC</p>

    <p align="center">
        <img src="./result_images/library_matchRANSAC.png?raw=true"/>
    </p>
    <p align="center">RANSAC</p>

    잘못된 매칭(왼쪽 아래 창문)이 사라짐을 볼 수 있다.

    - 사용한 파라미터  
        | parameter         | value |
        |-------------------|-------|
        | ratio_threshold   | 0.6   |
        | orient_agreement  | 5     |
        | scale_agreement   | 0.1   |

