# Epipolar Geometry

## Introduction

Epipolar Geometry에서 epipole을 찾고, epipole을 이용하여 epipolar line을 그린다.


<p align="center">
    <img src="./result_images/warrior_result.png?raw=true"/>
</p>
<p align="center">RANSAC image matching</p>


## Contents

1. Fundamental Matrix Estimation
2. Compute epipoles
3. Epipolar lines

## Process

1. Fundamental Matrix Estimation  
    서로 다른 위치에서 같은 대상을 찍은 두 이미지와 함께 두 이미지에서 상응하는 특징점들의 배열이 주어진다. 이 특징점들을 이용하여 fundamental matrix를 찾을 수 있다.

    - **fundamental matrix**  
        $q^T Fp=0$  
        - **q**: 한 이미지의 특징점 좌표(homogeneous coordinate)
        - **p**: 다른 이미지의 특징점 좌표(homogeneous coordinate)
        - **F**: Fundamental Matrix

    - **8-points algorithm**  

        ![8-points algorithm](./Lab%20Description%20and%20Report/8-points-algorithm.png)

        8-points algorithm을 사용하여 fundamental matrix를 추정 가능하다.

2. Compute epipoles  
    추가 예정

3. Epipolar lines  
    추가 예정
