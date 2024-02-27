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
     8-points algorithm을 사용하여 fundamental matrix를 추정 가능하다.

<p align="center">
    <img src="./Lab%20Description%20and%20Report/8-points-algorithm.png?raw=true" />
</p>

2. Compute epipoles  
   $Fe = 0$을 만족시키는 점이 `epipole`이므로 $F$를 다시 특이값 분해를 하여 마지막 영공간의 특이 벡터를 구하여 `normalize`하여 계산한다.

   ```python
   def compute_epipoles(F):
       # solve Fe1 = 0 by SVD
       U, S, V = np.linalg.svd(F)
       e1 = V[-1]
       e1 = e1/e1[2] # normalization

       # solve F_te2 = 0 by SVD
       U, S, V = np.linalg.svd(F.T)
       e2 = V[-1]
       e2 = e2/e2[2] # normalizaiton

       return e1, e2
   ```

3. Epipolar lines  
   계산한 `epipole`과 각 특징점으로 이미지 양 끝에 대해 `epipolar line` 위의 점을 구하고 `epipolar line`을 이미지에 그려 시각화를 한다.

   ```python
   axes[0].scatter(x1, y1, color=colors, s=100, marker='.')
   axes[1].scatter(x2, y2, color=colors, s=100, marker='.')

   axes[0].plot([min_x1, max_x1], [min_y1, max_y1], color=colors)
   axes[1].plot([min_x2, max_x2], [min_y2, max_y2], color=colors)
   ```

<p align="center">
    <img src="./result_images/warrior_result.png?raw=true"/>
</p>
<p align="center">epipolar line - Warrior</p>

<p align="center">
    <img src="./result_images/graffiti_result.png?raw=true"/>
</p>
<p align="center">epipolar line - graffiti</p>
