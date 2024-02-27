# Panorama Stitching

## Introduction

이미지 매칭을 토대로 여러 이미지를 하나로 합쳐 Panorama 이미지를 만든다.

파노라마 이미지는 한 이미지에서 다른 이미지 공간으로 projection으로 만들 수 있다. 이동, 회 전, 크기 뿐만 아니라 비틀림까지 나타내기 위해서 homography를 사용한다. Homography는 매칭 되는 점 4개를 통해 구할 수 있고, 가장 잘 맞는 homography를 구하기 위해 RANSAC을 사용한다.

<p align="center">
    <img src="./result_images/Rainier_pano.png?raw=true"/>
</p>
<p align="center">Panorama image</p>

## Execution

you just **execute `main_pano.py`**.

`main_proj.py` and `test_pano.py` are for testing.

```sh
(.venv) $ python main_pano.py
```

- **main_proj.py**  
   you can test keypoints projection from one image to other image.
    <p align="center">
        <img src="./result_images/keypoint_projection.png?raw=true"/>
    </p>
- **test_pano.py**  
   make panorama image using cv2 for test.

## Contents

1. Keypoint Projections
2. RANSAC Homography

## Process

1. **Keypoint Projections**

projection test를 위해 `data/test.pkl`에 `data/Hanging1.png`, `data/Hanging2.png`의 매칭된 pair와 homography가 이미 만들어져 있다.

```python
# Make homogeneous
xy_points_hg = np.append(xy_points, np.ones((xy_points.shape[0], 1)), axis=1)
# Apply h matrix
xy_points_proj = h.dot(xy_points_hg.T).T
# Avoid Divide By Zero
is_zero_points = xy_points_proj[:, 2] == 0
xy_points_proj[is_zero_points, 2] = 1e-10

# Return to Regular Coordinate
xy_points_out = xy_points_proj[:, :2] / xy_points_proj[:, 2:]

return xy_points_out
```

`test.pkl`을 사용하여 `Hanging` 이미지 2개를 projection 시켜본다.

<p align="center">
    <img src="./result_images/keypoint_projection.png?raw=true"/>
</p>

2. **RANSAC Homography**  
   매칭된 `keypoints`들로 homography를 만든다. 각 image들의 keypoints는 같은 이름의 `.pkl` 파일로 작성되어있다.

<p align="center">
    <img src="./Lab Description and Report/homography.png?raw=true"/>
</p>

$||Ah - 0||$을 최소화하는 `h`를 구해야한다. 이는 $A^TA$의 가장 작은 `eigenvalue`에 대응하는 `eigenvector`로 구할 수 있다.

일부 `match keypoints(4쌍)`으로 계산한 homography를 사용하여, `keypoint projection`을 수행 후 inlier 개수를 센다. 이때 RANSAC 기법을 사용하여 여러번 반복 후 가장 많은 inlier를 얻은 homography를 선택한다.

> Refer to `RANSACHomography(xy_src, xy_ref, num_iter, tol)` of `solution.py`

계산한 homography로 cv2 라이브러리를 사용해 이미지를 하나로 합친다. 이미지들의 겹치는 부분은 단순하게 투명도를 0.5씩 할당하여 합친다.

이미지마다 빛이 다르기 때문에 겹친 부분의 선이 뚜렷하게 나타날 수 있다. 이를 해결하기 위해서 더 나은 blending 방법을 도입해야할 것이다. 예를 들어, `Laplacian pyramid`를 활용하여 해결할 수 있다.

<p align="center">
    <img src="./result_images/Foundation_pano.png?raw=true"/>
</p>
