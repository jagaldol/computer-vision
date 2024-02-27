# Canny Edge Detection

## Introduction

Canny edge detection의 구현을 직접 해본다.

Edge는 표면 법선, 깊이, 색상, 빛 불연속 등에 의해 발생한다. 픽셀에서 보았을 때 급격히
intensity value가 변하는 지점을 edge라고 간주할 수 있다. Canny edge detection에서는 Sobel Kernel을 사용하여 gradient를 계산하고 이를 바탕으로 edge를 탐색한다.

<p align="center">
    <img src="./images/iguana.bmp?raw=true" width="45%" />
    <img src="./result_images/canny_edge.png?raw=true" width="45%" />
</p>
<p align="center">original image(left) / canny edge detection(right)</p>

## Contents

1. Noise reduction
2. Find the intensity of gradient
3. Non-Maximum Suppression
4. Double threshold
5. Edge linking by hysteresis

## Process

1. Noise reduction
<p align="center">
    <img src="./result_images/noise_reduction_result.png?raw=true" />
</p>

이미지에는 기본적으로 노이즈 픽셀이 존재할 가능성이 높다. Gaussian convolution을 수행해 noise reduction을 한다.

2. Find the intensity of gradient
<p align="center">
    <img src="./result_images/intensity_gradient.png?raw=true" />
</p>

이미지의 edge를 구하기 위해 gradient를 계산한다. gradient는 sobel filter를 사용하여 계산한다.
sobel filter는 x, y 2개의 filter로 구성되어있다.

<p align="center">
    <img src="./assets/sobel_filter.png?raw=true" />
</p>

x, y 방향에서의 gradient, Ix와 Iy를 얻고, 이를 통해 gradient의 크기와 방향을 얻을 수 있다.

- **gradient magitude**  
   $||\nabla f|| = \sqrt{{I_x}^2 + {I_y}^2}$
- **gradient orientation**  
   $\theta = tan^{-1}(I_x/I_y)$

이는 python의 `NumPy.hypot()`와 `NumPy.arctan2()`를 사용하여 간단하게 구현 할 수 있다.

```python
def sobel_filters(img):
    # make xFilter, yFilter
    xFilter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
    yFilter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    # convolution
    Ix = convolve2d(img, xFilter)
    Iy = convolve2d(img, yFilter)

    # calculate gradient
    G = np.hypot(Ix, Iy)
    # mapping gradient into 0-255
    G = G / np.max(G) * 255
    # calculate theta
    theta = np.arctan2(Iy, Ix)
    return (G, theta)
```

3. Non-Maximum Suppression
<p align="center">
    <img src="./result_images/non_max_suppression.png?raw=true" />
</p>

각 픽셀들의 gradient 방향(theta)에 따라 local maxima를 계산한다. 주변 8개의 픽셀들과의 관계를 바탕으로 계산하기 때문에 theta값에 따라 0도, 45도, 90도, 135도로 분류하여 계산한다.

<p align="center">
    <img src="./assets/degrees.png?raw=true" />
</p>

가장 바깥을 제외한 픽셀들의 degree에 대해 0도, 45도, 90도, 135도로 근사 시켜, 좌우 극한을 구해 local maxima인지 판단한다. Local maxima인 값 만을 남기고 나머지는 0으로 부여해 non-maximum suppression을 수행한다.

4. Double threshold
<p align="center">
    <img src="./result_images/double_thresholding.png?raw=true" />
</p>

- $diff = max(image) - min(image)$
- $T_{high} = min(image) + diff \times 0.15$
- $T_{low} = min(image) + diff \times 0.03$

$T_{high}$이상이면 strong edge(value: 255), $T_{low}$이하면 weak edge(value: 80)으로 만들고 나머지는 0으로 없앤다.

5. Edge linking by hysteresis
<p align="center">
    <img src="./result_images/canny_edge.png?raw=true" />
</p>

strong edge와 이어져있는 weak edge만을 accept한다. strong edge에서 dfs를 수행해 최종 edge들을 찾아낸다.

## Function Overview

> [2. gaussian filter and hybrid image](https://github.com/jagaldol/computer-vision/tree/main/2.%20gaussian%20filter%20and%20hybrid%20image)의 함수를 포함하고 있습니다.

### `sobel_filters(img)`

이미지의 gradient 크기와 방향 리턴.

- `img`: Grayscale image. Numpy array of shape (H, W).

**returns:**

- `G`: gradient magnitude image with shape of (H, W).
- `theta`: direction of gradients with shape of (H, W).

### `non_max_suppression(G, theta)`

방향(theta)을 0도, 45도, 90도, 135도, 4가지로 분류하여 크기가 local maxima가 아닌 픽셀 값 제거.

- `G`: gradient magnitude image with shape of (H, W).
- `theta`: direction of gradients with shape of (H, W).

**returns:** non-maxima suppressed image.

### `double_thresholding(img)`

double threshold를 적용한 이미지 리턴.

- `img`: numpy array of shape (H, W) representing NMS edge response.

**returns:** double_thresholded image.

### `dfs(img, res, i, j, visited=[])`

시작지점(i, j)에서부터 인접한 weak edge를 dfs로 탐색하여 res에 저장.

- `img`: numpy array of shape (H, W) representing double_threshold edge response.
- `res`: numpy array of shape (H, W) to store dfs result.
- `i`: start index.
- `j`: start index.
- `visited`: visited list for dfs.

### `hysteresis(img)`

double_thresholding 이후 결과를 받아, strong edges와 연결된 weak edges를 찾아 edge로 만들어 리턴.

- `img`: numpy array of shape (H, W) representing double_threshold edge response.

**returns:** hysteresised image.
