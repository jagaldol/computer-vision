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

## Process(예정)
> lab 채점 이후 코드와 함께 추가 예정

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

**returns:**  non-maxima suppressed image.

### `double_thresholding(img)`

double threshold를 적용한 이미지 리턴.

- `img`: numpy array of shape (H, W) representing NMS edge response.

**returns:**  double_thresholded image.

### `dfs(img, res, i, j, visited=[])`

시작지점(i, j)에서부터 인접한 weak edge를 dfs로 탐색하여 res에 저장.

- `img`: numpy array of shape (H, W) representing double_threshold edge response.
- `res`: numpy array of shape (H, W) to store dfs result.
- `i`: start index.
- `j`: start index.
- `visited`: visited list for dfs.

### `hysteresis(img)`

double_thresholding 이후 결과를 받아, strong edges와 연결된 weak edges를 찾아 edge로 만들어 리턴.

-  `img`: numpy array of shape (H, W) representing double_threshold edge response.

**returns:**  hysteresised image.