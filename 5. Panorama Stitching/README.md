# Panorama Stitching
## Introduction
이미지 매칭을 토대로 여러 이미지를 하나로 합쳐 Panorama 이미지를 만든다.

파노라마 이미지는 한 이미지에서 다른 이미지 공간으로 projection으로 만들 수 있다. 이동, 회 전, 크기 뿐만 아니라 비틀림까지 나타내기 위해서 homography를 사용한다. Homography는 매칭 되는 점 4개를 통해 구할 수 있고, 가장 잘 맞는 homography를 구하기 위해 RANSAC을 사용한다.


<p align="center">
    <img src="./result_images/Rainier_pano.png?raw=true"/>
</p>
<p align="center">Panorama image</p>

## Execution

you just use `main_pano.py`. The rest of files are for testing.

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

1. Keypoint Projections
2. RANSAC Homography
