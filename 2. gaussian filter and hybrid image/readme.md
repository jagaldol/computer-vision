# Gaussian filter and Hybrid image
## Introduction
aligned images를 각각 low frequency와 high frequency로 filtering하여 두 이미지를 hybrid 시켜 본다.

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

## Process
1. Gaussian Filtering  
    <p align="center">
        <img src="./images/2b_dog.bmp?raw=true" width="45%" />
        <img src="./result_images/dog_convolution.png?raw=true" width="45%" />
    </p>

    이미지를 grayscale로 변환하여 단일 채널에 gaussian filter를 적용.

    ```python
    im = Image.open('./images/2b_dog.bmp')
    # convert to gray scale
    im = im.convert('L')
    im = np.asarray(im)

    # gaussian convolution with sigma 3
    im = gaussconvolve2d(im, 3)

    # convert to image format
    im = np.clip(im, 0, 255).astype(np.uint8)
    im = Image.fromarray(im)

    im.show()
    ```

2. Hybrid Image  
    <p align="center">
        <img src="./images/2a_cat.bmp?raw=true" width="45%" />
        <img src="./images/2b_dog.bmp?raw=true" width="45%" />
    </p>
    <p align="center">aligned images cat(left) / dog(right)</p>

    두 정렬된 이미지(고양이, 개)를 사용하여 hybrid 이미지를 생성한다.
    1. low frequency image 생성  
        <p align="center">
            <img src="./result_images/blur_image.png?raw=true" />
        </p>

        각 rgb 채널에 대해 convolution 수행 후 하나로 합침.

        ```python
        blur = Image.open('./images/2b_dog.bmp')
        blur = np.asarray(blur)
        sigma = 10

        # gaussian convolution into each rgb channel
        r = gaussconvolve2d(blur[:,:,0], sigma)
        g = gaussconvolve2d(blur[:,:,1], sigma)
        b = gaussconvolve2d(blur[:,:,2], sigma)

        # stack 3 channel and make into image
        blur = np.dstack([r, g, b])
        blur = np.clip(blur, 0, 255).astype(np.uint8)
        blurImage = Image.fromarray(blur)
        
        blurImage.show()
        ```

    2. high frequency image 생성  
        <p align="center">
            <img src="./result_images/high_freq_image.png?raw=true" />
        </p>

        원본 이미지에서 gaussian convolution을 적용한 이미지(blur)를 제거(high frequency image 생성).

        ```python
        sharpen = Image.open('./images/2a_cat.bmp')
        sharpen = np.asarray(sharpen)
        sigma = 10

        # gaussian convolution into each rgb channel
        r = gaussconvolve2d(sharpen[:,:,0], sigma)
        g = gaussconvolve2d(sharpen[:,:,1], sigma)
        b = gaussconvolve2d(sharpen[:,:,2], sigma)

        # stack 3 channel 
        blurredSharpen = np.dstack([r, g, b])
        blurredSharpen = np.clip(blurredSharpen, 0, 255).astype(np.int16)

        # generate high frequency image(original - blur)
        highFreqSharpen = sharpen - blurredSharpen

        # fit values from -128 to 127
        highFreqSharpen = np.clip(highFreqSharpen, -128, 127).astype(np.int16)

        # below is for visualization
        # add 128 for image format(0-255) and convert to image format(uint8)
        visulalizeHighFreqSharpen = highFreqSharpen + np.ones_like(highFreqSharpen) * 128
        visulalizeHighFreqSharpen = np.clip(visulalizeHighFreqSharpen, 0, 255).astype(np.uint8)
        visulalizeHighFreqSharpen = Image.fromarray(visulalizeHighFreqSharpen)
    
        visulalizeHighFreqSharpen.show()
        ```

    3. 두 이미지 합치기  
        <p align="center">
            <img src="./result_images/hybrid_image.png?raw=true" />
            <img src="./result_images/hybrid_image.png?raw=true" width="10%" />
        </p>
        <p align="center">최종 hybrid 이미지(좌우 같은 이미지)</p>

        생성한 두 이미지를 합쳐서 완성

        ```python
        # generate hybrid image(blurred one + another sharpen one)
        hybridImage = blur + highFreqSharpen

        # convert to image format
        hybridImage = np.clip(hybridImage, 0, 255).astype(np.uint8)
        hybridImage = Image.fromarray(hybridImage)

        hybridImage.show()
        ```

## Function Overview
### `gauss1d(sigma)`
sigma 값을 바탕으로 `normalize`된 1차원 gaussian filter를 생성

- `sigma`: standard deviation.

**returns:** 1-d gaussian filter.

### `gauss2d(sigma)`

sigma 값을 바탕으로 `normalize`된 2차원 gaussian filter를 생성

- `sigma`: standard deviation.

**returns:** 2-d gaussian filter.

### `convolve2d(array, filter)`

이미지 array에 filter로 convolution을 적용하여 리턴

- `array`: image. Numpy array of shape (H, W).
- `filter`: 2-d Numpy array.

**returns:**  filtered image. Numpy array of shape (H, W).

### `gaussconvolve2d(array, sigma)`

sigma 값을 바탕으로 `normalize`된 1차원 gaussian filter를 생성

- `array`: image. Numpy array of shape (H, W).
- `sigma`: standard deviation.

**returns:** gaussian filtered image. Numpy array of shape (H, W).
