
from PIL import Image
import numpy as np
import math

## Gaussian Filtering
### 1. boxfilter
def boxfilter(n):
    """
    Normalize된 NxN numpy 배열 생성
    """
    # n이 odd가 아니면 assert error
    assert n%2==1, "Dimension must be odd"
    # n x n 의 array를 n x n으로 나눠 normalize
    return np.ones((n, n)) / (n * n)


### 2.gauss1d(sigma)
def gauss1d(sigma):
    """
    1차원 gaussian filter 생성
    """
    length = math.ceil(sigma * 6)                       # 6배 적용 후 올림
    if length % 2 == 0:                                 # odd로 변경
        length += 1
    xs = np.arange(-(length // 2), length // 2 + 1, 1)  # 중간의 element가 0인 length 길이의 array 생성
    gaussian = np.exp(-xs**2 / (2 * sigma ** 2))        # array에 gauss함수 적용
    return gaussian / np.sum(gaussian)                  # normalize하여 return


### 3. gauss2d(sigma)
def gauss2d(sigma):
    """
    2차원 gaussian filter 생성
    """
    gaussian1d = gauss1d(sigma)             # 1-D gaussian filter 생성
    return np.outer(gaussian1d, gaussian1d) # 1-D gaussian filter 외적으로 2-D gaussfitler 생성


### 4. convovle2d(array, filter)
def convolve2d(array, filter):
    """
    filter를 적용한 image array 생성
    """
    height, width = array.shape     # image 크기 정보
    filterSize = filter.shape[0]    # filter 크기 정보

    paddingSize = filterSize // 2   # paddingsize

    modifiedArray = np.zeros((height + 2 * paddingSize, width + 2 * paddingSize))   # padding을 추가한 0으로된 빈 array 생성
    modifiedArray[paddingSize:-paddingSize, paddingSize: -paddingSize] = array      # image 정보를 빈 array에 추가(패딩 적용된 이미지 생성)

    result = np.zeros((height, width))                              # filter 적용 결과 저장할 빈 array

    np.flip(filter, axis=0)                                         # filter 각 차원에 대하여 뒤집기
    np.flip(filter, axis=1)

    for i in range(height):
        for j in range(width):
            window = modifiedArray[i:i+filterSize, j:j+filterSize]  # 각 pixel에 적용될 image part 추출
            result[i,j] = np.sum(window * filter)                   # 각 pixel에 filter 적용
    
    return result
    

def gaussconvolve2d(array, sigma):
    """
    gaussian filter를 적용한 image array 생성
    """
    filter = gauss2d(sigma)             # gaussian filter 생성
    return convolve2d(array, filter)    # image array에 gaussian filter 적용


def main():
    print("part1: Gaussian Filtering")
    print("1.")
    print("boxfilter(3) :", boxfilter(3))
    print("boxfilter(4) will make assert error. Please check report.")
    print("boxfilter(7) :", boxfilter(7))
    print("2.")
    print("gauss1d(0.3):", gauss1d(0.3))
    print("gauss1d(0.5):", gauss1d(0.5))
    print("gauss1d(1):", gauss1d(1))
    print("gauss1d(2):", gauss1d(2))
    print("3.")
    print("gauss2d(0.5):", gauss2d(0.5))
    print("gauss2d(1):", gauss2d(1))

    print("4.")
    fileName1 = '2b_dog.bmp'
    im = Image.open('./images/' + fileName1)                  # 개 사진 불러오기
    print('open '+ fileName1)
    im.show()
    im = im.convert('L')                        # gray scale 변경
    im = np.asarray(im)                         # array로 변경

    print('gaussconvolve2d to ' + fileName1 + '...')
    im = gaussconvolve2d(im, 3)                 # gaussian filter 적용 
    im = np.clip(im, 0, 255).astype(np.uint8)   # 0, 255로 값 제한 및 uint8로 변경
    im = Image.fromarray(im)                    # image로 변경
    im.save("./result_images/dog_convolution.png", "PNG")       # 로컬에 저장
    print("make dog_convolution.png")
    im.show()
    print()
    print("===================================================")
    print()
    print("Part 2: Hybrid Images")
    ## Hypbrid Images
    ### 1. Gaussian filtered low frequency image
    fileName2 = '2b_dog.bmp'                                  # 탑 사진 불러오기
    blur = Image.open('./images/' + fileName2)
    print('open '+ fileName2)
    blur.show()
    blur = np.asarray(blur)
    sigma = 10

    print('gaussconvolve2d to ' + fileName2 + ' r channel ...')
    r = gaussconvolve2d(blur[:,:,0], sigma)                    # 각 rgb 채널에 gaussian filter 적용
    print('gaussconvolve2d to ' + fileName2 + ' g channel ...')
    g = gaussconvolve2d(blur[:,:,1], sigma)
    print('gaussconvolve2d to ' + fileName2 + ' b channel ...')
    b = gaussconvolve2d(blur[:,:,2], sigma)

    blur = np.dstack([r, g, b])                                # rgb 채널 합침
    blur = np.clip(blur, 0, 255).astype(np.uint8)             # 0, 255로 값 제한 및 uint8로 변경
    blurImage = Image.fromarray(blur)
    blurImage.save("./result_images/blur_image.png", "PNG")                    # 결과 image 저장
    print("make blur.png")
    blurImage.show()


    fileName3 = '2a_cat.bmp'                                                             # sharpen 사진 불러오기
    sharpen = Image.open('./images/' + fileName3)
    print('open '+ fileName3)
    sharpen.show()                                  
    sharpen = np.asarray(sharpen)
    sigma = 10

    print('gaussconvolve2d to ' + fileName3 + ' r channel ...')
    r = gaussconvolve2d(sharpen[:,:,0], sigma)                                               # 각 rgb 채널에 gaussian filter 적용
    print('gaussconvolve2d to ' + fileName3 + ' g channel ...')
    g = gaussconvolve2d(sharpen[:,:,1], sigma)
    print('gaussconvolve2d to ' + fileName3 + ' b channel ...')
    b = gaussconvolve2d(sharpen[:,:,2], sigma)

    blurredSharpen = np.dstack([r, g, b])                                                    # rgb 채널 합침
    blurredSharpen = np.clip(blurredSharpen, 0, 255).astype(np.int16)                         # uint시 값 뺐을 때 overflow 될 수 있기에 int로 타입 지정

    highFreqSharpen = sharpen - blurredSharpen                                                 # 원본 - blur 이미지로 high frequency 이미지 생성

    highFreqSharpen = np.clip(highFreqSharpen, -128, 127).astype(np.int16)                    # -128에서 127로 제한

    visulalizeHighFreqSharpen = highFreqSharpen + np.ones_like(highFreqSharpen) * 128          # 출력을 위해 128더해 0 ~ 255 값 가지게 변경
    visulalizeHighFreqSharpen = np.clip(visulalizeHighFreqSharpen, 0, 255).astype(np.uint8)   # 타입 uint8로 변경
    visulalizeHighFreqSharpen = Image.fromarray(visulalizeHighFreqSharpen)
    print("make high_freq_image.png")
    visulalizeHighFreqSharpen.save("./result_images/high_freq_image.png", "PNG")                              # high frequency 결과 저장 및 확인

    visulalizeHighFreqSharpen.show()

    hybridImage = blur + highFreqSharpen                            # hybrid 이미지
    hybridImage = np.clip(hybridImage, 0, 255).astype(np.uint8)     # 0 ~ 255 제한 및 uint8 변경
    hybridImage = Image.fromarray(hybridImage)
    hybridImage.save("./result_images/hybrid_image.png", "PNG")              # 이미지 저장 및 출력
    print("make hybrid_image.png")
    hybridImage.show()


main()
