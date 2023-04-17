from PIL import Image
import math
import numpy as np


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


def gauss2d(sigma):
    """
    2차원 gaussian filter 생성
    """
    gaussian1d = gauss1d(sigma)             # 1-D gaussian filter 생성
    return np.outer(gaussian1d, gaussian1d) # 1-D gaussian filter 외적으로 2-D gaussfitler 생성


def convolve2d(array, filter):
    """
    filter를 적용한 image array 생성
    """
    height, width = array.shape     # image 크기 정보
    filterSize = filter.shape[0]    # filter 크기 정보

    paddingSize = filterSize // 2   # paddingsize

    modifiedArray = np.zeros((height + 2 * paddingSize, width + 2 * paddingSize))   # padding을 추가한 0으로된 빈 array 생성
    modifiedArray[paddingSize:-paddingSize, paddingSize: -paddingSize] = array      # image 정보를 빈 array에 추가(패딩 적용된 이미지 생성)

    result = np.zeros((height, width)).astype(np.float32)           # filter 적용 결과 저장할 빈 array

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


def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:, 
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    xFilter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)       # make xFilter, yFilter
    yFilter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)

    Ix = convolve2d(img, xFilter)   # convolution
    Iy = convolve2d(img, yFilter)
    
    G = np.hypot(Ix, Iy)            # calculate gradient
    G = G / np.max(G) * 255         # mapping gradient into 0-255
    theta = np.arctan2(Iy, Ix)      # calculate theta
    return (G, theta)


def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    H, W = G.shape
    res = np.zeros((H, W))                      # empty palette fill with 0s
    angle = theta * 180 / np.pi                 # change radian into degree
    angle[angle < 0] = angle[angle < 0] + 180   # align angles according to the direction

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            # 0도
            if (157.5 <= angle[i, j] <= 180) or (0 <= angle[i, j] < 22.5):
                lLimit = G[i, j - 1]
                rLimit = G[i, j + 1]

            # 45도
            elif 22.5 <= angle[i, j] < 67.5:
                lLimit = G[i - 1, j + 1]
                rLimit = G[i + 1, j - 1]

            # 90도
            elif 67.5 <= angle[i, j] < 112.5:
                lLimit = G[i - 1, j]
                rLimit = G[i + 1, j]

            # 135도
            elif 112.5 <= angle[i, j] < 157.5:
                lLimit = G[i - 1, j - 1]
                rLimit = G[i + 1, j + 1]

            if (G[i, j] > lLimit) and (G[i, j] > rLimit):   # 좌우보다 더 큰것만 저장
                res[i, j] = G[i, j]
    return res


def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """
    diff = np.max(img) - np.min(img)            # make two thresholds
    thresholdH = np.min(img) + diff * 0.15
    thresholdL = np.min(img) + diff * 0.03

    weak = 80
    strong = 255
    res = np.zeros(img.shape)
    res = np.where((thresholdL <= img) & (img < thresholdH), np.ones(img.shape) * weak, res)    # thresholds 사이면 weak
    res = np.where(thresholdH <= img, np.ones(img.shape) * strong, res)                         # high threshold 이상이면 strong
    return res


def dfs(img, res, i, j, visited=[]):
    # 호출된 시점의 시작점 (i, j)은 최초 호출이 아닌 이상 
    # strong 과 연결된 weak 포인트이므로 res에 strong 값을 준다
    res[i, j] = 255

    # 이미 방문했음을 표시한다
    visited.append((i, j))

    # (i, j)에 연결된 8가지 방향을 모두 검사하여 weak 포인트가 있다면 재귀적으로 호출
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)


def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    H, W = img.shape
    res = np.zeros((H, W))
    visited = []

    for i in range(1, H - 1):
        for j in range(1, W - 1):               # 모든 픽셀(테두리 제외)에 대하여
            if (img[i][j] == 255):              # strong pixel이면
                dfs(img, res, i, j, visited)    # dfs 수행
    
    return res


def main():
    print("1. Noise reduction")                                 # 1. Noise reduction
    print("open original image")
    img = Image.open('./images/iguana.bmp')                     # Open image

    print("make filtered_image")
    img_gray = img.convert('L')                                 # to gray scale

    img_array = np.asarray(img_gray)
    img_array = img_array.astype(np.float32)
    img_filtered = gaussconvolve2d(img_array, 1.6)              # gaussain filter
    img_filtered = np.clip(img_filtered, 0, 255)

    print("show noise reduction")                               # make 2 images into 1 image
    width, height = img.size
    img_sum = Image.new('RGB', (width * 2, height))

    img_filtered_result = img_filtered.astype(np.uint8)
    img_filtered_result = Image.fromarray(img_filtered_result)

    img_sum.paste(img, (0, 0))                                  # paste original
    img_sum.paste(img_filtered_result, (width, 0))              # paste noise reduction
    img_sum.show()
    img_sum.save('./result_images/noise_reduction_result.png', 'PNG')

    print("2. Finding the intensity gradient of the image")             # 2. intensity gradient
    G, theta = sobel_filters(img_filtered)
    img_grad = Image.fromarray(G.astype(np.uint8))
    img_grad.show()
    img_grad.save('./result_images/intensity_gradient.png', 'PNG')

    print("3. Non-maximum Suppression")                                 # 3. non-maximum suppression
    img_non_max = non_max_suppression(G,theta)
    img_non_max_result = Image.fromarray(img_non_max.astype(np.uint8))
    img_non_max_result.show()
    img_non_max_result.save('./result_images/non_max_suppression.png', 'PNG')

    print("4. Double threshold")                                        # 4. Double threshold
    img_threshold = double_thresholding(img_non_max)
    img_threshold_result = Image.fromarray(img_threshold.astype(np.uint8))
    img_threshold_result.show()
    img_threshold_result.save('./result_images/double_thresholding.png', 'PNG')

    print("5. Edge Tracking by hysteresis")                             # 5. Edge Tracking by hysteresis
    img_canny_edge = hysteresis(img_threshold)
    img_canny_edge_result = Image.fromarray(img_canny_edge.astype(np.uint8))
    img_canny_edge_result.show()
    img_canny_edge_result.save('result_images/canny_edge.png', 'PNG')


main()
