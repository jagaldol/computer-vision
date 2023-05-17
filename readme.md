# Computer Vision Practice

2023년 부산대학교 4-1 컴퓨터비전개론 실습입니다.

## Contents
* [Image Processing with PIL](https://github.com/jagaldol/computer-vision-4-1/tree/main/1.%20image%20processing%20with%20PIL)
* [gaussian filter and hybrid image](https://github.com/jagaldol/computer-vision-4-1/tree/main/2.%20gaussian%20filter%20and%20hybrid%20image)
* [canny edge detection](https://github.com/jagaldol/computer-vision-4-1/tree/main/3.%20canny%20edge%20detection)
* [RANSAC](https://github.com/jagaldol/computer-vision-4-1/tree/main/4.%20RANSAC)
* [Panorama Stitching](https://github.com/jagaldol/computer-vision/tree/main/5.%20Panorama%20Stitching)
* [Epipolar Geometry](https://github.com/jagaldol/computer-vision-4-1/tree/main/6.%20Epipolar%20Geometry)

## Installation
> using python 3.11.2

### set python virtual environment
* Linux  
    ```sh
    $ activate_venv.sh
    (.venv) $
    ```
* Windows  
    ```cmd
    > activate_venv.bat
    (.venv) >
    ```

### install packages
```shell
(venv) $ pip install -r requirements.txt
```

## Structure
```
computer-vision
├── {each lab}                      # Directory for each lab
|  ├── images
|  |  └── *                         # Image files required for the lab
|  ├── result_images
|  |  └── *                         # Image files created by the lab
|  ├── Lab Description and Report
|  |  └── [REPORT]*.pdf             # lab report pdf
|  |  └── description.pdf           $ lab description pdf
|  ├── *.py                         # lab with python code
|  └── *.ipynb                      # lab with jupyter notebook
├── ...
```