# Image Processiong with PIL
파이썬의 PIL과 NumPy를 사용한 기초적인 디지털 이미지 처리
```python
from PIL import Image
import numpy as np
```
## process
1. chipmunk_head.png  
    `Image.crop()`을 사용하여 머리 부분 좌표로 이미지 잘라내기
2. chipmunk_head_bright.png  
    50 더하여 모든 픽셀 밝게(픽셀의 intensity의 범위는 0-255로 최대값 255)  
    ```python
    for x in range(0,150):
        for y in range(0,150):
            im3_array[y,x] = min(im3_array[y,x] + 50, 255)
    ```
3. chipmunk_head_dark.png  
    0.5 곱하여 모든 픽셀 어둡게(image의 타입은 uint8로 맞춰야 함)  
    ```python
    im4_array = im4_array * 0.5
    im4_array = im4_array.astype('uint8')
    ```