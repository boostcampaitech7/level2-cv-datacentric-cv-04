import math
import numpy as np
from PIL import Image
from typing import Tuple, Union, List
import cv2
from shapely.geometry import Polygon

# 유클리드 거리 계산
def cal_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    '''Calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# 회전 행렬 계산
def get_rotate_mat(theta: float) -> np.ndarray:
    '''Get rotation matrix for the given angle'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

# 이미지 리사이즈
def resize_img(img: Image.Image, vertices: np.ndarray, size: int) -> Tuple[Image.Image, np.ndarray]:
    '''Resize image and vertices to target size'''
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    vertices = vertices * ratio if vertices.size > 0 else vertices
    return img, vertices

# 이미지 높이 조정
def adjust_height(img: Image.Image, vertices: np.ndarray, ratio: float = 0.2) -> Tuple[Image.Image, np.ndarray]:
    '''Adjust height of image with random ratio'''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)
    
    if vertices.size > 0:
        vertices = vertices.copy()
        vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
    return img, vertices

# 이미지 회전
def rotate_img(img: Image.Image, vertices: np.ndarray, angle_range: float = 10) -> Tuple[Image.Image, np.ndarray]:
    '''Rotate image and vertices within given angle range'''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    
    if vertices.size > 0:
        vertices = vertices.copy()
        rotate_mat = get_rotate_mat(-angle / 180 * math.pi)
        for i in range(vertices.shape[0]):
            for j in range(0, 8, 2):
                x, y = vertices[i, j] - center_x, vertices[i, j + 1] - center_y
                new_x, new_y = np.dot(rotate_mat, np.array([x, y]))
                vertices[i, j] = new_x + center_x
                vertices[i, j + 1] = new_y + center_y
    return img, vertices

# 크롭 영역 텍스트 영역 확인
def is_cross_text(start_loc: List[int], length: int, vertices: np.ndarray) -> bool:
    '''Check if crop area crosses text regions'''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, 
                  start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False

# 이미지 크롭
def crop_img(img: Image.Image, vertices: np.ndarray, labels: np.ndarray, 
             length: int) -> Tuple[Image.Image, np.ndarray]:
    '''Randomly crop image of given size'''
    h, w = img.height, img.width
    if h < length or w < length:
        img = img.resize((max(w, length), max(h, length)), Image.BILINEAR)
        
    vertices = vertices.copy()
    remain_h = img.height - length
    remain_w = img.width - length
    
    # Find valid crop position
    for _ in range(50):  # Maximum attempts
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        if not is_cross_text([start_w, start_h], length, vertices[labels==1]):
            box = (start_w, start_h, start_w + length, start_h + length)
            region = img.crop(box)
            if vertices.size > 0:
                vertices[:,[0,2,4,6]] -= start_w
                vertices[:,[1,3,5,7]] -= start_h
            return region, vertices
            
    # 유효한 위치를 찾지 못한 경우, 중앙 크롭
    start_w = (remain_w) // 2
    start_h = (remain_h) // 2
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)

    if vertices.size > 0:
        vertices = vertices.copy()
        vertices[:,[0,2,4,6]] -= start_w
        vertices[:,[1,3,5,7]] -= start_h
        
        # crop 영역 내부에 있는 box만 필터링
        valid_indices = []
        for i, vertex in enumerate(vertices):
            x_coords = vertex[::2]
            y_coords = vertex[1::2]
            if (0 <= x_coords.min() and x_coords.max() <= length and 
                0 <= y_coords.min() and y_coords.max() <= length):
                valid_indices.append(i)
        
        vertices = vertices[valid_indices]
        labels = labels[valid_indices]
    
    return region, vertices



# # Numba로 최적화된 노이즈 제거 함수
# from numba import jit
# import cv2

# @jit(nopython=True)
# def remove_noise(img_array):
#     """Numba로 최적화된 노이즈 제거 함수"""
#     result = img_array.copy()
#     rows, cols = img_array.shape
#     for i in range(1, rows-1):
#         for j in range(1, cols-1):
#             window = img_array[i-1:i+2, j-1:j+2]
#             result[i,j] = np.median(window)
#     return result


def lsa_processing(img_array):
    # L*a*b* 색공간으로 변환 - L채널이 인간의 시각에 더 가까운 밝기 표현
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # L채널에 대해 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l)
    
    # Bilateral 필터로 노이즈 제거하면서 경계는 보존
    smooth_l = cv2.bilateralFilter(enhanced_l, 9, 75, 75)

    return smooth_l


def preprocess_receipt(image):
    """영수증 이미지 전처리 함수"""
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    smooth_l = lsa_processing(img_array)

    # 적응형 이진화 - 이미지를 흑백 2진 이미지로 변환
    # gray: 입력 그레이스케일 이미지
    # 255: 최대 픽셀값 
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 가우시안 가중치 적용
    # cv2.THRESH_BINARY: 이진화 방식
    # 11: 블록 크기 (주변 픽셀 고려 범위)
    # 2: 평균이나 가중평균에서 차감할 값

    binary = cv2.adaptiveThreshold(
        smooth_l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 적응형 이진화는 일반 이진화와 달리 이미지의 각 영역별로 다른 임계값을 적용합니다.
    # 영수증처럼 조명이 불균일하거나 그림자가 있는 경우에 더 좋은 결과를 얻을 수 있습니다.
    # blockSize=11: 주변 11x11 픽셀을 보고 임계값을 계산 (홀수여야 함, 보통 3~19 사용)
    # C=2: 계산된 평균에서 빼는 값 (보통 2~3 사용)

    # 노이즈 제거 (median blur 사용)
    denoised = cv2.medianBlur(binary, 3)

    # 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)

    # numpy array를 PIL Image로 변환하여 반환

    return Image.fromarray(enhanced)

def clean_background(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    """배경 처리"""
    # PIL Image를 NumPy 배열로 변환
    if isinstance(img, Image.Image):
        img_array = np.array(img)
    else:
        img_array = img.copy()
    
    # 그레이스케일로 변환
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # 잡음 제거
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    # 원본 이미지에 마스크 적용
    if len(img_array.shape) == 3:
        # 컬러 이미지인 경우
        result = cv2.bitwise_and(img_array, img_array, mask=cleaned)
    else:
        # 그레이스케일 이미지인 경우
        result = cv2.bitwise_and(gray, gray, mask=cleaned)
    
    # NumPy 배열을 PIL Image로 변환하여 반환
    return Image.fromarray(result)