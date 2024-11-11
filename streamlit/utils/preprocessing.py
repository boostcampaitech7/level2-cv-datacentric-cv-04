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

from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor

@jit(nopython=True, parallel=True)
def _apply_mask_numba(array, mask):
    """Numba로 최적화된 마스크 적용 함수"""
    result = array.copy()
    for i in prange(array.shape[0]):
        for j in prange(array.shape[1]):
            if not mask[i, j]:
                result[i, j] = 0
    return result

def preprocess_receipt(image):
    """영수증 이미지 전처리 함수 - CPU 최적화 버전
    
    Args:
        image: PIL.Image 또는 numpy array 형식의 입력 이미지
    Returns:
        PIL.Image: 처리된 이미지
    """
    # PIL.Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    # ThreadPoolExecutor를 사용한 병렬 처리
    with ThreadPoolExecutor() as executor:
        lab_future = executor.submit(lab_processing, img_array)
        hsv_future = executor.submit(hsv_processing, img_array)
        
        lab_result = lab_future.result()
        hsv_result = hsv_future.result()
    
    # 벡터화된 연산으로 최적화
    combined = np.add(
        np.multiply(lab_result.astype(np.float32), 0.7),
        np.multiply(hsv_result.astype(np.float32), 0.3)
    ) / 3
    
    # 배경 제거 병렬 처리
    with ThreadPoolExecutor() as executor:
        lab_bg_future = executor.submit(background_removal, lab_result.astype(np.float32))
        hsv_bg_future = executor.submit(background_removal, hsv_result.astype(np.float32))
        
        bg_removed = np.add(lab_bg_future.result(), hsv_bg_future.result()) / 2
    
    # 최종 결과 계산
    result = np.add(combined, bg_removed).astype(np.uint8)
    
    return Image.fromarray(result)

def lab_processing(image):
    """LAB 색공간 기반 이미지 처리"""
    # PIL Image를 numpy array로 변환
    # PIL.Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # LAB 변환
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 마스크 생성
    mask = np.logical_or(
        np.logical_and(
            l > 90,
            np.logical_and(
                np.abs(a) < 155,
                np.abs(b) < 155
            )
        ),
        l < 1
    )
    
    # Numba 최적화된 마스크 적용
    return _apply_mask_numba(l, mask)

def hsv_processing(img_array):
    """HSV 색공간 기반 이미지 처리"""
    # PIL Image를 numpy array로 변환
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # 그레이스케일 이미지를 BGR로 변환
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # HSV 변환
    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 마스크 생성
    mask = np.logical_or(
        np.logical_and(s < 100, v > 10),
        v < 1
    )
    
    # Numba 최적화된 마스크 적용
    return _apply_mask_numba(v, mask)

@jit(nopython=True)
def _apply_morphology(image, kernel_size):
    """모폴로지 연산을 위한 Numba 최적화 함수"""
    height, width = image.shape
    result = np.zeros_like(image)
    
    for i in prange(kernel_size//2, height - kernel_size//2):
        for j in prange(kernel_size//2, width - kernel_size//2):
            window = image[i-kernel_size//2:i+kernel_size//2+1,
                         j-kernel_size//2:j+kernel_size//2+1]
            result[i, j] = np.max(window)
    
    return result

def background_removal(image):
    """배경 제거 함수 - CPU 최적화 버전"""
    # 동적 커널 크기 계산
    blur_radius = max(5, min(image.shape) // 10)
    if blur_radius % 2 == 0:
        blur_radius += 1
    
    # 가우시안 블러
    blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 1)
    
    # 최적화된 모폴로지 연산
    kernel_size = 3
    for _ in range(5):
        blurred = _apply_morphology(blurred, kernel_size)
    
    # 차이 계산
    return cv2.absdiff(image, blurred)


def find_optimal_threshold(gray_image):
    """히스토그램 분포를 그대로 활용하여 최적 임계값 찾기"""
    # 히스토그램 계산
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    
    # 픽셀 총 개수
    total_pixels = gray_image.shape[0] * gray_image.shape[1]
    
    # 각 밝기값의 비율 계산
    pixel_probabilities = hist / total_pixels
    
    max_variance = 0
    optimal_threshold = 0
    
    # 평균 밝기와 표준편차 계산
    mean_brightness = np.sum(np.arange(256) * pixel_probabilities)
    # 이미지의 밝기 특성에 따라 패널티 강도 조정
    if mean_brightness > 180:  # 매우 밝은 이미지
        return 0
        
    
    # 모든 가능한 임계값에 대해 검사
    for threshold in range(256):
        # 임계값을 기준으로 두 클래스로 분할
        w0 = np.sum(pixel_probabilities[:threshold])
        w1 = 1 - w0
        
        # 가중치가 0인 경우 스킵
        if w0 == 0 or w1 == 0:
            continue
        
        # 각 클래스의 평균 계산
        mu0 = np.sum(np.arange(threshold) * pixel_probabilities[:threshold]) / w0
        mu1 = np.sum(np.arange(threshold, 256) * pixel_probabilities[threshold:]) / w1
        
        # 클래스 간 분산 계산
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        # 최대 분산을 가지는 임계값 저장
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = threshold
    
    return optimal_threshold
    
def get_receipt_contour(contours):
    """영수증 윤곽선을 찾는 함수
    
    Args:
        contours: findContours로 찾은 윤곽선들의 리스트
    Returns:
        receipt_contour: 영수증으로 판단되는 윤곽선 (없으면 None)
    """
    if not contours:
        return None
    
    best_contour = None
    max_area = 0
    
    for contour in contours:
        # 윤곽선의 길이
        peri = cv2.arcLength(contour, True)
        # 윤곽선 근사화
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 사각형 형태인지 확인 (꼭지점이 4개)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            
            # 경계 사각형 구하기
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            
            # 면적이 일정 크기 이상이고 종횡비가 적절한 범위 내인 경우
            if area > max_area and 0.2 < aspect_ratio < 5:
                max_area = area
                best_contour = approx
    
    return best_contour
    
def color_processing(image):
    img_array = np.array(image)
    img_array = ((lab_processing(img_array).astype(np.float32) + hsv_processing(img_array).astype(np.float32)) / 2).astype(np.uint8)
    
    return img_array

def improved_background_removal_rgb(image):
    result = preprocess_receipt(image)
    result = 255 - cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    image += result
    return Image.fromarray(image)


def detect_receipt(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(blurred, rectKernel)
    #_, binary = cv2.threshold(dilated, 120, 255, cv2.THRESH_BINARY)
    # 엣지 검출 (Canny)
    #edges = cv2.Canny(binary, 50, 150)
    #edged = cv2.Canny(dilated, 100, 100, apertureSize=3)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    edges = 255 - edges
    edges = cv2.Canny(edges, 50, 250, apertureSize=5)
    
    kernel = np.ones((3,3), np.uint8)
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    return connected



def select_receipt(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edges = cv2.Laplacian(dilated, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    #edges = 255 - edges
    edges = cv2.Canny(edges, 50, 150, apertureSize=5)
    
    # 윤곽선 검출
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 윤곽선 선택 (영수증으로 가정)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 윤곽선 근사화 - Douglas-Peucker 알고리즘 사용
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)  # 근사화 정확도 파라미터
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 4개의 코너 포인트로 근사화 (영수증은 직사각형)
        if len(approx) >= 4:
            # 결과 이미지에 근사화된 윤곽선 그리기
            result = np.zeros_like(image)
            cv2.drawContours(result, [approx], -1, 255, 2)
            return result
    return result

def draw_contours(image):    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    edges = 255 - edges
    edges = cv2.Canny(edges, 50, 150, apertureSize=5)
    
    kernel = np.ones((3,3), np.uint8)
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=5)
    
    # 허프 변환으로 직선 검출
    lines = cv2.HoughLinesP(connected, 1, np.pi/180, 
                           threshold=50, minLineLength=100, maxLineGap=10)
    
    if lines is None:
        raise ValueError("영수증 검출 실패")
    
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 수직/수평 직선만 필터링 (영수증은 보통 직사각형)
    filtered_lines = []
    extension_length = 50  # 선을 양쪽으로 30픽셀씩 연장

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180.0 / np.pi)
        
        if angle < 15 or (angle > 80 and angle < 100) or angle > 170:
            filtered_lines.append(line[0])
            
            # 선의 방향 벡터 계산
            dx = x2 - x1
            dy = y2 - y1
            norm = np.sqrt(dx*dx + dy*dy)
            
            if norm != 0:
                # 정규화된 방향 벡터
                dx = dx / norm
                dy = dy / norm
                
                # 연장된 시작점과 끝점 계산
                new_x1 = int(x1 - extension_length * dx)
                new_y1 = int(y1 - extension_length * dy)
                new_x2 = int(x2 + extension_length * dx)
                new_y2 = int(y2 + extension_length * dy)
                
                # 연장된 선 그리기
                cv2.line(rgb_image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)   
    return rgb_image


def detect_rectangle(image):
    """영수증 검출 및 보정 함수"""
       
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = np.uint8(np.absolute(edges))
    edges = 255 - edges
    edges = cv2.Canny(edges, 50, 150, apertureSize=5)
    
    # 모폴로지 닫힘 연산을 위한 큰 커널 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    
    # 닫힘 연산 강하게 적용 
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # 사각형 후보들을 저장할 리스트
    rectangle_contours = []
    
    # 면적이 큰 순서대로 정렬
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    
    for contour in largest_contours:
        # approximate the contour by a more primitive polygon shape
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.032 * peri, True)
        
        # 사각형 검증
        if len(approx) == 4:  # 꼭지점이 4개인 경우
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
        
    # 검출된 사각형 그리기
    result_image = image.copy()

    # BGR to RGB 변환
    rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    rgb_result_with_rectangle = cv2.drawContours(rgb_result, rectangle_contours, -1, (255, 255, 0), 2)
    return rgb_result_with_rectangle