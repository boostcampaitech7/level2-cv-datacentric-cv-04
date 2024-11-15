import os.path as osp
import math
import json
from PIL import Image

import torch
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset
from shapely.geometry import Polygon

from numba import njit
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
@njit
def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices

@njit
def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
    r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
    r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
    r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
        offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v

@njit
def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4,2)).T
    if anchor is None:
        anchor = v[:,:1]
    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

@njit
def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max

@njit
def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err

@njit
def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, start_w + length, start_h + length,
                  start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert(ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
        new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:,[0,2,4,6]] -= start_w
    new_vertices[:,[1,3,5,7]] -= start_h
    return region, new_vertices



@njit
def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def resize_img(img, vertices, size):
    h, w = img.height, img.width
    ratio = size / max(h, w)
    if w > h:
        img = img.resize((size, int(h * ratio)), Image.BILINEAR)
    else:
        img = img.resize((int(w * ratio), size), Image.BILINEAR)
    if vertices is not None:
        new_vertices = vertices * ratio
        return img, new_vertices
    else:
        return img

def flip_img(img, vertices, flip_type="horizontal"):
    """이미지와 vertices를 좌우 또는 상하 반전
    
    Args:
        img: PIL Image
        vertices: 텍스트 영역의 좌표 (n,8) 
        flip_type: "horizontal" 또는 "vertical"
        
    Returns:
        img: 반전된 PIL Image
        new_vertices: 반전된 vertices 좌표
    """
    w, h = img.width, img.height
    
    if flip_type == "horizontal":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if vertices is not None and vertices.size > 0:
            new_vertices = vertices.copy()
            new_vertices[:, [0,2,4,6]] = w - vertices[:, [0,2,4,6]]
    elif flip_type == "vertical":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if vertices is not None and vertices.size > 0:
            new_vertices = vertices.copy()
            new_vertices[:, [1,3,5,7]] = h - vertices[:, [1,3,5,7]]
    else:
        raise ValueError("flip_type must be 'horizontal' or 'vertical'")
        
    if vertices is not None:
        return img, new_vertices
    return img

def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    if vertices is not None:
        new_vertices = vertices.copy()
        if vertices.size > 0:
            new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
        return img, new_vertices
    else:
        return img, None


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
    return img, new_vertices


def generate_roi_mask(image, vertices, labels):
    mask = np.ones(image.shape[:2], dtype=np.float32)
    ignored_polys = []
    for vertice, label in zip(vertices, labels):
        if label == 0:
            ignored_polys.append(np.around(vertice.reshape((4, 2))).astype(np.int32))
    cv2.fillPoly(mask, ignored_polys, 0)
    return mask


def filter_vertices(vertices, labels, ignore_under=0, drop_under=0):
    if drop_under == 0 and ignore_under == 0:
        return vertices, labels

    new_vertices, new_labels = vertices.copy(), labels.copy()

    areas = np.array([Polygon(v.reshape((4, 2))).convex_hull.area for v in vertices])
    labels[areas < ignore_under] = 0

    if drop_under > 0:
        passed = areas >= drop_under
        new_vertices, new_labels = new_vertices[passed], new_labels[passed]

    return new_vertices, new_labels

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

def lab_processing(img_array):
    """LAB 색공간 기반 이미지 처리"""
    # PIL Image를 numpy array로 변환
    if isinstance(img_array, Image.Image):
        img_array = np.array(img_array)
    
    # 그레이스케일 이미지를 BGR로 변환
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
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
    _, s, v = cv2.split(hsv)
    
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

class SceneTextDataset:
    def __init__(self, root_dir, split='train', image_size=2048, crop_size=1024, 
                 ignore_under_threshold=10, drop_under_threshold=1,
                 color_jitter=True, normalize=True):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.crop_size = crop_size
        self.color_jitter = color_jitter
        self.normalize = normalize
        self.ignore_under_threshold = ignore_under_threshold
        self.drop_under_threshold = drop_under_threshold

        # merged_receipts 폴더 구조에 맞게 수정
        self.image_dir = osp.join(root_dir, 'images', split)
        self.annotation_path = osp.join(root_dir, f'{split}.json')
        
        if not osp.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not osp.exists(self.annotation_path):
            raise ValueError(f"Annotation file not found: {self.annotation_path}")
            
        self.annotations = []
        
        # UFO 형식 데이터 로드
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        for image_name, annotation in annotations['images'].items():
            img_path = osp.join(self.image_dir, image_name)
            if osp.exists(img_path):  # 이미지 파일이 실제로 존재하는지 확인
                self.annotations.append({
                    'file_name': image_name,
                    'words': annotation.get('words', dict()),
                    'img_path': img_path
                })
            else:
                print(f"Warning: Image not found: {img_path}")

    def __len__(self):
        # image_fnames 대신 annotations 사용
        return len(self.annotations)

    def __getitem__(self, idx):
        # image_fname 대신 annotation 사용
        annotation = self.annotations[idx]
        image_fpath = annotation['img_path']
        
        vertices, labels = [], []
        for word_info in annotation['words'].values():
            points = np.array(word_info['points'])
            if points.shape[0] > 4:
                continue
            vertices.append(points.flatten())
            labels.append(1)
        vertices = np.array(vertices, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        vertices, labels = filter_vertices(
            vertices,
            labels,
            ignore_under=self.ignore_under_threshold,
            drop_under=self.drop_under_threshold
        )

        image = Image.open(image_fpath)
        # 전처리 적용
        # image = preprocess_receipt(image)
        # 30% 확률로 상하 반전 (필요한 경우)
        if np.random.random() > 0.5:
            image, vertices = flip_img(image, vertices, flip_type="vertical")
        image, vertices = resize_img(image, vertices, self.image_size)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices, angle_range=15)
        image, vertices = crop_img(image, vertices, labels, self.crop_size)

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)

        funcs = []
        if self.color_jitter:
            funcs.append(A.ColorJitter())
        if self.normalize:
            funcs.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = A.Compose(funcs)

        image = transform(image=image)['image']
        word_bboxes = np.reshape(vertices, (-1, 4, 2))
        roi_mask = generate_roi_mask(image, vertices, labels)

        return image, word_bboxes, roi_mask