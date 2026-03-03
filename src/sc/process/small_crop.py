import os
import glob
import cv2
import random
import math

def create_classification_crop(image, bbox):
    """
    객체 면적이 25 이하가 되도록 크롭하고, 인터폴레이션(패딩)을 최소화하는 위치를 선정합니다.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox

    # 1. 크롭 사이즈 계산 (객체 면적 기준)
    obj_area = w * h
    if obj_area <= 0:
        return None
       
    # 최종 이미지 면적(10000) 대비 객체 면적이 25 이하가 되기 위한 최소 크롭 사이즈
    crop_size = int(math.ceil(20 * math.sqrt(obj_area)))
   
    # 예외 처리: 계산된 크롭 사이즈가 객체 가로/세로보다 작으면 안됨
    crop_size = max(crop_size, w, h)
   
    # 2. 크롭 시작점(좌상단) 범위 계산 (인터폴레이션 최소화 로직)
    # [조건 A] 객체를 온전히 포함해야 함 -> 최소: x+w-crop_size, 최대: x
    # [조건 B] 패딩이 없으려면 이미지 안에 있어야 함 -> 최소: 0, 최대: img_w-crop_size
   
    # A와 B의 교집합 범위 (패딩이 발생하지 않는 이상적인 범위)
    ideal_min_x = max(int(x + w - crop_size), 0)
    ideal_max_x = min(int(x), img_w - crop_size)
   
    ideal_min_y = max(int(y + h - crop_size), 0)
    ideal_max_y = min(int(y), img_h - crop_size)
   
    # --- X 좌표 랜덤 추출 ---
    if ideal_min_x <= ideal_max_x:
        # 이미지 내에서 크롭이 가능하여 패딩을 안 해도 되는 경우
        crop_x1 = random.randint(ideal_min_x, ideal_max_x)
    else:
        # 크롭 사이즈가 이미지 가로폭보다 커서 '무조건' 패딩이 발생하는 경우
        # 객체 포함 조건(A)만 만족하는 범위에서 추출
        crop_x1 = random.randint(int(x + w - crop_size), int(x))
       
    # --- Y 좌표 랜덤 추출 ---
    if ideal_min_y <= ideal_max_y:
        # 이미지 내에서 크롭이 가능하여 패딩을 안 해도 되는 경우
        crop_y1 = random.randint(ideal_min_y, ideal_max_y)
    else:
        # 무조건 패딩이 발생하는 경우
        crop_y1 = random.randint(int(y + h - crop_size), int(y))
       
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size
   
    # 3. 유효 영역 및 여백(Padding) 계산
    valid_x1 = max(0, crop_x1)
    valid_y1 = max(0, crop_y1)
    valid_x2 = min(img_w, crop_x2)
    valid_y2 = min(img_h, crop_y2)
   
    cropped_img = image[valid_y1:valid_y2, valid_x1:valid_x2]
   
    # 크롭 영역이 이미지 밖으로 나갔을 때 채워야 할 픽셀 수 (ideal 범위에서 뽑혔다면 모두 0이 됨)
    pad_top = valid_y1 - crop_y1
    pad_bottom = crop_y2 - valid_y2
    pad_left = valid_x1 - crop_x1
    pad_right = crop_x2 - valid_x2
   
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped_img = cv2.copyMakeBorder(
            cropped_img,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REPLICATE
        )
       
    # 4. 최종 이미지 100x100으로 리사이즈
    final_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_AREA)
   
    return final_img


def convert_yolo_to_pixel(yolo_bbox, img_w, img_h):
    x_c_norm, y_c_norm, w_norm, h_norm = yolo_bbox
   
    w = w_norm * img_w
    h = h_norm * img_h
    x_min = (x_c_norm - w_norm / 2) * img_w
    y_min = (y_c_norm - h_norm / 2) * img_h
   
    return [int(x_min), int(y_min), int(w), int(h)]

def process_yolov8_dataset(images_dir, labels_dir, output_dir, context):
    image_paths = glob.glob(os.path.join(images_dir, "*.*"))
   
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
       
        if not os.path.exists(label_path):
            continue
           
        image = cv2.imread(img_path)
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {img_path}")
            continue
           
        img_h, img_w = image.shape[:2]
       
        with open(label_path, 'r') as f:
            lines = f.readlines()
           
        for obj_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
               
            class_id = parts[0]
            # 정규화된 좌표값 [x_center, y_center, w, h] 추출
            yolo_bbox = [float(x) for x in parts[1:5]]
           
            # 절대 픽셀 좌표로 변환
            pixel_bbox = convert_yolo_to_pixel(yolo_bbox, img_w, img_h)
           
            # 이미지 생성
            cropped_img = create_classification_crop(image, pixel_bbox)
            if cropped_img is None:
                continue
        
            class_dir = os.path.join(output_dir, str(class_id))
            os.makedirs(class_dir, exist_ok=True)
        
            save_path = os.path.join(class_dir, f"{base_name}_{obj_idx}.jpg")
            cv2.imwrite(save_path, cropped_img)
           
    print(f"✅ 변환 완료! 데이터가 '{output_dir}'에 저장되었습니다.")


