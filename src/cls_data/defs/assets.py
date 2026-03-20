import os
import dagster as dg

import sys
sys.path.append('/home/sy/sy/cls_data/src/cls_data')
from process import process_yolov8_dataset

# 동적 파티션 정의: 세션 폴더명을 파티션 키로 사용
session_partitions = dg.DynamicPartitionsDefinition(name="sessions")

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

@dg.asset(partitions_def=session_partitions)
def raw_image_data(context: dg.AssetExecutionContext):
    session_id = context.partition_key
    image_path = os.path.join(RAW_DATA_DIR, session_id, "camera")
    context.log.info(f"[{session_id}] 이미지 경로: {image_path}")
   
    return {"session": session_id, "type": "image", "path": image_path}

@dg.asset(partitions_def=session_partitions)
def raw_radar_data(context: dg.AssetExecutionContext):
    session_id = context.partition_key
    radar_path = os.path.join(RAW_DATA_DIR, session_id, "radar")
    context.log.info(f"[{session_id}] 레이다 경로: {radar_path}")
   
    return {"session": session_id, "type": "radar", "path": radar_path}

@dg.asset(
    partitions_def=session_partitions,
    code_version="v3" 
)
def cropped_image_data(context: dg.AssetExecutionContext, raw_image_data):
    session_id = context.partition_key
    input_path = raw_image_data["path"]
    
    version = context.assets_def.code_version_by_key[context.asset_key]
    output_path = os.path.join(PROCESSED_DATA_DIR, "cropped_images", version, session_id)
    os.makedirs(output_path, exist_ok=True)
    
    classes = os.listdir(input_path)
    train_proc_count = {}
    train_img_count = {}
    test_proc_count = {}
    test_img_count = {}
    
    for c in classes:
        image_path = os.path.join(input_path, c, 'images', 'train')
        save_count, img_count = process_yolov8_dataset(image_path, os.path.join(input_path, c, 'labels', 'train'),
                                                       os.path.join(output_path, c, 'train'), context)
        train_proc_count[c] = save_count
        train_img_count[c] = img_count
        
        image_path = os.path.join(input_path, c, 'images', 'test')
        save_count, img_count = process_yolov8_dataset(image_path, os.path.join(input_path, c, 'labels', 'test'),
                                                       os.path.join(output_path, c, 'test'), context)
        test_proc_count[c] = save_count
        test_img_count[c] = img_count
    
    final_count = sum(train_proc_count.values()) + sum(test_proc_count.values())

    context.log.info(f"[{session_id}] 이미지 {final_count}장 크롭 완료. 저장 경로: {output_path}")
    
    context.add_output_metadata({
        "source_session": session_id,
        "classes": classes,
        "train_proc_count": train_proc_count,
        "test_proc_count": test_proc_count,
        "save_path": output_path,
        "version": version
    })
    
    # 다음 에셋(예: synced_data)에서 크롭된 이미지 경로를 사용할 수 있도록 정보 반환
    return {
        "session": session_id,
        "type": "cropped_image",
        "path": output_path,
        "classes": classes,
        "train_proc_count": train_proc_count,
        "test_proc_count": test_proc_count,
    }