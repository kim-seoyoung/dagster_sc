import os
import dagster as dg

import sys
sys.path.append('/home/sy/sy/sc/src/sc')
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
    code_version="v2" 
)
def cropped_image_data(context: dg.AssetExecutionContext, raw_image_data):
    session_id = context.partition_key
    input_path = raw_image_data["path"]
    
    version = context.assets_def.code_version_by_key[context.asset_key]
    output_path = os.path.join(PROCESSED_DATA_DIR, "cropped_images", version, session_id)
    os.makedirs(output_path, exist_ok=True)
    
    files = os.listdir(input_path)
    processed_count = 0
    
    for file_name in files:
        # 이미지 파일만 필터링
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
            
        in_file = os.path.join(input_path, file_name)
        out_file = os.path.join(output_path, file_name)
        
        # --- [실제 크롭 로직 시작] ---
        save_count = process_yolov8_dataset(input_path, input_path,
                                             output_path, context)
        processed_count += save_count
        # --- [실제 크롭 로직 끝] ---

    context.log.info(f"[{session_id}] 이미지 {processed_count}장 크롭 완료. 저장 경로: {output_path}")
    
    context.add_output_metadata({
        "source_session": session_id,
        "processed_count": processed_count,
        "save_path": output_path,
        "version": version
    })
    
    # 다음 에셋(예: synced_data)에서 크롭된 이미지 경로를 사용할 수 있도록 정보 반환
    return {
        "session": session_id,
        "type": "cropped_image",
        "path": output_path,
        "count": processed_count
    }