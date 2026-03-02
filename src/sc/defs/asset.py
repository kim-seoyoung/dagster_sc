import os
import dagster as dg
import shutil

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
    code_version="v1"  # 크롭 영역(좌표)을 변경하게 되면 이 값을 "v2"로 올리세요!
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
        # 1. OpenCV를 사용하는 경우 (주석 해제 후 사용)
        # img = cv2.imread(in_file)
        # if img is not None:
        #     # 예: 세로 100~500, 가로 200~800 영역 크롭 [y1:y2, x1:x2]
        #     cropped_img = img[100:500, 200:800] 
        #     cv2.imwrite(out_file, cropped_img)
        #     processed_count += 1
        
        # 2. 크롭 로직 작성 전, 테스트를 위해 단순 복사만 해보려면 아래 코드 사용
        shutil.copy(in_file, out_file)
        processed_count += 1
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