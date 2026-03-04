import os
import json
import subprocess
import dagster as dg
from dagster import AssetIn, Config, Dict
from pydantic import Field
from .assets import cropped_image_data, session_partitions

from ..process.create_dataset_splits import create_dataset_splits


# -------------------------------------------------------------------
#  Model Training Asset Patterns
# -------------------------------------------------------------------
#
#  You can define different types of model training assets here.
#  Below is an example:
#
#  1. `efficientnetv2`: Trains a single model using data from a selected list of
#     sessions that you provide when you launch the run.
#     - This is useful for experimenting with specific subsets of your data.
#
#  You can copy this pattern to create as many model assets as you need.
#
# -------------------------------------------------------------------


class SelectiveTrainingConfig(Config):
    session_ids: list[str] = Field(
        description="A list of session IDs to use for training. e.g., ['session_A', 'session_B']"
    )
    model_name: str = Field(
        default="selective_model.pt",
        description="The name for the output model file."
    )
    batch_size: int = Field(
        default=8,
        description="Batch size for training."
    )
    num_epochs: int = Field(
        default=10,
        description="Number of training epochs."
    )

@dg.asset(
    compute_kind="pytorch",
    ins={"cropped_image_data": dg.AssetIn(dagster_type=Dict)},
    code_version='v1'
)
def efficientnetv2(context: dg.AssetExecutionContext, config: SelectiveTrainingConfig, cropped_image_data: dict):
    """
    Trains a single model using data from a selected list of sessions.
    
    You can specify which sessions to use in the run configuration.
    In the Dagster UI, go to Launchpad and provide a config like this:
    
    ops:
      efficientnetv2:
        config:
          session_ids:
            - "session_A"
            - "session_C"
          model_name: "my_special_model.pt"

    """
    version = context.assets_def.code_version_by_key[context.asset_key]
    asset_name = context.asset_key.path[-1]
    selected_session_ids = config.session_ids
    model_name = config.model_name
    
    # Collect paths from the selected partitions
    selected_data_paths = []
    for session_id in selected_session_ids:
        if session_id in cropped_image_data:
            selected_data_paths.append(cropped_image_data[session_id]["path"])
        else:
            context.log.warning(
                f"Session ID '{session_id}' provided in config was not found in upstream "
                f"asset partitions. Skipping."
            )

    if not selected_data_paths:
        context.log.warning("No valid data partitions found for the selected session IDs. Skipping training.")
        return
    
    output_dir = os.path.join("./model", asset_name, version)
    os.makedirs(output_dir, exist_ok=True)
    
    create_dataset_splits(selected_data_paths, output_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)

    training_script_path = "/home/sy/sy/simple_classification/run_main.sh"
    
    output_model_path = os.path.join(output_dir, asset_name, version)
    
    # --- [Build Command] ---
    command = [
        "bash",
        training_script_path,
        "--data_dir", output_dir,
        "--save_dir", output_model_path,
        "--model_type", "efficientnet_v2_s",
        "--batch_size", str(config.batch_size),
        "--epochs", str(config.num_epochs)
    ]
    
    context.log.info(f"Running selective training command: {' '.join(command)}")

    # --- [Execute Command] ---
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        context.log.info(f"Training script stdout: {result.stdout}")
        if result.stderr:
            context.log.warning(f"Training script stderr: {result.stderr}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        context.log.error(f"Training script failed: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            context.log.error(f"Stderr: {e.stderr}")
            context.log.error(f"Stdout: {e.stdout}")
        raise

    context.log.info(f"Selective model training complete. Model saved to: {output_model_path}")

    metadata = {
        "model_path": output_model_path,
        "source_sessions": selected_session_ids,
        "version": version
    }

    metrics_path = os.path.join(output_model_path, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            metadata.update(metrics)
            context.log.info(f"Loaded metrics from {metrics_path}")
        except Exception as e:
            context.log.error(f"Failed to load metrics from {metrics_path}: {e}")

    context.add_output_metadata(metadata)

    return {
        "type": "EfficientNetV2",
        "path": output_model_path,
    }