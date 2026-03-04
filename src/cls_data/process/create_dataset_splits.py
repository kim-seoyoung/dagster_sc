import os
import random
import argparse
from pathlib import Path

def create_dataset_splits(data_dirs, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Scans multiple directories for images, splits them into train, validation, and test sets,
    and creates .txt files with image paths and their corresponding class labels.

    The directory structure is expected to be:
    data_dir/
    ↗↗ class_1/
    ←←← image_1.jpg
    ←←← ...
    ↗↗ class_2/
    ←←← image_n.jpg
    ←←← ...
    """
    if not isinstance(data_dirs, (list, tuple)):
        data_dirs = [data_dirs]

    if not (train_ratio + val_ratio + test_ratio) == 1.0:
        raise ValueError("The sum of train, val, and test ratios must be 1.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    
    # Collect all unique class names across all directories
    class_names_set = set()
    for d_dir in data_dirs:
        d_path = Path(d_dir)
        if d_path.exists():
            class_names_set.update([d.name for d in d_path.iterdir() if d.is_dir()])
    
    class_names = sorted(list(class_names_set))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for d_dir in data_dirs:
        d_path = Path(d_dir)
        if not d_path.exists():
            print(f"Warning: Directory {d_path} does not exist. Skipping.")
            continue
            
        for class_name in class_names:
            class_dir = d_path / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                    image_paths.append((img_path, class_to_idx[class_name], d_path.parent))

    random.shuffle(image_paths)

    num_images = len(image_paths)
    num_train = int(num_images * train_ratio)
    num_val = int(num_images * val_ratio)

    train_set = image_paths[:num_train]
    val_set = image_paths[num_train:num_train + num_val]
    test_set = image_paths[num_train + num_val:]

    def write_to_file(filename, dataset):
        with open(output_dir / filename, 'w') as f:
            for img_path, label, parent_path in dataset:
                relative_path = img_path.relative_to(parent_path)
                f.write(f"{relative_path} {label}\n")

    write_to_file('train.txt', train_set)
    write_to_file('val.txt', val_set)
    write_to_file('test.txt', test_set)

    print(f"Created dataset splits in {output_dir}:")
    print(f"  - train.txt: {len(train_set)} images")
    print(f"  - val.txt: {len(val_set)} images")
    print(f"  - test.txt: {len(test_set)} images")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset splits for training, validation, and testing.')
    parser.add_argument('--data_dirs', nargs='+', type=str, default=['./data'], help='Paths to the dataset directories (e.g., --data_dirs ./data1 ./data2).')
    parser.add_argument('--output_dir', type=str, default='./data', help='Directory to save the .txt files.')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation data.')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of test data.')
    
    args = parser.parse_args()

    create_dataset_splits(args.data_dirs, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)