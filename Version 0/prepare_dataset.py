import os
import shutil
from pathlib import Path

raw_dir = Path("dataset/raw/specie")
train_a = Path("dataset/processed/train/A")
train_b = Path("dataset/processed/train/B")

# Create processed directories if missing
train_a.mkdir(parents=True, exist_ok=True)
train_b.mkdir(parents=True, exist_ok=True)

if not raw_dir.exists():
    print(f"Directory {raw_dir} does not exist.")
    exit(1)

files = os.listdir(raw_dir)
ossa_files = [f for f in files if "_ossa_" in f]

paired_count = 0
for ossa_file in ossa_files:
    # Map input file to target output file name pattern
    output_file = ossa_file.replace("_ossa_", "_output_")
    
    src_a = raw_dir / ossa_file
    src_b = raw_dir / output_file
    
    if src_b.exists():
        # Create identical filename for both A and B subdirectories
        unified_name = ossa_file.replace("_ossa_", "_")
        
        shutil.copy(src_a, train_a / unified_name)
        shutil.copy(src_b, train_b / unified_name)
        paired_count += 1

print(f"Successfully paired and populated {paired_count} samples into dataset/processed/train/")

