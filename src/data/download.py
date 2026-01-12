import os
import zipfile
from pathlib import Path
import urllib.request
import sys

def download_file(url, dest_path):
    """Download file with progress bar"""
    print(f"Downloading {dest_path.name}...")
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            sys.stdout.write(f"\rProgress: {percent:.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, show_progress)
    print(f"\n✓ Downloaded: {dest_path}")

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to: {extract_to}")

def download_div2k(data_dir="data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    (data_dir / "div2k").mkdir(exist_ok=True)

    print("="*60)
    print("DOWNLOADING DATASETS FOR ADASR")
    print("="*60)

    # 1. DIV2K TRAIN HR
    div2k_train_hr_zip = data_dir / "div2k" / "DIV2K_train_HR.zip"
    div2k_train_hr_dir = data_dir / "div2k" / "DIV2K_train_HR"

    if div2k_train_hr_dir.exists() and len(list(div2k_train_hr_dir.glob("*.png"))) > 0:
        print(f"✓ DIV2K_train_HR already exists")
    else:
        url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
        try:
            download_file(url, div2k_train_hr_zip)
            extract_zip(div2k_train_hr_zip, data_dir / "div2k")
            div2k_train_hr_zip.unlink() # Cleanup
        except Exception as e:
            print(f"✗ Error downloading DIV2K_train_HR: {e}")

    # 2. DIV2K VALID HR
    div2k_valid_hr_zip = data_dir / "div2k" / "DIV2K_valid_HR.zip"
    div2k_valid_hr_dir = data_dir / "div2k" / "DIV2K_valid_HR"

    if div2k_valid_hr_dir.exists() and len(list(div2k_valid_hr_dir.glob("*.png"))) > 0:
        print(f"✓ DIV2K_valid_HR already exists")
    else:
        url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        try:
            download_file(url, div2k_valid_hr_zip)
            extract_zip(div2k_valid_hr_zip, data_dir / "div2k")
            div2k_valid_hr_zip.unlink()
        except Exception as e:
            print(f"✗ Error downloading DIV2K_valid_HR: {e}")

    # 3. DIV2K VALID LR (X4)
    div2k_valid_lr_zip = data_dir / "div2k" / "DIV2K_valid_LR_bicubic.zip"
    div2k_valid_lr_dir = data_dir / "div2k" / "DIV2K_valid_LR_bicubic"

    if div2k_valid_lr_dir.exists() and (div2k_valid_lr_dir / "X4").exists():
        print(f"✓ DIV2K_valid_LR_bicubic/X4 already exists")
    else:
        url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic.zip"
        try:
            download_file(url, div2k_valid_lr_zip)
            extract_zip(div2k_valid_lr_zip, data_dir / "div2k")
            div2k_valid_lr_zip.unlink()
        except Exception as e:
            print(f"✗ Error downloading DIV2K_valid_LR: {e}")

if __name__ == "__main__":
    download_div2k()
