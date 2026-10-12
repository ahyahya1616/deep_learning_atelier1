import os
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NYSE_DIR = os.path.join(DATA_DIR, "nyse")
MAINT_DIR = os.path.join(DATA_DIR, "predictive_maintenance")

os.makedirs(NYSE_DIR, exist_ok=True)
os.makedirs(MAINT_DIR, exist_ok=True)

def download_and_extract(url, target_dir, zip_name):
    zip_path = os.path.join(target_dir, zip_name)
    print(f"Downloading from {url} to {zip_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        total_length = response.headers.get('content-length')
        if total_length:
            total_length = int(total_length)
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            out_file.write(chunk)
            if total_length:
                print(f"  {downloaded / (1024*1024):.1f}MB / {total_length / (1024*1024):.1f}MB ({100*downloaded/total_length:.1f}%)", end='\r')
            else:
                print(f"  {downloaded / (1024*1024):.1f}MB downloaded", end='\r')
    print("\nDownload complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extracted to {target_dir}")
    os.remove(zip_path)
    print(f"Files in {target_dir}: {os.listdir(target_dir)}\n")

if __name__ == "__main__":
    print("=== Downloading Datasets for Lab 1 ===")
    
    # 1. Predictive Maintenance Dataset
    maint_url = "https://www.kaggle.com/api/v1/datasets/download/shivamb/machine-predictive-maintenance-classification"
    download_and_extract(maint_url, MAINT_DIR, "maint.zip")
    
    # 2. NYSE Dataset
    nyse_url = "https://www.kaggle.com/api/v1/datasets/download/dgawlik/nyse"
    download_and_extract(nyse_url, NYSE_DIR, "nyse.zip")
    
    print("All datasets downloaded and extracted successfully!")
