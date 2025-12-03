import os
import requests
import hashlib
from tqdm import tqdm
import torch
from urllib.parse import urlparse, parse_qs


# Add your new models here
_links = [
    # Original CUTIE models (optional)
    #('https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth',
    # '6fb97de7ea32f4856f2e63d146a09f31'),
    #('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth',
    # 'a6071de6136982e396851903ab4c083a'),

    # NEW HuggingFace models (no MD5 known — set to None)
    ('https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/cutie-surgenetseg-50k.pth?download=true',
     None),
    ('https://huggingface.co/TimJaspersTue/SurgeNetSeg/resolve/main/ritm-surgenetseg-50k.pth?download=true',
     None),
]


def extract_filename(url: str) -> str:
    """
    HF links often contain ?download=true so we must strip query parameters.
    """
    path = urlparse(url).path
    return os.path.basename(path)


def md5_of_file(path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_models_if_needed() -> str:
    weight_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')
    os.makedirs(weight_dir, exist_ok=True)

    for link, expected_md5 in _links:
        filename = extract_filename(link)
        filepath = os.path.join(weight_dir, filename)

        needs_download = True

        if os.path.exists(filepath):
            if expected_md5 is None:
                # If no MD5 provided, assume OK and skip download
                needs_download = False
            else:
                # Verify checksum
                actual_md5 = md5_of_file(filepath)
                if actual_md5 == expected_md5:
                    needs_download = False

        if needs_download:
            print(f'Downloading {filename} to {weight_dir}...')
            r = requests.get(link, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True)

            with open(filepath, 'wb') as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()

            if total_size != 0 and t.n != total_size:
                raise RuntimeError(f"Error while downloading {filename}")

            if expected_md5:
                # Validate MD5 if expected value exists
                actual_md5 = md5_of_file(filepath)
                if actual_md5 != expected_md5:
                    raise RuntimeError(f"MD5 mismatch for {filename}")

    return weight_dir


if __name__ == '__main__':
    download_models_if_needed()
