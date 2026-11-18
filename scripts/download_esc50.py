#!/usr/bin/env python3
"""
ESC-50 Dataset Downloader

Downloads the ESC-50 dataset for LatentAudio training.

ESC-50: Environmental Sound Classification Dataset
https://github.com/karolpiczak/ESC-50

Usage:
    python download_esc50.py --output /path/to/download
    python download_esc50.py --output /path/to/download --extract-to /path/to/extract
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

warnings.filterwarnings("ignore")

DATASET_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ZENODO_URL = "https://zenodo.org/record/5835185/files/ESC-50-master.zip"

EXPECTED_MD5 = {
    'master.zip': None,
}

@dataclass
class DownloadProgress:
    total_size: int = 0
    downloaded: int = 0
    last_percent: float = 0

    def update(self, count: int, block_size: int):
        self.downloaded += count * block_size
        percent = (self.downloaded / self.total_size) * 100 if self.total_size > 0 else 0
        if percent - self.last_percent >= 1.0:
            self.last_percent = percent
            print(f"\r  Downloaded: {self.downloaded / 1024 / 1024:.1f} MB ({percent:.1f}%)", end='', flush=True)

    def finish(self):
        print()


class Downloader:
    def __init__(self, url: str, output_path: Path):
        self.url = url
        self.output_path = output_path
        self.temp_path = output_path.with_suffix('.tmp')

    def get_file_size(self) -> int:
        """Get file size from URL without downloading."""
        try:
            with urllib.request.urlopen(self.url, timeout=10) as response:
                return int(response.headers.get('Content-Length', 0))
        except Exception:
            return 0

    def download_with_urllib(self, progress: DownloadProgress) -> bool:
        """Download file using urllib (built-in, no dependencies)."""
        try:
            progress.total_size = self.get_file_size()
            print(f"  Source: {self.url}")
            print(f"  Destination: {self.output_path}")

            if self.output_path.exists():
                print("  File already exists. Skipping download.")
                return True

            if self.temp_path.exists():
                print("  Resuming partial download...")
                mode = 'ab'
            else:
                mode = 'wb'

            print(f"  Downloading...", end=' ', flush=True)
            req = urllib.request.Request(self.url, headers={'User-Agent': 'LatentAudio-Downloader/1.0'})

            with urllib.request.urlopen(req, timeout=300) as response:
                progress.total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0

                with open(self.temp_path, mode) as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / progress.total_size * 100) if progress.total_size > 0 else 0
                        if percent - progress.last_percent >= 1.0:
                            progress.last_percent = percent
                            print(f"\r  Downloading... {percent:.1f}%", end='', flush=True)

            print("  Done!")

            if self.temp_path.exists():
                self.temp_path.rename(self.output_path)

            return True

        except Exception as e:
            print(f"\n  Error: {e}")
            if self.temp_path.exists():
                print("  Partial download saved for resume.")
            return False

    def download_with_curl(self, progress: DownloadProgress) -> bool:
        """Download file using curl (more reliable for large files)."""
        try:
            print(f"  Source: {self.url}")
            print(f"  Destination: {self.output_path}")

            if self.output_path.exists():
                print("  File already exists. Skipping download.")
                return True

            if self.temp_path.exists():
                print("  Resuming partial download...")
                resume_flag = '-C -'
            else:
                resume_flag = ''

            cmd = [
                'curl', '-L', '-o', str(self.temp_path),
                resume_flag, '--retry', '3', '--retry-delay', '5',
                '--silent', '--show-error', self.url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                if self.temp_path.exists():
                    self.temp_path.rename(self.output_path)
                print("  Download complete!")
                return True
            else:
                print(f"  Error: {result.stderr}")
                return False

        except FileNotFoundError:
            print("  curl not found, falling back to urllib")
            return False
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def download(self, use_curl: bool = True) -> bool:
        """Download the dataset."""
        progress = DownloadProgress()

        if use_curl:
            return self.download_with_curl(progress)
        else:
            return self.download_with_urllib(progress)

    def verify(self) -> bool:
        """Verify downloaded file exists and has content."""
        if not self.output_path.exists():
            print("  ERROR: Downloaded file not found")
            return False

        size = self.output_path.stat().st_size
        if size < 1000:
            print(f"  ERROR: File too small ({size} bytes), download may have failed")
            return False

        print(f"  Verified: {size / 1024 / 1024:.1f} MB")
        return True


def extract_zip(zip_path: Path, extract_to: Path, clean_zip: bool = True) -> bool:
    """Extract ZIP file to directory."""
    if not zip_path.exists():
        print(f"  ERROR: ZIP file not found: {zip_path}")
        return False

    try:
        print(f"  Extracting to: {extract_to}")

        extract_to.mkdir(parents=True, exist_ok=True)

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        extracted_folder = extract_to / "ESC-50-master"
        if extracted_folder.exists() and extracted_folder != extract_to:
            print("  Cleanup: Flattening directory structure...")
            for item in extracted_folder.iterdir():
                dest = extract_to / item.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(item), str(dest))
            extracted_folder.rmdir()

        if clean_zip:
            print("  Removing ZIP file...")
            zip_path.unlink()

        print("  Extraction complete!")
        return True

    except Exception as e:
        print(f"  ERROR: Extraction failed: {e}")
        return False


def verify_dataset(directory: Path, expected_clips: int = 2000) -> bool:
    """Verify extracted dataset has expected files."""
    if not directory.exists():
        print(f"  ERROR: Directory not found: {directory}")
        return False

    wav_files = list(directory.glob('*.wav'))
    count = len(wav_files)

    if count >= expected_clips * 0.9:
        print(f"  Verified: {count} audio clips found")
        return True
    else:
        print(f"  WARNING: Expected ~{expected_clips} clips, found {count}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ESC-50 dataset for LatentAudio training'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='ESC-50.zip',
        help='Path for downloaded ZIP file (default: ESC-50.zip)'
    )
    parser.add_argument(
        '--extract-to', '-e',
        type=str,
        default='ESC-50',
        help='Directory to extract dataset (default: ESC-50)'
    )
    parser.add_argument(
        '--no-extract', '-x',
        action='store_true',
        help='Download but do not extract'
    )
    parser.add_argument(
        '--keep-zip', '-k',
        action='store_true',
        help='Keep ZIP file after extraction'
    )
    parser.add_argument(
        '--use-curl', '-c',
        action='store_true',
        help='Use curl for download (more reliable)'
    )
    parser.add_argument(
        '--zenodo', '-z',
        action='store_true',
        help='Download from Zenodo instead of GitHub'
    )

    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    extract_path = Path(args.extract_to).resolve()

    print()
    print("=" * 60)
    print("ESC-50 Dataset Downloader")
    print("=" * 60)
    print()
    print("Dataset: ESC-50 (Environmental Sound Classification)")
    print("Size:    ~600 MB (2,000 audio clips)")
    print("License: CC-BY-4.0")
    print()

    url = ZENODO_URL if args.zenodo else DATASET_URL

    downloader = Downloader(url, output_path)

    if not downloader.download(use_curl=args.use_curl):
        print("\nDownload failed. Trying alternative source...")
        alt_url = ZENODO_URL if not args.zenodo else DATASET_URL
        downloader = Downloader(alt_url, output_path)
        if not downloader.download(use_curl=args.use_curl):
            print("\nERROR: All download sources failed.")
            sys.exit(1)

    if not downloader.verify():
        sys.exit(1)

    if not args.no_extract:
        if not extract_zip(output_path, extract_path, clean_zip=not args.keep_zip):
            sys.exit(1)

        if not verify_dataset(extract_path):
            print("  Note: Dataset may be in subdirectory. Checking...")
            for subdir in extract_path.iterdir():
                if subdir.is_dir() and (subdir / "audio").exists():
                    if verify_dataset(subdir / "audio"):
                        print(f"  Found at: {subdir / 'audio'}")
                        break

    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run preprocessing script:")
    print(f"     python scripts/preprocess_esc50.py -i {extract_path} -o ESC-50-preprocessed")
    print()
    print("  2. Load preprocessed data in LatentAudio GUI")
    print("  3. Train with 20,000 samples (vs current 400)")
    print()


if __name__ == '__main__':
    main()
