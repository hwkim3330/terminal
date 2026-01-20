#!/usr/bin/env python3
"""
NVIDIA Clara SDK 설정 스크립트
의료 영상 AI 개발을 위한 Clara 환경 구성
"""

import os
import subprocess
import sys
from pathlib import Path
import requests
import json

CLARA_DIR = Path("/mnt/data/lfm_agi/clara")
DOCKER_DIR = Path("/mnt/data/lfm_agi/docker_images")

def run_command(cmd: str, check: bool = True):
    """명령어 실행"""
    print(f"실행 중: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"오류: {result.stderr}")
        return False
    print(result.stdout)
    return True

def check_requirements():
    """시스템 요구사항 확인"""
    print("🔍 시스템 요구사항 확인...")
    
    # Docker 확인
    if not run_command("docker --version", check=False):
        print("❌ Docker가 설치되지 않음")
        return False
    
    # NVIDIA Docker 확인
    if not run_command("nvidia-docker --version", check=False):
        print("⚠️ NVIDIA Docker가 설치되지 않음, nvidia-container-toolkit 사용")
    
    # GPU 확인
    if not run_command("nvidia-smi", check=False):
        print("❌ NVIDIA GPU 또는 드라이버가 설치되지 않음")
        return False
    
    print("✅ 시스템 요구사항 확인 완료")
    return True

def download_clara():
    """Clara SDK 다운로드"""
    print("📥 NVIDIA Clara SDK 다운로드...")
    
    # Clara 이미지 목록
    clara_images = [
        "nvcr.io/nvidia/clara/clara-train-sdk:v4.0",
        "nvcr.io/nvidia/clara/clara-holoscan:v0.4.0",
        "nvcr.io/nvidia/clara/clara-train-sdk:v3.1",
        "nvcr.io/nvidia/monai/monai:0.9.1"
    ]
    
    for image in clara_images:
        try:
            print(f"\n📦 다운로드 중: {image}")
            
            # 이미지 풀
            cmd = f"docker pull {image}"
            if run_command(cmd):
                print(f"✅ 완료: {image}")
                
                # 이미지를 tar로 저장
                image_name = image.replace("/", "_").replace(":", "_")
                tar_path = DOCKER_DIR / f"{image_name}.tar"
                
                print(f"💾 이미지 저장 중: {tar_path}")
                save_cmd = f"docker save {image} > {tar_path}"
                run_command(save_cmd)
                
            else:
                print(f"❌ 실패: {image}")
                
        except Exception as e:
            print(f"❌ 오류 {image}: {e}")

def setup_clara_workspace():
    """Clara 작업 공간 설정"""
    print("🏗️ Clara 작업 공간 설정...")
    
    # 디렉토리 구조 생성
    directories = [
        "models",
        "data/input",
        "data/output", 
        "configs",
        "apps",
        "notebooks",
        "pipelines"
    ]
    
    for dir_name in directories:
        dir_path = CLARA_DIR / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 생성: {dir_path}")
    
    # Clara 설정 파일 생성
    clara_config = {
        "clara_version": "4.0",
        "workspace": str(CLARA_DIR),
        "models": {
            "segmentation": str(CLARA_DIR / "models/segmentation"),
            "classification": str(CLARA_DIR / "models/classification"),
            "detection": str(CLARA_DIR / "models/detection")
        },
        "data": {
            "input": str(CLARA_DIR / "data/input"),
            "output": str(CLARA_DIR / "data/output")
        },
        "docker_images": list(CLARA_DIR.glob("docker_images/*.tar"))
    }
    
    with open(CLARA_DIR / "clara_config.json", "w") as f:
        json.dump(clara_config, f, indent=2, default=str)
    
    print("✅ Clara 작업 공간 설정 완료")

def create_clara_examples():
    """Clara 예제 생성"""
    print("📝 Clara 예제 생성...")
    
    # 기본 세그멘테이션 예제
    segmentation_config = """
{
  "epochs": 1250,
  "num_training_epoch_per_valid": 20,
  "learning_rate": 1e-4,
  "multi_gpu": false,
  
  "model": {
    "name": "SegResNet",
    "args": {
      "spatial_dims": 3,
      "init_filters": 8,
      "in_channels": 1,
      "out_channels": 2,
      "dropout_prob": 0.2
    }
  },
  
  "pre_transforms": [
    {
      "name": "LoadImaged",
      "args": {
        "keys": ["image", "label"]
      }
    },
    {
      "name": "AddChanneld",
      "args": {
        "keys": ["image", "label"]
      }
    },
    {
      "name": "Orientationd",
      "args": {
        "keys": ["image", "label"],
        "axcodes": "RAS"
      }
    },
    {
      "name": "Spacingd",
      "args": {
        "keys": ["image", "label"],
        "pixdim": [1.0, 1.0, 1.0],
        "mode": ["bilinear", "nearest"]
      }
    },
    {
      "name": "ScaleIntensityRanged",
      "args": {
        "keys": ["image"],
        "a_min": -175,
        "a_max": 250,
        "b_min": 0.0,
        "b_max": 1.0,
        "clip": true
      }
    }
  ],
  
  "dataset": {
    "data_list_file_path": "{DATASET_JSON}",
    "data_file_base_dir": "{DATASET_ROOT}",
    "data_list_key": "training"
  },
  
  "loss": {
    "name": "DiceLoss",
    "args": {
      "softmax": true,
      "to_onehot_y": true,
      "squared_pred": true
    }
  },
  
  "optimizer": {
    "name": "Adam",
    "args": {
      "lr": "{learning_rate}"
    }
  }
}
"""
    
    with open(CLARA_DIR / "configs/segmentation_config.json", "w") as f:
        f.write(segmentation_config)
    
    # Docker 실행 스크립트 생성
    docker_script = """#!/bin/bash
# Clara Docker 실행 스크립트

CLARA_WORKSPACE="/mnt/data/lfm_agi/clara"
DOCKER_IMAGE="nvcr.io/nvidia/clara/clara-train-sdk:v4.0"

echo "🚀 Clara Docker 컨테이너 시작..."

docker run --gpus all \\
    --rm -it \\
    --shm-size=1g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    -v ${CLARA_WORKSPACE}:/workspace \\
    -v /mnt/data:/mnt/data \\
    -p 8888:8888 \\
    -p 6006:6006 \\
    ${DOCKER_IMAGE} \\
    /bin/bash

echo "✅ Clara Docker 세션 종료"
"""
    
    with open(CLARA_DIR / "run_clara.sh", "w") as f:
        f.write(docker_script)
    
    # 실행 권한 부여
    os.chmod(CLARA_DIR / "run_clara.sh", 0o755)
    
    print("✅ Clara 예제 생성 완료")

def download_medical_datasets():
    """의료 영상 데이터셋 다운로드"""
    print("🏥 의료 영상 데이터셋 다운로드...")
    
    datasets = [
        {
            "name": "MSD Decathlon",
            "description": "의료 세그멘테이션 챌린지 데이터",
            "url": "http://medicaldecathlon.com/",
            "note": "수동 다운로드 필요"
        },
        {
            "name": "MONAI Tutorials Data", 
            "description": "MONAI 튜토리얼 데이터",
            "url": "https://github.com/Project-MONAI/tutorials/tree/main/3d_segmentation",
            "note": "Git clone 필요"
        }
    ]
    
    # 데이터셋 정보 저장
    with open(CLARA_DIR / "datasets_info.json", "w") as f:
        json.dump(datasets, f, indent=2)
    
    print("📋 데이터셋 정보 저장 완료")
    print("⚠️ 의료 데이터는 라이선스 제약으로 수동 다운로드가 필요할 수 있습니다.")

def create_clara_notebooks():
    """Clara Jupyter 노트북 생성"""
    print("📓 Clara Jupyter 노트북 생성...")
    
    notebook_content = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NVIDIA Clara 시작하기\\n",
    "\\n",
    "이 노트북은 Clara SDK를 사용한 의료 영상 AI 개발을 위한 기본 예제입니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\\n",
    "import numpy as np\\n",
    "import torch\\n",
    "import monai\\n",
    "from monai.transforms import *\\n",
    "from monai.data import *\\n",
    "from monai.engines import *\\n",
    "\\n",
    "print(f'MONAI version: {monai.__version__}')\\n",
    "print(f'PyTorch version: {torch.__version__}')\\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 기본 변환 파이프라인"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 기본 변환 정의\\n",
    "transforms = Compose([\\n",
    "    LoadImaged(keys=['image', 'label']),\\n",
    "    AddChanneld(keys=['image', 'label']),\\n",
    "    Orientationd(keys=['image', 'label'], axcodes='RAS'),\\n",
    "    Spacingd(keys=['image', 'label'], pixdim=[1.0, 1.0, 1.0]),\\n",
    "    ScaleIntensityRanged(keys=['image'], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),\\n",
    "    ToTensord(keys=['image', 'label'])\\n",
    "])\\n",
    "\\n",
    "print('변환 파이프라인 생성 완료')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""
    
    with open(CLARA_DIR / "notebooks/clara_tutorial.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("✅ Clara 노트북 생성 완료")

def main():
    """메인 함수"""
    print("🏥 NVIDIA Clara 설정 시작")
    
    # 디렉토리 생성
    CLARA_DIR.mkdir(exist_ok=True)
    DOCKER_DIR.mkdir(exist_ok=True)
    
    # 시스템 요구사항 확인
    if not check_requirements():
        print("❌ 시스템 요구사항을 확인하세요")
        return
    
    print("\nClara 설정 옵션:")
    print("1. 기본 작업 공간만 설정")
    print("2. Docker 이미지 다운로드")
    print("3. 전체 설정 (작업공간 + 이미지)")
    print("4. 예제 및 노트북만 생성")
    
    choice = input("선택 (1-4): ").strip()
    
    if choice == "1":
        setup_clara_workspace()
        create_clara_examples()
        create_clara_notebooks()
        download_medical_datasets()
        
    elif choice == "2":
        download_clara()
        
    elif choice == "3":
        setup_clara_workspace()
        create_clara_examples()
        create_clara_notebooks()
        download_medical_datasets()
        download_clara()
        
    elif choice == "4":
        create_clara_examples()
        create_clara_notebooks()
        download_medical_datasets()
        
    else:
        print("잘못된 선택입니다.")
        return
    
    print(f"\n🎉 Clara 설정 완료!")
    print(f"📁 위치: {CLARA_DIR}")
    print(f"🚀 실행: {CLARA_DIR}/run_clara.sh")

if __name__ == "__main__":
    main()