#!/usr/bin/env python3
"""
LFM AGI 모델 및 데이터 다운로드 스크립트
"""

import os
import asyncio
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import json

BASE_DIR = Path("/mnt/data/lfm_agi")
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
CACHE_DIR = BASE_DIR / "huggingface_cache"
OLLAMA_DIR = BASE_DIR / "ollama_models"

# 환경 변수 설정
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["OLLAMA_MODELS"] = str(OLLAMA_DIR)

def download_file(url: str, filename: str):
    """파일 다운로드 with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename.name,
        total=total_size,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def run_command(cmd: str):
    """명령어 실행"""
    print(f"실행 중: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"오류: {result.stderr}")
        return False
    print(f"완료: {cmd}")
    return True

class ModelDownloader:
    def __init__(self):
        self.downloaded_models = []
        
    def download_huggingface_models(self):
        """Hugging Face 모델 다운로드"""
        models = [
            # LFM 계열 모델
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large", 
            
            # 한국어 모델
            "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
            "beomi/Llama-3-Open-Ko-8B",
            "beomi/KoAlpaca-Polyglot-5.8B",
            "klue/roberta-large",
            "klue/bert-base",
            
            # 멀티모달 모델
            "microsoft/kosmos-2-patch14-224",
            "Salesforce/blip2-opt-2.7b",
            "openai/clip-vit-large-patch14",
            
            # 임베딩 모델
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "jhgan/ko-sroberta-multitask",
            
            # 코드 모델
            "microsoft/CodeBERT-base",
            "bigcode/starcoder"
        ]
        
        print("🤖 Hugging Face 모델 다운로드 시작...")
        
        for model in models:
            try:
                print(f"\n📥 다운로드 중: {model}")
                
                # git lfs 사용하여 모델 클론
                model_dir = MODELS_DIR / model.replace("/", "_")
                if not model_dir.exists():
                    cmd = f"cd {MODELS_DIR} && git lfs clone https://huggingface.co/{model} {model.replace('/', '_')}"
                    if run_command(cmd):
                        self.downloaded_models.append(model)
                else:
                    print(f"✅ 이미 존재: {model}")
                    
            except Exception as e:
                print(f"❌ 다운로드 실패 {model}: {e}")
        
    def download_ollama_models(self):
        """Ollama 모델 다운로드"""
        models = [
            "llama3.1:8b",
            "llama3.1:70b",
            "qwen2.5:7b",
            "qwen2.5:32b",
            "mistral:7b",
            "gemma2:9b",
            "phi3:medium",
            "codellama:7b",
            "codellama:34b"
        ]
        
        print("\n🦙 Ollama 모델 다운로드 시작...")
        
        for model in models:
            try:
                print(f"\n📥 다운로드 중: {model}")
                cmd = f"OLLAMA_MODELS={OLLAMA_DIR} ollama pull {model}"
                if run_command(cmd):
                    self.downloaded_models.append(f"ollama:{model}")
                    
            except Exception as e:
                print(f"❌ 다운로드 실패 {model}: {e}")
    
    def download_datasets(self):
        """한국어 데이터셋 다운로드"""
        datasets = [
            # 대화 데이터셋
            {
                "name": "korean_conversation",
                "url": "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
                "filename": "korean_conversation.csv"
            },
            
            # 뉴스 데이터셋
            {
                "name": "korean_news",
                "url": "https://raw.githubusercontent.com/lovit/korean_news_dataset/master/data/news_2018_sample.txt",
                "filename": "korean_news.txt"
            },
            
            # 감정 분석 데이터
            {
                "name": "korean_sentiment",
                "url": "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/Data/SentiWord_Dict.txt",
                "filename": "korean_sentiment.txt"
            }
        ]
        
        print("\n📊 한국어 데이터셋 다운로드 시작...")
        
        for dataset in datasets:
            try:
                filepath = DATASETS_DIR / dataset["filename"]
                if not filepath.exists():
                    print(f"\n📥 다운로드 중: {dataset['name']}")
                    download_file(dataset["url"], filepath)
                    print(f"✅ 완료: {dataset['name']}")
                else:
                    print(f"✅ 이미 존재: {dataset['name']}")
                    
            except Exception as e:
                print(f"❌ 다운로드 실패 {dataset['name']}: {e}")
    
    def setup_korean_nlp_resources(self):
        """한국어 NLP 리소스 설정"""
        print("\n🇰🇷 한국어 NLP 리소스 설정...")
        
        try:
            # KoNLPy 데이터 다운로드
            import konlpy
            from konlpy.tag import Okt, Komoran, Hannanum, Mecab
            
            # 형태소 분석기 테스트
            okt = Okt()
            test_text = "안녕하세요. 한국어 형태소 분석 테스트입니다."
            result = okt.morphs(test_text)
            print(f"✅ KoNLPy 설정 완료: {result[:5]}")
            
        except Exception as e:
            print(f"⚠️ KoNLPy 설정 실패: {e}")
            print("pip install konlpy 로 설치하세요.")
    
    def create_model_index(self):
        """모델 인덱스 파일 생성"""
        index = {
            "downloaded_at": str(datetime.now()),
            "models": {
                "huggingface": [],
                "ollama": [],
                "local": []
            },
            "datasets": [],
            "total_size_gb": 0
        }
        
        # 다운로드된 모델 스캔
        for model_dir in MODELS_DIR.iterdir():
            if model_dir.is_dir():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                index["models"]["huggingface"].append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_mb": round(size / (1024*1024), 2)
                })
        
        # Ollama 모델 스캔
        if OLLAMA_DIR.exists():
            for model_file in OLLAMA_DIR.rglob("*"):
                if model_file.is_file():
                    size = model_file.stat().st_size
                    index["models"]["ollama"].append({
                        "name": model_file.name,
                        "path": str(model_file),
                        "size_mb": round(size / (1024*1024), 2)
                    })
        
        # 데이터셋 스캔
        for dataset_file in DATASETS_DIR.iterdir():
            if dataset_file.is_file():
                size = dataset_file.stat().st_size
                index["datasets"].append({
                    "name": dataset_file.name,
                    "path": str(dataset_file),
                    "size_mb": round(size / (1024*1024), 2)
                })
        
        # 총 크기 계산
        total_size = sum(model["size_mb"] for model in index["models"]["huggingface"])
        total_size += sum(model["size_mb"] for model in index["models"]["ollama"])
        total_size += sum(dataset["size_mb"] for dataset in index["datasets"])
        index["total_size_gb"] = round(total_size / 1024, 2)
        
        # 인덱스 파일 저장
        with open(BASE_DIR / "model_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 모델 인덱스 생성 완료: {index['total_size_gb']} GB")
        return index

def main():
    """메인 함수"""
    print("🚀 LFM AGI 모델 & 데이터 다운로드 시작")
    print(f"📂 기본 경로: {BASE_DIR}")
    
    # 디렉토리 생성 확인
    MODELS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    OLLAMA_DIR.mkdir(exist_ok=True)
    
    downloader = ModelDownloader()
    
    # 사용자 선택
    print("\n다운로드할 항목을 선택하세요:")
    print("1. Hugging Face 모델")
    print("2. Ollama 모델")
    print("3. 한국어 데이터셋")
    print("4. 전체")
    print("5. 모델 인덱스만 생성")
    
    choice = input("선택 (1-5): ").strip()
    
    if choice == "1":
        downloader.download_huggingface_models()
    elif choice == "2":
        downloader.download_ollama_models()
    elif choice == "3":
        downloader.download_datasets()
        downloader.setup_korean_nlp_resources()
    elif choice == "4":
        downloader.download_huggingface_models()
        downloader.download_ollama_models()
        downloader.download_datasets()
        downloader.setup_korean_nlp_resources()
    elif choice == "5":
        pass
    else:
        print("잘못된 선택입니다.")
        return
    
    # 모델 인덱스 생성
    from datetime import datetime
    index = downloader.create_model_index()
    
    print(f"\n🎉 다운로드 완료!")
    print(f"📊 총 크기: {index['total_size_gb']} GB")
    print(f"📁 위치: {BASE_DIR}")

if __name__ == "__main__":
    main()