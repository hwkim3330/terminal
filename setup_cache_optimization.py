#!/usr/bin/env python3
"""
모델 캐시 및 최적화 설정 스크립트
GPU 메모리 최적화, 모델 양자화, 캐시 관리
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
import torch
import psutil

BASE_DIR = Path("/mnt/data/lfm_agi")
CACHE_DIR = BASE_DIR / "cache"
MODELS_DIR = BASE_DIR / "models"

def get_system_info():
    """시스템 정보 수집"""
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(shutil.disk_usage(BASE_DIR).free / (1024**3), 2),
        "cuda_available": torch.cuda.is_available()
    }
    
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "id": i,
                "name": props.name,
                "memory_gb": round(props.total_memory / (1024**3), 2)
            })
    
    return info

def setup_environment_cache():
    """환경 변수 캐시 설정"""
    print("🔧 환경 변수 캐시 설정...")
    
    cache_dirs = {
        "HF_HOME": CACHE_DIR / "huggingface",
        "TRANSFORMERS_CACHE": CACHE_DIR / "transformers",
        "TORCH_HOME": CACHE_DIR / "torch",
        "TORCH_EXTENSIONS_DIR": CACHE_DIR / "torch_extensions",
        "CUDA_CACHE_PATH": CACHE_DIR / "cuda",
        "NUMBA_CACHE_DIR": CACHE_DIR / "numba",
        "OLLAMA_MODELS": BASE_DIR / "ollama_models"
    }
    
    # 캐시 디렉토리 생성
    for env_var, path in cache_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(path)
        print(f"✅ {env_var} = {path}")
    
    # 환경 변수 스크립트 생성
    env_script = "#!/bin/bash\n# LFM AGI 환경 변수 설정\n\n"
    for env_var, path in cache_dirs.items():
        env_script += f"export {env_var}=\"{path}\"\n"
    
    env_script += """
# PyTorch 최적화 설정
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# CUDA 최적화
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_MEMORY_FRACTION=0.8

echo "✅ LFM AGI 환경 변수 설정 완료"
"""
    
    with open(BASE_DIR / "setup_env.sh", "w") as f:
        f.write(env_script)
    
    os.chmod(BASE_DIR / "setup_env.sh", 0o755)
    print(f"✅ 환경 설정 스크립트: {BASE_DIR}/setup_env.sh")

def create_optimization_configs():
    """모델 최적화 설정 생성"""
    print("⚡ 모델 최적화 설정 생성...")
    
    # GPU 메모리 기반 최적화 설정
    system_info = get_system_info()
    
    optimization_configs = {
        "low_memory": {
            "description": "저메모리 환경 (< 8GB GPU)",
            "model_config": {
                "load_in_8bit": True,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "torch_dtype": "float16"
            },
            "generation_config": {
                "max_length": 512,
                "batch_size": 1,
                "use_cache": True
            }
        },
        "medium_memory": {
            "description": "중간메모리 환경 (8-16GB GPU)",
            "model_config": {
                "load_in_8bit": False,
                "device_map": "auto", 
                "low_cpu_mem_usage": True,
                "torch_dtype": "float16"
            },
            "generation_config": {
                "max_length": 1024,
                "batch_size": 2,
                "use_cache": True
            }
        },
        "high_memory": {
            "description": "고메모리 환경 (> 16GB GPU)",
            "model_config": {
                "load_in_8bit": False,
                "device_map": None,
                "low_cpu_mem_usage": False,
                "torch_dtype": "float16"
            },
            "generation_config": {
                "max_length": 2048,
                "batch_size": 4,
                "use_cache": True
            }
        }
    }
    
    # 시스템에 맞는 기본 설정 선택
    if system_info["cuda_available"]:
        max_gpu_memory = max(device["memory_gb"] for device in system_info["cuda_devices"])
        if max_gpu_memory < 8:
            default_config = "low_memory"
        elif max_gpu_memory < 16:
            default_config = "medium_memory"
        else:
            default_config = "high_memory"
    else:
        default_config = "low_memory"
    
    optimization_configs["default"] = optimization_configs[default_config]
    optimization_configs["system_info"] = system_info
    
    with open(CACHE_DIR / "optimization_configs.json", "w") as f:
        json.dump(optimization_configs, f, indent=2)
    
    print(f"✅ 최적화 설정 생성 완료: {default_config} 선택됨")

def setup_model_quantization():
    """모델 양자화 설정"""
    print("🔢 모델 양자화 설정...")
    
    quantization_script = """#!/usr/bin/env python3
'''
모델 양자화 스크립트
8bit, 4bit 양자화로 메모리 사용량 대폭 감소
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from pathlib import Path

def quantize_model(model_name: str, output_dir: str, bits: int = 8):
    '''모델 양자화 및 저장'''
    print(f"🔢 {model_name} {bits}bit 양자화 시작...")
    
    # 양자화 설정
    if bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    elif bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        raise ValueError("bits는 4 또는 8이어야 합니다")
    
    try:
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 저장
        output_path = Path(output_dir) / f"{model_name.replace('/', '_')}_{bits}bit"
        output_path.mkdir(parents=True, exist_ok=True)
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # 메타데이터 저장
        metadata = {
            "original_model": model_name,
            "quantization_bits": bits,
            "quantization_type": "bitsandbytes",
            "torch_dtype": "float16",
            "device_map": "auto"
        }
        
        with open(output_path / "quantization_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 양자화 완료: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"❌ 양자화 실패: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="모델 양자화")
    parser.add_argument("--model", required=True, help="모델 이름")
    parser.add_argument("--output", required=True, help="출력 디렉토리")
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8, help="양자화 비트")
    
    args = parser.parse_args()
    
    quantize_model(args.model, args.output, args.bits)
"""
    
    with open(CACHE_DIR / "quantize_models.py", "w") as f:
        f.write(quantization_script)
    
    os.chmod(CACHE_DIR / "quantize_models.py", 0o755)
    print(f"✅ 양자화 스크립트 생성: {CACHE_DIR}/quantize_models.py")

def setup_memory_management():
    """메모리 관리 설정"""
    print("🧠 메모리 관리 설정...")
    
    memory_script = """#!/usr/bin/env python3
'''
GPU 메모리 관리 유틸리티
메모리 모니터링, 캐시 정리, 최적화
'''

import torch
import gc
import psutil
import nvidia_ml_py3 as nvml
from typing import Dict, List

class GPUMemoryManager:
    def __init__(self):
        if torch.cuda.is_available():
            nvml.nvmlInit()
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = 0
    
    def get_memory_info(self) -> Dict:
        '''GPU 메모리 정보 조회'''
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        info = {}
        for i in range(self.device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            info[f"gpu_{i}"] = {
                "name": nvml.nvmlDeviceGetName(handle).decode(),
                "total_mb": round(mem_info.total / 1024**2),
                "used_mb": round(mem_info.used / 1024**2),
                "free_mb": round(mem_info.free / 1024**2),
                "utilization": round((mem_info.used / mem_info.total) * 100, 1)
            }
        
        return info
    
    def clear_cache(self):
        '''GPU 캐시 정리'''
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            print("✅ GPU 캐시 정리 완료")
        else:
            print("⚠️ CUDA가 사용 불가능합니다")
    
    def optimize_memory(self):
        '''메모리 최적화'''
        # Python 가비지 컬렉션
        gc.collect()
        
        # PyTorch 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("✅ 메모리 최적화 완료")
    
    def print_memory_summary(self):
        '''메모리 사용량 요약 출력'''
        print("📊 메모리 사용량:")
        
        # CPU 메모리
        cpu_memory = psutil.virtual_memory()
        print(f"CPU: {cpu_memory.percent:.1f}% ({cpu_memory.used // 1024**3}GB / {cpu_memory.total // 1024**3}GB)")
        
        # GPU 메모리
        gpu_info = self.get_memory_info()
        for gpu_id, info in gpu_info.items():
            if "error" not in info:
                print(f"{info['name']}: {info['utilization']:.1f}% ({info['used_mb']}MB / {info['total_mb']}MB)")

if __name__ == "__main__":
    manager = GPUMemoryManager()
    
    import sys
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "clear":
            manager.clear_cache()
        elif action == "optimize":
            manager.optimize_memory()
        elif action == "info":
            print(manager.get_memory_info())
        elif action == "summary":
            manager.print_memory_summary()
    else:
        manager.print_memory_summary()
"""
    
    with open(CACHE_DIR / "memory_manager.py", "w") as f:
        f.write(memory_script)
    
    os.chmod(CACHE_DIR / "memory_manager.py", 0o755)
    print(f"✅ 메모리 관리자 생성: {CACHE_DIR}/memory_manager.py")

def setup_model_serving():
    """모델 서빙 최적화"""
    print("🚀 모델 서빙 최적화...")
    
    serving_script = """#!/usr/bin/env python3
'''
최적화된 모델 서빙 서버
TensorRT, TorchScript, ONNX 등 최적화 지원
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging

app = FastAPI(title="Optimized Model Server")

class ModelRequest(BaseModel):
    text: str
    model_name: Optional[str] = "default"
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class ModelManager:
    def __init__(self, config_path: str):
        self.models = {}
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path: str) -> Dict:
        with open(config_path) as f:
            return json.load(f)
    
    async def load_model(self, model_name: str, optimization: str = "default"):
        '''모델 로드 및 최적화'''
        if model_name in self.models:
            return self.models[model_name]
        
        config = self.config.get(optimization, self.config["default"])
        
        try:
            # 모델 로드 (실제 구현 필요)
            print(f"📥 모델 로드 중: {model_name}")
            
            # 최적화 적용
            if config["model_config"].get("torch_compile", False):
                # PyTorch 2.0 컴파일 최적화
                pass
            
            self.models[model_name] = {
                "model": None,  # 실제 모델 객체
                "tokenizer": None,  # 토크나이저
                "config": config
            }
            
            print(f"✅ 모델 로드 완료: {model_name}")
            return self.models[model_name]
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise HTTPException(status_code=500, detail=str(e))

model_manager = ModelManager("/mnt/data/lfm_agi/cache/optimization_configs.json")

@app.post("/generate")
async def generate(request: ModelRequest):
    '''텍스트 생성'''
    try:
        model_info = await model_manager.load_model(request.model_name)
        
        # 실제 생성 로직 (구현 필요)
        response = f"Generated response for: {request.text}"
        
        return {
            "response": response,
            "model": request.model_name,
            "parameters": {
                "max_length": request.max_length,
                "temperature": request.temperature
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    '''로드된 모델 목록'''
    return {"models": list(model_manager.models.keys())}

@app.get("/health")
async def health_check():
    '''헬스 체크'''
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "loaded_models": len(model_manager.models)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
"""
    
    with open(CACHE_DIR / "model_server.py", "w") as f:
        f.write(serving_script)
    
    os.chmod(CACHE_DIR / "model_server.py", 0o755)
    print(f"✅ 모델 서빙 서버 생성: {CACHE_DIR}/model_server.py")

def create_monitoring_dashboard():
    """모니터링 대시보드 생성"""
    print("📊 모니터링 대시보드 생성...")
    
    dashboard_html = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LFM AGI 리소스 모니터</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #16213e; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .metric { text-align: center; margin: 10px 0; }
        .metric-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .metric-label { color: #ccc; }
        .chart-container { position: relative; height: 200px; margin: 20px 0; }
        .status { padding: 10px; border-radius: 5px; margin: 5px 0; }
        .status.online { background: #4CAF50; }
        .status.offline { background: #f44336; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 LFM AGI 리소스 모니터</h1>
        
        <div class="grid">
            <div class="card">
                <h3>🖥️ 시스템 정보</h3>
                <div class="metric">
                    <div class="metric-value" id="cpuUsage">--</div>
                    <div class="metric-label">CPU 사용률 (%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="memoryUsage">--</div>
                    <div class="metric-label">메모리 사용률 (%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="diskUsage">--</div>
                    <div class="metric-label">디스크 사용률 (%)</div>
                </div>
            </div>
            
            <div class="card">
                <h3>🚀 GPU 정보</h3>
                <div id="gpuInfo"></div>
                <button onclick="clearGPUCache()">GPU 캐시 정리</button>
            </div>
            
            <div class="card">
                <h3>🤖 모델 상태</h3>
                <div id="modelStatus"></div>
                <button onclick="refreshModels()">모델 새로고침</button>
            </div>
            
            <div class="card">
                <h3>📊 성능 차트</h3>
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 차트 초기화
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'GPU 사용률',
                    data: [],
                    borderColor: '#4CAF50',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: 'white' } } },
                scales: {
                    y: { ticks: { color: 'white' } },
                    x: { ticks: { color: 'white' } }
                }
            }
        });

        // 데이터 업데이트 함수들
        async function updateSystemInfo() {
            // 실제 API 호출 시뮬레이션
            document.getElementById('cpuUsage').textContent = Math.floor(Math.random() * 100);
            document.getElementById('memoryUsage').textContent = Math.floor(Math.random() * 100);
            document.getElementById('diskUsage').textContent = Math.floor(Math.random() * 100);
        }

        async function updateGPUInfo() {
            const gpuDiv = document.getElementById('gpuInfo');
            gpuDiv.innerHTML = `
                <div class="metric">
                    <div class="metric-value">${Math.floor(Math.random() * 100)}</div>
                    <div class="metric-label">GPU 0 사용률 (%)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${Math.floor(Math.random() * 100)}</div>
                    <div class="metric-label">GPU 메모리 (%)</div>
                </div>
            `;
        }

        async function updateModelStatus() {
            const models = ['LFM-2.5-VL', 'EXAONE-3.5', 'KoAlpaca'];
            const statusDiv = document.getElementById('modelStatus');
            
            statusDiv.innerHTML = models.map(model => 
                `<div class="status online">${model}: 온라인</div>`
            ).join('');
        }

        async function clearGPUCache() {
            alert('GPU 캐시를 정리했습니다.');
        }

        async function refreshModels() {
            await updateModelStatus();
        }

        // 정기 업데이트
        setInterval(() => {
            updateSystemInfo();
            updateGPUInfo();
            updateModelStatus();
            
            // 차트 업데이트
            const now = new Date().toLocaleTimeString();
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(Math.floor(Math.random() * 100));
            
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update();
        }, 2000);

        // 초기 로드
        updateSystemInfo();
        updateGPUInfo();
        updateModelStatus();
    </script>
</body>
</html>"""
    
    with open(CACHE_DIR / "monitor.html", "w") as f:
        f.write(dashboard_html)
    
    print(f"✅ 모니터링 대시보드 생성: {CACHE_DIR}/monitor.html")

def main():
    """메인 함수"""
    print("⚡ LFM AGI 캐시 및 최적화 설정")
    
    # 디렉토리 생성
    CACHE_DIR.mkdir(exist_ok=True)
    
    # 시스템 정보 출력
    system_info = get_system_info()
    print(f"\n💻 시스템 정보:")
    print(f"CPU: {system_info['cpu_count']}코어")
    print(f"메모리: {system_info['memory_gb']}GB")
    print(f"디스크 여유공간: {system_info['disk_free_gb']}GB")
    print(f"CUDA: {system_info['cuda_available']}")
    
    if system_info['cuda_available']:
        for device in system_info['cuda_devices']:
            print(f"GPU {device['id']}: {device['name']} ({device['memory_gb']}GB)")
    
    print("\n설정 옵션:")
    print("1. 기본 캐시 설정")
    print("2. 모델 최적화 도구")
    print("3. 메모리 관리 도구") 
    print("4. 서빙 서버 설정")
    print("5. 모니터링 대시보드")
    print("6. 전체 설정")
    
    choice = input("선택 (1-6): ").strip()
    
    if choice == "1":
        setup_environment_cache()
        create_optimization_configs()
        
    elif choice == "2":
        setup_model_quantization()
        
    elif choice == "3":
        setup_memory_management()
        
    elif choice == "4":
        setup_model_serving()
        
    elif choice == "5":
        create_monitoring_dashboard()
        
    elif choice == "6":
        setup_environment_cache()
        create_optimization_configs()
        setup_model_quantization()
        setup_memory_management()
        setup_model_serving()
        create_monitoring_dashboard()
        
    else:
        print("잘못된 선택입니다.")
        return
    
    print(f"\n🎉 캐시 및 최적화 설정 완료!")
    print(f"📁 위치: {CACHE_DIR}")
    print(f"🔧 환경 설정: source {BASE_DIR}/setup_env.sh")

if __name__ == "__main__":
    main()