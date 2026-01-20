#!/usr/bin/env python3
"""
LFM AGI 통합 설정 스크립트
모든 리소스와 환경을 한번에 설정
"""

import subprocess
import sys
from pathlib import Path
import time

BASE_DIR = Path("/mnt/data/lfm_agi")

def run_script(script_path: str, options: str = ""):
    """스크립트 실행"""
    cmd = f"cd {BASE_DIR} && python3 {script_path} {options}"
    print(f"🚀 실행: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print(f"✅ 완료: {script_path}")
            return True
        else:
            print(f"❌ 실패: {script_path}")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ 타임아웃: {script_path}")
        return False
    except Exception as e:
        print(f"💥 오류: {e}")
        return False

def check_dependencies():
    """의존성 확인"""
    print("🔍 의존성 확인...")
    
    required_packages = ["torch", "transformers", "fastapi", "uvicorn"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 설치됨")
        except ImportError:
            print(f"❌ {package} 누락")
            missing.append(package)
    
    if missing:
        print(f"⚠️ 누락된 패키지: {missing}")
        print("pip install -r requirements.txt 로 설치하세요")
        return False
    
    return True

def main():
    """메인 설정 실행"""
    print("""
╔══════════════════════════════════════════════╗
║           LFM AGI 통합 설정                   ║
║     데이터 파티션 리소스 구성                 ║
╚══════════════════════════════════════════════╝
    """)
    
    print(f"📂 기본 경로: {BASE_DIR}")
    print(f"💾 여유공간: {round(BASE_DIR.stat().st_size / (1024**3), 2)}GB")
    
    # 의존성 확인
    if not check_dependencies():
        print("❌ 의존성을 먼저 해결하세요")
        return
    
    print("\n설정 옵션을 선택하세요:")
    print("1. 📥 모델 다운로드만")
    print("2. 🏥 Clara 설정만")
    print("3. ⚡ 최적화 설정만")
    print("4. 🐳 Docker 이미지 빌드만")
    print("5. 🚀 전체 설정 (추천)")
    print("6. 📊 상태 확인만")
    
    choice = input("\n선택 (1-6): ").strip()
    
    start_time = time.time()
    
    if choice == "1":
        print("\n📥 모델 다운로드 시작...")
        run_script("download_models.py")
        
    elif choice == "2":
        print("\n🏥 Clara 환경 설정...")
        run_script("setup_clara.py")
        
    elif choice == "3":
        print("\n⚡ 캐시 및 최적화 설정...")
        run_script("setup_cache_optimization.py")
        
    elif choice == "4":
        print("\n🐳 Docker 이미지 빌드...")
        build_cmd = f"cd {BASE_DIR}/docker && docker-compose build"
        subprocess.run(build_cmd, shell=True)
        
    elif choice == "5":
        print("\n🚀 전체 설정 시작...")
        
        steps = [
            ("📥 모델 다운로드", "download_models.py", "4"),  # 전체 다운로드
            ("⚡ 캐시 최적화", "setup_cache_optimization.py", "6"),  # 전체 설정
            ("🏥 Clara 환경", "setup_clara.py", "1"),  # 기본 작업공간
        ]
        
        for step_name, script, option in steps:
            print(f"\n{step_name}...")
            if not run_script(script, option):
                print(f"❌ {step_name} 실패")
                break
            time.sleep(2)
        
        # Docker 이미지 빌드
        print("\n🐳 Docker 이미지 빌드...")
        build_cmd = f"cd {BASE_DIR}/docker && docker-compose build"
        subprocess.run(build_cmd, shell=True)
        
    elif choice == "6":
        print("\n📊 시스템 상태 확인...")
        
        # 디스크 사용량
        disk_usage = subprocess.run(f"du -sh {BASE_DIR}/*", shell=True, capture_output=True, text=True)
        print("💾 디스크 사용량:")
        print(disk_usage.stdout)
        
        # GPU 정보
        gpu_info = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", 
                                 shell=True, capture_output=True, text=True)
        if gpu_info.returncode == 0:
            print("🚀 GPU 정보:")
            print(gpu_info.stdout)
        
        # 모델 파일 확인
        models_dir = BASE_DIR / "models"
        if models_dir.exists():
            model_count = len(list(models_dir.iterdir()))
            print(f"🤖 모델 수: {model_count}")
        
        return
        
    else:
        print("❌ 잘못된 선택입니다")
        return
    
    elapsed = time.time() - start_time
    print(f"\n🎉 설정 완료! (소요 시간: {elapsed/60:.1f}분)")
    
    # 다음 단계 안내
    print("\n📋 다음 단계:")
    print(f"1. 환경 변수 설정: source {BASE_DIR}/setup_env.sh")
    print(f"2. AGI 시스템 실행: cd /home/kim/lfm_agi && python3 run_agi.py")
    print(f"3. Docker 실행: cd {BASE_DIR}/docker && docker-compose up -d")
    print(f"4. 모니터링: {BASE_DIR}/cache/monitor.html")
    
    print(f"\n📁 모든 데이터 위치: {BASE_DIR}")
    print("🚀 LFM AGI 시스템 준비 완료!")

if __name__ == "__main__":
    main()