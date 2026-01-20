#!/usr/bin/env python3
"""
CARLA 자율주행 시뮬레이터 설정 스크립트
자율주행 AI 개발을 위한 CARLA 환경 구성
"""

import os
import subprocess
import sys
from pathlib import Path
import requests
import json
import shutil
from urllib.parse import urlparse
import tarfile
import zipfile

CARLA_DIR = Path("/mnt/data/lfm_agi/carla")
BASE_DIR = Path("/mnt/data/lfm_agi")

def run_command(cmd: str, check: bool = True):
    """명령어 실행"""
    print(f"실행 중: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"오류: {result.stderr}")
        return False
    print(result.stdout if result.stdout else "완료")
    return True

def check_requirements():
    """시스템 요구사항 확인"""
    print("🔍 CARLA 시스템 요구사항 확인...")
    
    # GPU 확인
    gpu_check = subprocess.run("nvidia-smi", shell=True, capture_output=True)
    if gpu_check.returncode != 0:
        print("❌ NVIDIA GPU 또는 드라이버가 설치되지 않음")
        print("CARLA는 GPU 가속이 권장됩니다.")
        return False
    
    # 디스크 공간 확인 (CARLA는 약 20-30GB 필요)
    disk_usage = shutil.disk_usage(BASE_DIR)
    free_gb = disk_usage.free / (1024**3)
    
    if free_gb < 50:
        print(f"⚠️ 디스크 여유 공간 부족: {free_gb:.1f}GB (50GB 권장)")
        return False
    
    print("✅ 시스템 요구사항 확인 완료")
    return True

def download_carla():
    """CARLA 다운로드 및 설치"""
    print("📥 CARLA 시뮬레이터 다운로드...")
    
    # CARLA 릴리스 정보
    carla_versions = {
        "0.9.15": {
            "linux": "https://github.com/carla-simulator/carla/releases/download/0.9.15/CARLA_0.9.15.tar.gz",
            "additional_maps": "https://github.com/carla-simulator/carla/releases/download/0.9.15/AdditionalMaps_0.9.15.tar.gz",
            "size_gb": 8.5
        },
        "0.9.14": {
            "linux": "https://github.com/carla-simulator/carla/releases/download/0.9.14/CARLA_0.9.14.tar.gz", 
            "additional_maps": "https://github.com/carla-simulator/carla/releases/download/0.9.14/AdditionalMaps_0.9.14.tar.gz",
            "size_gb": 7.8
        }
    }
    
    print("사용 가능한 CARLA 버전:")
    for version, info in carla_versions.items():
        print(f"{version}: {info['size_gb']}GB")
    
    version = input("다운로드할 버전 (0.9.15): ").strip() or "0.9.15"
    
    if version not in carla_versions:
        print("❌ 잘못된 버전입니다")
        return False
    
    version_info = carla_versions[version]
    carla_install_dir = CARLA_DIR / f"CARLA_{version}"
    
    # 이미 다운로드되어 있는지 확인
    if carla_install_dir.exists() and (carla_install_dir / "CarlaUE4.sh").exists():
        print(f"✅ CARLA {version}이 이미 설치되어 있습니다")
        return str(carla_install_dir)
    
    carla_install_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 메인 CARLA 다운로드
        print(f"\n📦 CARLA {version} 메인 패키지 다운로드...")
        carla_tar = CARLA_DIR / f"CARLA_{version}.tar.gz"
        
        if not carla_tar.exists():
            download_cmd = f"wget -O {carla_tar} {version_info['linux']}"
            if not run_command(download_cmd):
                print("❌ CARLA 다운로드 실패")
                return False
        
        # 압축 해제
        print("📂 CARLA 압축 해제 중...")
        extract_cmd = f"cd {carla_install_dir} && tar -xzf {carla_tar}"
        if not run_command(extract_cmd):
            print("❌ CARLA 압축 해제 실패")
            return False
        
        # 추가 맵 다운로드 (선택사항)
        download_maps = input("추가 맵을 다운로드하시겠습니까? (y/N): ").strip().lower()
        if download_maps == 'y':
            print(f"\n🗺️ 추가 맵 다운로드...")
            maps_tar = CARLA_DIR / f"AdditionalMaps_{version}.tar.gz"
            
            if not maps_tar.exists():
                download_cmd = f"wget -O {maps_tar} {version_info['additional_maps']}"
                if not run_command(download_cmd):
                    print("⚠️ 추가 맵 다운로드 실패 (선택사항)")
                else:
                    # 맵 압축 해제
                    extract_cmd = f"cd {carla_install_dir} && tar -xzf {maps_tar}"
                    run_command(extract_cmd)
        
        # 실행 권한 부여
        carla_sh = carla_install_dir / "CarlaUE4.sh"
        if carla_sh.exists():
            os.chmod(carla_sh, 0o755)
            print(f"✅ CARLA {version} 설치 완료: {carla_install_dir}")
            return str(carla_install_dir)
        else:
            print("❌ CARLA 실행 파일을 찾을 수 없습니다")
            return False
            
    except Exception as e:
        print(f"❌ CARLA 설치 오류: {e}")
        return False

def setup_carla_python():
    """CARLA Python API 설정"""
    print("🐍 CARLA Python API 설정...")
    
    # CARLA Python 패키지 설치
    carla_python_commands = [
        "pip install carla",
        "pip install pygame",
        "pip install numpy",
        "pip install opencv-python",
        "pip install matplotlib",
        "pip install scipy",
        "pip install pillow"
    ]
    
    for cmd in carla_python_commands:
        if not run_command(cmd, check=False):
            print(f"⚠️ {cmd} 설치 실패")
    
    # CARLA Python 예제 다운로드
    print("📥 CARLA Python 예제 다운로드...")
    examples_dir = CARLA_DIR / "python_examples"
    examples_dir.mkdir(exist_ok=True)
    
    # GitHub에서 예제 다운로드
    git_cmd = f"cd {examples_dir} && git clone https://github.com/carla-simulator/carla.git carla_repo"
    if run_command(git_cmd, check=False):
        # 예제 파일 복사
        repo_examples = examples_dir / "carla_repo/PythonAPI/examples"
        if repo_examples.exists():
            copy_cmd = f"cp -r {repo_examples}/* {examples_dir}/"
            run_command(copy_cmd, check=False)
            
        # 불필요한 repo 제거
        shutil.rmtree(examples_dir / "carla_repo", ignore_errors=True)
    
    print("✅ CARLA Python API 설정 완료")

def create_carla_scripts():
    """CARLA 실행 스크립트 및 예제 생성"""
    print("📝 CARLA 실행 스크립트 생성...")
    
    # 서버 실행 스크립트
    server_script = """#!/bin/bash
# CARLA 서버 실행 스크립트

CARLA_ROOT="/mnt/data/lfm_agi/carla/CARLA_0.9.15"
DISPLAY_NUM=${DISPLAY_NUM:-0}

echo "🚗 CARLA 서버 시작..."
echo "📂 CARLA 경로: $CARLA_ROOT"

# GPU 정보 확인
echo "🖥️ GPU 정보:"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv

# CARLA 서버 실행
cd "$CARLA_ROOT"

# 헤드리스 모드로 실행 (GUI 없음)
if [ "$1" = "headless" ]; then
    echo "🔧 헤드리스 모드 실행..."
    DISPLAY= ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 -carla-streaming-port=2001 -quality-level=Low
else
    echo "🖥️ GUI 모드 실행..."
    ./CarlaUE4.sh -carla-rpc-port=2000 -carla-streaming-port=2001
fi
"""
    
    with open(CARLA_DIR / "start_carla_server.sh", "w") as f:
        f.write(server_script)
    os.chmod(CARLA_DIR / "start_carla_server.sh", 0o755)
    
    # 클라이언트 테스트 스크립트
    client_script = """#!/usr/bin/env python3
'''
CARLA 클라이언트 테스트 스크립트
기본 연결 및 차량 생성 테스트
'''

import carla
import random
import time
import numpy as np
import cv2

def main():
    print("🚗 CARLA 클라이언트 시작...")
    
    # CARLA 서버에 연결
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        print("✅ CARLA 서버 연결 성공")
        print(f"🌍 버전: {client.get_server_version()}")
        
        # 월드 가져오기
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 날씨 설정
        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=30.0,
            sun_altitude_angle=70.0
        )
        world.set_weather(weather)
        print("🌤️ 날씨 설정 완료")
        
        # 차량 블루프린트 선택
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        # 스폰 포인트 가져오기
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        # 차량 생성
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"🚙 차량 생성 완료: {vehicle.type_id}")
        
        # 카메라 블루프린트 설정
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # 카메라를 차량에 부착
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        print("📷 카메라 부착 완료")
        
        # 자동 조종 활성화
        vehicle.set_autopilot(True)
        print("🤖 자동 조종 활성화")
        
        # 10초 동안 실행
        print("⏰ 10초 동안 시뮬레이션 실행...")
        time.sleep(10)
        
        print("🧹 정리 중...")
        camera.destroy()
        vehicle.destroy()
        print("✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        print("CARLA 서버가 실행 중인지 확인하세요.")

if __name__ == '__main__':
    main()
"""
    
    with open(CARLA_DIR / "test_carla_client.py", "w") as f:
        f.write(client_script)
    os.chmod(CARLA_DIR / "test_carla_client.py", 0o755)
    
    # 자율주행 AI 예제
    ai_script = """#!/usr/bin/env python3
'''
CARLA 자율주행 AI 예제
간단한 차선 추종 및 장애물 회피
'''

import carla
import cv2
import numpy as np
import time
import math

class SimpleAutopilot:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        self.image_data = None
        self.collision_flag = False
        
    def connect(self):
        '''CARLA 서버 연결'''
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print("✅ CARLA 연결 완료")
        
    def spawn_vehicle(self):
        '''차량 생성'''
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print("🚙 차량 생성 완료")
        
    def setup_sensors(self):
        '''센서 설정'''
        blueprint_library = self.world.get_blueprint_library()
        
        # RGB 카메라
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self.process_image)
        
        # 충돌 센서
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self.collision_callback)
        
        print("📷 센서 설정 완료")
        
    def process_image(self, image):
        '''이미지 처리'''
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # RGBA에서 RGB로
        self.image_data = array
        
    def collision_callback(self, event):
        '''충돌 콜백'''
        self.collision_flag = True
        print("💥 충돌 감지!")
        
    def detect_lanes(self, image):
        '''차선 감지'''
        if image is None:
            return None
            
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 가우시안 블러
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny 엣지 검출
        edges = cv2.Canny(blur, 50, 150)
        
        # 관심 영역 마스크
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width//2 - 50, height//2),
            (width//2 + 50, height//2),
            (width, height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough 변환으로 직선 검출
        lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 100, minLineLength=100, maxLineGap=50)
        
        return lines
        
    def calculate_steering(self, lines):
        '''조향각 계산'''
        if lines is None:
            return 0.0
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            
            if slope < -0.5:  # 왼쪽 차선
                left_lines.append(line)
            elif slope > 0.5:  # 오른쪽 차선
                right_lines.append(line)
        
        # 중앙점 계산
        center_offset = 0.0
        if left_lines and right_lines:
            # 양쪽 차선이 모두 감지된 경우
            center_offset = 0.0  # 중앙 유지
        elif left_lines:
            # 왼쪽 차선만 감지된 경우
            center_offset = -0.1  # 오른쪽으로 조향
        elif right_lines:
            # 오른쪽 차선만 감지된 경우
            center_offset = 0.1  # 왼쪽으로 조향
            
        return center_offset
        
    def run_autopilot(self, duration=60):
        '''자율주행 실행'''
        print(f"🤖 자율주행 시작 ({duration}초)")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.collision_flag:
                print("💥 충돌로 인한 정지")
                break
                
            # 이미지 분석
            if self.image_data is not None:
                lines = self.detect_lanes(self.image_data)
                steering = self.calculate_steering(lines)
                
                # 차량 제어
                control = carla.VehicleControl()
                control.throttle = 0.3
                control.steer = steering
                control.brake = 0.0
                
                self.vehicle.apply_control(control)
                
                # 시각화 (선택사항)
                if lines is not None:
                    display_image = self.image_data.copy()
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        cv2.line(display_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    cv2.imshow("CARLA Autopilot", cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            
            time.sleep(0.1)
            
        cv2.destroyAllWindows()
        print("✅ 자율주행 완료")
        
    def cleanup(self):
        '''정리'''
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("🧹 정리 완료")

def main():
    autopilot = SimpleAutopilot()
    
    try:
        autopilot.connect()
        autopilot.spawn_vehicle()
        autopilot.setup_sensors()
        
        time.sleep(2)  # 센서 초기화 대기
        
        autopilot.run_autopilot(30)  # 30초 자율주행
        
    except Exception as e:
        print(f"❌ 오류: {e}")
    finally:
        autopilot.cleanup()

if __name__ == '__main__':
    main()
"""
    
    with open(CARLA_DIR / "simple_autopilot.py", "w") as f:
        f.write(ai_script)
    os.chmod(CARLA_DIR / "simple_autopilot.py", 0o755)
    
    print("✅ CARLA 스크립트 생성 완료")

def setup_carla_docker():
    """CARLA Docker 환경 설정"""
    print("🐳 CARLA Docker 환경 설정...")
    
    dockerfile_content = """# CARLA Docker 이미지
FROM carlasim/carla:0.9.15

# 추가 Python 패키지 설치
RUN apt-get update && apt-get install -y \\
    python3-pip \\
    python3-opencv \\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \\
    numpy \\
    opencv-python \\
    matplotlib \\
    pygame \\
    scipy \\
    pillow

# 작업 디렉토리 설정
WORKDIR /carla

# 포트 노출
EXPOSE 2000 2001 2002

# 시작 스크립트
CMD ["./CarlaUE4.sh", "-RenderOffScreen"]
"""
    
    docker_dir = CARLA_DIR / "docker"
    docker_dir.mkdir(exist_ok=True)
    
    with open(docker_dir / "Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Docker Compose 파일
    compose_content = """version: '3.8'

services:
  carla-server:
    build: .
    image: carla-custom:latest
    container_name: carla-server
    restart: unless-stopped
    
    # GPU 지원
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    # 포트 매핑
    ports:
      - "2000:2000"  # RPC 포트
      - "2001:2001"  # Streaming 포트
      - "2002:2002"  # Secondary 포트
    
    # 볼륨 마운트
    volumes:
      - ../python_examples:/carla/examples
      - carla-data:/carla/Import
    
    # 환경 변수
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
      - DISPLAY=${DISPLAY}
    
    # 네트워크
    networks:
      - carla-network

  carla-client:
    image: python:3.8
    container_name: carla-client
    restart: "no"
    profiles: ["client"]
    
    volumes:
      - ../:/workspace
    
    working_dir: /workspace
    
    depends_on:
      - carla-server
    
    networks:
      - carla-network
    
    command: ["python", "test_carla_client.py"]

networks:
  carla-network:
    driver: bridge

volumes:
  carla-data:
"""
    
    with open(docker_dir / "docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print(f"✅ CARLA Docker 설정 완료: {docker_dir}")

def create_carla_config():
    """CARLA 설정 파일 생성"""
    config = {
        "carla": {
            "version": "0.9.15",
            "install_path": str(CARLA_DIR / "CARLA_0.9.15"),
            "server_port": 2000,
            "streaming_port": 2001
        },
        "simulation": {
            "synchronous_mode": True,
            "fixed_delta_seconds": 0.05,
            "no_rendering_mode": False,
            "quality_level": "Low"
        },
        "autopilot": {
            "max_speed": 30.0,
            "target_fps": 20,
            "image_width": 640,
            "image_height": 480
        },
        "sensors": {
            "camera": {
                "fov": 90,
                "image_size_x": 640,
                "image_size_y": 480
            },
            "lidar": {
                "channels": 32,
                "range": 50.0,
                "points_per_second": 56000
            }
        }
    }
    
    with open(CARLA_DIR / "carla_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ CARLA 설정 파일 생성: {CARLA_DIR}/carla_config.json")

def main():
    """메인 함수"""
    print("""
    🚗 CARLA 자율주행 시뮬레이터 설정
    """)
    
    # 디렉토리 생성
    CARLA_DIR.mkdir(exist_ok=True)
    
    # 시스템 요구사항 확인
    if not check_requirements():
        print("❌ 시스템 요구사항을 확인하세요")
        return
    
    print("\nCARLA 설정 옵션:")
    print("1. 🚗 CARLA 시뮬레이터 다운로드")
    print("2. 🐍 Python API 설정")
    print("3. 📝 스크립트 및 예제 생성")
    print("4. 🐳 Docker 환경 설정")
    print("5. 🚀 전체 설정 (추천)")
    print("6. ✅ 설치 확인 및 테스트")
    
    choice = input("선택 (1-6): ").strip()
    
    if choice == "1":
        carla_path = download_carla()
        if carla_path:
            create_carla_config()
    
    elif choice == "2":
        setup_carla_python()
    
    elif choice == "3":
        create_carla_scripts()
        create_carla_config()
    
    elif choice == "4":
        setup_carla_docker()
    
    elif choice == "5":
        print("\n🚀 전체 CARLA 환경 설정 시작...")
        
        carla_path = download_carla()
        if carla_path:
            setup_carla_python()
            create_carla_scripts()
            setup_carla_docker()
            create_carla_config()
            
            print(f"\n🎉 CARLA 설정 완료!")
            print(f"📁 위치: {CARLA_DIR}")
            print(f"🚗 서버 실행: {CARLA_DIR}/start_carla_server.sh")
            print(f"🐍 클라이언트 테스트: python3 {CARLA_DIR}/test_carla_client.py")
            print(f"🤖 자율주행 예제: python3 {CARLA_DIR}/simple_autopilot.py")
        else:
            print("❌ CARLA 다운로드에 실패했습니다")
    
    elif choice == "6":
        # 설치 확인
        carla_install = CARLA_DIR / "CARLA_0.9.15"
        if carla_install.exists():
            print("✅ CARLA 시뮬레이터 설치됨")
            
            # Python 패키지 확인
            try:
                import carla
                print("✅ CARLA Python API 설치됨")
            except ImportError:
                print("❌ CARLA Python API 누락")
            
            # 스크립트 확인
            if (CARLA_DIR / "start_carla_server.sh").exists():
                print("✅ 실행 스크립트 준비됨")
            
            print(f"\n📊 설치 크기: {sum(f.stat().st_size for f in CARLA_DIR.rglob('*') if f.is_file()) / (1024**3):.1f} GB")
        else:
            print("❌ CARLA가 설치되지 않았습니다")
    
    else:
        print("❌ 잘못된 선택입니다")

if __name__ == "__main__":
    main()