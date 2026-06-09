"""포즈 관련 경량 타입. (cv2/mediapipe 의존 없음)"""
from typing import Dict, Tuple

# 랜드마크: index -> (x, y, z, visibility), x/y/z 는 정규화 좌표
Landmarks = Dict[int, Tuple[float, float, float, float]]
