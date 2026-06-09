"""관절 각도 계산 유틸. 랜드마크는 (x, y, z) 정규화 좌표 튜플."""
import math
from typing import Sequence

import numpy as np

Point = Sequence[float]


def joint_angle(a: Point, b: Point, c: Point) -> float:
    """꼭짓점 b 에서 a-b-c 가 이루는 각도(도). 3D 벡터 기준."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cos = float(np.dot(ba, bc) / denom)
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def segment_angle_to_horizontal(a: Point, b: Point) -> float:
    """선분 a-b 가 수평선과 이루는 각도(도, 0~90). 이미지 평면(x,y)만 사용.

    등/정강이의 '기울기'를 판단할 때 사용한다. 0=수평, 90=수직.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])
    return math.degrees(math.atan2(dy, dx + 1e-9))


def midpoint(a: Point, b: Point) -> Point:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return tuple((a + b) / 2.0)
