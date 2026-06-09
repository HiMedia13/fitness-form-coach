"""운동 레지스트리. 이름 → Exercise 클래스 매핑."""
from typing import Dict, List, Type

from .base import Exercise
from .deadlift import Deadlift
from .lunge import Lunge
from .overhead_press import OverheadPress
from .plank import Plank
from .pushup import Pushup
from .squat import Squat

_REGISTRY: Dict[str, Type[Exercise]] = {
    Squat.name: Squat,
    Pushup.name: Pushup,
    Plank.name: Plank,
    Deadlift.name: Deadlift,
    Lunge.name: Lunge,
    OverheadPress.name: OverheadPress,
}


def create(name: str) -> Exercise:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"알 수 없는 운동: {name}. 사용 가능: {', '.join(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> List[str]:
    return list(_REGISTRY)
