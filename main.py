"""실시간 운동 자세 코칭 에이전트.

웹캠 → MediaPipe 포즈 → 관절 각도/렙 카운팅(로컬) → 렙/세트 완료 시 Claude 코칭.

사용법:
    python main.py --exercise squat
    python main.py -e pushup --camera 1

실행 중 키:
    1 스쿼트 / 2 푸시업 / 3 플랭크 / 4 데드리프트 로 전환
    q 또는 ESC : 종료
"""
import argparse
import time

import cv2

import config
from src.coach.agent import AsyncCoach, build_coach
from src.exercises import registry
from src.exercises.base import HoldResult, RepResult
from src.pose.estimator import PoseEstimator
from src.ui.overlay import Overlay

_HOTKEYS = {ord("1"): "squat", ord("2"): "pushup",
            ord("3"): "plank", ord("4"): "deadlift"}


def _issues_payload(issues):
    return [{"code": i.code, "message_ko": i.message_ko, "severity": i.severity}
            for i in issues]


def _rep_summary(exercise, result: RepResult):
    return {
        "exercise": exercise.name,
        "event": "rep",
        "rep_index": result.rep_index,
        "metrics": result.metrics,
        "depth": result.depth,
        "tempo_s": result.tempo_s,
        "detected_issues": _issues_payload(result.issues),
    }


def _hold_summary(exercise, result: HoldResult):
    return {
        "exercise": exercise.name,
        "event": "hold",
        "duration_s": result.duration_s,
        "metrics": exercise.last_metrics,
        "detected_issues": _issues_payload(result.issues),
    }


def main():
    parser = argparse.ArgumentParser(description="실시간 운동 자세 코칭 에이전트")
    parser.add_argument("-e", "--exercise", default="squat",
                        choices=registry.available(),
                        help="운동 종류 (기본: squat)")
    parser.add_argument("-c", "--camera", type=int, default=config.CAMERA_INDEX,
                        help="웹캠 장치 인덱스")
    args = parser.parse_args()

    exercise = registry.create(args.exercise)
    estimator = PoseEstimator()
    overlay = Overlay()
    async_coach = AsyncCoach(build_coach())

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"웹캠({args.camera})을 열 수 없습니다.")
        return

    print(f"코칭 시작: {exercise.name_ko}  (모델: {config.COACH_MODEL})")
    print("키: 1 스쿼트 / 2 푸시업 / 3 플랭크 / 4 데드리프트 / q 종료")

    window = "FitCoach - 실시간 자세 코칭"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            t = time.perf_counter()

            landmarks = estimator.process(frame)
            if landmarks is not None:
                estimator.draw_skeleton(frame)
                result = exercise.update(landmarks, t)
                if isinstance(result, RepResult):
                    async_coach.submit(_rep_summary(exercise, result))
                elif isinstance(result, HoldResult):
                    async_coach.submit(_hold_summary(exercise, result))

            hold_s = exercise.hold_duration if exercise.mode == "hold" else None
            frame = overlay.render(
                frame,
                exercise_ko=exercise.name_ko,
                rep_count=exercise.rep_count,
                phase=exercise.phase,
                hold_s=hold_s,
                coaching=async_coach.latest,
                busy=async_coach.busy,
            )

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in _HOTKEYS:
                name = _HOTKEYS[key]
                if name != exercise.name:
                    exercise = registry.create(name)
                    print(f"운동 전환: {exercise.name_ko}")
    finally:
        async_coach.stop()
        estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
