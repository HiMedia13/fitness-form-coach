"""운동 자세 코칭 에이전트 (실시간 웹캠 + 영상 파일).

웹캠/영상 → MediaPipe 포즈 → 관절 각도/렙 카운팅(로컬) → 렙/세트 완료 시 Claude 코칭.

사용법:
    python main.py --exercise squat                # 웹캠
    python main.py -e pushup --camera 1            # 다른 웹캠
    python main.py -e squat --video clips/set1.mp4 # 영상 파일 분석

실행 중 키:
    1 스쿼트 / 2 푸시업 / 3 플랭크 / 4 데드리프트 / 5 런지 / 6 오버헤드프레스 로 전환
    s 세트 종료(종합 코칭) / space 일시정지(영상) / q 또는 ESC : 종료
"""
import argparse
import time

import cv2

import config
from src.coach.agent import AsyncCoach, build_coach
from src.exercises import registry
from src.exercises.base import HoldResult, RepResult
from src.pose.estimator import PoseEstimator
from src.session import SetTracker
from src.ui.overlay import Overlay

_HOTKEYS = {ord("1"): "squat", ord("2"): "pushup",
            ord("3"): "plank", ord("4"): "deadlift",
            ord("5"): "lunge", ord("6"): "overhead_press"}


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


def _finalize_set(set_tracker, exercise, async_coach, reason=""):
    """현재 세트를 마감해 종합 코칭을 요청하고 카운터를 초기화한다."""
    summary = set_tracker.finish(exercise.name)
    if summary is None:
        return
    async_coach.submit(summary)
    n = summary.get("total_reps", summary.get("total_holds", 0))
    tag = f" ({reason})" if reason else ""
    print(f"세트 {summary['set_index']} 종료{tag}: {n}개 → 종합 코칭 요청")
    exercise.reset_counts()


def main():
    parser = argparse.ArgumentParser(description="운동 자세 코칭 에이전트")
    parser.add_argument("-e", "--exercise", default="squat",
                        choices=registry.available(),
                        help="운동 종류 (기본: squat)")
    parser.add_argument("-c", "--camera", type=int, default=config.CAMERA_INDEX,
                        help="웹캠 장치 인덱스 (--video 미지정 시)")
    parser.add_argument("-v", "--video", default=None,
                        help="분석할 영상 파일 경로 (지정 시 웹캠 대신 영상 분석)")
    args = parser.parse_args()

    exercise = registry.create(args.exercise)
    estimator = PoseEstimator()
    overlay = Overlay()
    async_coach = AsyncCoach(build_coach())
    set_tracker = SetTracker()

    is_video = args.video is not None
    source = args.video if is_video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        kind = f"영상 파일({args.video})" if is_video else f"웹캠({args.camera})"
        print(f"{kind} 을(를) 열 수 없습니다.")
        return

    # 영상은 자체 타임라인(프레임/FPS)으로 시간을 계산해야 처리 속도와 무관하게
    # 템포가 정확하다. 웹캠은 벽시계(perf_counter)를 쓴다.
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (fps and fps == fps and fps > 0):  # 0/NaN 방어
        fps = 30.0
    frame_delay = max(1, int(1000 / fps)) if is_video else 1

    src_label = f"영상 {args.video}" if is_video else f"웹캠 {args.camera}"
    print(f"코칭 시작: {exercise.name_ko}  ({src_label}, 모델: {config.COACH_MODEL})")
    print("키: 1~6 운동 전환 / s 세트 종료(종합 코칭) / "
          "space 일시정지 / q 종료")

    window = "FitCoach - 자세 코칭"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    paused = False
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if is_video:
                        print("영상 끝.")
                    break

                if is_video:
                    # 영상 자체 시간(초). POS_MSEC 가 신뢰 가능하면 사용, 아니면 프레임/fps.
                    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    t = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else frame_idx / fps
                else:
                    frame = cv2.flip(frame, 1)  # 웹캠 거울 모드
                    t = time.perf_counter()
                frame_idx += 1

                landmarks = estimator.process(frame)
                if landmarks is not None:
                    estimator.draw_skeleton(frame)
                    result = exercise.update(landmarks, t)
                    if isinstance(result, RepResult):
                        async_coach.submit(_rep_summary(exercise, result))
                        set_tracker.add_rep(result, t)
                    elif isinstance(result, HoldResult):
                        async_coach.submit(_hold_summary(exercise, result))
                        set_tracker.add_hold(result, t)

                # 휴식(무동작) 감지 시 세트 자동 종료 → 종합 코칭
                if set_tracker.should_auto_finish(t):
                    _finalize_set(set_tracker, exercise, async_coach, "휴식 감지")

                hold_s = exercise.hold_duration if exercise.mode == "hold" else None
                display = overlay.render(
                    frame,
                    exercise_ko=exercise.name_ko,
                    rep_count=exercise.rep_count,
                    phase=exercise.phase,
                    hold_s=hold_s,
                    coaching=async_coach.latest,
                    busy=async_coach.busy,
                    set_index=set_tracker.set_index,
                )
                cv2.imshow(window, display)

            key = cv2.waitKey(frame_delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and is_video:
                paused = not paused
            if key == ord("s"):
                _finalize_set(set_tracker, exercise, async_coach, "수동")
            if key in _HOTKEYS:
                name = _HOTKEYS[key]
                if name != exercise.name:
                    # 운동 전환 전 현재 세트 마감
                    _finalize_set(set_tracker, exercise, async_coach, "운동 전환")
                    exercise = registry.create(name)
                    print(f"운동 전환: {exercise.name_ko}")
    finally:
        # 종료 전 남은 세트 마감
        _finalize_set(set_tracker, exercise, async_coach, "종료")
        print(f"종료: {exercise.name_ko}")
        async_coach.stop()
        estimator.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
