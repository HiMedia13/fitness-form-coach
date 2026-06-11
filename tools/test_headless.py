"""헤드리스 파이프라인 테스트 (GUI 창 없이 영상 분석).

실제 운동 영상에 대해 포즈 추출 → 렙 카운팅 → 운동 자동 인식 → 세트 요약 → 코칭을
콘솔로 검증한다. cv2.imshow 를 쓰지 않으므로 디스플레이 없는 환경에서도 동작한다.

사용:
    python tools/test_headless.py <영상경로> [운동명]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from src.coach.agent import RuleBasedCoach  # noqa: E402
from src.exercises import registry  # noqa: E402
from src.exercises.base import HoldResult, RepResult  # noqa: E402
from src.pose.estimator import PoseEstimator  # noqa: E402
from src.recognition import ExerciseRecognizer  # noqa: E402
from src.session import SetTracker  # noqa: E402


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else None
    ex_name = sys.argv[2] if len(sys.argv) > 2 else "squat"
    if not video or not os.path.exists(video):
        print(f"영상 파일을 찾을 수 없습니다: {video}")
        sys.exit(1)

    exercise = registry.create(ex_name)
    estimator = PoseEstimator()
    recognizer = ExerciseRecognizer()
    set_tracker = SetTracker()
    coach = RuleBasedCoach()

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (fps and fps == fps and fps > 0):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"영상: {os.path.basename(video)}  fps={fps:.1f}  frames={total_frames}  "
          f"운동(초기)={exercise.name_ko}")
    print("-" * 64)

    frame_idx = 0
    detected_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else frame_idx / fps
        frame_idx += 1

        lm = estimator.process(frame)
        if lm is not None:
            detected_frames += 1
            recognizer.feed(lm)
            result = exercise.update(lm, t)
            if isinstance(result, RepResult):
                set_tracker.add_rep(result, t)
                print(f"[t={t:5.1f}s] 렙 {result.rep_index:2d}  "
                      f"depth={result.depth:5.1f}  tempo={result.tempo_s:.1f}s  "
                      f"issues={[i.code for i in result.issues]}")
            elif isinstance(result, HoldResult):
                set_tracker.add_hold(result, t)
                print(f"[t={t:5.1f}s] 유지 {result.duration_s:.1f}s")

    name, conf, scores = recognizer.predict()
    summary = set_tracker.finish(exercise.name)
    cap.release()
    estimator.close()

    print("-" * 64)
    print(f"포즈 인식 프레임: {detected_frames}/{frame_idx}")
    print(f"총 렙: {exercise.rep_count}")
    print(f"운동 자동 인식: {name}  신뢰도={conf:.2f}")
    print("  점수표:", {k: round(v, 2) for k, v in scores.items()})

    if summary is not None:
        if name:
            summary["detected_exercise"] = name
            summary["detection_confidence"] = round(conf, 2)
        print("-" * 64)
        print("세트 요약:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        c = coach.coach(summary)
        print("-" * 64)
        print(f"코칭: 점수 {c.form_score} ({c.severity}) · {c.headline}")
        for cue in c.cues:
            print(f"  • {cue}")
        print(f"  {c.encouragement}")
    else:
        print("세트 데이터 없음 (렙 미감지)")


if __name__ == "__main__":
    main()
