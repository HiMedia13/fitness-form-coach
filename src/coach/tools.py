"""세트 심층 분석 에이전트가 사용하는 도구 정의 + 디스패치.

도구는 방금 끝난 세트의 렙 기록(RepResult 리스트)을 읽기만 한다. Claude 가
list_reps → get_rep_stats / get_rep_trajectory 로 원인을 파고든 뒤
submit_coaching(터미널)으로 최종 코칭을 제출한다.
"""
import json
from typing import Dict, List

# 읽기 도구 스키마 (Claude 에 전달)
READ_TOOLS = [
    {
        "name": "list_reps",
        "description": "세트의 모든 렙 개요(번호, 깊이, 템포, 감지된 이슈, 사용 가능한 "
                       "관절 각도 metric 키)를 반환한다.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_rep_stats",
        "description": "한 렙의 관절별 최소/최대/평균 각도, 깊이, 템포, 이슈를 반환한다.",
        "input_schema": {
            "type": "object",
            "properties": {"rep_index": {"type": "integer", "description": "렙 번호(1부터)"}},
            "required": ["rep_index"],
        },
    },
    {
        "name": "get_rep_trajectory",
        "description": "한 렙에서 특정 관절 각도(metric)의 시간순 궤적 샘플을 반환한다. "
                       "바닥 도달 지점, 스티킹 포인트, 반동, 좌우 비대칭 분석에 사용.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rep_index": {"type": "integer"},
                "metric": {"type": "string", "description": "예: knee_angle, hip_angle, back_angle"},
            },
            "required": ["rep_index", "metric"],
        },
    },
]

# 터미널 도구: 최종 코칭 제출 (Coaching 스키마와 동일 필드, strict)
SUBMIT_TOOL = {
    "name": "submit_coaching",
    "description": "분석을 마친 뒤 최종 코칭을 제출한다. 모든 텍스트는 한국어.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "form_score": {"type": "integer", "description": "0~100 종합 점수"},
            "severity": {"type": "string", "enum": ["good", "minor", "major"]},
            "headline": {"type": "string", "description": "한 줄 총평"},
            "cues": {"type": "array", "items": {"type": "string"},
                     "description": "다음 세트 핵심 교정 큐 1~3개"},
            "encouragement": {"type": "string"},
        },
        "required": ["form_score", "severity", "headline", "cues", "encouragement"],
        "additionalProperties": False,
    },
}

ALL_TOOLS = READ_TOOLS + [SUBMIT_TOOL]


def _downsample(seq: List, n: int = 14) -> List:
    if len(seq) <= n:
        return seq
    step = (len(seq) - 1) / (n - 1)
    return [seq[round(i * step)] for i in range(n)]


def _metric_keys(record) -> List[str]:
    keys = set(record.metrics.keys())
    for _, m in record.trajectory:
        keys.update(m.keys())
    return sorted(keys)


def dispatch(name: str, args: Dict, rep_map: Dict[int, object]) -> str:
    """읽기 도구를 실행해 JSON 문자열을 반환한다."""
    if name == "list_reps":
        reps = []
        for idx in sorted(rep_map):
            r = rep_map[idx]
            reps.append({
                "rep_index": r.rep_index,
                "depth": round(r.depth, 1),
                "tempo_s": r.tempo_s,
                "issues": [i.code for i in r.issues],
                "metrics": _metric_keys(r),
            })
        return json.dumps({"reps": reps}, ensure_ascii=False)

    if name == "get_rep_stats":
        r = rep_map.get(int(args["rep_index"]))
        if r is None:
            return json.dumps({"error": "해당 렙 없음"}, ensure_ascii=False)
        agg: Dict[str, List[float]] = {}
        for _, m in r.trajectory:
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        stats = {k: {"min": round(min(vs), 1), "max": round(max(vs), 1),
                     "avg": round(sum(vs) / len(vs), 1)}
                 for k, vs in agg.items() if vs}
        return json.dumps({
            "rep_index": r.rep_index,
            "depth": round(r.depth, 1),
            "tempo_s": r.tempo_s,
            "issues": [{"code": i.code, "severity": i.severity,
                        "message_ko": i.message_ko} for i in r.issues],
            "metric_stats": stats,
        }, ensure_ascii=False)

    if name == "get_rep_trajectory":
        r = rep_map.get(int(args["rep_index"]))
        if r is None:
            return json.dumps({"error": "해당 렙 없음"}, ensure_ascii=False)
        metric = args["metric"]
        series = [(t, m[metric]) for t, m in r.trajectory if metric in m]
        if not series:
            return json.dumps(
                {"error": f"metric '{metric}' 없음", "available": _metric_keys(r)},
                ensure_ascii=False)
        pts = [{"t": round(t, 2), "v": round(v, 1)} for t, v in _downsample(series)]
        return json.dumps({"rep_index": r.rep_index, "metric": metric, "points": pts},
                          ensure_ascii=False)

    return json.dumps({"error": f"알 수 없는 도구: {name}"}, ensure_ascii=False)
