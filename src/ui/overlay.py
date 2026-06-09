"""화면 오버레이: 한글 텍스트(PIL), 점수/렙/코칭 패널."""
import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

_FALLBACK_FONTS = [
    config.KOREAN_FONT_PATH,
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FALLBACK_FONTS:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


class Overlay:
    def __init__(self) -> None:
        self.font_lg = _find_font(34)
        self.font_md = _find_font(24)
        self.font_sm = _find_font(18)

    def _put_text(self, img_pil, draw, xy, text, font, fill):
        draw.text(xy, text, font=font, fill=fill)

    def render(
        self,
        frame_bgr,
        exercise_ko: str,
        rep_count: int,
        phase: str,
        hold_s: Optional[float],
        coaching,            # Coaching | None
        busy: bool,
        set_index: int = 1,
    ):
        """frame 위에 정보 패널을 그려 반환한다."""
        # PIL 로 변환 (한글 렌더링)
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = img.size

        # 상단 상태 바
        draw.rectangle([0, 0, w, 70], fill=(0, 0, 0, 150))
        status = f"세트 {set_index}  ·  {exercise_ko}"
        if hold_s is not None:
            status += f"   유지 {hold_s:.0f}s"
        else:
            status += f"   {rep_count} 회   [{phase}]"
        draw.text((16, 16), status, font=self.font_lg, fill=(255, 255, 255))
        if busy:
            draw.text((w - 150, 22), "분석 중…", font=self.font_md,
                      fill=(255, 220, 120))

        # 하단 코칭 패널
        if coaching is not None:
            panel_h = 170
            draw.rectangle([0, h - panel_h, w, h], fill=(0, 0, 0, 170))

            color = {
                "good": (120, 230, 120),
                "minor": (255, 220, 120),
                "major": (255, 110, 110),
            }.get(coaching.severity, (255, 255, 255))

            draw.text((16, h - panel_h + 12),
                      f"점수 {coaching.form_score}  ·  {coaching.headline}",
                      font=self.font_lg, fill=color)

            y = h - panel_h + 60
            for cue in coaching.cues[:3]:
                draw.text((24, y), f"• {cue}", font=self.font_md,
                          fill=(255, 255, 255))
                y += 30

            draw.text((16, h - 32), coaching.encouragement,
                      font=self.font_sm, fill=(200, 200, 200))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
