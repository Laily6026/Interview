#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""발명자 인터뷰 전사기의 공통 전사 및 결과 저장 로직."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class TranscriptionProfile:
    key: str
    label: str
    description: str
    options: dict


PROFILES = {
    "high_recall": TranscriptionProfile(
        key="high_recall",
        label="누락 최소화 (권장)",
        description="민감한 VAD와 앞뒤 1초 여백으로 작은 음성을 살리면서 긴 무음 환각을 막습니다.",
        options={
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.15,
                "min_speech_duration_ms": 0,
                "min_silence_duration_ms": 1500,
                "speech_pad_ms": 1000,
            },
            "no_speech_threshold": 0.7,
            "log_prob_threshold": -1.1,
            "condition_on_previous_text": True,
            "hallucination_silence_threshold": 2.0,
        },
    ),
    "balanced": TranscriptionProfile(
        key="balanced",
        label="균형",
        description="보수적인 VAD로 긴 무음을 줄이되 약한 음성도 최대한 보존합니다.",
        options={
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.35,
                "min_speech_duration_ms": 0,
                "min_silence_duration_ms": 1200,
                "speech_pad_ms": 600,
            },
            "no_speech_threshold": 0.7,
            "log_prob_threshold": -1.1,
            "condition_on_previous_text": True,
        },
    ),
    "fast": TranscriptionProfile(
        key="fast",
        label="빠르게",
        description="명확한 음성만 VAD로 선별합니다. 작은 목소리는 빠질 수 있습니다.",
        options={
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 0,
                "min_silence_duration_ms": 700,
                "speech_pad_ms": 400,
            },
            "no_speech_threshold": 0.6,
            "log_prob_threshold": -1.0,
            "condition_on_previous_text": True,
        },
    ),
}


@dataclass
class SegmentRecord:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class ReviewGap:
    start: float
    end: float
    previous_text: str = ""
    next_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptionResult:
    source: Path
    duration: float
    language: str
    language_probability: float
    profile_key: str
    segments: list[SegmentRecord] = field(default_factory=list)
    gaps: list[ReviewGap] = field(default_factory=list)
    txt_path: Optional[Path] = None
    srt_path: Optional[Path] = None
    md_path: Optional[Path] = None
    review_path: Optional[Path] = None


def format_timestamp(seconds: float, srt: bool = False) -> str:
    seconds = max(0.0, seconds)
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if srt:
        milliseconds = int(round((seconds - total) * 1000))
        if milliseconds == 1000:
            total += 1
            hours, remainder = divmod(total, 3600)
            minutes, secs = divmod(remainder, 60)
            milliseconds = 0
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def find_review_gaps(
    segments: Iterable[SegmentRecord],
    duration: float,
    minimum_gap_seconds: float = 15.0,
) -> list[ReviewGap]:
    """전사 세그먼트 사이의 긴 공백을 찾아 청취 검토 목록으로 만든다."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    gaps: list[ReviewGap] = []
    previous_end = 0.0
    previous_text = ""

    for segment in ordered:
        start = max(0.0, segment.start)
        if start - previous_end >= minimum_gap_seconds:
            gaps.append(
                ReviewGap(
                    start=previous_end,
                    end=start,
                    previous_text=previous_text,
                    next_text=segment.text,
                )
            )
        if segment.end > previous_end:
            previous_end = segment.end
            previous_text = segment.text

    if duration - previous_end >= minimum_gap_seconds:
        gaps.append(
            ReviewGap(
                start=previous_end,
                end=duration,
                previous_text=previous_text,
            )
        )
    return gaps


def build_summary_template(result: TranscriptionResult) -> str:
    full_text = " ".join(segment.text for segment in result.segments)
    return f"""# 발명자 인터뷰 정리 초안

- **녹음 파일**: {result.source.name}
- **녹음 길이**: {format_timestamp(result.duration)}
- **전사 모드**: {PROFILES[result.profile_key].label}
- **인터뷰 일시**: (기입)
- **발명자**: (기입)
- **사건번호 / 관리번호**: (기입)

---

## 1. 발명의 배경 / 해결하려는 과제
(전사문에서 해당 내용을 발췌·정리)

## 2. 발명의 핵심 구성
(전사문에서 해당 내용을 발췌·정리)

## 3. 종래기술 대비 차이점 / 효과
(전사문에서 해당 내용을 발췌·정리)

## 4. 실시예 / 변형예
(전사문에서 해당 내용을 발췌·정리)

## 5. 추가 확인 필요 사항 (발명자 회신 요청)
- [ ]

---

## 전체 전사문 (원문)

{full_text}
"""


def build_review_report(result: TranscriptionResult) -> str:
    lines = [
        "# 전사 검토 구간",
        "",
        "아래 구간은 전사 세그먼트 사이의 공백이 길어 원본을 다시 들어볼 후보입니다.",
        "실제 무음일 수도 있으며, 이 목록 자체가 누락을 뜻하지는 않습니다.",
        "",
    ]
    if not result.gaps:
        lines.append("지정한 기준 이상의 긴 공백이 없습니다.")
        return "\n".join(lines) + "\n"

    for index, gap in enumerate(result.gaps, start=1):
        lines.append(
            f"{index}. [{format_timestamp(gap.start)} ~ {format_timestamp(gap.end)}] "
            f"({gap.duration:.1f}초)"
        )
        if gap.previous_text:
            lines.append(f"   - 직전: {gap.previous_text[-100:]}")
        if gap.next_text:
            lines.append(f"   - 직후: {gap.next_text[:100]}")
    return "\n".join(lines) + "\n"


def write_outputs(
    result: TranscriptionResult,
    *,
    include_timestamps: bool = True,
    make_srt: bool = True,
    output_dir: Optional[Path] = None,
) -> TranscriptionResult:
    """Windows 메모장에서도 한글을 안정적으로 여는 UTF-8 BOM으로 저장한다."""
    output_dir = output_dir or result.source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result.source.stem
    result.txt_path = output_dir / f"{stem}_전사.txt"
    result.srt_path = output_dir / f"{stem}_자막.srt" if make_srt else None
    result.md_path = output_dir / f"{stem}_정리초안.md"
    result.review_path = output_dir / f"{stem}_전사_검토구간.md"

    txt_lines = []
    srt_blocks = []
    for index, segment in enumerate(result.segments, start=1):
        if include_timestamps:
            txt_lines.append(f"[{format_timestamp(segment.start)}] {segment.text}")
        else:
            txt_lines.append(segment.text)
        srt_blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment.start, srt=True)} --> "
            f"{format_timestamp(segment.end, srt=True)}\n"
            f"{segment.text}\n"
        )

    result.txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8-sig")
    if result.srt_path is not None:
        result.srt_path.write_text("\n".join(srt_blocks), encoding="utf-8-sig")
    result.md_path.write_text(build_summary_template(result), encoding="utf-8-sig")
    result.review_path.write_text(build_review_report(result), encoding="utf-8-sig")
    return result


class TranscriberEngine:
    def __init__(self):
        self.model = None
        self.loaded_model_name: Optional[str] = None

    def load_model(
        self,
        model_name: str,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        if self.model is not None and self.loaded_model_name == model_name:
            return
        from faster_whisper import WhisperModel

        if progress_callback:
            progress_callback(0.0, f"모델 로딩 중: {model_name}")
        self.model = WhisperModel(model_name, device="auto", compute_type="auto")
        self.loaded_model_name = model_name

    def transcribe(
        self,
        source: Path,
        *,
        model_name: str = "large-v3",
        profile_key: str = "high_recall",
        prompt: Optional[str] = None,
        gap_seconds: float = 15.0,
        include_timestamps: bool = True,
        make_srt: bool = True,
        output_dir: Optional[Path] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        if profile_key not in PROFILES:
            raise ValueError(f"지원하지 않는 전사 모드입니다: {profile_key}")
        if not source.exists():
            raise FileNotFoundError(source)

        self.load_model(model_name, progress_callback=progress_callback)
        profile = PROFILES[profile_key]
        options = dict(profile.options)
        if "vad_parameters" in options:
            options["vad_parameters"] = dict(options["vad_parameters"])

        segments_iter, info = self.model.transcribe(
            str(source),
            language="ko",
            initial_prompt=prompt,
            hotwords=prompt,
            beam_size=5,
            word_timestamps=True,
            **options,
        )

        segments: list[SegmentRecord] = []
        last_reported = -1
        for raw_segment in segments_iter:
            text = raw_segment.text.strip()
            if not text:
                continue
            segment = SegmentRecord(
                start=float(raw_segment.start),
                end=float(raw_segment.end),
                text=text,
                avg_logprob=float(raw_segment.avg_logprob),
                no_speech_prob=float(raw_segment.no_speech_prob),
            )
            segments.append(segment)
            percent = min(100, int(segment.end / max(info.duration, 0.001) * 100))
            if progress_callback and percent >= last_reported + 5:
                last_reported = percent
                progress_callback(
                    percent / 100.0,
                    f"{format_timestamp(segment.end)} / {format_timestamp(info.duration)} 처리 중",
                )

        result = TranscriptionResult(
            source=source,
            duration=float(info.duration),
            language=str(info.language),
            language_probability=float(info.language_probability),
            profile_key=profile_key,
            segments=segments,
        )
        result.gaps = find_review_gaps(segments, result.duration, gap_seconds)
        write_outputs(
            result,
            include_timestamps=include_timestamps,
            make_srt=make_srt,
            output_dir=output_dir,
        )
        if progress_callback:
            progress_callback(1.0, "전사 및 결과 저장 완료")
        return result
