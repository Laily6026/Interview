#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""발명자 인터뷰 전사기의 공통 전사 및 결과 저장 로직."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import io
import json
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Optional
from urllib.parse import quote
import wave


ProgressCallback = Callable[[float, str], None]
APP_VERSION = "1.3"


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
            "condition_on_previous_text": False,
            "hallucination_silence_threshold": 1.5,
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
            "condition_on_previous_text": False,
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
            "condition_on_previous_text": False,
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
    compression_ratio: float = 0.0
    audio_rms_db: Optional[float] = None
    voiced_fraction: Optional[float] = None


@dataclass
class QualityIssue:
    start: float
    end: float
    kind: str
    reason: str
    text: str = ""


@dataclass
class RetryAttempt:
    start: float
    end: float
    reasons: list[str] = field(default_factory=list)
    replace_existing: bool = False
    candidate_count: int = 0
    accepted: bool = False
    note: str = ""


@dataclass
class ReviewGap:
    start: float
    end: float
    previous_text: str = ""
    next_text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VertexReviewConfig:
    """사용자가 명시적으로 켠 Vertex AI 선택 검토 설정."""

    service_account_json: Optional[str] = None
    project_id: Optional[str] = None
    location: str = "global"
    model: str = "gemini-3.7-flash"
    padding_seconds: float = 8.0
    max_clip_seconds: float = 120.0


@dataclass
class GeminiCandidateSegment:
    start: float
    end: float
    text: str
    speaker: str = ""
    uncertain: bool = False


@dataclass
class GeminiReviewAttempt:
    gap_start: float
    gap_end: float
    clip_start: float
    clip_end: float
    status: str
    model: str
    candidates: list[GeminiCandidateSegment] = field(default_factory=list)
    note: str = ""
    input_token_count: Optional[int] = None
    total_token_count: Optional[int] = None


@dataclass
class TranscriptionResult:
    source: Path
    duration: float
    language: str
    language_probability: float
    profile_key: str
    segments: list[SegmentRecord] = field(default_factory=list)
    gaps: list[ReviewGap] = field(default_factory=list)
    quality_issues: list[QualityIssue] = field(default_factory=list)
    retry_attempts: list[RetryAttempt] = field(default_factory=list)
    gemini_reviews: list[GeminiReviewAttempt] = field(default_factory=list)
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


def normalize_hotwords(hotwords: Optional[str]) -> Optional[str]:
    """쉼표·줄바꿈으로 입력한 용어를 faster-whisper용 목록으로 정리한다."""
    if not hotwords:
        return None
    terms = [term.strip() for term in re.split(r"[,;\n]+", hotwords) if term.strip()]
    return ", ".join(dict.fromkeys(terms)) or None


def _review_clip_bounds(
    gap: ReviewGap,
    duration: float,
    padding_seconds: float,
    max_clip_seconds: float,
) -> Optional[tuple[float, float]]:
    """공백 전체를 보존하면서 가능한 범위에서 앞뒤 문맥을 붙인다."""
    if gap.duration <= 0 or gap.duration > max_clip_seconds:
        return None
    padding_seconds = max(0.0, padding_seconds)
    clip_start = max(0.0, gap.start - padding_seconds)
    clip_end = min(duration, gap.end + padding_seconds)
    if clip_end - clip_start <= max_clip_seconds:
        return clip_start, clip_end

    available_padding = max_clip_seconds - gap.duration
    before = min(padding_seconds, available_padding / 2.0, gap.start)
    after = min(padding_seconds, available_padding - before, duration - gap.end)
    remaining = available_padding - before - after
    if remaining > 0:
        add_before = min(remaining, gap.start - before)
        before += add_before
        remaining -= add_before
    if remaining > 0:
        after += min(remaining, duration - gap.end - after)
    return gap.start - before, gap.end + after


def _wav_base64(audio, start: float, end: float, sample_rate: int = 16000) -> str:
    """faster-whisper의 mono float32 오디오 일부를 메모리 WAV로 변환한다."""
    import numpy as np

    start_sample = max(0, int(round(start * sample_rate)))
    end_sample = min(len(audio), int(round(end * sample_rate)))
    clip = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
    if clip.size == 0:
        raise ValueError("Gemini에 보낼 오디오 구간이 비어 있습니다.")
    pcm = (np.clip(clip, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _json_from_gemini_text(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Gemini 응답 JSON의 최상위 값이 객체가 아닙니다.")
    return value


class VertexGeminiReviewer:
    """Vertex AI Gemini에 선택한 오디오 클립을 보내 검토 후보만 받는다."""

    _SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    def __init__(self, config: VertexReviewConfig):
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2 import service_account

        source = config.service_account_json or os.environ.get("VERTEX_SA_JSON")
        if not source:
            raise ValueError(
                "Vertex 서비스 계정 JSON 파일을 선택하거나 VERTEX_SA_JSON 환경 변수를 설정해 주세요."
            )
        source_text = str(source).strip()
        if source_text.startswith("{"):
            info = json.loads(source_text)
        else:
            source_path = Path(source_text).expanduser()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Vertex 서비스 계정 JSON을 찾을 수 없습니다: {source_text}"
                )
            info = json.loads(source_path.read_text(encoding="utf-8-sig"))

        self.config = config
        self.project_id = (
            config.project_id
            or os.environ.get("VERTEX_PROJECT_ID")
            or info.get("project_id")
        )
        if not self.project_id:
            raise ValueError("Vertex AI 프로젝트 ID를 확인할 수 없습니다.")
        self.credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=self._SCOPES,
        )
        self._google_request = GoogleRequest()

    def _endpoint(self) -> str:
        location = (self.config.location or "global").strip()
        host = (
            "aiplatform.googleapis.com"
            if location == "global"
            else f"{location}-aiplatform.googleapis.com"
        )
        project = quote(self.project_id, safe="-._")
        encoded_location = quote(location, safe="-._")
        model = quote(self.config.model, safe="-._")
        return (
            f"https://{host}/v1/projects/{project}/locations/{encoded_location}/"
            f"publishers/google/models/{model}:generateContent"
        )

    def _generate(self, payload: dict) -> dict:
        import httpx

        if not self.credentials.valid:
            self.credentials.refresh(self._google_request)
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                self._endpoint(),
                headers={"Authorization": f"Bearer {self.credentials.token}"},
                json=payload,
            )
        if response.is_error:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except (ValueError, AttributeError):
                detail = response.text
            raise RuntimeError(f"Vertex AI HTTP {response.status_code}: {detail[:500]}")
        return response.json()

    def review_gap(
        self,
        gap: ReviewGap,
        *,
        audio_base64: str,
        clip_start: float,
        clip_end: float,
        hotwords: Optional[str] = None,
    ) -> GeminiReviewAttempt:
        prompt = f"""당신은 한국어 기술 회의 전사 검토자입니다.
첨부 오디오는 원본의 {format_timestamp(clip_start)}~{format_timestamp(clip_end)} 구간입니다.
로컬 전사에서 비어 있는 구간은 {format_timestamp(gap.start)}~{format_timestamp(gap.end)}입니다.
직전 문맥: {gap.previous_text or '(없음)'}
직후 문맥: {gap.next_text or '(없음)'}
참고 기술용어: {hotwords or '(없음)'}

실제로 들리는 한국어 발화만 전사하세요. 추측으로 문장을 만들지 말고, 불명확하면 uncertain을 true로 표시하세요.
start와 end는 첨부 클립 시작을 0초로 한 상대 시간입니다. 화자 구분이 불확실하면 speaker를 빈 문자열로 두세요.
이 결과는 사람이 검토할 후보이며 기존 전사에 자동 반영되지 않습니다."""
        schema = {
            "type": "OBJECT",
            "properties": {
                "segments": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "speaker": {"type": "STRING"},
                            "text": {"type": "STRING"},
                            "uncertain": {"type": "BOOLEAN"},
                        },
                        "required": ["start", "end", "speaker", "text", "uncertain"],
                    },
                },
                "note": {"type": "STRING"},
            },
            "required": ["segments", "note"],
        }
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "audio/wav", "data": audio_base64}},
                ],
            }],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        response = self._generate(payload)
        usage = response.get("usageMetadata", {})
        base = dict(
            gap_start=gap.start,
            gap_end=gap.end,
            clip_start=clip_start,
            clip_end=clip_end,
            model=self.config.model,
            input_token_count=usage.get("promptTokenCount"),
            total_token_count=usage.get("totalTokenCount"),
        )
        block_reason = response.get("promptFeedback", {}).get("blockReason")
        candidates = response.get("candidates") or []
        finish_reason = candidates[0].get("finishReason") if candidates else None
        if block_reason or finish_reason in {"PROHIBITED_CONTENT", "SAFETY", "BLOCKLIST"}:
            reason = block_reason or finish_reason
            return GeminiReviewAttempt(
                **base,
                status="blocked",
                note=f"Vertex AI 정책에 의해 차단됨: {reason}. 자동 재요청하지 않았습니다.",
            )
        if not candidates:
            return GeminiReviewAttempt(
                **base,
                status="error",
                note="Vertex AI가 후보 응답을 반환하지 않았습니다.",
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        response_text = "".join(str(part.get("text", "")) for part in parts)
        parsed = _json_from_gemini_text(response_text)
        clip_duration = max(0.0, clip_end - clip_start)
        candidate_segments: list[GeminiCandidateSegment] = []
        for raw in parsed.get("segments", []):
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            relative_start = max(0.0, min(float(raw.get("start", 0.0)), clip_duration))
            relative_end = max(relative_start, min(float(raw.get("end", relative_start)), clip_duration))
            candidate_segments.append(
                GeminiCandidateSegment(
                    start=clip_start + relative_start,
                    end=clip_start + relative_end,
                    speaker=str(raw.get("speaker", "")).strip(),
                    text=text,
                    uncertain=bool(raw.get("uncertain", False)),
                )
            )
        return GeminiReviewAttempt(
            **base,
            status="success",
            candidates=candidate_segments,
            note=str(parsed.get("note", "")).strip(),
        )


def review_gaps_with_gemini(
    result: TranscriptionResult,
    config: VertexReviewConfig,
    audio,
    *,
    hotwords: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[GeminiReviewAttempt]:
    """남은 공백을 검토하되 오류가 로컬 전사 결과 저장을 막지 않게 한다."""
    if not result.gaps:
        return []
    try:
        reviewer = VertexGeminiReviewer(config)
    except Exception as exc:
        return [
            GeminiReviewAttempt(
                gap_start=gap.start,
                gap_end=gap.end,
                clip_start=gap.start,
                clip_end=gap.end,
                status="error",
                model=config.model,
                note=f"Gemini 검토를 시작하지 못했습니다: {exc}",
            )
            for gap in result.gaps
        ]

    attempts: list[GeminiReviewAttempt] = []
    for index, gap in enumerate(result.gaps, start=1):
        bounds = _review_clip_bounds(
            gap,
            result.duration,
            config.padding_seconds,
            config.max_clip_seconds,
        )
        if bounds is None:
            attempts.append(
                GeminiReviewAttempt(
                    gap_start=gap.start,
                    gap_end=gap.end,
                    clip_start=gap.start,
                    clip_end=gap.end,
                    status="skipped",
                    model=config.model,
                    note=f"공백이 {config.max_clip_seconds:.0f}초를 초과해 외부 전송하지 않았습니다.",
                )
            )
            continue
        clip_start, clip_end = bounds
        if progress_callback:
            progress_callback(
                0.96 + 0.03 * index / len(result.gaps),
                f"Gemini 선택 검토 {index}/{len(result.gaps)}",
            )
        try:
            audio_base64 = _wav_base64(audio, clip_start, clip_end)
            attempt = reviewer.review_gap(
                gap,
                audio_base64=audio_base64,
                clip_start=clip_start,
                clip_end=clip_end,
                hotwords=hotwords,
            )
        except Exception as exc:
            attempt = GeminiReviewAttempt(
                gap_start=gap.start,
                gap_end=gap.end,
                clip_start=clip_start,
                clip_end=clip_end,
                status="error",
                model=config.model,
                note=f"Gemini 검토 오류: {exc}",
            )
        attempts.append(attempt)
    return attempts


def _normalized_text(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", text.lower())


def _text_similarity(left: str, right: str) -> float:
    left_normalized = _normalized_text(left)
    right_normalized = _normalized_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def find_quality_issues(
    segments: Iterable[SegmentRecord],
    *,
    prompt_texts: Iterable[str] = (),
    maximum_segment_seconds: float = 30.0,
    short_fragment_length: int = 2,
    short_fragment_count: int = 5,
    short_fragment_window_seconds: float = 30.0,
) -> list[QualityIssue]:
    """긴 압축 세그먼트, 프롬프트 복사, 반복과 짧은 파편 연속을 찾는다."""
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    prompts = [text for text in prompt_texts if text and len(_normalized_text(text)) >= 8]
    issues: list[QualityIssue] = []

    for segment in ordered:
        duration = max(0.001, segment.end - segment.start)
        density = len(_normalized_text(segment.text)) / duration
        if duration > maximum_segment_seconds:
            issues.append(
                QualityIssue(
                    segment.start,
                    segment.end,
                    "long_segment",
                    f"단일 세그먼트가 {duration:.1f}초로 너무 깁니다.",
                    segment.text,
                )
            )
        elif duration >= 20.0 and density < 0.5:
            issues.append(
                QualityIssue(
                    segment.start,
                    segment.end,
                    "sparse_segment",
                    f"텍스트 밀도가 {density:.2f}자/초로 낮습니다.",
                    segment.text,
                )
            )

        for prompt in prompts:
            if _text_similarity(segment.text, prompt) >= 0.85:
                issues.append(
                    QualityIssue(
                        segment.start,
                        segment.end,
                        "prompt_echo",
                        "입력 프롬프트와 거의 같은 문장이 출력되었습니다.",
                        segment.text,
                    )
                )
                break

    repeated_start = 0
    while repeated_start < len(ordered):
        normalized = _normalized_text(ordered[repeated_start].text)
        if len(normalized) < 8:
            repeated_start += 1
            continue
        repeated_end = repeated_start + 1
        while (
            repeated_end < len(ordered)
            and _text_similarity(ordered[repeated_end].text, ordered[repeated_start].text) >= 0.96
        ):
            repeated_end += 1
        if repeated_end - repeated_start >= 2:
            issues.append(
                QualityIssue(
                    ordered[repeated_start].start,
                    ordered[repeated_end - 1].end,
                    "repeated_text",
                    f"유사 문장이 {repeated_end - repeated_start}회 연속 반복되었습니다.",
                    ordered[repeated_start].text,
                )
            )
        repeated_start = repeated_end

    short_indices = [
        index
        for index, segment in enumerate(ordered)
        if len(_normalized_text(segment.text)) <= short_fragment_length
    ]
    cursor = 0
    while cursor < len(short_indices):
        start_index = short_indices[cursor]
        last_cursor = cursor
        while (
            last_cursor + 1 < len(short_indices)
            and ordered[short_indices[last_cursor + 1]].start - ordered[start_index].start
            <= short_fragment_window_seconds
        ):
            last_cursor += 1
        if last_cursor - cursor + 1 >= short_fragment_count:
            end_index = short_indices[last_cursor]
            issues.append(
                QualityIssue(
                    ordered[start_index].start,
                    ordered[end_index].end,
                    "short_fragment_streak",
                    f"{short_fragment_window_seconds:.0f}초 안에 {last_cursor - cursor + 1}개의 짧은 파편이 연속되었습니다.",
                )
            )
            cursor = last_cursor + 1
        else:
            cursor += 1

    return sorted(issues, key=lambda item: (item.start, item.end, item.kind))


def build_retry_attempts(
    issues: Iterable[QualityIssue],
    gaps: Iterable[ReviewGap],
    *,
    duration: float,
    retry_gaps: bool = True,
    padding_seconds: float = 3.0,
    maximum_gap_retry_seconds: float = 180.0,
) -> list[RetryAttempt]:
    """서로 가까운 품질 문제를 합쳐 재전사할 시간 범위를 만든다."""
    candidates: list[RetryAttempt] = []
    for issue in issues:
        candidates.append(
            RetryAttempt(
                start=max(0.0, issue.start - padding_seconds),
                end=min(duration, issue.end + padding_seconds),
                reasons=[issue.reason],
                replace_existing=True,
            )
        )
    if retry_gaps:
        for gap in gaps:
            if gap.duration <= maximum_gap_retry_seconds:
                candidates.append(
                    RetryAttempt(
                        start=max(0.0, gap.start - padding_seconds),
                        end=min(duration, gap.end + padding_seconds),
                        reasons=[f"{gap.duration:.1f}초 전사 공백"],
                        replace_existing=False,
                    )
                )

    candidates.sort(key=lambda item: (item.start, item.end))
    merged: list[RetryAttempt] = []
    for candidate in candidates:
        if merged and candidate.start <= merged[-1].end + 1.0:
            current = merged[-1]
            current.end = max(current.end, candidate.end)
            current.reasons.extend(reason for reason in candidate.reasons if reason not in current.reasons)
            current.replace_existing = current.replace_existing or candidate.replace_existing
        else:
            merged.append(candidate)
    return merged


def normalize_audio_for_retry(audio, *, sample_rate: int = 16000):
    """저음량 회의 구간용 완만한 자동 게인을 적용한다."""
    import numpy as np

    normalized = np.asarray(audio, dtype=np.float32).copy()
    frame_size = max(1, int(sample_rate * 0.5))
    smoothed_gain = 1.0
    target_rms = 0.06
    for start in range(0, normalized.size, frame_size):
        end = min(normalized.size, start + frame_size)
        frame = normalized[start:end]
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64) + 1e-12))
        desired_gain = 1.0 if rms < 0.001 else min(6.0, max(1.0, target_rms / rms))
        smoothed_gain = 0.7 * smoothed_gain + 0.3 * desired_gain
        normalized[start:end] = np.clip(frame * smoothed_gain, -0.98, 0.98)
    return normalized


def annotate_audio_energy(
    segments: Iterable[SegmentRecord],
    audio,
    *,
    sample_rate: int = 16000,
) -> None:
    """재전사 후보가 실제 음성 에너지를 포함하는지 원본 음원에서 측정한다."""
    import math
    import numpy as np

    for segment in segments:
        start = max(0, int(segment.start * sample_rate))
        end = min(len(audio), max(start + 1, int(segment.end * sample_rate)))
        clip = audio[start:end]
        if clip.size == 0:
            segment.audio_rms_db = -120.0
            segment.voiced_fraction = 0.0
            continue
        rms = float(np.sqrt(np.mean(np.square(clip), dtype=np.float64) + 1e-12))
        segment.audio_rms_db = 20.0 * math.log10(max(rms, 1e-6))
        segment.voiced_fraction = float(np.mean(np.abs(clip) > 0.01))


def _segments_from_iterator(segments_iter) -> list[SegmentRecord]:
    records: list[SegmentRecord] = []
    for raw_segment in segments_iter:
        text = raw_segment.text.strip()
        if not text:
            continue
        records.append(
            SegmentRecord(
                start=float(raw_segment.start),
                end=float(raw_segment.end),
                text=text,
                avg_logprob=float(getattr(raw_segment, "avg_logprob", 0.0)),
                no_speech_prob=float(getattr(raw_segment, "no_speech_prob", 0.0)),
                compression_ratio=float(getattr(raw_segment, "compression_ratio", 0.0)),
            )
        )
    return records


def _looks_like_generic_hallucination(text: str) -> bool:
    phrases = (
        "시청해 주셔서",
        "시청해주셔서",
        "구독과 좋아요",
        "자막 제공",
        "이곳에 오신 것을 환영",
        "한국국토정보공사",
        "이 영상은 제작지원",
    )
    return any(phrase in text for phrase in phrases)


def _candidate_is_usable(
    segments: list[SegmentRecord],
    prompts: Iterable[str],
    *,
    require_audio_evidence: bool = False,
) -> bool:
    if not segments or sum(len(_normalized_text(item.text)) for item in segments) < 4:
        return False
    if any(_looks_like_generic_hallucination(item.text) for item in segments):
        return False
    if require_audio_evidence and any(
        item.audio_rms_db is None
        or item.voiced_fraction is None
        or item.audio_rms_db < -40.0
        or item.voiced_fraction < 0.25
        for item in segments
    ):
        return False
    if find_quality_issues(segments, prompt_texts=prompts, maximum_segment_seconds=45.0):
        serious = {"long_segment", "prompt_echo", "repeated_text", "short_fragment_streak"}
        if any(
            issue.kind in serious
            for issue in find_quality_issues(
                segments, prompt_texts=prompts, maximum_segment_seconds=45.0
            )
        ):
            return False
    average_logprob = sum(item.avg_logprob for item in segments) / len(segments)
    average_no_speech = sum(item.no_speech_prob for item in segments) / len(segments)
    return average_logprob >= -1.1 and average_no_speech <= 0.75


def merge_retry_segments(
    original: Iterable[SegmentRecord],
    candidates: Iterable[SegmentRecord],
    attempts: list[RetryAttempt],
    *,
    prompt_texts: Iterable[str] = (),
    require_audio_evidence: bool = False,
) -> list[SegmentRecord]:
    """재시도별 품질을 확인하고 채택된 구간만 원 전사에 병합한다."""
    merged = list(original)
    candidate_list = list(candidates)
    for attempt in attempts:
        selected = [
            segment
            for segment in candidate_list
            if attempt.start <= (segment.start + segment.end) / 2 <= attempt.end
        ]
        attempt.candidate_count = len(selected)
        if require_audio_evidence:
            selected = [
                segment
                for segment in selected
                if segment.audio_rms_db is not None
                and segment.voiced_fraction is not None
                and segment.audio_rms_db >= -40.0
                and segment.voiced_fraction >= 0.25
                and not _looks_like_generic_hallucination(segment.text)
            ]
        if not _candidate_is_usable(
            selected,
            prompt_texts,
            require_audio_evidence=require_audio_evidence,
        ):
            attempt.note = "재전사 결과가 비었거나 품질 기준을 통과하지 못해 원문을 유지했습니다."
            continue

        existing = [
            segment
            for segment in merged
            if segment.end > attempt.start and segment.start < attempt.end
        ]
        if attempt.replace_existing:
            original_max_duration = max(
                (segment.end - segment.start for segment in existing), default=0.0
            )
            candidate_max_duration = max(
                (segment.end - segment.start for segment in selected), default=0.0
            )
            if len(selected) < 2 and candidate_max_duration >= original_max_duration:
                attempt.note = "재전사 결과가 기존 이상 구간보다 개선되지 않아 원문을 유지했습니다."
                continue
            merged = [
                segment
                for segment in merged
                if segment.end <= attempt.start or segment.start >= attempt.end
            ]
        else:
            selected = [
                segment
                for segment in selected
                if not any(
                    segment.end > current.start and segment.start < current.end
                    for current in existing
                )
            ]
            if not selected:
                attempt.note = "기존 세그먼트와 겹쳐 추가할 새 발화가 없었습니다."
                continue

        merged.extend(selected)
        attempt.accepted = True
        attempt.note = f"재전사 세그먼트 {len(selected)}개를 채택했습니다."

    merged.sort(key=lambda item: (item.start, item.end))
    deduplicated: list[SegmentRecord] = []
    for segment in merged:
        if (
            deduplicated
            and segment.start < deduplicated[-1].end
            and _text_similarity(segment.text, deduplicated[-1].text) >= 0.96
        ):
            previous = deduplicated[-1]
            if segment.avg_logprob > previous.avg_logprob:
                deduplicated[-1] = segment
            continue
        deduplicated.append(segment)
    return deduplicated


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
        "v1.3은 긴 공백뿐 아니라 비정상적으로 긴 세그먼트, 프롬프트 반복, 짧은 파편 연속을 검사합니다.",
        "자동 재전사 결과가 품질 기준을 통과한 경우에만 최종 전사문에 반영합니다.",
        "",
        "## 자동 재전사 결과",
        "",
    ]
    if result.retry_attempts:
        for index, attempt in enumerate(result.retry_attempts, start=1):
            status = "채택" if attempt.accepted else "원문 유지"
            lines.append(
                f"{index}. [{format_timestamp(attempt.start)} ~ {format_timestamp(attempt.end)}] "
                f"**{status}** - {'; '.join(attempt.reasons)}"
            )
            lines.append(f"   - {attempt.note}")
    else:
        lines.append("자동 재전사가 필요한 이상 구간이 없었습니다.")

    lines.extend(["", "## 1차 전사에서 감지한 품질 문제", ""])
    if result.quality_issues:
        for index, issue in enumerate(result.quality_issues, start=1):
            lines.append(
                f"{index}. [{format_timestamp(issue.start)} ~ {format_timestamp(issue.end)}] "
                f"{issue.reason}"
            )
            if issue.text:
                lines.append(f"   - 전사: {issue.text[:160]}")
    else:
        lines.append("별도 품질 문제가 감지되지 않았습니다.")

    lines.extend(["", "## 최종 전사에 남은 긴 공백", ""])
    if not result.gaps:
        lines.append("지정한 기준 이상의 긴 공백이 없습니다.")
    else:
        for index, gap in enumerate(result.gaps, start=1):
            lines.append(
                f"{index}. [{format_timestamp(gap.start)} ~ {format_timestamp(gap.end)}] "
                f"({gap.duration:.1f}초)"
            )
            if gap.previous_text:
                lines.append(f"   - 직전: {gap.previous_text[-100:]}")
            if gap.next_text:
                lines.append(f"   - 직후: {gap.next_text[:100]}")

    lines.extend(["", "## Gemini 3.7 Flash 선택 검토 후보", ""])
    lines.append("이 절의 후보는 최종 전사문에 자동 반영되지 않습니다. 반드시 원음을 듣고 확인하세요.")
    if not result.gemini_reviews:
        lines.append("Gemini 외부 검토를 사용하지 않았거나 검토할 긴 공백이 없었습니다.")
    else:
        lines.append("사용자가 선택한 공백 오디오 클립과 앞뒤 문맥만 Google Vertex AI로 전송했습니다.")
        status_labels = {
            "success": "후보 생성",
            "blocked": "정책 차단",
            "error": "오류",
            "skipped": "전송 안 함",
        }
        for index, attempt in enumerate(result.gemini_reviews, start=1):
            status = status_labels.get(attempt.status, attempt.status)
            lines.append(
                f"{index}. [{format_timestamp(attempt.gap_start)} ~ {format_timestamp(attempt.gap_end)}] "
                f"**{status}** ({attempt.model})"
            )
            lines.append(
                f"   - 전송 구간: {format_timestamp(attempt.clip_start)} ~ "
                f"{format_timestamp(attempt.clip_end)}"
            )
            if attempt.note:
                lines.append(f"   - 메모: {attempt.note}")
            if attempt.input_token_count is not None:
                lines.append(
                    f"   - 토큰: 입력 {attempt.input_token_count:,}, "
                    f"전체 {attempt.total_token_count or 0:,}"
                )
            if not attempt.candidates and attempt.status == "success":
                lines.append("   - 들리는 발화 후보 없음")
            for candidate in attempt.candidates:
                speaker = f" {candidate.speaker}:" if candidate.speaker else ""
                uncertain = " [불확실]" if candidate.uncertain else ""
                lines.append(
                    f"   - [{format_timestamp(candidate.start)} ~ "
                    f"{format_timestamp(candidate.end)}]{speaker} {candidate.text}{uncertain}"
                )
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
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
        auto_retry: bool = True,
        retry_gaps: bool = True,
        normalize_retry_audio: bool = True,
        require_retry_audio_evidence: bool = True,
        gap_seconds: float = 15.0,
        include_timestamps: bool = True,
        make_srt: bool = True,
        output_dir: Optional[Path] = None,
        vertex_review: Optional[VertexReviewConfig] = None,
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

        # v1.1의 prompt 인자는 하위 호환을 위해 기술용어 목록으로만 취급한다.
        normalized_hotwords = normalize_hotwords(hotwords if hotwords is not None else prompt)
        prompt_texts = [text for text in (initial_prompt, normalized_hotwords) if text]

        segments_iter, info = self.model.transcribe(
            str(source),
            language="ko",
            initial_prompt=initial_prompt,
            hotwords=normalized_hotwords,
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
                compression_ratio=float(getattr(raw_segment, "compression_ratio", 0.0)),
            )
            segments.append(segment)
            percent = min(100, int(segment.end / max(info.duration, 0.001) * 100))
            if progress_callback and percent >= last_reported + 5:
                last_reported = percent
                progress_callback(
                    percent / 125.0,
                    f"1차 전사 {format_timestamp(segment.end)} / {format_timestamp(info.duration)}",
                )

        initial_gaps = find_review_gaps(segments, float(info.duration), gap_seconds)
        quality_issues = find_quality_issues(segments, prompt_texts=prompt_texts)
        retry_attempts: list[RetryAttempt] = []
        original_retry_audio = None
        if auto_retry:
            retry_attempts = build_retry_attempts(
                quality_issues,
                initial_gaps,
                duration=float(info.duration),
                retry_gaps=retry_gaps,
            )

        if retry_attempts:
            if progress_callback:
                progress_callback(
                    0.82,
                    f"이상 구간 {len(retry_attempts)}개 자동 재전사 준비 중",
                )
            retry_audio = str(source)
            if normalize_retry_audio or require_retry_audio_evidence:
                from faster_whisper.audio import decode_audio

                original_retry_audio = decode_audio(str(source))
                retry_audio = (
                    normalize_audio_for_retry(original_retry_audio)
                    if normalize_retry_audio
                    else original_retry_audio
                )
            clip_timestamps = [
                timestamp
                for attempt in retry_attempts
                for timestamp in (attempt.start, attempt.end)
            ]
            retry_iter, _retry_info = self.model.transcribe(
                retry_audio,
                language="ko",
                initial_prompt=None,
                hotwords=normalized_hotwords,
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.65,
                log_prob_threshold=-1.0,
                hallucination_silence_threshold=1.5,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                clip_timestamps=clip_timestamps,
            )
            retry_segments = _segments_from_iterator(retry_iter)
            if original_retry_audio is not None:
                annotate_audio_energy(retry_segments, original_retry_audio)
            segments = merge_retry_segments(
                segments,
                retry_segments,
                retry_attempts,
                prompt_texts=prompt_texts,
                require_audio_evidence=require_retry_audio_evidence,
            )
            if progress_callback:
                accepted = sum(attempt.accepted for attempt in retry_attempts)
                progress_callback(
                    0.95,
                    f"자동 재전사 완료: {accepted}/{len(retry_attempts)}개 구간 채택",
                )

        result = TranscriptionResult(
            source=source,
            duration=float(info.duration),
            language=str(info.language),
            language_probability=float(info.language_probability),
            profile_key=profile_key,
            segments=segments,
            quality_issues=quality_issues,
            retry_attempts=retry_attempts,
        )
        result.gaps = find_review_gaps(segments, result.duration, gap_seconds)
        if vertex_review is not None and result.gaps:
            try:
                if original_retry_audio is None:
                    from faster_whisper.audio import decode_audio

                    original_retry_audio = decode_audio(str(source))
                result.gemini_reviews = review_gaps_with_gemini(
                    result,
                    vertex_review,
                    original_retry_audio,
                    hotwords=normalized_hotwords,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                result.gemini_reviews = [
                    GeminiReviewAttempt(
                        gap_start=gap.start,
                        gap_end=gap.end,
                        clip_start=gap.start,
                        clip_end=gap.end,
                        status="error",
                        model=vertex_review.model,
                        note=f"Gemini 검토 준비 오류: {exc}",
                    )
                    for gap in result.gaps
                ]
        write_outputs(
            result,
            include_timestamps=include_timestamps,
            make_srt=make_srt,
            output_dir=output_dir,
        )
        if progress_callback:
            progress_callback(1.0, "전사 및 결과 저장 완료")
        return result
