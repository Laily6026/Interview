#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""발명자 인터뷰 전사기 CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber_core import APP_VERSION, PROFILES, TranscriberEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="faster-whisper 기반 한국어 인터뷰 전사기 v1.2")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    parser.add_argument("files", nargs="+", type=Path, help="전사할 오디오 또는 동영상 파일")
    parser.add_argument("--model", default="large-v3", help="Whisper 모델 이름")
    parser.add_argument(
        "--mode",
        choices=tuple(PROFILES),
        default="high_recall",
        help="high_recall(누락 최소화), balanced(균형), fast(빠르게)",
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="쉼표로 구분한 기술용어 및 고유명사",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="전사 배경 설명. 필요한 경우에만 사용",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="이전 버전 호환 옵션. --hotwords와 같은 의미로 처리",
    )
    parser.add_argument("--no-auto-retry", action="store_true", help="이상 구간 자동 재전사 끄기")
    parser.add_argument("--no-retry-gaps", action="store_true", help="긴 공백 자동 재전사 끄기")
    parser.add_argument("--no-normalize-retry", action="store_true", help="재전사 저음량 보정 끄기")
    parser.add_argument("--gap-seconds", type=float, default=15.0, help="검토 목록에 넣을 공백 길이")
    parser.add_argument("--output-dir", type=Path, default=None, help="결과를 저장할 별도 폴더")
    parser.add_argument("--no-timestamps", action="store_true", help="TXT 타임스탬프 생략")
    parser.add_argument("--no-srt", action="store_true", help="SRT 자막을 만들지 않음")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = TranscriberEngine()

    def show_progress(progress: float, message: str) -> None:
        print(f"[{progress * 100:5.1f}%] {message}", flush=True)

    failed = False
    for source in args.files:
        try:
            print(f"\n전사 시작: {source}")
            result = engine.transcribe(
                source,
                model_name=args.model,
                profile_key=args.mode,
                prompt=args.prompt,
                initial_prompt=args.initial_prompt,
                hotwords=args.hotwords,
                auto_retry=not args.no_auto_retry,
                retry_gaps=not args.no_retry_gaps,
                normalize_retry_audio=not args.no_normalize_retry,
                gap_seconds=args.gap_seconds,
                include_timestamps=not args.no_timestamps,
                make_srt=not args.no_srt,
                output_dir=args.output_dir,
                progress_callback=show_progress,
            )
            accepted_retries = sum(attempt.accepted for attempt in result.retry_attempts)
            print(
                f"완료: {len(result.segments)}개 세그먼트, "
                f"자동 재전사 {accepted_retries}개 구간 채택"
            )
            print(f"전사문: {result.txt_path}")
            print(f"검토 구간: {result.review_path}")
        except Exception as exc:
            failed = True
            print(f"오류: {source}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
