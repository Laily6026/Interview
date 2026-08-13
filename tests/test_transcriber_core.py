from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from transcriber_core import (
    PROFILES,
    QualityIssue,
    RetryAttempt,
    SegmentRecord,
    TranscriptionResult,
    TranscriberEngine,
    build_retry_attempts,
    find_quality_issues,
    find_review_gaps,
    format_timestamp,
    merge_retry_segments,
    normalize_hotwords,
    write_outputs,
)


class ProfileTests(TestCase):
    def test_high_recall_uses_sensitive_vad_and_wide_padding(self):
        options = PROFILES["high_recall"].options
        self.assertTrue(options["vad_filter"])
        self.assertLessEqual(options["vad_parameters"]["threshold"], 0.15)
        self.assertGreaterEqual(options["vad_parameters"]["speech_pad_ms"], 1000)
        self.assertFalse(options["condition_on_previous_text"])

    def test_all_profiles_reset_previous_text_by_default(self):
        self.assertTrue(
            all(not profile.options["condition_on_previous_text"] for profile in PROFILES.values())
        )


class PromptTests(TestCase):
    def test_normalizes_and_deduplicates_hotwords(self):
        self.assertEqual(normalize_hotwords("LOT, AMR\nLOT; HMI"), "LOT, AMR, HMI")


class QualityTests(TestCase):
    def test_finds_long_prompt_echo_and_short_fragment_streak(self):
        prompt = "특허, 출원, 청구항, 명세서 등의 용어가 사용됩니다."
        segments = [
            SegmentRecord(0.0, 45.0, prompt),
            SegmentRecord(50.0, 51.0, "아"),
            SegmentRecord(52.0, 53.0, "어"),
            SegmentRecord(54.0, 55.0, "이"),
            SegmentRecord(56.0, 57.0, "그"),
            SegmentRecord(58.0, 59.0, "네"),
        ]
        kinds = {issue.kind for issue in find_quality_issues(segments, prompt_texts=[prompt])}
        self.assertIn("long_segment", kinds)
        self.assertIn("prompt_echo", kinds)
        self.assertIn("short_fragment_streak", kinds)

    def test_build_retry_attempts_merges_nearby_issue_and_gap(self):
        issues = [QualityIssue(10.0, 40.0, "long_segment", "긴 구간")]
        gaps = [SimpleNamespace(start=40.5, end=55.0, duration=14.5)]
        attempts = build_retry_attempts(issues, gaps, duration=100.0)
        self.assertEqual(len(attempts), 1)
        self.assertTrue(attempts[0].replace_existing)
        self.assertEqual(attempts[0].start, 7.0)
        self.assertEqual(attempts[0].end, 58.0)

    def test_merge_retry_replaces_long_segment_with_better_chunks(self):
        original = [SegmentRecord(10.0, 80.0, "짧게 뭉친 문장", avg_logprob=-0.5)]
        candidates = [
            SegmentRecord(10.0, 20.0, "첫 번째 복구 문장입니다", avg_logprob=-0.2, no_speech_prob=0.1),
            SegmentRecord(20.0, 30.0, "두 번째 복구 문장입니다", avg_logprob=-0.2, no_speech_prob=0.1),
        ]
        attempts = [RetryAttempt(7.0, 83.0, ["긴 구간"], replace_existing=True)]
        merged = merge_retry_segments(original, candidates, attempts)
        self.assertEqual([item.text for item in merged], [item.text for item in candidates])
        self.assertTrue(attempts[0].accepted)

    def test_merge_retry_rejects_low_energy_and_generic_hallucinations(self):
        original = [SegmentRecord(10.0, 80.0, "기존 문장", avg_logprob=-0.5)]
        candidates = [
            SegmentRecord(
                12.0,
                15.0,
                "한국국토정보공사",
                avg_logprob=-0.1,
                no_speech_prob=0.1,
                audio_rms_db=-35.0,
                voiced_fraction=0.4,
            ),
            SegmentRecord(
                20.0,
                24.0,
                "조용한 구간의 환각",
                avg_logprob=-0.1,
                no_speech_prob=0.1,
                audio_rms_db=-42.0,
                voiced_fraction=0.2,
            ),
        ]
        attempts = [RetryAttempt(7.0, 83.0, ["긴 구간"], replace_existing=True)]
        merged = merge_retry_segments(
            original,
            candidates,
            attempts,
            require_audio_evidence=True,
        )
        self.assertEqual([item.text for item in merged], ["기존 문장"])
        self.assertFalse(attempts[0].accepted)


class TimestampTests(TestCase):
    def test_plain_timestamp(self):
        self.assertEqual(format_timestamp(3661.9), "01:01:01")

    def test_srt_timestamp_rounds_carry(self):
        self.assertEqual(format_timestamp(59.9996, srt=True), "00:01:00,000")


class GapTests(TestCase):
    def test_finds_internal_and_trailing_gaps(self):
        segments = [
            SegmentRecord(0.0, 3.0, "첫 문장"),
            SegmentRecord(20.0, 25.0, "둘째 문장"),
        ]
        gaps = find_review_gaps(segments, duration=50.0, minimum_gap_seconds=15.0)
        self.assertEqual([(gap.start, gap.end) for gap in gaps], [(3.0, 20.0), (25.0, 50.0)])
        self.assertEqual(gaps[0].previous_text, "첫 문장")
        self.assertEqual(gaps[0].next_text, "둘째 문장")


class OutputTests(TestCase):
    def test_writes_korean_bom_and_sequential_srt(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "회의.mp4"
            source.touch()
            result = TranscriptionResult(
                source=source,
                duration=30.0,
                language="ko",
                language_probability=1.0,
                profile_key="high_recall",
                segments=[
                    SegmentRecord(1.0, 2.0, "안녕하세요"),
                    SegmentRecord(5.0, 6.0, "두 번째 문장"),
                ],
            )
            result.gaps = find_review_gaps(result.segments, result.duration, 10.0)
            write_outputs(result)

            self.assertTrue(result.txt_path.read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertIn("안녕하세요", result.txt_path.read_text(encoding="utf-8-sig"))
            srt = result.srt_path.read_text(encoding="utf-8-sig")
            self.assertIn("1\n00:00:01,000 --> 00:00:02,000", srt)
            self.assertIn("2\n00:00:05,000 --> 00:00:06,000", srt)
            review = result.review_path.read_text(encoding="utf-8-sig")
            self.assertIn("최종 전사에 남은 긴 공백", review)
            self.assertIn("24.0초", review)


class EngineTests(TestCase):
    def test_engine_passes_high_recall_options_and_custom_output_dir(self):
        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def transcribe(self, source, **kwargs):
                self.kwargs = kwargs
                segment = SimpleNamespace(
                    start=1.0,
                    end=2.0,
                    text=" 테스트 전사 ",
                    avg_logprob=-0.2,
                    no_speech_prob=0.1,
                )
                info = SimpleNamespace(duration=10.0, language="ko", language_probability=0.99)
                return iter([segment]), info

        with TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "회의.mp4"
            source.touch()
            output_dir = temp / "outputs"
            fake_model = FakeModel()
            engine = TranscriberEngine()
            engine.model = fake_model
            engine.loaded_model_name = "large-v3"

            result = engine.transcribe(
                source,
                model_name="large-v3",
                profile_key="high_recall",
                prompt="전문용어",
                initial_prompt="발명자 인터뷰 배경",
                output_dir=output_dir,
            )

            self.assertEqual(fake_model.kwargs["language"], "ko")
            self.assertEqual(fake_model.kwargs["hotwords"], "전문용어")
            self.assertEqual(fake_model.kwargs["initial_prompt"], "발명자 인터뷰 배경")
            self.assertTrue(fake_model.kwargs["vad_filter"])
            self.assertEqual(fake_model.kwargs["vad_parameters"]["threshold"], 0.15)
            self.assertEqual(result.txt_path.parent, output_dir)
            self.assertIn("테스트 전사", result.txt_path.read_text(encoding="utf-8-sig"))

    def test_engine_retries_long_segment_without_prompt_or_vad(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def transcribe(self, source, **kwargs):
                self.calls.append(kwargs)
                info = SimpleNamespace(duration=100.0, language="ko", language_probability=0.99)
                if len(self.calls) == 1:
                    return iter(
                        [
                            SimpleNamespace(
                                start=10.0,
                                end=80.0,
                                text=" 너무 길게 뭉친 문장 ",
                                avg_logprob=-0.4,
                                no_speech_prob=0.1,
                                compression_ratio=1.0,
                            )
                        ]
                    ), info
                return iter(
                    [
                        SimpleNamespace(
                            start=10.0,
                            end=20.0,
                            text=" 첫 번째 복구 문장 ",
                            avg_logprob=-0.2,
                            no_speech_prob=0.1,
                            compression_ratio=1.0,
                        ),
                        SimpleNamespace(
                            start=20.0,
                            end=30.0,
                            text=" 두 번째 복구 문장 ",
                            avg_logprob=-0.2,
                            no_speech_prob=0.1,
                            compression_ratio=1.0,
                        ),
                    ]
                ), info

        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "회의.wav"
            source.touch()
            engine = TranscriberEngine()
            engine.model = FakeModel()
            engine.loaded_model_name = "large-v3"
            result = engine.transcribe(
                source,
                hotwords="LOT, AMR",
                initial_prompt="발명 배경",
                retry_gaps=False,
                normalize_retry_audio=False,
                require_retry_audio_evidence=False,
                output_dir=Path(temp_dir) / "outputs",
            )

            self.assertEqual(len(engine.model.calls), 2)
            retry_kwargs = engine.model.calls[1]
            self.assertIsNone(retry_kwargs["initial_prompt"])
            self.assertFalse(retry_kwargs["vad_filter"])
            self.assertFalse(retry_kwargs["condition_on_previous_text"])
            self.assertEqual(retry_kwargs["hotwords"], "LOT, AMR")
            self.assertEqual([item.text for item in result.segments], ["첫 번째 복구 문장", "두 번째 복구 문장"])
            self.assertTrue(result.retry_attempts[0].accepted)
