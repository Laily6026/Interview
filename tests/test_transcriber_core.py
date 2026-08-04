from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from transcriber_core import (
    PROFILES,
    SegmentRecord,
    TranscriptionResult,
    TranscriberEngine,
    find_review_gaps,
    format_timestamp,
    write_outputs,
)


class ProfileTests(TestCase):
    def test_high_recall_uses_sensitive_vad_and_wide_padding(self):
        options = PROFILES["high_recall"].options
        self.assertTrue(options["vad_filter"])
        self.assertLessEqual(options["vad_parameters"]["threshold"], 0.15)
        self.assertGreaterEqual(options["vad_parameters"]["speech_pad_ms"], 1000)


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
            self.assertIn("실제 무음일 수도", result.review_path.read_text(encoding="utf-8-sig"))


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
                output_dir=output_dir,
            )

            self.assertEqual(fake_model.kwargs["language"], "ko")
            self.assertEqual(fake_model.kwargs["hotwords"], "전문용어")
            self.assertTrue(fake_model.kwargs["vad_filter"])
            self.assertEqual(fake_model.kwargs["vad_parameters"]["threshold"], 0.15)
            self.assertEqual(result.txt_path.parent, output_dir)
            self.assertIn("테스트 전사", result.txt_path.read_text(encoding="utf-8-sig"))
