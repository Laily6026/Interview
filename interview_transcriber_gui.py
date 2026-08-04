#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""발명자 인터뷰 전사기 GUI."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Iterable

from transcriber_core import PROFILES, TranscriberEngine, format_timestamp


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac",
    ".wma", ".aac", ".webm", ".opus", ".mkv", ".avi", ".mov",
}
DEFAULT_PROMPT = (
    "발명자 인터뷰 녹취록입니다. 특허, 출원, 청구항, 명세서, 실시예, "
    "선행기술, 종래기술, 발명의 효과, 구성요소 등의 용어가 사용됩니다."
)
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
PROFILE_LABEL_TO_KEY = {profile.label: key for key, profile in PROFILES.items()}


class TranscriberApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("발명자 인터뷰 전사기 — 로컬 처리")
        self.root.geometry("820x720")
        self.root.minsize(720, 620)

        self.files: list[Path] = []
        self.engine = TranscriberEngine()
        self.running = False
        self.event_queue: queue.Queue[tuple] = queue.Queue()
        self.last_output_dir: Path | None = None
        self.last_txt_path: Path | None = None
        self.last_review_path: Path | None = None

        self._build_ui()
        self._poll_events()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 5}

        file_frame = ttk.LabelFrame(self.root, text="1. 음성·영상 파일 선택")
        file_frame.pack(fill="x", **pad)
        button_row = ttk.Frame(file_frame)
        button_row.pack(fill="x", padx=8, pady=6)
        ttk.Button(button_row, text="파일 선택", command=self.select_files).pack(side="left")
        ttk.Button(button_row, text="폴더 선택 (일괄)", command=self.select_folder).pack(side="left", padx=6)
        ttk.Button(button_row, text="목록 비우기", command=self.clear_files).pack(side="left")
        self.file_listbox = tk.Listbox(file_frame, height=4)
        self.file_listbox.pack(fill="x", padx=8, pady=(0, 8))

        option_frame = ttk.LabelFrame(self.root, text="2. 전사 옵션")
        option_frame.pack(fill="x", **pad)

        first_row = ttk.Frame(option_frame)
        first_row.pack(fill="x", padx=8, pady=4)
        ttk.Label(first_row, text="모델:").pack(side="left")
        self.model_var = tk.StringVar(value="large-v3")
        ttk.Combobox(
            first_row,
            textvariable=self.model_var,
            values=MODEL_CHOICES,
            state="readonly",
            width=11,
        ).pack(side="left", padx=(4, 18))

        ttk.Label(first_row, text="모드:").pack(side="left")
        self.profile_var = tk.StringVar(value=PROFILES["high_recall"].label)
        profile_combo = ttk.Combobox(
            first_row,
            textvariable=self.profile_var,
            values=[profile.label for profile in PROFILES.values()],
            state="readonly",
            width=18,
        )
        profile_combo.pack(side="left", padx=(4, 8))
        profile_combo.bind("<<ComboboxSelected>>", self._update_profile_description)

        self.profile_description_var = tk.StringVar()
        self._update_profile_description()
        ttk.Label(
            option_frame,
            textvariable=self.profile_description_var,
            foreground="#555555",
            wraplength=760,
        ).pack(fill="x", padx=8, pady=(0, 4))

        second_row = ttk.Frame(option_frame)
        second_row.pack(fill="x", padx=8, pady=4)
        self.timestamps_var = tk.BooleanVar(value=True)
        self.srt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(second_row, text="TXT에 타임스탬프 포함", variable=self.timestamps_var).pack(side="left")
        ttk.Checkbutton(second_row, text="자막(.srt) 생성", variable=self.srt_var).pack(side="left", padx=12)
        ttk.Label(second_row, text="검토 공백 기준(초):").pack(side="left", padx=(12, 4))
        self.gap_seconds_var = tk.StringVar(value="15")
        ttk.Entry(second_row, textvariable=self.gap_seconds_var, width=6).pack(side="left")

        prompt_row = ttk.Frame(option_frame)
        prompt_row.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(
            prompt_row,
            text="기술용어 힌트 (사건별 고유명사·전문용어를 쉼표로 추가):",
        ).pack(anchor="w")
        self.prompt_entry = tk.Text(prompt_row, height=3, wrap="word")
        self.prompt_entry.insert("1.0", DEFAULT_PROMPT)
        self.prompt_entry.pack(fill="x", pady=(2, 0))

        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="▶ 전사 시작", command=self.start_transcription)
        self.run_button.pack(side="left")
        self.progress = ttk.Progressbar(run_frame, mode="determinate", maximum=100, length=240)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(run_frame, textvariable=self.status_var).pack(side="left")

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        ttk.Button(bottom_frame, text="결과 폴더 열기", command=self.open_output_folder).pack(side="left")
        ttk.Button(bottom_frame, text="전사문 열기", command=self.open_txt).pack(side="left", padx=6)
        ttk.Button(bottom_frame, text="검토 구간 열기", command=self.open_review).pack(side="left")
        ttk.Label(bottom_frame, text="모든 처리는 PC에서 로컬로 수행됩니다.").pack(side="right")

        log_frame = ttk.LabelFrame(self.root, text="3. 진행 상황")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_box = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Malgun Gothic", 10),
        )
        self.log_box.pack(fill="both", expand=True, padx=8, pady=8)

    def _update_profile_description(self, _event=None) -> None:
        key = PROFILE_LABEL_TO_KEY.get(self.profile_var.get(), "high_recall")
        self.profile_description_var.set(PROFILES[key].description)

    def select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="음성·영상 파일 선택",
            filetypes=[
                ("오디오·비디오 파일", " ".join(f"*{ext}" for ext in sorted(AUDIO_EXTENSIONS))),
                ("모든 파일", "*.*"),
            ],
        )
        self._add_files(Path(path) for path in paths)

    def select_folder(self) -> None:
        folder = filedialog.askdirectory(title="녹음 폴더 선택")
        if not folder:
            return
        found = sorted(
            path for path in Path(folder).iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not found:
            messagebox.showwarning("알림", "폴더에 지원하는 오디오·비디오 파일이 없습니다.")
            return
        self._add_files(found)

    def _add_files(self, paths: Iterable[Path]) -> None:
        for path in paths:
            if path not in self.files:
                self.files.append(path)
                self.file_listbox.insert("end", path.name)

    def clear_files(self) -> None:
        if self.running:
            return
        self.files.clear()
        self.file_listbox.delete(0, "end")

    def _queue_log(self, message: str) -> None:
        self.event_queue.put(("log", message))

    def _queue_progress(self, value: float, message: str) -> None:
        self.event_queue.put(("progress", value, message))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                if event[0] == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", event[1] + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif event[0] == "progress":
                    self.progress["value"] = event[1] * 100
                    self.status_var.set(event[2])
        except queue.Empty:
            pass
        self.root.after(150, self._poll_events)

    def start_transcription(self) -> None:
        if self.running:
            return
        if not self.files:
            messagebox.showwarning("알림", "먼저 음성 또는 영상 파일을 선택해 주세요.")
            return
        try:
            gap_seconds = float(self.gap_seconds_var.get())
            if gap_seconds <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("알림", "검토 공백 기준은 0보다 큰 숫자로 입력해 주세요.")
            return

        self.running = True
        self.run_button.configure(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("작업 준비 중")
        worker_args = (
            gap_seconds,
            self.model_var.get(),
            PROFILE_LABEL_TO_KEY.get(self.profile_var.get(), "high_recall"),
            self.prompt_entry.get("1.0", "end").strip() or None,
            self.timestamps_var.get(),
            self.srt_var.get(),
            list(self.files),
        )
        threading.Thread(target=self._worker, args=worker_args, daemon=True).start()

    def _worker(
        self,
        gap_seconds: float,
        model_name: str,
        profile_key: str,
        prompt: str | None,
        include_timestamps: bool,
        make_srt: bool,
        files: list[Path],
    ) -> None:
        try:
            started = time.time()

            self._queue_log(f"모델: {model_name}")
            self._queue_log(f"전사 모드: {PROFILES[profile_key].label}")
            for file_index, audio_path in enumerate(files, start=1):
                self._queue_log(f"\n[{file_index}/{len(files)}] 전사 시작: {audio_path.name}")
                result = self.engine.transcribe(
                    audio_path,
                    model_name=model_name,
                    profile_key=profile_key,
                    prompt=prompt,
                    gap_seconds=gap_seconds,
                    include_timestamps=include_timestamps,
                    make_srt=make_srt,
                    progress_callback=self._queue_progress,
                )
                self.last_output_dir = audio_path.parent
                self.last_txt_path = result.txt_path
                self.last_review_path = result.review_path
                self._queue_log(
                    f"완료: {len(result.segments)}개 세그먼트, "
                    f"검토 후보 {len(result.gaps)}개"
                )
                self._queue_log(f"전사문: {result.txt_path.name}")
                self._queue_log(f"검토 구간: {result.review_path.name}")

            self._queue_log(f"\n모든 작업 완료 ({time.time() - started:.0f}초)")
            self._queue_progress(1.0, "완료")
        except ImportError:
            self._queue_log("오류: faster-whisper가 설치되어 있지 않습니다.")
            self._queue_log("명령 프롬프트에서: pip install -r requirements.txt")
        except Exception as exc:
            self._queue_log(f"오류 발생: {exc}")
            self._queue_progress(0.0, "오류")
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_button.configure(state="normal"))

    def open_output_folder(self) -> None:
        target = self.last_output_dir or (self.files[0].parent if self.files else None)
        if target is None:
            messagebox.showinfo("알림", "아직 결과가 없습니다.")
            return
        self._open_path(target)

    def open_txt(self) -> None:
        self._open_result(self.last_txt_path, "아직 생성된 전사문이 없습니다.")

    def open_review(self) -> None:
        self._open_result(self.last_review_path, "아직 생성된 검토 구간 파일이 없습니다.")

    def _open_result(self, path: Path | None, missing_message: str) -> None:
        if path is None or not path.exists():
            messagebox.showinfo("알림", missing_message)
            return
        self._open_path(path)

    @staticmethod
    def _open_path(path: Path) -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
