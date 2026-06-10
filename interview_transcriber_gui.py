#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
발명자 인터뷰 전사 프로그램 (GUI 버전)
========================================

faster-whisper 기반 로컬 음성 인식 GUI 도구.
음성 파일을 외부 서버에 전송하지 않아 발명 내용 보안에 안전합니다.

실행:  python interview_transcriber_gui.py
필요:  pip install faster-whisper
       ffmpeg 설치 (https://ffmpeg.org)

.exe 변환:
    pip install pyinstaller
    pyinstaller --onefile --noconsole --name "인터뷰전사기" interview_transcriber_gui.py
    → dist 폴더에 인터뷰전사기.exe 생성
"""

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac",
                    ".wma", ".aac", ".webm", ".opus", ".mkv", ".avi", ".mov"}

DEFAULT_PROMPT = (
    "발명자 인터뷰 녹취록입니다. 특허, 출원, 청구항, 명세서, 실시예, "
    "선행기술, 종래기술, 발명의 효과, 구성요소 등의 용어가 사용됩니다."
)

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


def format_timestamp(seconds: float, srt: bool = False) -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if srt:
        ms = int((seconds - total) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_summary_template(filename: str, duration: float, sentences: list) -> str:
    full_text = " ".join(sentences)
    return f"""# 발명자 인터뷰 정리 초안

- **녹음 파일**: {filename}
- **녹음 길이**: {format_timestamp(duration)}
- **인터뷰 일시**: (기입)
- **발명자**: (기입)
- **사건번호 / 관리번호**: (기입)

---

## 1. 발명의 배경 / 해결하고자 하는 과제
(전사문에서 해당 내용 발췌·정리)

## 2. 발명의 핵심 구성
(전사문에서 해당 내용 발췌·정리)

## 3. 종래기술 대비 차별점 / 효과
(전사문에서 해당 내용 발췌·정리)

## 4. 실시예 / 변형예
(전사문에서 해당 내용 발췌·정리)

## 5. 추가 확인 필요 사항 (발명자 회신 요청)
- [ ]

---

## 전체 전사문 (원문)

{full_text}
"""


class TranscriberApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("발명자 인터뷰 전사기 (로컬 처리 · 보안 안전)")
        root.geometry("780x640")
        root.minsize(680, 560)

        self.files: list[Path] = []
        self.model = None
        self.loaded_model_name = None
        self.running = False
        self.log_queue: queue.Queue = queue.Queue()
        self.last_output_dir: Path | None = None
        self.last_txt_path: Path | None = None

        self._build_ui()
        self._poll_log_queue()

    # ---------------- UI 구성 ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # 파일 선택 영역
        frame_file = ttk.LabelFrame(self.root, text="1. 음성 파일 선택")
        frame_file.pack(fill="x", **pad)

        btn_row = ttk.Frame(frame_file)
        btn_row.pack(fill="x", padx=8, pady=6)
        ttk.Button(btn_row, text="파일 선택", command=self.select_files).pack(side="left")
        ttk.Button(btn_row, text="폴더 선택 (일괄)", command=self.select_folder).pack(side="left", padx=6)
        ttk.Button(btn_row, text="목록 비우기", command=self.clear_files).pack(side="left")

        self.file_listbox = tk.Listbox(frame_file, height=4)
        self.file_listbox.pack(fill="x", padx=8, pady=(0, 8))

        # 옵션 영역
        frame_opt = ttk.LabelFrame(self.root, text="2. 옵션")
        frame_opt.pack(fill="x", **pad)

        row1 = ttk.Frame(frame_opt)
        row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="모델:").pack(side="left")
        self.model_var = tk.StringVar(value="medium")
        ttk.Combobox(row1, textvariable=self.model_var, values=MODEL_CHOICES,
                     state="readonly", width=10).pack(side="left", padx=(4, 16))
        ttk.Label(row1, text="(medium=균형, large-v3=최고 정확도·느림)").pack(side="left")

        self.timestamps_var = tk.BooleanVar(value=True)
        self.srt_var = tk.BooleanVar(value=True)
        row2 = ttk.Frame(frame_opt)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Checkbutton(row2, text="타임스탬프 포함", variable=self.timestamps_var).pack(side="left")
        ttk.Checkbutton(row2, text="자막(.srt) 생성", variable=self.srt_var).pack(side="left", padx=12)

        row3 = ttk.Frame(frame_opt)
        row3.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(row3, text="기술용어 힌트 (인식률 향상, 사건별 핵심 용어 추가 권장):").pack(anchor="w")
        self.prompt_entry = tk.Text(row3, height=2, wrap="word")
        self.prompt_entry.insert("1.0", DEFAULT_PROMPT)
        self.prompt_entry.pack(fill="x", pady=(2, 0))

        # 실행 영역
        frame_run = ttk.Frame(self.root)
        frame_run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(frame_run, text="▶ 전사 시작", command=self.start_transcription)
        self.run_btn.pack(side="left")
        self.progress = ttk.Progressbar(frame_run, mode="indeterminate", length=200)
        self.progress.pack(side="left", padx=12)
        self.status_var = tk.StringVar(value="대기 중")
        ttk.Label(frame_run, textvariable=self.status_var).pack(side="left")

        # 결과/로그 영역
        frame_log = ttk.LabelFrame(self.root, text="3. 진행 상황 및 결과 미리보기")
        frame_log.pack(fill="both", expand=True, **pad)
        self.log_box = scrolledtext.ScrolledText(frame_log, wrap="word", state="disabled",
                                                 font=("Malgun Gothic", 10))
        self.log_box.pack(fill="both", expand=True, padx=8, pady=8)

        # 하단 버튼
        frame_bottom = ttk.Frame(self.root)
        frame_bottom.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(frame_bottom, text="결과 폴더 열기", command=self.open_output_folder).pack(side="left")
        ttk.Button(frame_bottom, text="전사문(.txt) 열기", command=self.open_txt).pack(side="left", padx=6)

    # ---------------- 파일 선택 ----------------
    def select_files(self):
        paths = filedialog.askopenfilenames(
            title="음성 파일 선택",
            filetypes=[("오디오/비디오 파일",
                        " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))),
                       ("모든 파일", "*.*")])
        for p in paths:
            path = Path(p)
            if path not in self.files:
                self.files.append(path)
                self.file_listbox.insert("end", path.name)

    def select_folder(self):
        folder = filedialog.askdirectory(title="녹음 폴더 선택")
        if not folder:
            return
        found = sorted(p for p in Path(folder).iterdir()
                       if p.suffix.lower() in AUDIO_EXTENSIONS)
        if not found:
            messagebox.showwarning("알림", "폴더에 오디오 파일이 없습니다.")
            return
        for path in found:
            if path not in self.files:
                self.files.append(path)
                self.file_listbox.insert("end", path.name)

    def clear_files(self):
        self.files.clear()
        self.file_listbox.delete(0, "end")

    # ---------------- 로그 ----------------
    def log(self, msg: str):
        self.log_queue.put(msg)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(150, self._poll_log_queue)

    # ---------------- 전사 실행 ----------------
    def start_transcription(self):
        if self.running:
            return
        if not self.files:
            messagebox.showwarning("알림", "먼저 음성 파일을 선택해 주세요.")
            return
        self.running = True
        self.run_btn.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("작업 중...")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                self.log("✘ faster-whisper가 설치되어 있지 않습니다.")
                self.log("   명령 프롬프트에서:  pip install faster-whisper")
                return

            model_name = self.model_var.get()
            if self.model is None or self.loaded_model_name != model_name:
                self.log(f"모델 로딩 중: {model_name} (최초 1회는 다운로드로 수 분 걸릴 수 있습니다)")
                self.model = WhisperModel(model_name, device="auto", compute_type="auto")
                self.loaded_model_name = model_name
                self.log("모델 로딩 완료\n")

            prompt = self.prompt_entry.get("1.0", "end").strip() or None
            with_ts = self.timestamps_var.get()
            make_srt = self.srt_var.get()

            for audio_path in list(self.files):
                self._transcribe_one(audio_path, prompt, with_ts, make_srt)

            self.log("\n■ 모든 작업이 끝났습니다. [결과 폴더 열기] 버튼으로 확인하세요.")
        except Exception as e:
            self.log(f"✘ 오류 발생: {e}")
        finally:
            self.running = False
            self.root.after(0, self._finish_ui)

    def _finish_ui(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.status_var.set("대기 중")

    def _transcribe_one(self, audio_path: Path, prompt, with_ts: bool, make_srt: bool):
        self.log(f"▶ 전사 시작: {audio_path.name}")
        start = time.time()

        segments, info = self.model.transcribe(
            str(audio_path),
            language="ko",
            initial_prompt=prompt,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 700},
            beam_size=5,
        )
        self.log(f"   오디오 길이: {format_timestamp(info.duration)}")

        out_dir = audio_path.parent
        stem = audio_path.stem
        txt_path = out_dir / f"{stem}_전사.txt"
        srt_path = out_dir / f"{stem}_자막.srt"
        md_path = out_dir / f"{stem}_정리초안.md"

        txt_lines, srt_blocks, plain = [], [], []
        for i, seg in enumerate(segments, start=1):
            text = seg.text.strip()
            if not text:
                continue
            plain.append(text)
            txt_lines.append(f"[{format_timestamp(seg.start)}] {text}" if with_ts else text)
            srt_blocks.append(
                f"{i}\n{format_timestamp(seg.start, srt=True)} --> "
                f"{format_timestamp(seg.end, srt=True)}\n{text}\n")
            if i % 15 == 0:
                self.log(f"   ... {format_timestamp(seg.end)} 지점 처리 중")
                # 중간 미리보기 한 줄
                self.log(f"      └ \"{text[:60]}\"")

        txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
        if make_srt:
            srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
        md_path.write_text(
            build_summary_template(audio_path.name, info.duration, plain),
            encoding="utf-8")

        self.last_output_dir = out_dir
        self.last_txt_path = txt_path

        elapsed = time.time() - start
        self.log(f"✔ 완료 ({elapsed:.0f}초) → {txt_path.name}")

        # 결과 앞부분 미리보기
        preview = "\n".join(txt_lines[:8])
        self.log("---- 결과 미리보기 ----")
        self.log(preview)
        self.log("----------------------\n")

    # ---------------- 결과 열기 ----------------
    def open_output_folder(self):
        target = self.last_output_dir or (self.files[0].parent if self.files else None)
        if not target:
            messagebox.showinfo("알림", "아직 결과가 없습니다.")
            return
        self._open_path(target)

    def open_txt(self):
        if not self.last_txt_path or not self.last_txt_path.exists():
            messagebox.showinfo("알림", "아직 생성된 전사문이 없습니다.")
            return
        self._open_path(self.last_txt_path)

    @staticmethod
    def _open_path(path: Path):
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


def main():
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