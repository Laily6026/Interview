# 발명자 인터뷰 전사기

발명자 인터뷰 녹음(mp3, m4a, mp4 등)을 텍스트로 변환하는 프로그램입니다.
faster-whisper 기반으로 **모든 처리가 로컬 PC에서 이루어지므로**
음성 파일이 외부 서버로 전송되지 않습니다.

## 일반 사용자: .exe로 사용 (권장)

1. 이 저장소의 **[Releases](../../releases)** 페이지에서 최신 `인터뷰전사기.exe` 다운로드
2. 더블클릭 실행 (최초 실행은 압축 해제로 10~30초 걸릴 수 있음)
3. 음성 파일 선택 → 옵션 설정 → [▶ 전사 시작]
4. 결과는 음성 파일과 같은 폴더에 생성됩니다:
   - `파일명_전사.txt` — 타임스탬프 포함 전사문
   - `파일명_자막.srt` — 녹음 재생 대조용 자막
   - `파일명_정리초안.md` — 인터뷰 정리 템플릿(전문 포함)

> 최초 1회 모델 다운로드를 위해 인터넷 연결이 필요합니다.
> (medium 약 1.5GB / large-v3 약 3GB, `C:\Users\사용자명\.cache\huggingface`에 저장)
> 다운로드 후에는 오프라인에서 동작합니다.

## 개발자: 소스로 실행

```cmd
pip install -r requirements.txt
python interview_transcriber_gui.py
```

CLI 버전:

```cmd
python interview_transcriber.py 녹음파일.m4a --model large-v3
```

## .exe 빌드 방법

```cmd
pip install -r requirements.txt
python -m PyInstaller --onefile --noconsole --name "인터뷰전사기" --collect-all faster_whisper interview_transcriber_gui.py
```

→ `dist\인터뷰전사기.exe` 생성. 새 버전은 Releases에 업로드하여 배포합니다.

## 사용 팁

- **기술용어 힌트**: 사건별 핵심 용어(예: "리튬이차전지, 양극활물질")를
  힌트 입력란에 추가하면 전문용어 인식률이 올라갑니다.
- **모델 선택**: medium = 속도·정확도 균형 / large-v3 = 최고 정확도(느림)
- **보안 주의**: 전사 결과물(txt/srt/md)은 발명 내용이 포함되므로
  저장소에 커밋하지 마세요 (.gitignore로 차단되어 있음).
