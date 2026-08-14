# 발명자 인터뷰 전사기 v1.3

발명자 인터뷰 녹음·영상(mp3, m4a, mp4 등)을 한국어 텍스트로 변환하는 Windows용 전사 도구입니다. 기본 전사는 `faster-whisper`와 `large-v3`를 사용해 PC에서 로컬로 처리합니다.

## v1.3 주요 변경

- 최종 전사에 남은 긴 공백만 Vertex AI의 `gemini-3.7-flash`로 선택 검토할 수 있습니다.
- Gemini 기능은 기본적으로 꺼져 있으며, GUI에서 외부 전송 확인을 거쳐야 실행됩니다.
- 공백 전체와 최대 8초의 앞뒤 문맥만 메모리 WAV로 전송합니다. 전체 원본 파일을 업로드하거나 원격 파일로 보관하지 않습니다.
- Gemini 응답은 `*_전사_검토구간.md`에 후보로만 기록하며 최종 전사문에 자동 병합하지 않습니다.
- 정책 차단이 발생한 구간은 다른 프롬프트로 우회하거나 자동 재요청하지 않습니다.
- 서비스 계정 키는 실행 파일에 포함하거나 저장하지 않습니다.

훌루테크 회의 녹음 A/B 시험에서는 Gemini가 일부 기술용어와 Whisper의 누락 구간을 보완했지만, 일부 구간은 정책 차단되었고 불명확한 음성을 확신하는 경우도 있었습니다. 따라서 v1.3은 Gemini를 전체 전사의 대체 모델이 아닌 사람 검토용 선택 후보 생성기로 사용합니다.

## 전사 방식

| 모드 | 동작 | 권장 상황 |
|---|---|---|
| 누락 최소화 (기본) | 민감한 VAD와 앞뒤 1초 여백 | 회의·인터뷰, 작은 목소리 |
| 균형 | 보수적인 VAD | 무음이 매우 긴 녹음 |
| 빠르게 | 명확한 음성 위주 | 짧고 깨끗한 녹음 |

1차 전사 후 비정상적으로 긴 세그먼트, 프롬프트 반복, 짧은 파편 연속, 긴 공백을 검사합니다. 이상 구간만 VAD를 끄고 재전사한 뒤 품질 기준과 오디오 에너지 기준을 통과한 후보만 로컬 전사문에 병합합니다.

## 설치 및 GUI 실행

```powershell
python -m pip install -r requirements.txt
python interview_transcriber_gui.py
```

1. 음성·영상 파일 또는 폴더를 선택합니다.
2. 보통 `large-v3`와 `누락 최소화 (권장)`을 사용합니다.
3. 기술용어와 고유명사를 쉼표로 구분해 hotwords에 입력합니다.
4. 필요한 경우에만 짧은 배경 설명을 initial prompt에 입력합니다.
5. Gemini 검토가 필요하면 아래 Vertex AI 설정을 완료한 뒤 체크합니다.
6. 완료 후 전사문과 검토 구간 파일을 함께 확인합니다.

## Gemini 선택 검토 설정

Vertex AI API 사용 권한이 있는 Google Cloud 서비스 계정 JSON이 필요합니다. GUI의 `서비스 계정 JSON`에서 파일을 선택합니다. 프로젝트 ID는 JSON의 `project_id`를 기본으로 사용하며 필요할 때만 직접 입력합니다.

환경 변수로 설정할 수도 있습니다.

```powershell
$env:VERTEX_SA_JSON = "C:\secure\vertex-service-account.json"
$env:VERTEX_PROJECT_ID = "my-project-id"  # 선택 사항
```

주의 사항:

- 체크하면 남은 긴 공백의 오디오 클립과 직전·직후 전사문이 Google Vertex AI로 전송됩니다.
- 구간당 전송 길이는 기본 최대 120초입니다. 더 긴 공백은 전송하지 않고 검토 보고서에 표시합니다.
- 결과에는 오류·정책 차단·토큰 사용량과 후보 문장이 기록됩니다.
- API 비용, 데이터 처리 위치, 조직의 보안 정책을 확인한 후 사용하세요.
- 서비스 계정 JSON과 회의 원본·전사 결과는 저장소에 커밋하지 마세요.

## 생성 파일

- `파일명_전사.txt`: 타임스탬프 포함 전사문
- `파일명_자막.srt`: 영상 대조용 자막
- `파일명_정리초안.md`: 인터뷰 정리 템플릿과 전체 전사문
- `파일명_전사_검토구간.md`: 자동 재전사 판단, 남은 공백, Gemini 검토 후보

모든 결과는 Windows 메모장에서 한글이 안정적으로 열리도록 UTF-8 BOM으로 저장됩니다.

## CLI

로컬 전사:

```powershell
python interview_transcriber.py "회의.mp4" --model large-v3 --mode high_recall
```

Gemini 선택 검토 포함:

```powershell
python interview_transcriber.py "회의.mp4" `
  --gemini-review `
  --vertex-service-account "C:\secure\vertex-service-account.json" `
  --vertex-location global `
  --gemini-model gemini-3.7-flash
```

주요 옵션:

```text
--mode high_recall|balanced|fast
--hotwords "LOT, AMR, HMI, 연결관"
--initial-prompt "필요한 경우에만 사용하는 짧은 회의 배경"
--gap-seconds 15
--output-dir "결과폴더"
--no-auto-retry
--no-retry-gaps
--no-normalize-retry
--no-timestamps
--no-srt
--gemini-review
--vertex-service-account PATH
--vertex-project-id PROJECT_ID
--vertex-location global
--gemini-model gemini-3.7-flash
--gemini-max-clip-seconds 120
```

## Windows EXE 빌드와 검증

```powershell
python -m PyInstaller "인터뷰전사기.spec" --noconfirm --clean
dist\InterviewTranscriber.exe --package-smoke-test
```

배포 자산 이름은 브라우저 다운로드 시 한글 파일명이 변형되지 않도록 `InterviewTranscriber.exe`를 사용합니다. 최초 로컬 전사 시 Whisper 모델 다운로드를 위해 인터넷 연결이 필요하고, 이후에는 Gemini 선택 검토를 사용하지 않는 한 오프라인으로 동작합니다.
