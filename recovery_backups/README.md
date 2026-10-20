# GitHub와 로컬의 비행 자료

코드·실행 설명·검증 결과는 저장소 기본 브랜치에 기록합니다. 대용량 실행 캐시와 원자료, 원본 압축 백업은 같은 저장소의 [flight-recovery-2026-10-20 Release](https://github.com/bamboosalt09-bam/drosophila_nav/releases/tag/flight-recovery-2026-10-20)에 보존합니다. SHA256과 다운로드 주소는 `github_assets.json`에 있습니다.

새 작업환경에서 저장소를 내려받은 다음 다음 명령으로 실행 캐시를 복원하십시오. Python 표준 라이브러리만 필요하며, 현재 코드 파일을 과거 백업으로 덮어쓰지 않습니다.

```bash
python flight_recovery/restore_artifacts.py
```

접촉 행렬·교정 응답·원자료·결과까지 전부 복원하려면 다음을 사용하십시오.

```bash
python flight_recovery/restore_artifacts.py --all
```

로컬의 `.venv`, Python 캐시, 중첩 Git 저장소 자체는 업로드하지 않습니다. Python 의존성은 `flight_recovery/requirements.txt`로 설치할 수 있습니다. 공개 Flybody 참조는 고정 커밋의 Git submodule로 연결하며, 기본 실행에는 동봉된 질량·관성 캐시를 사용합니다. 원본 메시까지 필요할 때만 `git submodule update --init reference/flybody_recovered`를 실행하십시오.

2026-10-18 실행 결과는 원자료 기반 공학적 재구성입니다. 이전 `nav_v2` 파일·4,000행 반응 표·드론 변환기가 복원되었다는 의미는 아닙니다.
