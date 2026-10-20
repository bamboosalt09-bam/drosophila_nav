# 여기서 작업을 이어가십시오

현재 비행 재구성의 코드는 `flight_recovery/`, 검증 결과는 `results/flight_recovery_2026-10-18/`에 있습니다. 저장소 기본 브랜치와 GitHub Release를 함께 사용하여 다른 환경에서 이어갈 수 있습니다.

1. `python flight_recovery/restore_artifacts.py`로 실행 캐시를 내려받으십시오. 파일별 SHA256을 검사합니다.
2. `flight_recovery/requirements.txt`의 Python 의존성을 준비하십시오. 검증 환경은 Python 3.13.15입니다.
3. `flight_recovery/README.md`와 `results/flight_recovery_2026-10-18/results.md`를 읽으십시오.
4. 새 출력 폴더를 지정하여 비행을 실행하십시오.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl python flight_recovery/run.py --duration 20 --output results/new_flight_trial
```

캐시 전체·원자료·원본 결과까지 필요하면 `python flight_recovery/restore_artifacts.py --all`을 사용하십시오. [Release 및 보존 방식](recovery_backups/README.md)을 참고하십시오.

두 고정 표식 조건에서 20초 비행을 실행했고 각각 연속 무접촉 19.932초를 확인했습니다. 신경 차단·표식 없음에서는 추력 0, 비행 중 모터 차단에서는 지면 복귀를 확인했습니다. 독립 감각 혼합, 국소 추력·중력, 시간 간격 반감 물리 재생과 몸체 없는 3D 화면을 검증했습니다.

이는 학습 출력층·자세 안정기·대체 추력의 공학적 비행입니다. 자연 초파리의 생리 교정, 이전 학습 가중치와의 동일 복원, 이전 4,000행 반응 표, 드론 운동 변환기, 새로운 장애물 시험과 실물 비행은 미완료입니다. 과거 기록의 수치를 이번 결과로 혼용하지 마십시오.
