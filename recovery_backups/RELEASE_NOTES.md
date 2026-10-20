# 비행 재구성 코드와 실행 자료 보존

2026-10-18에 실행한 대체 추력 비행의 코드, 학습 출력층, 파생 그래프, 물리 기록과 원자료를 다른 환경에서 다시 사용할 수 있도록 보존합니다.

- 코드·인계 문서·검증 결과: 이 태그와 저장소 기본 브랜치
- 실행에 필요한 캐시: flight_runtime_2026-10-18.tar.gz
- 접촉 행렬과 교정 응답을 포함한 전체 파생 자료: flight_derived_2026-10-18.tar.gz
- 원본 검증 기록과 몸체 없는 3D 궤적: flight_results_2026-10-18.tar.gz
- 원래 MaleCNS 자료: body-annotations, body-neurotransmitters, connectome-weights 세 Feather 파일
- 파일 크기·SHA256·다운로드 주소: github_assets.json

저장소를 clone한 뒤 `python flight_recovery/restore_artifacts.py`를 실행하면 현재 소스 코드를 덮어쓰지 않고 필요한 캐시를 복원합니다. `--all`은 전체 파생 자료·원자료·결과를 복원합니다.

두 20초 지면 출발 시험은 각각 19.932초 연속 무접촉 비행을 유지했습니다. 신경 차단·표식 없음·비행 중 모터 차단·독립 감각 혼합·국소 추력/중력·시간 간격 반감 물리 재생을 검증했습니다. 이는 학습 출력층·자세 안정기·대체 추력을 사용하는 공학 모델입니다. 자연 초파리의 생리 교정이나 이전 nav_v2 결과의 동일 복원은 아닙니다.

원자료 출처: FlyEM MaleCNS v1.0 public flat connectome (CC-BY), https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/ . Flybody 참조는 TuragaLab/flybody 고정 커밋 d015e9bfe441bd90ae431bac24c55cb74bdbce26이며 저장소의 submodule 및 FLYBODY_LICENSE를 참고하십시오. Three.js는 MIT입니다.
