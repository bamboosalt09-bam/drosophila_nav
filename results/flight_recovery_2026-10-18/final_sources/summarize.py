"""Build a measured-results report and a body-free self-contained 3D replay."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'results/flight_recovery_2026-10-18'
CASES=[('ground_corrected_20s','고정 고도 목표 · 20초'),('opposite_corrected_20s','반대쪽 · 다른 고도 목표 · 20초'),('cut_corrected_3s','신경 연결 차단'),('absent_corrected_3s','시각 표식 없음'),('motor_off_corrected_6s','3초부터 모터 차단')]

def main():
    rows=[];reports={}
    for name,label in CASES:
        folder=OUT/name;report=json.loads((folder/'report.json').read_text());z=np.load(folder/'trace.npz')['records'];idx=np.unique(np.r_[np.arange(0,len(z),5),len(z)-1]);r=z[idx]
        reports[name]=report
        rows.append(dict(name=name,label=label,t=r[:,0].tolist(),p=r[:,1:4].tolist(),tilt=r[:,15].tolist(),contact=r[:,14].tolist(),force=(r[:,16:20].sum(axis=1)/(report['plant']['source_mass_kg']*9.81)).tolist(),target=report['fixed_target_m'],absent=report['absent'],duration=report['duration_s'],airborne=report['continuous_airborne_s'],height=report['final_height_m']))
    template=(ROOT/'viewer_template.html').read_text();template=template.replace('__THREE__',(ROOT/'three.min.js').read_text()).replace('__DATA__',json.dumps(rows,ensure_ascii=False));(OUT/'viewer.html').write_text(template)
    selected=[reports[n] for n,_ in CASES[:2]];cut=reports[CASES[2][0]];absent=reports[CASES[3][0]];off=reports[CASES[4][0]]
    physics=json.loads((OUT/'physics_replay.json').read_text());readout=json.loads((OUT/'independent_readout.json').read_text())
    checks={'two_20s_ground_launches_sustain_flight':all(r['sustained_flight'] for r in selected),'cut_has_zero_MN_modulation_and_thrust':cut['max_MN_modulation_hz']==0 and cut['max_thrust_weight_ratio']==0,'absent_marker_has_no_thrust':absent['max_thrust_weight_ratio']<1e-10,'midair_motor_off_lands':off['last_1s_contacts']>0 and off['final_height_m']<.002,'physics_force_gravity_replay':physics['all_pass'],'independent_mixed_sensor_readout':readout['all_pass']}
    summary={'scope':'Fresh engineering reconstruction; not bitwise recovery of the missing 2026-10-06 packet','checks':checks,'all_pass':all(checks.values()),'cases':reports,'natural_flapping_flight_verified':False,'previous_4000_response_table_restored':False,'previous_drone_converter_restored':False,'new_obstacle_trial_performed':False,'real_hardware_verified':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 비행 재구성 및 재검증 — 2026-10-18','',
      '이전 결과 폴더·가중치는 현재 작업공간/Git에서 찾지 못했고 원래 업로드도 다시 열 수 없었습니다. 따라서 과거 수치의 동일 재현이 아닌, 공개 원자료와 남아 있는 방법 기록으로 새로 구성한 공학적 비행 시험입니다. 아래 수치는 이번에 실행한 물리 기록에서 계산했습니다.','',
      '## 판정','',f"전체 비행 검증: **{'통과' if summary['all_pass'] else '미통과'}**. 지면 출발·연속 무접촉·고도·자세·실제 추력과 연결 차단 대조를 사용했습니다. 자연 초파리의 날갯짓이나 생리 교정 판정은 아닙니다.",'',
      '| 조건 | 연속 무접촉 | 마지막 높이 | 마지막 1초 최대 기울기 | 마지막 1초 추력/체중 |','|---|---:|---:|---:|---:|']
    for name,label in CASES:
        r=reports[name];lines.append(f"| {label} | {r['continuous_airborne_s']:.3f}초 | {r['final_height_m']*100:.2f}cm | {r['last_1s_max_tilt_deg']:.6f}° | {r['last_1s_mean_thrust_weight_ratio']:.6f} |")
    lines+=['','## 무엇을 복구했는지','',
      'MaleCNS v1.0 원자료 3종을 다시 내려받아 주석 ID 양 끝점을 가진 실제 211,577개 노드·26,028,386개 연결 쌍·125,365,933개 시냅스와 MN 815개를 재구성했습니다. 흥분/억제 부호를 적용한 현재 연산자는 25,117,038쌍입니다. 이전 보고의 부호 연산자 24,775,486쌍과 일치하지 않으므로 그래프 동역학까지 같은 파일로 복원되었다고 해석하지 않습니다. histamine은 이전 후보의 NT 규칙에 맞춰 계산에서 제외했으며 생물학적으로 무효라는 의미는 아닙니다.','',
      '음식 ORN 213개로 카메라 기반 가상 유인 단서를 전달합니다. 구면 광선 카메라는 실제 MuJoCo 물체 교차와 RGB 속성만 읽고 표식의 상대 방향·면적을 산출합니다. 목표 좌표는 장면 배치와 채점에만 사용하며 제어기로 전달하지 않습니다. 다른 시각·균형곤·종상·존스턴 기류/중력/진동·목·hair plate·chordotonal·체모 입력도 실제 주석 세포에 공학적 배정으로 전달합니다. 정상 수용체 생리와 광학 모델을 복원한 것은 아닙니다.','',
      '발화율 기준 75Hz, 갱신 20ms, 시간상수 20ms, 부호 연결 이득 0.9, 감각 이득 0.04의 안정된 선형 동역학을 사용합니다. 새 MN 출력층은 가상 감각 채널을 구별하도록 선형 교정했습니다. 초기 출력층은 비행은 유지했지만 방향이 흔들렸고, 감각 과도응답 간섭을 제거한 후보로 두 개 20초 시험을 다시 진행했습니다. 물리적으로 측정한 자연 MN–근육 대응을 복원했다는 뜻은 아닙니다.','',
      '공식 TuragaLab/flybody의 질량·관성을 SI로 집계한 6자유도 강체와 4개 국소 추력, 2ms 모터 지연, 명시적 항력·자세/속도 안정기, 질량 없는 마찰 없는 COM 착지 구를 사용합니다. 원 몸체의 관절 자세 qpos0에서 집계했으며 이전 모델의 정확한 다리 자세·시각 형상을 재현한 것은 아닙니다. 시작 이후 위치/자세를 강제로 쓰지 않습니다.','',
      '센서와 내부 제어는 2ms, 신경은 지난 20ms 감각을 평균해 갱신합니다. 카메라는 30Hz·전달 지연 10ms입니다. 감각→MN→속도 요청→자세 안정기→추력→실제 물리 상태→감각 순서로 실행합니다.','',
      '## 독립 확인','',
      '각 국소 추력의 힘·지레 토크를 좌표 변환과 대조했고, 무추력 100ms 낙하를 중력 해석식과 비교했습니다. 최종 비행의 실제 모터 명령과 수동 힘을 저장했습니다. 재생에서는 기록된 모터 명령을 적용하고 동일 항력 법칙을 재생 몸체의 상태에 적용해 같은 시간 간격/절반 간격으로 물리를 다시 적분했습니다. 수치는 physics_replay.json에 있습니다. 최초에는 기준 궤적의 항력을 고정한 별도 시험에서 0.427mm 차이로 미통과했습니다. 이를 재생 몸체의 속도에 따른 같은 항력 법칙으로 수정했고 기준을 완화하지 않았습니다. 이것은 기록된 명령의 물리 재생이며 감각 폐루프 전체의 시간 간격 수렴 시험과는 다릅니다.','',
      '출력층은 교정에 쓰지 않은 seed의 12개 감각 조합에서 20ms마다 다른 감각 채널을 무작위 변화시켜 별도로 검증했습니다. 신경 연결 차단, 표식 없음, 비행 중 3초 이후 모터 차단도 각각 실제 실행했습니다.','',
      '이전 4,000행 반응 표·드론 물리 변환기·장애물 비행 결과는 이번에 복구/재검증하지 않았습니다. 이번 판정 범위는 대체 추력 몸체가 실제 물리 계산에서 이륙·지속 비행하는지입니다. 실물 드론과 자연 초파리 비행의 검증도 아닙니다.','',
      '## 재실행','',
      '프로젝트 루트에서 기존 `.venv`를 사용합니다. 필요한 패키지는 numpy/scipy/pandas/pyarrow/mujoco/matplotlib입니다. dm_control은 필요하지 않습니다. 동봉된 질량·관성 캐시가 있으면 Flybody 메시 없이 실행할 수 있고, 캐시 재생성 시에는 원본 XML/메시 경로가 필요합니다.','',
      '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python flight_recovery/run.py --duration 20 --output results/new_flight_trial',
      'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl .venv/bin/python flight_recovery/run.py --duration 3 --cut --output results/new_cut_trial','```','',
      '파생 그래프를 다시 만들려면 `scripts/get_malecns.py`, `flight_recovery/build_graph.py`, `flight_recovery/neural.py` 순서로 실행합니다. 각 원자료의 SHA256과 그래프 정보는 flight_recovery/derived/graph.json에 있습니다.','',
      '[몸체 없는 3D 궤적](viewer.html) · [종합 수치](summary.json) · [물리 독립 검증](physics_replay.json) · [감각 독립 검증](independent_readout.json)']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
    for name,label in CASES:
        z=np.load(OUT/name/'trace.npz')['records'];axes[0].plot(z[:,0],z[:,3]*100,label=name);axes[1].plot(z[:,0],z[:,16:20].sum(axis=1)/(reports[name]['plant']['source_mass_kg']*9.81))
    axes[0].set_ylabel('COM height [cm]');axes[0].legend(fontsize=7);axes[1].set_ylabel('Thrust / weight');axes[1].set_xlabel('Simulation time [s]');fig.tight_layout();fig.savefig(OUT/'flight_checks.png',dpi=160)
    print(json.dumps({'all_pass':summary['all_pass'],'checks':checks}))
    if not summary['all_pass']:raise RuntimeError('Flight verification did not pass')

if __name__=='__main__':main()
