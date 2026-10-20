"""New measured flight reproduction, with no writes to trajectory after launch."""
from pathlib import Path
import argparse,json,time,hashlib,shutil
import numpy as np
import mujoco
from neural import Brain,Readout
from plant import build,Camera,control

ROOT=Path(__file__).resolve().parent

def run(a):
    if a.output.exists():raise FileExistsError('Use a fresh output directory')
    a.output.mkdir(parents=True)
    for filename in ['run.py','neural.py','plant.py','build_graph.py']:
        dest=a.output/'executed_sources'/filename;dest.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/filename,dest)
    model,data,I,mass,mix,limit,metadata,xml=build(a.dt,a.target,a.absent)
    (a.output/'model.xml').write_text(xml)
    if a.start_height is not None:data.qpos[2]=a.start_height;mujoco.mj_forward(model,data)
    brain=Brain(cut=a.cut);readout=Readout();camera=Camera(model)
    dt_control=.002;substeps=int(round(dt_control/a.dt))
    if abs(substeps*a.dt-dt_control)>1e-12:raise ValueError('dt must divide 2ms exactly')
    queue=[];proxy=np.zeros(7);next_cam=0.;velocity=np.zeros(3);active=False;mn=np.full(815,75.);code=np.zeros(7)
    history=[];records=[];mnrecords=[];inputs=[];brain_times=[];start=time.perf_counter();prev_velocity=np.zeros(3);receptor_sum=np.zeros(len(brain.channels));receptor_count=0
    for step in range(int(round(a.duration/dt_control))+1):
        now=step*dt_control
        mujoco.mj_forward(model,data)
        R=data.xmat[1].reshape(3,3);vb=data.sensordata[3:6].copy();omega=data.sensordata[:3].copy();accel=(vb-prev_velocity)/dt_control;prev_velocity=vb
        if now+1e-10>=next_cam:
            queue.append((now+.010,camera.sense(data)));next_cam+=1/30
        while queue and queue[0][0]<=now+1e-10:proxy=queue.pop(0)[1]
        # Update at the end of the past 20ms window; no future sensor samples.
        if step and step%10==0:
            supplied=receptor_sum/max(receptor_count,1);t=time.perf_counter();mn=brain.step(supplied);brain_times.append(time.perf_counter()-t)
            velocity,active,code=readout.predict(mn);mnrecords.append(mn.copy());inputs.append(supplied.copy());receptor_sum[:]=0;receptor_count=0
        contact=data.ncon>0
        receptor_sum+=brain.inputs(proxy,R,vb,omega,accel,contact);receptor_count+=1
        control(model,data,I,mass,mix,limit,velocity,active)
        if a.motor_off_after is not None and now>=a.motor_off_after:data.ctrl[:]=0
        position=data.xpos[1].copy();tilt=float(np.degrees(np.arccos(np.clip(R[2,2],-1,1))))
        records.append(np.r_[now,position,data.qpos[3:7],vb,omega,float(contact),tilt,data.act.copy(),velocity,float(active),proxy,code])
        if not np.isfinite(data.qpos).all() or not np.isfinite(mn).all():raise RuntimeError('Nonfinite physical/neural state')
        if step%2500==0:print(json.dumps({'time_s':now,'height_m':position[2],'contact':bool(contact),'active':bool(active),'tilt_deg':tilt,'max_evoked_MN_Hz':float(np.max(abs(mn-75)))}),flush=True)
        if step==int(round(a.duration/dt_control)):break
        for _ in range(substeps):mujoco.mj_step(model,data)
    trace=np.asarray(records)
    # Column layout is immutable and recorded here to keep measurements auditable.
    layout={'time':0,'position':[1,4],'quaternion':[4,8],'body_velocity':[8,11],'gyro':[11,14],'contact':14,'tilt':15,'rotor_force':[16,20],'neural_velocity':[20,23],'active':23,'camera_proxy':[24,31],'decoded_proxy':[31,38]}
    np.savez_compressed(a.output/'trace.npz',records=trace,motor_rates=np.asarray(mnrecords),receptor_inputs=np.asarray(inputs),brain_step_seconds=np.asarray(brain_times))
    airborne=trace[:,14]==0;longest=count=0
    for yes in airborne:count=count+1 if yes else 0;longest=max(longest,count)
    tail=trace[:,0]>=a.duration-1;distance=np.linalg.norm(trace[:,1:4]-np.asarray(a.target),axis=1)
    report={'duration_s':a.duration,'cut':a.cut,'absent':a.absent,'motor_off_after_s':a.motor_off_after,'start_height_m':float(trace[0,3]),'max_height_m':float(trace[:,3].max()),'final_height_m':float(trace[-1,3]),'continuous_airborne_s':max(0,longest-1)*dt_control,'last_1s_contacts':int(trace[tail,14].sum()),'last_1s_max_tilt_deg':float(trace[tail,15].max()),'initial_goal_distance_m':float(distance[0]),'final_goal_distance_m':float(distance[-1]),'fixed_target_m':a.target,'max_thrust_weight_ratio':float(trace[:,16:20].sum(axis=1).max()/(mass*9.81)),'last_1s_mean_thrust_weight_ratio':float(trace[tail,16:20].sum(axis=1).mean()/(mass*9.81)),'max_MN_modulation_hz':float(np.max(abs(np.asarray(mnrecords)-75))) if mnrecords else 0,'max_neural_speed_m_s':float(np.linalg.norm(trace[:,20:23],axis=1).max()),'wall_seconds':time.perf_counter()-start,'layout':layout,'plant':metadata,'scope':'Fresh reconstruction of an engineered rate/readout/thruster candidate, not bitwise recovery or natural flapping flight','brain_metadata':json.loads((ROOT/'derived/graph.json').read_text()),'trace_sha256':hashlib.sha256((a.output/'trace.npz').read_bytes()).hexdigest()}
    report['sustained_flight']=bool(report['continuous_airborne_s']>=max(1.,a.duration-1.) and report['max_height_m']>.05 and report['last_1s_contacts']==0 and report['last_1s_max_tilt_deg']<15)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--duration',type=float,default=20.);p.add_argument('--target',nargs=3,type=float,default=[.2,.15,.3]);p.add_argument('--dt',type=float,default=25e-6);p.add_argument('--cut',action='store_true');p.add_argument('--absent',action='store_true');p.add_argument('--start-height',type=float);p.add_argument('--motor-off-after',type=float);run(p.parse_args())
