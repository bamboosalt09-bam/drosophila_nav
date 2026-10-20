"""Check real force transmission, gravity and numerical replay of flown commands."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
from plant import build

def witnesses():
    m,d,I,mass,mix,limit,meta,xml=build()
    d.qpos[2]=1.;d.qpos[3:7]=[np.cos(.2),np.sin(.2),0,0];mujoco.mj_forward(m,d)
    R=d.xmat[1].reshape(3,3).copy();errors=[]
    for k in range(4):
        d.act[:]=0;d.act[k]=mass*9.81/4;mujoco.mj_forward(m,d)
        force=R@np.array([0.,0.,d.act[k]])
        arm=m.site_pos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'rotor'+str(k))]
        torque=np.cross(arm,[0.,0.,d.act[k]])+np.array([0.,0.,[1,-1,1,-1][k]*.00003*d.act[k]])
        errors.append(float(np.max(abs(d.qfrc_actuator-np.r_[force,torque]))))
    m,d,*_=build();d.qpos[2]=1;mujoco.mj_forward(m,d)
    for _ in range(4000):mujoco.mj_step(m,d)
    analytic=1-.5*9.81*.1**2
    result={'four_rotor_force_torque_max_errors':errors,'freefall_100ms_height_error_m':float(abs(d.qpos[2]-analytic)),'all_pass':bool(max(errors)<1e-15 and abs(d.qpos[2]-analytic)<1e-10)}
    return result

def replay(folder,dt):
    r=json.loads((folder/'report.json').read_text());z=np.load(folder/'trace.npz')['records']
    if z.shape[1]<48:raise ValueError('Replay requires saved command/passive force columns')
    m,d,I,mass,*_=build(dt,r['fixed_target_m'],r['absent']);d.qpos[2]=r['start_height_m'];mujoco.mj_forward(m,d)
    errors=[];angle=[]
    for i,row in enumerate(z):
        mujoco.mj_forward(m,d);errors.append(float(np.linalg.norm(d.xpos[1]-row[1:4])))
        dot=abs(float(d.qpos[3:7]@row[4:8]));angle.append(float(np.degrees(2*np.arccos(np.clip(dot,-1,1)))))
        if i==len(z)-1:break
        d.ctrl[:]=row[38:42]
        # Passive drag is a constitutive law of the replayed body, not an
        # exogenous future force taken from the reference trajectory.
        R=d.xmat[1].reshape(3,3)
        d.qfrc_applied[:3]=-2*mass*(R@d.sensordata[3:6])
        d.qfrc_applied[3:]=-10*(I@d.sensordata[:3])
        for _ in range(int(round(.002/dt))):mujoco.mj_step(m,d)
    return {'dt_s':dt,'samples':len(z),'passive_force':'Same declared drag law evaluated on the replayed body every 2ms','maximum_position_error_m':max(errors),'maximum_angle_error_deg':max(angle),'gate_position_m':.0001,'gate_angle_deg':.1,'all_pass':max(errors)<.0001 and max(angle)<.1}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--trace-folder',type=Path);a=p.parse_args()
    result={'force_and_gravity':witnesses()}
    if a.trace_folder:
        result['same_step_replay']=replay(a.trace_folder,25e-6)
        result['half_step_replay']=replay(a.trace_folder,12.5e-6)
    result['all_pass']=all(v['all_pass'] for v in result.values());a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['all_pass']:raise RuntimeError('Physics verification failed')
