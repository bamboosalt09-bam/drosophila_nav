"""Flybody inertia plus an explicitly engineered four-thruster rigid body."""
from pathlib import Path
import json
import hashlib
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent

def build(dt=25e-6,target=(.2,.15,.3),absent=False):
    source=BASE/'reference/flybody_recovered/flybody/fruitfly/assets/fruitfly.xml'
    cache=ROOT/'derived/flybody_mass_properties.npz'
    if cache.is_file():
        props=np.load(cache);m=float(props['mass']);com=props['com'];inertia=props['inertia']
    else:
        original=mujoco.MjModel.from_xml_path(str(source));state=mujoco.MjData(original);mujoco.mj_forward(original,state)
        mass=original.body_mass[1:]*1e-3;com=state.subtree_com[1]*.01;inertia=np.zeros((3,3))
        for i,body_mass in enumerate(mass,1):
            R=state.ximat[i].reshape(3,3);dx=state.xipos[i]*.01-com
            inertia+=R@np.diag(original.body_inertia[i]*1e-7)@R.T+body_mass*(np.eye(3)*(dx@dx)-np.outer(dx,dx))
        m=float(mass.sum());np.savez_compressed(cache,mass=m,com=com,inertia=inertia)
        (ROOT/'derived/flybody_source.json').write_text(json.dumps({'source':'https://github.com/TuragaLab/flybody','commit':'d015e9bfe441bd90ae431bac24c55cb74bdbce26','xml_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'pose':'Native qpos0','native_units':'g, cm, s','mass_kg':m},indent=2)+'\n')
    arm=.0008;kappa=.00003;limit=.8*m*9.81
    full=[inertia[0,0],inertia[1,1],inertia[2,2],inertia[0,1],inertia[0,2],inertia[1,2]]
    xy=np.array([[arm,arm],[-arm,arm],[-arm,-arm],[arm,-arm]])
    signs=np.array([1,-1,1,-1]);sites='';actuators=''
    for i,(x,y) in enumerate(xy):
        sites+=f'<site name="rotor{i}" pos="{x} {y} 0" size=".0001"/>'
        actuators+=f'<general name="rotor{i}" site="rotor{i}" gear="0 0 1 0 0 {signs[i]*kappa}" dyntype="filterexact" dynprm=".002" ctrllimited="true" ctrlrange="0 {limit}"/>'
    xyz=' '.join(map(str,target));colour='.15 .15 .15 1' if absent else '1 .02 .01 1'
    xml=f'''<mujoco model="reconstructed_fly_equivalent">
      <compiler angle="radian"/><option timestep="{dt}" gravity="0 0 -9.81" integrator="RK4"/>
      <worldbody><geom name="floor" type="plane" size="5 5 .01" rgba=".2 .25 .3 1"/>
      <geom name="marker" type="sphere" pos="{xyz}" size=".025" rgba="{colour}" contype="0" conaffinity="0"/>
      <body name="fly_equivalent" pos="0 0 .0011"><freejoint name="root"/>
      <inertial pos="0 0 0" mass="{m}" fullinertia="{' '.join(map(str,full))}"/>
      <geom name="massless_skid" type="sphere" size=".0011" mass="0" friction="0 0 0" condim="1" priority="1" solref=".001 1" group="3"/>
      <site name="imu" size=".0001"/>{sites}</body></worldbody><actuator>{actuators}</actuator>
      <sensor><gyro name="gyro" site="imu"/><velocimeter name="velocity" site="imu"/><accelerometer name="accel" site="imu"/></sensor>
      </mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    mixing=np.vstack([np.ones(4),xy[:,1],-xy[:,0],signs*kappa])
    metadata={'source':'TuragaLab/flybody fruitfly.xml','source_native_units':'g, cm, s','source_mass_kg':m,'source_COM_m':com.tolist(),'rigid_inertia_kg_m2':inertia.tolist(),'maximum_total_thrust_weight_ratio':3.2,'motor_lag_s':.002,'drag_rate_per_s':2,'angular_drag_rate_per_s':10,'landing':'Massless frictionless COM sphere; not biological feet','geometry_scope':'Source inertia is aggregated at native qpos0; visual anatomical meshes are not used in this flight-only equivalent plant','integrator':'RK4','physics_dt_s':dt,'pose_writes_after_initialization':False}
    return model,data,inertia,m,np.linalg.inv(mixing),limit,metadata,xml

class Camera:
    def __init__(self,model,n=16384):
        self.model=model
        z=1-2*(np.arange(n)+.5)/n;phi=np.arange(n)*np.pi*(3-np.sqrt(5));r=np.sqrt(1-z*z)
        self.rays=np.column_stack([r*np.cos(phi),r*np.sin(phi),z]);self.n=n
        self.ids=np.empty(n,np.int32);self.dist=np.empty(n);self.mask=np.array([1,1,1,0,0,0],np.uint8)
    def sense(self,data):
        R=data.xmat[1].reshape(3,3);origin=data.xpos[1].copy()+R@np.array([.0015,0,0])
        world=np.ascontiguousarray(self.rays@R.T)
        mujoco.mj_multiRay(self.model,data,origin,world.ravel(),self.mask,1,1,self.ids,self.dist,None,self.n,10.)
        visible=self.ids>=0;selected=np.zeros(self.n,bool)
        colors=self.model.geom_rgba[self.ids[visible],:3]
        selected[visible]=(colors[:,0]>.7)&(colors[:,0]>2*colors[:,1])&(colors[:,0]>2*colors[:,2])
        if not selected.any():return np.zeros(7)
        direction=self.rays[selected].mean(axis=0);direction/=np.linalg.norm(direction)
        area=selected.mean();return np.r_[np.maximum(np.column_stack([direction,-direction]).ravel(),0),np.sqrt(area/.04)]

def control(model,data,inertia,mass,mix_inv,limit,velocity_body,active):
    R=data.xmat[1].reshape(3,3);omega=data.sensordata[:3].copy();vb=data.sensordata[3:6].copy();v=R@vb
    # Explicit passive drag, separate from local propulsor forces.
    data.qfrc_applied[:3]=-2*mass*v;data.qfrc_applied[3:]=-10*(inertia@omega)
    if not active:data.ctrl[:]=0;return
    desired=R@np.asarray(velocity_body)
    total=np.array([0.,0.,9.81])+10*(desired-v)+2*v
    z=total/np.linalg.norm(total);y=np.cross(z,[1.,0.,0.]);y/=max(np.linalg.norm(y),1e-12);x=np.cross(y,z);Rd=np.column_stack([x,y,z])
    error=.5*(Rd.T@R-R.T@Rd);e=np.array([error[2,1],error[0,2],error[1,0]])
    torque=inertia@(-900*e-60*omega)+np.cross(omega,inertia@omega)+10*inertia@omega
    force=max(0,mass*(total@R[:,2]));commands=mix_inv@np.r_[force,torque]
    data.ctrl[:]=np.clip(commands,0,limit)
