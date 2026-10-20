"""Reconstructed linear rate candidate, not recovered natural fly physiology."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parent

class Brain:
    def __init__(self,cut=False):
        self.W=sparse.load_npz(ROOT/'derived/operator.npz')
        b=np.load(ROOT/'derived/binding.npz');self.motor=b['motor'];self.food=b['food'];self.ids=b['ids']
        a=pd.read_parquet(ROOT/'derived/annotations.parquet');sub=a['subclass'].fillna('');typ=a['type'].fillna('')
        self.groups={
            'retina':np.flatnonzero(typ.str.match(r'^R[1-8]')),
            'haltere':np.flatnonzero(sub.eq('haltere')),
            'campaniform':np.flatnonzero(sub.eq('campaniform sensilla')),
            'JO_wind':np.flatnonzero(sub.eq('wind_gravity')),
            'JO_gravity':np.flatnonzero(sub.eq('wind_gravity')),
            'JO_vibration':np.flatnonzero(sub.eq('auditory')),
            'neck':np.flatnonzero(sub.eq('neck')),
            'hair_plate':np.flatnonzero(sub.eq('hair plate')),
            'chordotonal':np.flatnonzero(sub.eq('chordotonal organ')),
            'bristle':np.flatnonzero(sub.str.contains('bristle') & a['class'].fillna('').str.contains('mechanosensory')),
        }
        self.channels=[];columns=[]
        def add(name,indices,banks):
            for k in range(banks):
                v=np.zeros(len(self.ids));v[indices[np.arange(len(indices))%banks==k]]=.04
                self.channels.append(name+str(k));columns.append(v)
        add('camera_proxy',self.food,7)
        for name,idx in self.groups.items():add(name,idx,6 if name not in ['JO_vibration','neck','hair_plate','bristle'] else 1)
        self.B=np.column_stack(columns);self.delta=np.zeros(len(self.ids));self.cut=cut;self.time=0.
    def reset(self):self.delta[:]=0;self.time=0.
    def step(self,channels):
        # Additive input on real receptor IDs. The input gains are engineering choices.
        drive=self.B@np.asarray(channels)
        recurrent=np.zeros_like(self.delta) if self.cut else self.W@self.delta
        self.delta+=(1-np.exp(-1))*(recurrent+drive-self.delta);self.time+=.02
        return 75+150*self.delta[self.motor]
    def inputs(self,proxy,R,velocity_body,omega,accel_body,contact):
        values=list(proxy)
        def split(v):
            v=np.clip(np.asarray(v),-1,1);return np.maximum(np.column_stack([v,-v]).ravel(),0).tolist()
        # Ideal sensors and engineered receptor assignments; no physiological tuning claim.
        sensory={
            'retina':np.asarray(proxy[:6])*.5,
            'haltere':split(np.asarray(omega)/10),
            'campaniform':split(np.asarray(accel_body)/19.62),
            'JO_wind':split(-np.asarray(velocity_body)/.2),
            'JO_gravity':split(R.T@np.array([0.,0.,1.])),
            'JO_vibration':[min(np.linalg.norm(omega)/20,1)],
            'neck':[.2], 'hair_plate':[.2],
            'chordotonal':split(np.asarray(velocity_body)/.2), 'bristle':[float(contact)],
        }
        for name in self.groups:values.extend(sensory[name])
        return np.asarray(values)

def calibrate():
    brain=Brain();X=[];Y=[];steps=30
    for k in range(len(brain.channels)):
        brain.reset();v=np.zeros(len(brain.channels));v[k]=1
        for t in range(steps):
            mn=brain.step(v)-75
            if t>=5:
                X.append(mn);y=np.zeros(7)
                if k<7:y[k]=1
                Y.append(y)
        if k%8==0:print(json.dumps({'calibration_channel':k,'channels':len(brain.channels)}),flush=True)
    X=np.array(X);Y=np.array(Y);std=np.maximum(np.sqrt(np.mean(X*X,axis=0)),1e-7);Z=X/std
    weights=np.linalg.lstsq(np.vstack([Z,np.eye(len(std))*1e-4]),np.vstack([Y,np.zeros((len(std),7))]),rcond=1e-10)[0]/std[:,None]
    errors=np.max(np.abs(X@weights-Y),axis=0)
    np.savez_compressed(ROOT/'derived/readout.npz',weights=weights,motor=brain.motor,ids=brain.ids[brain.motor])
    report={'source':'Fresh linear readout calibration after original weights could not be recovered','labels':'Injected camera-proxy channel amplitudes, not measured muscles','channels':brain.channels,'receptor_counts':{k:len(v) for k,v in brain.groups.items()},'food_ORNs':len(brain.food),'max_calibration_errors':errors.tolist(),'bias':0,'all_baseline_motor_rates_give_zero_command':True,'natural_neural_policy':False}
    (ROOT/'derived/readout.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if max(errors)>.15:raise RuntimeError('Readout calibration failed fixed .15 channel error limit')

class Readout:
    def __init__(self):self.weights=np.load(ROOT/'derived/readout.npz')['weights']
    def predict(self,mn):
        code=(np.asarray(mn)-75)@self.weights
        direction=code[:6:2]-code[1:6:2];presence=float(np.linalg.norm(direction))
        speed=.08*np.clip((.65-code[6])/.25,0,1)
        active=presence>.2
        velocity=speed*direction/max(presence,1e-12) if active else np.zeros(3)
        return velocity,active,code

if __name__=='__main__':calibrate()
