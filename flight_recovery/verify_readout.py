"""Independent random mixture/impulse witness for the corrected MN readout."""
from pathlib import Path
import json, numpy as np
from neural import Brain,Readout,ROOT

def verify():
    b=Brain();policy=Readout();rng=np.random.default_rng(18102026);records=[]
    for case in range(12):
        b.reset();direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        proxy=np.r_[np.maximum(np.column_stack([direction,-direction]).ravel(),0),rng.uniform(.1,.6)]
        if case%4==0:proxy[:]=0
        errors=[];angles=[]
        for t in range(35):
            x=rng.uniform(0,1,len(b.channels));x[:7]=proxy
            mn=b.step(x);_,active,code=policy.predict(mn)
            if t>=15:
                errors.append(float(np.max(abs(code-proxy))))
                got=code[:6:2]-code[1:6:2]
                if proxy.any():angles.append(float(np.degrees(np.arccos(np.clip((got@direction)/np.linalg.norm(got),-1,1)))))
        records.append({'case':case,'proxy_present':bool(proxy.any()),'max_channel_error':max(errors),'max_direction_error_deg':max(angles) if angles else None,'final_active':bool(active)})
    result={'seed':18102026,'training_cases_reused':False,'nuisance':'Independent uniformly changing input on every non-proxy channel at every 20ms step','cases':records,'all_pass':all(r['max_channel_error']<.02 and (r['max_direction_error_deg'] is None or r['max_direction_error_deg']<3) and r['final_active']==r['proxy_present'] for r in records)}
    (ROOT.parent/'results/flight_recovery_2026-10-18/independent_readout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['all_pass']:raise RuntimeError('Independent mixed sensory readout check failed')

if __name__=='__main__':verify()
