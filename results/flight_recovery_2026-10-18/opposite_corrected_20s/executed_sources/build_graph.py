"""Rebuild the annotated-endpoint MaleCNS graph, without thresholding contacts."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'flight_recovery/derived'

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    src=BASE/'data/malecns'
    ann=pd.read_feather(src/'body-annotations-male-cns-v1.0-minconf-0.5.feather').sort_values('bodyId').reset_index(drop=True)
    ids=ann.bodyId.to_numpy(np.int64);index=pd.Index(ids)
    nt=pd.read_feather(src/'body-neurotransmitters-male-cns-v1.0.feather',columns=['body','consensus_nt']).drop_duplicates('body').set_index('body').consensus_nt
    # Preserve the previous candidate's declared NT rule. Histamine is not
    # modelled in that rule; zero here is an omission, not biological silence.
    sign=ann.bodyId.map(nt).map({'acetylcholine':1,'gaba':-1,'glutamate':-1,'dopamine':1,'octopamine':1,'serotonin':1}).fillna(0).to_numpy(np.float64)
    typ=ann['type'].fillna('');sc=ann.superclass.fillna('')
    motor=np.flatnonzero(sc.isin(['vnc_motor','cb_motor']))
    food=np.flatnonzero(typ.isin(['ORN_DM1','ORN_DM2','ORN_DM3','ORN_DM4']) & ann.rootSide.isin(['L','R']))
    sensory=np.flatnonzero(sc.str.contains('sensory'))
    started=time.perf_counter();rr=[];cc=[];vv=[];total=0;retained=0
    source=src/'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
    with pa.memory_map(str(source),'r') as mmap:
        reader=pa.ipc.open_file(mmap)
        for k in range(reader.num_record_batches):
            b=reader.get_batch(k);pre=index.get_indexer(b.column(b.schema.get_field_index('body_pre')).to_numpy());post=index.get_indexer(b.column(b.schema.get_field_index('body_post')).to_numpy());weight=b.column(b.schema.get_field_index('weight')).to_numpy()
            keep=(pre>=0)&(post>=0);rr.append(post[keep].astype(np.int32));cc.append(pre[keep].astype(np.int32));vv.append(weight[keep].astype(np.int32));total+=len(pre);retained+=int(keep.sum())
            if k%400==0:print(json.dumps({'batch':k,'rows':total,'kept':retained}),flush=True)
    counts=sp.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(ids),len(ids)))
    del rr,cc,vv
    sp.save_npz(OUT/'contacts.npz',counts)
    incoming=np.asarray(counts.sum(axis=1)).ravel()
    operator=counts.astype(np.float64);operator.data*=sign[operator.indices];operator.eliminate_zeros()
    operator=sp.diags(.9/np.maximum(incoming,1))@operator
    sp.save_npz(OUT/'operator.npz',operator)
    np.savez_compressed(OUT/'binding.npz',ids=ids,sign=sign,motor=motor,food=food,sensory=sensory,incoming=incoming)
    ann.to_parquet(OUT/'annotations.parquet',index=False)
    report={'source':'FlyEM MaleCNS v1.0 public flat connectome, CC-BY','nodes':len(ids),'raw_contact_rows':total,'annotated_pairs':counts.nnz,'signed_pairs':operator.nnz,'synapses':int(counts.sum()),'motor_neurons':len(motor),'food_ORNs':len(food),'all_annotated_sensory':len(sensory),'graph_orientation':'row postsynaptic, column presynaptic','gain':.9,'normalization':'total unsigned incoming synapses','histamine_sign':0,'histamine_note':'Not modelled by the previous candidate NT rule; biological inhibition is not being denied','seconds':time.perf_counter()-started,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.feather')}}
    (OUT/'graph.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':build()
