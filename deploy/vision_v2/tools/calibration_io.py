"""Pinned calibration registry and coordinate transforms through original left pixels."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(profile=None):
 import numpy as np,cv2
 registry=json.loads((ROOT/'active_calibration.json').read_text());profile=profile or os.environ.get('BUCKET_CALIBRATION_PROFILE',registry['active'])
 assert profile in registry['profiles'];record=registry['profiles'][profile];path=(ROOT/record['path']).resolve();assert path.is_relative_to(ROOT)
 digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==record['sha256'],'calibration hash changed'
 c=json.loads(path.read_text());assert c['image_size']==[1280,720] and c['units']=='mm'
 matrices={k:np.ascontiguousarray(c[k],dtype=np.float64) for k in ['K1','D1','K2','D2','R','T','R1','R2','P1','P2','Q']}
 matrices['T']=matrices['T'].reshape(3,1);matrices['D1']=matrices['D1'].reshape(-1,1);matrices['D2']=matrices['D2'].reshape(-1,1)
 check=cv2.stereoRectify(matrices['K1'],matrices['D1'],matrices['K2'],matrices['D2'],(1280,720),matrices['R'],matrices['T'])
 for key,value in zip(['R1','R2','P1','P2','Q'],check[:5]):assert np.allclose(matrices[key],value,rtol=0,atol=1e-5),(profile,key,'stored/recomputed geometry differs')
 return dict(profile=profile,path=path,sha256=digest,data=c,matrices=matrices,valid_rois=check[-2:])

def raw_to_rectified(xy,profile):
 import cv2,numpy as np
 m=load(profile)['matrices'];p=np.array(xy,np.float64).reshape(-1,1,2)
 return cv2.undistortPoints(p,m['K1'],m['D1'],R=m['R1'],P=m['P1']).reshape(-1,2)

def rectified_to_raw(xy,profile):
 import cv2,numpy as np
 m=load(profile)['matrices'];p=np.array(xy,np.float64).reshape(-1,2)
 rays=np.column_stack([p,np.ones(len(p))])@np.linalg.inv(m['P1'][:,:3]).T
 rays=rays@m['R1'];points,_=cv2.projectPoints(rays,np.zeros(3),np.zeros(3),m['K1'],m['D1'])
 return points.reshape(-1,2)

def reference_for_profile(reference,profile):
 out=dict(reference);points=raw_to_rectified([reference['raw_left_x_px'],reference['raw_left_y_px']],profile)[0]
 out.update(mouth_x_px=float(points[0]),mouth_y_px=float(points[1]),calibration_profile=profile,calibration_sha256=load(profile)['sha256'],mapping='same independent point via raw-left ray; not a prediction or additional annotation')
 for key in ['bucket_polygon_raw','body_polygon_raw','rim_polygon_raw']:
  if reference.get(key):out[key.replace('_raw','_rectified')]=raw_to_rectified(reference[key],profile).tolist()
 return out
