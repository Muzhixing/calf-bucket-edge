import json
from support import ROOT,sha

def verify():
 model=json.loads((ROOT/'models.json').read_text());assert len(model)==1 and model[0]['name']=='seed2_REG_FP16'
 assert model[0]['sha256']==sha(ROOT/model[0]['path'])=='bb3d9d9deacf3f099232414139040eae81ea3c362976cbf3a7a13af5cca06c4d'
 original=json.loads((ROOT/'config/model_bindings.json').read_text());assert original[model[0]['name']]['sha256']==model[0]['sha256']
 manifest=json.loads((ROOT/'config/runtime_hashes.json').read_text())
 for p,h in manifest.items():assert sha(ROOT/p)==h,('V2 runtime changed',p)
 # Original detector/ranger unchanged. Only the new loader supplies geometry.
 original=json.loads((ROOT/'config/original_core_hashes.json').read_text())
 for p,h in original.items():
  if p!='app/infrastructure/camera_config.py':assert sha(ROOT/p)==h,('unexpected baseline algorithm change',p)
 print('VERIFIED_V2',len(manifest),'files; frozen model and original detector/ranger',flush=True)
 return manifest
if __name__=='__main__':verify()
