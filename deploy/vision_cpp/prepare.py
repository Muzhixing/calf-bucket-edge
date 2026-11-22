"""One-time board asset preparation; no Python interpreter remains in live use."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
v2=Path(sys.argv[1] if len(sys.argv)>1 else '/home/elf/deploy/vision_v2').resolve()
sys.path[:0]=[str(v2/'tools'),str(v2)]
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
assets=ROOT/'assets';assets.mkdir(exist_ok=True)
models=json.loads((v2/'models.json').read_text())
if len(models)!=1 or models[0]['name']!='seed2_REG_FP16':
    raise RuntimeError('Expected exactly the frozen V1 model')
model=v2/models[0]['path']
model_sha=hashlib.sha256(model.read_bytes()).hexdigest()
original_binding=json.loads((v2/'config/model_bindings.json').read_text())
if model_sha!=models[0]['sha256'] or model_sha!=original_binding['seed2_REG_FP16']['sha256'] or model_sha!='bb3d9d9deacf3f099232414139040eae81ea3c362976cbf3a7a13af5cca06c4d':
    raise RuntimeError('Frozen model binding mismatch')
shutil.copyfile(model,assets/'seed2_REG_FP16.rknn')
selection=ROOT/'calibration_source.json'
selected=json.loads(selection.read_text())
source=(ROOT/selected['path']).resolve()
if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest()!=selected['sha256']:
    raise RuntimeError('Selected calibration source/hash mismatch')
data=json.loads(source.read_text())
if data['units']!='mm' or data['image_size']!=[1280,720]:
    raise RuntimeError('Selected calibration units/resolution mismatch')
matrices={key:np.array(data[key],np.float64) for key in ('K1','D1','K2','D2','R','T','R1','R2','P1','P2','Q')}
if any(not np.isfinite(value).all() for value in matrices.values()):
    raise RuntimeError('Nonfinite calibration')
rect=cv2.stereoRectify(matrices['K1'],matrices['D1'],matrices['K2'],matrices['D2'],(1280,720),matrices['R'],matrices['T'])
for key,value in zip(('R1','R2','P1','P2','Q'),rect[:5]):
    if not np.allclose(matrices[key],value,rtol=0,atol=1e-5):
        raise RuntimeError('Selected rectification mismatch: '+key)
shutil.copyfile(source,assets/source.name)
binding=dict(profile=selected['profile'],source=str(source),source_sha256=selected['sha256'])
# Remove only the obsolete generated calibration copy after validating the new source.
(assets/'v2_radial2.json').unlink(missing_ok=True)
f=cv2.FileStorage(str(assets/'calibration.yml'),cv2.FILE_STORAGE_WRITE)
for key in ('K1','D1','K2','D2','R1','R2','P1','P2','T'):f.write(key,matrices[key])
f.release()
binding['yaml_sha256']=hashlib.sha256((assets/'calibration.yml').read_bytes()).hexdigest()
(assets/'calibration_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',21,index=2)
base=Image.new('RGB',(1280,720),(17,24,24));d=ImageDraw.Draw(base)
for xy,text in [((16,2),'桶身距离'),((248,2),'检测 / 对齐'),((524,2),'距离状态'),((16,78),'校正左目'),((928,4),'原始左目'),((928,235),'原始右目'),((928,469),'最近有效视差'),((1021,585),'暂无有效视差'),((16,624),'桶口中心未独立定位 · 圆点为桶框中心'),((16,657),'橙线框为距离来源帧的采样区')]:
    d.text(xy,text,font=font,fill=(191,210,204))
base.save(assets/'hdmi_base.png')
for i,text in enumerate(['未检出桶','待对齐','两轴已对齐','新测','沿用','无有效值']):
    im=Image.new('RGB',(175,38),(17,24,24));ImageDraw.Draw(im).text((0,3),text,font=font,fill=(229,239,235));im.save(assets/f'label{i}.png')
shutil.copyfile(v2/'tools/preview.html',assets/'preview.html')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'calibration.sha256').write_text(sha(assets/'calibration.yml')+'\n')
(assets/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in assets.iterdir() if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
(ROOT/'runs').mkdir(exist_ok=True)
print('PREPARED',ROOT)
