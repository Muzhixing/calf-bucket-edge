"""One-time board asset preparation; no Python interpreter remains in live use."""
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
v2=Path(sys.argv[1] if len(sys.argv)>1 else '/home/elf/deploy/vision_v2').resolve()
sys.path[:0]=[str(v2/'tools'),str(v2)]
from verify import verify
verify()
from app.infrastructure import camera_config as c
import cv2
from PIL import Image,ImageDraw,ImageFont
assets=ROOT/'assets';assets.mkdir(exist_ok=True)
model=v2/json.loads((v2/'models.json').read_text())[0]['path']
shutil.copyfile(model,assets/'seed2_REG_FP16.rknn')
shutil.copyfile(v2/'calibrations/v2_radial2.json',assets/'v2_radial2.json')
assert hashlib.sha256((assets/'v2_radial2.json').read_bytes()).hexdigest()=='21c1c049edc92503f26fee3ce0ba9e70984db5155f48ded53b67002eb976a12a'
f=cv2.FileStorage(str(assets/'calibration.yml'),cv2.FILE_STORAGE_WRITE)
for key,mat in [('K1',c.left_camera_matrix),('D1',c.left_distortion),('K2',c.right_camera_matrix),('D2',c.right_distortion),('R1',c.R1),('R2',c.R2),('P1',c.P1),('P2',c.P2),('T',c.T)]:f.write(key,mat)
f.release()
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
