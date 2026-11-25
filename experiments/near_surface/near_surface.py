"""Experimental near-surface stereo estimator. No camera, serial or motion I/O.

Accepts rectified stereo frames and a bucket box. Reports the median depth of
textured surface pixels, never bucket-mouth clearance. Not enabled in production.
"""
import cv2
import numpy as np

class NearSurface:
    def __init__(self,factor_m_px):
        if not np.isfinite(factor_m_px) or factor_m_px<=0:raise ValueError('invalid_stereo_factor')
        self.factor=float(factor_m_px);self.scale=.5;self.range_full=256
        def make(minimum):
            return cv2.StereoSGBM_create(minimum,128,7,8*3*49,32*3*49,1,63,12,120,2,cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left=make(0);self.right=make(-128)
    def measure(self,left,right,box):
        result={'valid':False,'distance_m':None,'distance_kind':'camera_forward_Z_textured_bucket_surface',
                'search_range_full_px':256,'nearest_searchable_Z_m':self.factor/256,'experimental':True}
        if left.shape!=right.shape or left.shape[:2]!=(720,1280):raise ValueError('invalid_rectified_shape')
        x1,y1,x2,y2=map(float,box)
        if not all(np.isfinite(v) for v in box) or not 0<=x1<x2<=1280 or not 0<=y1<y2<=720:raise ValueError('invalid_box')
        def grey(a):return cv2.cvtColor(a,cv2.COLOR_BGR2GRAY) if a.ndim==3 else a
        l,r=[cv2.resize(grey(a),(640,360),interpolation=cv2.INTER_AREA) for a in (left,right)]
        dl=self.left.compute(l,r).astype(np.float32)/16
        dr=self.right.compute(r,l).astype(np.float32)/16
        yy,xx=np.indices(dl.shape,dtype=np.float32);rx=xx-dl
        reverse=cv2.remap(dr,rx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-129)
        f=l.astype(np.float32);mean=cv2.boxFilter(f,-1,(9,9));variance=np.maximum(0,cv2.boxFilter(f*f,-1,(9,9))-mean*mean)
        texture=variance>=36
        valid=(dl>2)&(dl<126)&(rx>=0)&(rx<639)&(reverse>-128)&(np.abs(dl+reverse)<=1)&texture
        # Avoid the rim, the basket and background outside the detected bucket.
        w,h=x2-x1,y2-y1
        roi=[int((x1+.18*w)/2),int((y1+.28*h)/2),int((x1+.82*w)/2),int((y1+.85*h)/2)]
        a,b,c,d=roi
        if c-a<12 or d-b<12:result['reason']='target_too_small';return result
        m=valid[b:d,a:c];area=m.size;coverage=float(m.mean());texture_fraction=float(texture[b:d,a:c].mean())
        result.update(roi_full_px=[2*v for v in roi],surface_support_fraction=coverage,texture_fraction=texture_fraction,support_pixels=int(m.sum()))
        if texture_fraction<.12:result['reason']='insufficient_surface_texture';return result
        if coverage<.12 or m.sum()<200:result['reason']='insufficient_consistent_stereo';return result
        z=self.factor/(dl[b:d,a:c][m]/self.scale)
        median=np.median(z);mad=np.median(abs(z-median));keep=abs(z-median)<=max(.005,3*1.4826*mad);z=z[keep]
        if len(z)<.1*area:result['reason']='insufficient_depth_consensus';return result
        p10,median,p90=np.quantile(z,[.1,.5,.9]);result['depth_p10_median_p90_m']=[float(p10),float(median),float(p90)]
        if (p90-p10)/median>.15:result['reason']='ambiguous_surface_depth';return result
        if not .19<=median<=1.2:result['reason']='outside_provisional_working_range';return result
        result.update(valid=True,distance_m=float(median),reason='textured_surface_consistent')
        return result
