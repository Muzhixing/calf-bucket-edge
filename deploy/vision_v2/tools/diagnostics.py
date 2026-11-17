"""Read-only audit of the frozen range calculation; never changes its return value."""
import cv2
import numpy as np


def range_audit(service, calib, disparity, left, top, right, bottom):
    if disparity is None:
        return dict(value_m=None, attempts=[])
    h,w=disparity.shape
    attempts=[]
    params=[('strict',service.min_valid_ratio,service.max_std_ratio,service.min_median_disp,service.roi_expand_ratio,None),
            ('relaxed',service.relax_valid_ratio,service.relax_max_std_ratio,service.relax_min_median_disp,max(service.roi_expand_ratio,0.35),0.6)]
    for name,ratio,stdmax,dispmin,expand,crop in params:
        bw,bh=max(1,right-left),max(1,bottom-top)
        dx,dy=int(bw*expand),int(bh*expand)
        x1,x2=max(0,min(left-dx,w-1)),max(0,min(right+dx,w-1))
        y1,y2=max(0,min(top-dy,h-1)),max(0,min(bottom+dy,h-1))
        expanded=[int(x1),int(y1),int(x2),int(y2)]
        if crop:
            rw,rh=x2-x1,y2-y1
            x1,x2=x1+int(rw*(.5-crop/2)),x1+int(rw*(.5+crop/2))
            y1,y2=y1+int(rh*(.5-crop/2)),y1+int(rh*(.5+crop/2))
        roi=disparity[y1:y2,x1:x2]
        a=dict(branch=name,expanded_xyxy=expanded,sample_xyxy=[int(x1),int(y1),int(x2),int(y2)],value_m=None)
        attempts.append(a)
        if roi.size==0: a['reason']='empty'; continue
        valid=roi[roi>0]; valid=valid[np.isfinite(valid)]
        a['valid_ratio']=float(valid.size/roi.size)
        if valid.size<roi.size*ratio: a['reason']='valid_ratio'; continue
        lo,hi=np.percentile(valid,[10,90]); valid=valid[(valid>=lo)&(valid<=hi)]
        if valid.size<roi.size*(ratio*.6): a['reason']='percentile_count'; continue
        med=float(np.median(valid)); mad=float(np.median(np.abs(valid-med)))+1e-6
        inliers=valid[np.abs(valid-med)<=3*1.4826*mad]
        if inliers.size<roi.size*(ratio*.6): a['reason']='mad_count'; continue
        med=float(np.median(inliers)); std=float(np.std(inliers))
        a.update(median_disparity_px=med,std_disparity_px=std,inlier_count=int(inliers.size))
        if med<dispmin or std/max(1e-6,med)>stdmax: a['reason']='disparity_quality'; continue
        baseline_m=float(np.linalg.norm(calib.T))/1000
        z=float(calib.P1[0,0])*baseline_m/med
        if not service.distance_min_m<=z<=service.distance_max_m: a['reason']='range_limit'; continue
        a.update(value_m=z,reason='accepted')
        return dict(value_m=z,chosen_branch=name,attempts=attempts)
    return dict(value_m=None,chosen_branch=None,attempts=attempts)


def overlay(image,row,reference=None):
    im=image.copy(); h,w=im.shape[:2]
    for x in (588.8,640,691.2): cv2.line(im,(round(x),0),(round(x),h-1),(180,180,180),1)
    s=row.get('selected')
    if s:
        cv2.rectangle(im,(s['left'],s['top']),(s['right'],s['bottom']),(0,220,0),2)
        cv2.drawMarker(im,(round(s['center_x']),round((s['top']+s['bottom'])/2)),(0,220,0),cv2.MARKER_CROSS,22,2)
    for a in (row.get('range_attempt') or {}).get('attempts',[]):
        x1,y1,x2,y2=a['sample_xyxy']; cv2.rectangle(im,(x1,y1),(x2,y2),(0,170,255),2 if a['value_m'] is not None else 1)
        cv2.putText(im,a['branch']+':'+a['reason'],(max(0,x1),max(95,y1-6)),0,.5,(0,170,255),1)
    if reference:
        cv2.drawMarker(im,(round(reference['mouth_x_px']),round(reference['mouth_y_px'])),(255,0,255),cv2.MARKER_TILTED_CROSS,24,2)
    z=row.get('distance_m'); ztext=f'{z:.4f}' if z is not None else 'NA'
    texts=[f"V2 frame {row['frame_id']} | green=BUCKET BOX center",
           f"Z={ztext}m [{row['measurement_status']}] source frame={row.get('distance_sample_frame')}",
           f"centered={row.get('centered')} Z<=0.50={row.get('close_enough')} | orange=range sample ROI"]
    for i,t in enumerate(texts):
        cv2.putText(im,t,(10,24+i*24),0,.57,(0,0,0),3); cv2.putText(im,t,(10,24+i*24),0,.57,(255,255,255),1)
    return im
