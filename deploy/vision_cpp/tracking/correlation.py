"""Optional OpenCV CSRT proposal, never a visibility authority.

Explicitly enabled by the runtime's --csrt flag, or in matched replay.
Only runs after all existing current-frame tracking paths reject. Each CSRT
proposal still passes the frozen tracker's identity/pair/geometry checks.
"""
import cv2
import numpy as np


def install(vision):
    if getattr(vision.TargetTracker,'csrt_enabled',False):
        return vision
    if not hasattr(cv2, 'TrackerCSRT_create'):
        raise RuntimeError('csrt_unavailable')
    Base = vision.TargetTracker

    class CsrtTracker(Base):
        csrt_enabled=True
        def __init__(self, image, box, stamp):
            super().__init__(image, box, stamp)
            self.csrt = None
            self.csrt_previous_image = image
            self.csrt_attempts = 0

        def recover(self,image,stamp,detections=()):
            result=super().recover(image,stamp,detections)
            if result is not None:
                # A recovered box belongs to this frame, not the pre-loss image.
                self.csrt_previous_image=image
                self.csrt=None
            return result

        def update(self, image, stamp, detections=(),flow_only=False):
            previous_stamp = self.stamp
            try:
                result = super().update(image, stamp, detections,flow_only=flow_only)
                self.csrt_previous_image = image
                self.csrt = None
                return result
            except RuntimeError as original:
                if flow_only:raise
                if not 0 < stamp-previous_stamp <= .20:
                    self.csrt = None
                    raise original
                self.csrt_attempts += 1
                try:
                    if self.csrt is None:
                        params = cv2.TrackerCSRT_Params()
                        params.template_size = 128.
                        params.number_of_scales = 17
                        self.csrt = cv2.TrackerCSRT_create(params)
                        x1,y1,x2,y2 = np.rint(self.box/2).astype(int)
                        self.csrt.init(cv2.resize(self.csrt_previous_image,(640,360)),
                                       (int(x1),int(y1),int(x2-x1),int(y2-y1)))
                    ok, rect = self.csrt.update(cv2.resize(image,(640,360)))
                    if not ok:
                        raise RuntimeError('csrt_no_match')
                    x,y,w,h = rect
                    box = np.array([2*x,2*y,2*(x+w),2*(y+h)],float)
                    ratio = (box[2:]-box[:2])/(self.box[2:]-self.box[:2])
                    if not vision.valid_box(box) or np.any((ratio<.8)|(ratio>1.25)):
                        raise RuntimeError('csrt_geometry')
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    # CSRT's half-size integer rectangle can be several source
                    # pixels off. Refine only near its proposal using the last
                    # accepted appearance; identity still uses the reference bank.
                    candidates=[]
                    for size in (box[2:]-box[:2],self.box[2:]-self.box[:2]):
                        w,h=np.rint(size).astype(int)
                        bx,by=np.rint((box[:2]+box[2:]-size)/2).astype(int)
                        sx,sy=max(0,bx-8),max(0,by-8)
                        patch=gray[sy:min(720,by+h+8),sx:min(1280,bx+w+8)]
                        if patch.shape[0]<h or patch.shape[1]<w:continue
                        template=cv2.resize(self.previous_aligned,(w,h))
                        weights=cv2.resize(self.weights,(w,h))
                        response=vision.weighted_response(patch,template,weights)
                        _,score,_,pos=cv2.minMaxLoc(response)
                        if score<.70:continue
                        x,y=sx+pos[0],sy+pos[1]
                        refined=[x,y,x+w,y+h]
                        if not vision.valid_box(refined):continue
                        transform=self.box_transform(refined)
                        _,aligned,fixed,recent=self.alignment(gray,transform)
                        pair=vision.weighted_ncc(aligned,self.previous_aligned,self.weights)
                        if fixed>=.62 and recent>=.76 and pair>=.70:
                            candidates.append((min(recent,pair),transform))
                    if not candidates:
                        raise RuntimeError('csrt_identity')
                    transform=max(candidates,key=lambda x:x[0])[1]
                    result = self.accept(gray,transform,stamp,'csrt',detections)
                    self.csrt_previous_image = image
                    return result
                except (RuntimeError,cv2.error) as exc:
                    self.last_rejections.append({'stage':'csrt','reason':str(exc)})
                    self.csrt = None
                    raise original

    vision.TargetTracker = CsrtTracker
    return vision
