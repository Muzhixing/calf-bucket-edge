"""V2 JSON calibration loader; all original geometric values kept in immutable profiles."""
import cv2
import numpy as np
from tools.calibration_io import load

loaded=load()
CALIBRATION_PROFILE=loaded['profile']
CALIBRATION_FILE=str(loaded['path'])
CALIBRATION_SHA256=loaded['sha256']
m=loaded['matrices']
left_camera_matrix,left_distortion=m['K1'],m['D1']
right_camera_matrix,right_distortion=m['K2'],m['D2']
R,T=m['R'],m['T']
R1,R2,P1,P2,Q=[m[k] for k in ['R1','R2','P1','P2','Q']]
image_size=(1280,720)
validPixROI1,validPixROI2=loaded['valid_rois']
left_map1,left_map2=cv2.initUndistortRectifyMap(left_camera_matrix,left_distortion,R1,P1,image_size,cv2.CV_16SC2)
right_map1,right_map2=cv2.initUndistortRectifyMap(right_camera_matrix,right_distortion,R2,P2,image_size,cv2.CV_16SC2)
print(f'CALIBRATION_LOADED={CALIBRATION_FILE} SHA256={CALIBRATION_SHA256} profile={CALIBRATION_PROFILE} eye_size=1280x720 SBS=2560x720 units=mm',flush=True)
