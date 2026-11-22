"""Timestamped constant-velocity estimation using OpenCV's Kalman implementation.

The filter estimates motion; it never establishes visibility or emits arm targets.
Association follows ByteTrack's high-score first, low-score continuation principle.
See THIRD_PARTY.md for upstream algorithms and scope of reuse.
"""
import cv2
import numpy as np


class MotionFilter:
    def __init__(self, dimensions=1, measurement_std=.12, acceleration_std=35., gate_floor=.8):
        self.n=dimensions;self.noise=measurement_std;self.accel=acceleration_std;self.floor=gate_floor
        self.kf=cv2.KalmanFilter(2*dimensions,dimensions,0,cv2.CV_64F)
        self.kf.measurementMatrix=np.c_[np.eye(dimensions),np.zeros((dimensions,dimensions))]
        self.stamp=None;self.started=None;self.count=0;self.rejected=None

    def reset(self,value,stamp):
        value=np.asarray(value,dtype=float).reshape(self.n)
        self.kf.statePost=np.r_[value,np.zeros(self.n)].reshape(-1,1)
        self.kf.errorCovPost=np.diag([self.noise**2]*self.n+[100*self.noise**2]*self.n)
        self.stamp=self.started=stamp;self.count=1;self.rejected=None
        return True

    def update(self,value,stamp,quality=1.):
        value=np.asarray(value,dtype=float).reshape(self.n)
        if not np.isfinite(value).all() or not np.isfinite(stamp):raise ValueError('invalid_motion_measurement')
        if self.stamp is None:return self.reset(value,stamp)
        dt=stamp-self.stamp
        if dt<=0:return False
        if dt>.35:return self.reset(value,stamp)
        identity=np.eye(self.n)
        self.kf.transitionMatrix=np.block([[identity,identity*dt],[np.zeros_like(identity),identity]])
        g=np.r_[identity*dt*dt/2,identity*dt]
        self.kf.processNoiseCov=(g@g.T)*self.accel**2+np.eye(2*self.n)*1e-8
        variance=(self.noise/max(.4,min(1.,quality)))**2
        self.kf.measurementNoiseCov=identity*variance
        predicted=self.kf.predict();innovation=value-predicted[:self.n,0]
        covariance=self.kf.errorCovPre[:self.n,:self.n]+identity*variance
        distance=float(innovation@np.linalg.solve(covariance,innovation))
        self.stamp=stamp
        if distance>16 and np.linalg.norm(innovation)>self.floor:
            if self.rejected is not None and np.linalg.norm(value-self.rejected)<self.floor:
                return self.reset(value,stamp)  # Consistent changed observation: restart without momentum.
            self.rejected=value.copy();self.count=0
            return False
        self.kf.correct(value.reshape(-1,1));self.rejected=None;self.count+=1
        return True

    @property
    def position(self):return self.kf.statePost[:self.n,0].copy()

    @property
    def velocity(self):return self.kf.statePost[self.n:,0].copy()

    def reliable_velocity(self,maximum):
        if self.stamp is None or self.count<4 or self.stamp-self.started<.10:return np.zeros(self.n)
        uncertainty=np.sqrt(np.maximum(0,np.diag(self.kf.errorCovPost)[self.n:]))
        speed=self.velocity
        return np.where(np.abs(speed)>np.maximum(1.5*uncertainty,maximum*.04),np.clip(speed,-maximum,maximum),0.)

    def predict_position(self,stamp,max_lead=.15,max_displacement=80):
        dt=max(0.,min(max_lead,stamp-self.stamp)) if self.stamp is not None else 0.
        return self.position+np.clip(self.velocity*dt,-max_displacement,max_displacement)
