"""Shared OpenCV spectra for the existing weighted NCC, same search grid."""
import cv2,numpy as np
from collections import OrderedDict

class SpectrumCache:
    """Bounded kernels only; current image spectra are never cached.

    The tracker replaces rather than mutates its appearance references/weights.
    bind() retains their objects and invalidates all kernels on replacement.
    """
    def __init__(self,max_bytes=32*1024*1024):
        self.max_bytes=max_bytes;self.bytes=0;self.entries=OrderedDict();self.source=None;self.weights=None

    def bind(self,source,weights):
        if source is not self.source or weights is not self.weights:
            self.entries.clear();self.bytes=0;self.source=source;self.weights=weights

    def get(self,key):
        value=self.entries.get(key)
        if value is not None:self.entries.move_to_end(key)
        return value

    def put(self,key,value):
        size=value[2].nbytes+value[3].nbytes
        if size>self.max_bytes:return
        while self.entries and (self.bytes+size>self.max_bytes or len(self.entries)>=8):
            _,old=self.entries.popitem(last=False);self.bytes-=old[2].nbytes+old[3].nbytes
        self.entries[key]=value;self.bytes+=size


def weighted_response(search,template,weights,cache=None):
    s=search.astype(np.float32);t=template.astype(np.float32);w=weights.astype(np.float32)
    total=float(w.sum());center=t-float((t*w).sum()/total)
    variance=float((center*center*w).sum())
    oh,ow=np.subtract(s.shape,t.shape)+1
    # Near-flat patches amplify cancellation in the variance subtraction;
    # retain the original backend there instead of changing acceptance margins.
    if min(t.shape)>=64 and max(oh,ow)<=40 and variance>=4*total:
        shape=tuple(cv2.getOptimalDFTSize(int(v)) for v in s.shape)
        def spectrum(a):
            padded=np.zeros(shape,np.float32);padded[:a.shape[0],:a.shape[1]]=a
            return cv2.dft(padded,nonzeroRows=a.shape[0])
        key=(shape,t.shape)
        kernels=cache.get(key) if cache is not None else None
        if kernels is None:
            kernels=(total,variance,spectrum(center*w),spectrum(w))
            if cache is not None:cache.put(key,kernels)
        total,variance,cf,wf=kernels
        sf=spectrum(s)
        def correlate(a,b):
            return cv2.idft(cv2.mulSpectrums(a,b,0,conjB=True),flags=cv2.DFT_SCALE,nonzeroRows=int(oh))[:oh,:ow]
        numerator=correlate(sf,cf)
        sums=correlate(sf,wf)
        squares=correlate(spectrum(s*s),wf)
    else:
        numerator=cv2.matchTemplate(s,center*w,cv2.TM_CCORR)
        sums=cv2.matchTemplate(s,w,cv2.TM_CCORR)
        squares=cv2.matchTemplate(s*s,w,cv2.TM_CCORR)
    denom=np.sqrt(np.maximum(squares-sums*sums/total,0)*max(variance,1e-6))
    result=np.full(numerator.shape,-1.,np.float32)
    np.divide(numerator,denom,out=result,where=denom>1.)
    return np.clip(result,-1,1)
