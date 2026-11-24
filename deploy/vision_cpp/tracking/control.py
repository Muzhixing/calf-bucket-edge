"""Timestamped image-based servoing; no target-stationary assumption."""
import math
from collections import deque
from motion import MotionFilter
from response import response_angles


def clamp(x, low, high):
    return max(low, min(high, x))


def alignment_radius(target):
    """Circular tolerance: radius is 10% of the tracked box's shorter side."""
    x1,y1,x2,y2=target['box']
    return max(1.,.10*min(x2-x1,y2-y1))


class AlignmentGate:
    """Latch a true stop inside the circle; fresh outer observations release it."""
    def __init__(self):self.reset()

    def reset(self):
        self.holding=False;self.radius=0.;self.stamp=-float('inf');self.outside_since=None
        self.blocked=(False,False);self.constrained=False

    def update(self,target,position=None,limits=None,goal=None):
        stamp=target['timestamp']
        if stamp<=self.stamp:return self.holding
        self.stamp=stamp
        dx,dy=target['offset_px'];blocked=(False,False)
        if position is not None and limits is not None:
            # Actual limit telemetry is required; an in-flight boundary target
            # alone must not suppress a still-correctable error.
            a,b=position[:2];ga,gb=(goal or position)[:2]
            blocked=((a<=limits.a_min+.01 and ga<=limits.a_min+.01 and dx>0) or
                     (a>=limits.a_max-.01 and ga>=limits.a_max-.01 and dx<0),
                     (b<=limits.b_min+.01 and gb<=limits.b_min+.01 and dy<0) or
                     (b>=limits.b_max-.01 and gb>=limits.b_max-.01 and dy>0))
        if blocked!=self.blocked:
            self.holding=False;self.outside_since=None
        self.blocked=blocked
        distance=math.hypot(0 if blocked[0] else dx,0 if blocked[1] else dy)
        if not self.holding:
            self.radius=alignment_radius(target)
            if distance<=self.radius:self.holding=True;self.outside_since=None
        elif distance>1.5*self.radius:
            if self.outside_since is None:self.outside_since=stamp
            elif stamp-self.outside_since>=.06:self.holding=False;self.outside_since=None
        else:self.outside_since=None
        self.constrained=self.holding and any(blocked) and math.hypot(dx,dy)>self.radius
        return self.holding

    def status(self):
        return {'alignment_hold':self.holding,'alignment_radius_px':self.radius,
                'alignment_resume_radius_px':1.5*self.radius,
                'alignment_blocked_axes':[axis for axis,blocked in zip(('A','B'),self.blocked) if blocked],
                'alignment_hold_mode':'range_limit' if self.constrained else 'center' if self.holding else 'following'}


class FollowPolicy:
    def __init__(self, limits, fx, fy, cx, cy, response_profile=None):
        if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
            raise ValueError('invalid_tracking_intrinsics')
        self.limits, self.fx, self.fy, self.cx, self.cy = limits, fx, fy, cx, cy
        # Rectified image center need not equal the calibrated principal point.
        self.center_ray = math.atan((640-cx)/fx)
        # Field-observed local response: about 20 pixels / protocol degree.
        # This empirical scale includes the arm's offset camera and servo ratio.
        # Do not assume the camera rotates about its own optical center.
        self.profile=response_profile;self.active_response=response_profile
        self._response_parameters()
        self.reset()

    def _response_parameters(self):
        r=self.active_response
        self.units_per_rad=self.fx*(1+((640-self.cx)/self.fx)**2)/(r.gain if r else 20.)
        self.feedback_delay=r.delay+r.tau if r else .14

    def configure_context(self,state,metadata,target):
        r=self.profile
        if r is not None and r.scope is not None:
            from vision import iou
            selected=metadata.get('selected')
            matched=selected is not None and iou(target['box'],[selected[k] for k in ('left','top','right','bottom')])>=.5
            if not matched or not r.eligible(state,metadata):r=None
        if r is not self.active_response:
            self.active_response=r;self._response_parameters();self.reset()

    def frame_angle(self,port,stamp):
        r=self.active_response
        entries=list(getattr(port,'history',()))
        if r is None or len(entries)<2 or stamp<entries[0][0]+r.delay+max(.1,3*r.tau):
            return port.angle_at(stamp-self.feedback_delay)
        return float(response_angles(entries,[stamp],r.delay,r.tau)[0])

    def reset(self):
        self.stamp = None; self.world = None; self.velocity = 0.
        self.last_send = None; self.command = None; self.rate = 0.
        self.diagnostics = {};self.observations=deque();self.filtered_world=None
        self.motion=MotionFilter()
        self.visual_motion=MotionFilter();self.visual_position=0.;self.previous_image_x=None
        self.horizontal_ready_since=None;self.last_b_slot=None

    def observe(self, target, frame_a):
        stamp = target['timestamp']
        if self.stamp == stamp:
            return
        dx = target['offset_px'][0]
        angle = math.atan((640+dx-self.cx)/self.fx)-self.center_ray
        world = frame_a-angle*self.units_per_rad
        dt = stamp-self.stamp if self.stamp is not None else 0
        # A clipped box's visible center shifts as more of the bucket enters.
        # Use its observed bearing, but do not interpret that crop change as
        # physical target velocity and add predictive lead to it.
        restart=(not 0.015 <= dt <= .35 or target.get('partial_visible',False)
                 or target.get('method') in ('reacquired','detector_bridge'))
        if restart:
            self.observations.clear();self.velocity=0.;self.filtered_world=world
            self.motion.reset([world],stamp)
        self.observations.append((stamp,world))
        while self.observations and stamp-self.observations[0][0]>.30:
            self.observations.popleft()
        accepted=True if restart else self.motion.update([world],stamp,target.get('quality',1.))
        self.velocity=float(self.motion.reliable_velocity(25.)[0]) if accepted else 0.
        source='telemetry';ego=target.get('ego_motion') or {}
        if not isinstance(ego,dict):ego={}
        prior=ego.get('previous_timestamp');confidence=ego.get('confidence')
        # Background evidence only changes velocity feed-forward. It cannot
        # create a target, alter current positional error, or bypass hold/stop.
        # Removing camera-induced velocity while keeping a mis-scaled positional
        # response can increase corrections. Until a scoped response is verified,
        # background motion is diagnostic only (see offline ablation report).
        if (self.active_response is not None and not restart and self.previous_image_x is not None and ego.get('valid') is True
                and isinstance(confidence,(int,float)) and math.isfinite(confidence) and confidence>=.60
                and ego.get('timestamp')==stamp and isinstance(prior,(int,float)) and math.isfinite(prior)
                and abs(prior-self.stamp)<1e-6):
            try:
                matrix=ego['matrix']
                if len(matrix)!=2 or any(len(row)!=3 for row in matrix) or not all(math.isfinite(v) for row in matrix for v in row):
                    raise ValueError('invalid_ego_transform')
                x,y=self.previous_image_x
                camera_x=matrix[0][0]*x+matrix[0][1]*y+matrix[0][2]
                if abs(camera_x-x)>1200*dt+2:raise ValueError('large_ego_transform')
                camera_angle=math.atan((camera_x-self.cx)/self.fx)-self.center_ray
                self.visual_position+=(camera_angle-angle)*self.units_per_rad
                if self.visual_motion.stamp is None:self.visual_motion.reset([self.visual_position],stamp)
                else:self.visual_motion.update([self.visual_position],stamp,target.get('quality',1.))
                self.velocity=float(self.visual_motion.reliable_velocity(25.)[0]);source='background'
            except (ValueError,TypeError,KeyError,OverflowError):
                self.visual_motion=MotionFilter();self.visual_position=0.
        else:self.visual_motion=MotionFilter();self.visual_position=0.
        self.previous_image_x=(640+dx,360+target['offset_px'][1])
        self.filtered_world=float(self.motion.position[0])
        self.world, self.stamp = world, stamp
        self.diagnostics = {'bearing_error_deg': math.degrees(angle),
                            'world_target_A': world, 'velocity_A_dps': self.velocity,
                            'frame_A_estimate': frame_a,'motion_measurement_accepted':accepted,
                            'motion_direction':1 if self.velocity>0 else -1 if self.velocity<0 else 0,
                            'velocity_source':source,'response_source':'profile' if self.active_response else 'empirical_default',
                            'effective_response_s':self.feedback_delay}

    def next_a(self, target, frame_a, current, now):
        self.observe(target, frame_a)
        dt = clamp(now-self.last_send, .01, .15) if self.last_send is not None else .08
        if self.command is None:
            self.command = float(current)
        # Only extrapolate an observed identity, and for a bounded fresh interval.
        lead = min(.25, max(0., now-self.stamp)+self.feedback_delay)
        predicted = self.filtered_world+clamp(self.velocity*lead, -1., 1.)
        desired = clamp(predicted, self.limits.a_min, self.limits.a_max)
        if abs(target['offset_px'][0]) <= 10 and abs(self.velocity) < 2 and abs(desired-current)<.55:
            desired = float(current)
        remaining=desired-self.command
        # Braking distance bounds the ramp before reaching a moving setpoint.
        rate_limit=min(30.,math.sqrt(2*180*abs(remaining)))
        wanted_rate=clamp(remaining/dt,-rate_limit,rate_limit)
        acceleration=360 if self.rate*wanted_rate<=0 or abs(wanted_rate)<abs(self.rate) else 180
        self.rate += clamp(wanted_rate-self.rate,-acceleration*dt,acceleration*dt)
        step = self.rate*dt
        if step*(desired-self.command) > 0 and abs(step) > abs(desired-self.command):
            step = desired-self.command; self.rate = 0.
        self.command = clamp(self.command+step, self.limits.a_min, self.limits.a_max)
        self.last_send = now
        result = int(clamp(round(self.command), max(self.limits.a_min,current-4),
                           min(self.limits.a_max,current+4)))
        self.diagnostics.update({'lookahead_s': lead, 'desired_A': desired,
                                 'command_A': result, 'rate_A_dps': self.rate})
        return result

    def next_b(self, target, current, metadata):
        dy = target['offset_px'][1]
        if abs(dy) <= 15:
            return int(round(current))
        z = metadata.get('distance_m')
        selected = metadata.get('selected')
        age = metadata.get('distance_age_ms')
        matched = False
        if selected:
            from vision import iou
            matched = iou(target['box'], [selected[k] for k in ('left','top','right','bottom')]) >= .5
        # A needs bearing, B needs translation. Use only a fresh, associated Z;
        # detection/depth gaps retain the field gain, never block A tracking.
        if matched and isinstance(z,(int,float)) and math.isfinite(z) and .15 <= z <= 3 and isinstance(age,(int,float)) and 0 <= age <= 350:
            gain = clamp(z*1000/self.fy, .25, 2.)
        else:
            gain = 1.
        correction = dy*gain
        # Large offsets use the geometric correction without the former 30%
        # under-command. Do not create extra travel just to reach cruise speed.
        factor = self.limits.b_far_gain if abs(correction)>=self.limits.b_fast_error else .7
        delta = clamp(round(correction*factor), -self.limits.b_max_step, self.limits.b_max_step)
        return int(clamp(round(current)+delta, self.limits.b_min, self.limits.b_max))

    def allow_b(self,target,now):
        """Independent vertical correction; A continues while B is moving."""
        return abs(target['offset_px'][1])>15

    def choose_axis(self,a,old_a,b_due,now):
        """A gets first slot, but cannot starve an independent B correction."""
        if self.last_b_slot is None:self.last_b_slot=now
        if b_due and (a==old_a or now-self.last_b_slot>=self.limits.b_schedule_interval):
            self.last_b_slot=now;return 1
        return 0
