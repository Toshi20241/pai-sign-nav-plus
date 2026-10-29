"""Image-only branch selection and a standard waypoint feedback controller."""
from __future__ import annotations

import math
def clip(x, lo, hi):
    return min(hi, max(lo, x))



def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def waypoint_command(x, y, theta, target, max_speed=.30):
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    if distance < .048:
        return 0., 0., True
    error = wrap_angle(math.atan2(dy, dx) - theta)
    omega = float(clip(3.2 * error, -1.35, 1.35))
    linear = min(max_speed, 1.0 * distance) * max(0., math.cos(error))
    if abs(error) > .75:
        linear = 0.
    return linear, omega, False


def alignment_command(theta, target_theta=math.pi/2):
    error = wrap_angle(target_theta-theta)
    return 0., float(clip(3.2*error, -1.2, 1.2)), abs(error) < .02


class DifferentialDrive:
    """No-slip planar wheel model; no rigid-body contact solver or slip model."""
    radius = .055
    axle = .29
    tau = .12
    acceleration_limit = 32.0

    def __init__(self, start, wheel_gain=(1., 1.)):
        self.x, self.y, self.theta = map(float, start)
        self.left = self.right = 0.
        self.gain_left, self.gain_right = wheel_gain
        self.path_length = 0.
        self.max_observed_acceleration = 0.

    def step(self, linear, angular, dt):
        left_target = clip((linear-angular*self.axle/2)/self.radius, -10., 10.)
        right_target = clip((linear+angular*self.axle/2)/self.radius, -10., 10.)
        dl = float(clip((left_target-self.left)/self.tau,
                           -self.acceleration_limit, self.acceleration_limit))
        dr = float(clip((right_target-self.right)/self.tau,
                           -self.acceleration_limit, self.acceleration_limit))
        self.left += dl * dt
        self.right += dr * dt
        v = self.radius*(self.left*self.gain_left+self.right*self.gain_right)/2
        omega = self.radius*(self.right*self.gain_right-self.left*self.gain_left)/self.axle
        middle = self.theta + omega*dt/2
        self.x += v*math.cos(middle)*dt
        self.y += v*math.sin(middle)*dt
        self.theta = wrap_angle(self.theta+omega*dt)
        self.path_length += abs(v)*dt
        self.max_observed_acceleration = max(self.max_observed_acceleration, abs(dl), abs(dr))
        return v, omega

