"""시각 기반 가상 유인 단서 추적 — visual virtual attraction cue.

    3D 시각 입력 -> 표적 검출 -> 센트로이드 추적 -> 상대 방향/편차 추출
                 -> 가상 유인 단서 -> 초파리 항법 회로

The user's definition, kept verbatim because the name invites the wrong
implementation otherwise:

    가상 유인 단서(virtual attraction cue) 모듈: 카메라 영상에서 표적을
    검출하고 센트로이드를 추적한 뒤, 화면 중심에 대한 표적의 상대 방향·편차·
    검출 유효성 등을 저차원 신호로 변환한다. 이 신호는 실제 화학적 페로몬을
    감지하는 것이 아니라, 시각적 표적 정보를 초파리의 유인성 감각 단서에
    대응되는 형태로 단순화한 입력이다. 이후 이동 방향의 결정은 이 모듈이
    아니라 초파리 유래 항법 회로가 수행한다.

**This is a sensory front-end, not a navigation controller.**  It reports where
the target appears.  It never decides what to do about it.  That boundary is
the same one master doc section 3.5 puts on the body plugin, applied to the
sensor side, and it is enforced structurally here: `AttractionCue` is frozen,
carries no gain, and exposes no method returning a command.  If this module
ever picks a heading, the study stops measuring the fly circuit and starts
measuring this module.

Master doc section 22 already left this slot open:

    environment -> sensor / goal-vector preprocessing -> desired heading
    representation -> fly core -> steering

Two cameras, because depth has to come from somewhere -- stereo disparity.

MODELLING ASSUMPTION, stated because it is a choice and not a fact: the target
is assumed to be the bright thing in the scene, which is what makes an
intensity-weighted centroid the right tracker for it.  Tracking something that
is not bright is not a change to this algorithm -- it is a change of SENSOR.
Swap the camera for infrared, or for any front end that emits an intensity map
in which the target stands out, and the same centroid maths applies unchanged.
That is why `sense()` weights by an explicit per-sample intensity instead of
hard-coding "the target": the intensity is the sensor's business.

ponytail: the target is projected analytically rather than rendered.  A
rendered blob and an analytically projected sphere give the same centroid, and
this is ~40x cheaper per camera (measured: 0.28 ms against 12.0 ms).  Swap in a
real render the day detection itself -- clutter, texture, a learned detector --
is part of the question.  The geometry below is written so that swap is a
drop-in: same camera model, same centroid convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class StereoRig:
    """Two forward-facing cameras, separated along the body's y axis."""

    baseline_m: float = 0.25
    hfov_deg: float = 90.0
    vfov_deg: float = 60.0
    width_px: int = 256
    height_px: int = 192
    # A pinhole cannot reach the compound eye's field: tan blows up before
    # 180 deg.  With `fisheye` the projection is equidistant (u linear in
    # angle, r = f*theta), which is what a real wide-angle lens does and what
    # makes a like-for-like comparison against the eyes possible at all.
    # Stereo range is derived from a pinhole disparity relation and is NOT
    # corrected for this, so `range_m` is unreliable in fisheye mode; nothing
    # in the steering path reads it -- only `bearing_deg` and `strength`.
    fisheye: bool = False

    @property
    def fx(self) -> float:
        """Focal length in pixels.  A plain camera is a PINHOLE: pixels are
        linear in tan(angle), not in angle.  Getting this wrong made stereo
        range read 7.85 m for a target at 10 m, because b*f/disparity assumes
        exactly this model."""
        if self.fisheye:
            return (self.width_px / 2) / (math.radians(self.hfov_deg) / 2)
        return (self.width_px / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def fy(self) -> float:
        if self.fisheye:
            return (self.height_px / 2) / (math.radians(self.vfov_deg) / 2)
        return (self.height_px / 2) / math.tan(math.radians(self.vfov_deg) / 2)

    def project(self, rel: np.ndarray) -> Optional[np.ndarray]:
        """Camera-frame point -> (u, v) in pixels, or None if out of frame.

        `rel` is in camera axes: +x forward, +y left, +z up.  Image u grows to
        the RIGHT, so a target to the left lands at u < width/2.
        """
        if rel[0] <= 1e-6:
            return None                      # behind the image plane
        az = math.atan2(rel[1], rel[0])
        el = math.atan2(rel[2], math.hypot(rel[0], rel[1]))
        if (abs(az) > math.radians(self.hfov_deg) / 2
                or abs(el) > math.radians(self.vfov_deg) / 2):
            return None
        if self.fisheye:
            u = self.width_px / 2 - self.fx * az
            v = self.height_px / 2 - self.fy * el
        else:
            u = self.width_px / 2 - self.fx * math.tan(az)
            v = self.height_px / 2 - self.fy * math.tan(el)
        return np.array([u, v])

    def unproject(self, u: float, v: float):
        """(u, v) -> (azimuth, elevation) in radians.  Inverse of project."""
        if self.fisheye:
            return ((self.width_px / 2 - u) / self.fx,
                    (self.height_px / 2 - v) / self.fy)
        return (math.atan((self.width_px / 2 - u) / self.fx),
                math.atan((self.height_px / 2 - v) / self.fy))


@dataclass(frozen=True)
class AttractionCue:
    """The low-dimensional signal.  Deliberately not a command.

    bearing_deg     target centroid left/right of image centre, + is left
                    (same sign convention as the project's CCW-positive yaw)
    elevation_deg   above/below image centre, + is up
    deviation       |bearing| normalised by half the horizontal field of view,
                    0 at centre and 1 at the edge
    range_m         from stereo disparity; NaN when only one camera sees it
    strength        apparent size x detection validity -- the cue's intensity,
                    which is what the user suggested size/confidence be used for
    valid           whether any camera detected the target at all
    n_cameras       how many of the two saw it
    """

    bearing_deg: float
    elevation_deg: float
    deviation: float
    range_m: float
    strength: float
    valid: bool
    n_cameras: int

    def as_vector(self) -> np.ndarray:
        """The cue as the core would receive it.  Still not a command."""
        return np.array([self.bearing_deg, self.elevation_deg,
                         self.deviation, self.strength,
                         1.0 if self.valid else 0.0])


def _to_camera_frame(world_point: np.ndarray, cam_pos: np.ndarray,
                     heading: float) -> np.ndarray:
    d = world_point - cam_pos
    c, s = math.cos(-heading), math.sin(-heading)
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])


def _occluded_many(cam_pos: np.ndarray, points: np.ndarray, obstacles,
                   samples: int = 12) -> np.ndarray:
    """Which of `points` are hidden from `cam_pos`.  Vectorised.

    The per-point Python version cost 9.9 ms per step for 81 surface samples
    against 8 obstacles; this does the same arithmetic in a few numpy ops.
    Same sampled-ray approximation, same answers.
    """
    if not len(obstacles):
        return np.zeros(len(points), dtype=bool)
    t = np.linspace(0.05, 0.95, samples)[None, :, None]
    seg = cam_pos[None, None, :] + t * (points[:, None, :] - cam_pos[None, None, :])
    hit = np.zeros(len(points), dtype=bool)
    for o in obstacles:
        dx = seg[..., 0] - o.x
        dy = seg[..., 1] - o.y
        if o.height is None:
            dz = seg[..., 2] - o.z
            inside = (dx * dx + dy * dy + dz * dz) < o.radius ** 2
        else:
            inside = ((dx * dx + dy * dy) < o.radius ** 2) & \
                     (seg[..., 2] >= o.z) & (seg[..., 2] <= o.z + o.height)
        hit |= inside.any(axis=1)
    return hit


def _occluded(cam_pos: np.ndarray, target: np.ndarray, obstacles,
              samples: int = 24) -> bool:
    """Is the straight line from camera to target blocked?

    Sampled rather than solved: obstacles are cylinders and spheres and an
    exact test is a page of algebra for a result that only has to be right to
    within a sample spacing.
    ponytail: raise `samples` if a thin obstacle is ever slipped through.
    """
    for t in np.linspace(0.05, 0.95, samples):
        p = cam_pos + t * (target - cam_pos)
        if any(o.distance_to(p) < 0 for o in obstacles):
            return True
    return False


class AttractionCueSensor:
    """Detect the target in two cameras, report where it appears.

    It is given the world and the agent's POSE.  It is never given the agent's
    goal, its plant, its limits, or its controller -- there is nothing here to
    turn a cue into an action with.
    """

    # The compound eye spans +-136.8 deg azimuth and -78.8..+83.8 elevation.
    # A +-45 deg camera therefore gave the connectome a target the references
    # could not even see -- measured on map 14, 24 lamina cells had the target
    # at -74 deg while the camera reported it invisible.  Any claim that one
    # arm navigates better than another needs them looking at the same world,
    # so this rig matches the eye's field exactly.
    WIDE = None          # set below, once StereoRig is defined

    def __init__(self, rig: Optional[StereoRig] = None,
                 target_radius: float = 0.6):
        self.rig = rig or StereoRig()
        self.target_radius = target_radius

    def sense_many(self, p, heading, targets, obstacles):
        """The brightest target currently detected.

        With several beacons in the room the intensity-weighted centroid
        should follow whichever one dominates the image, and apparent size is
        exactly what `strength` measures -- so "brightest" is "nearest and
        unoccluded", which is what a real centroid tracker locks onto.  No
        knowledge of which target is reachable, or even of how many there
        are, leaks into the controller: it gets one cue, as before.

        ponytail: picks the strongest rather than blending several, so two
        equally bright beacons produce a lock rather than an average heading
        between them.  Blend them the day that ambiguity is the question.
        """
        best = None
        for t in targets:
            c = self.sense(p, heading, t, obstacles)
            if c.valid and (best is None or c.strength > best.strength):
                best = c
        return best if best is not None else self.sense(
            p, heading, targets[0], obstacles)

    def camera_positions(self, p: np.ndarray, heading: float) -> np.ndarray:
        """Left and right camera centres in world coordinates."""
        half = self.rig.baseline_m / 2
        left = np.array([-math.sin(heading), math.cos(heading), 0.0]) * half
        return np.array([p + left, p - left])

    def _surface_samples(self, cam: np.ndarray, target: np.ndarray,
                         n: int = 9) -> np.ndarray:
        """Points spread over the target's camera-facing hemisphere.

        The intensity-weighted centroid needs the target as an EXTENDED object,
        not a point: a half-occluded target must pull its centroid toward the
        visible half, which is the whole reason for weighting by intensity.
        """
        d = target - cam
        d = d / max(np.linalg.norm(d), 1e-9)
        # any two axes perpendicular to the view direction
        a = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(a, d)) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(d, a)
        e1 /= max(np.linalg.norm(e1), 1e-9)
        e2 = np.cross(d, e1)
        g = np.linspace(-1.0, 1.0, n)
        gx, gy = np.meshgrid(g, g)
        keep = (gx ** 2 + gy ** 2) <= 1.0          # the visible disk
        gx, gy = gx[keep], gy[keep]
        # lift onto the near hemisphere so occlusion tests hit real surface
        gz = np.sqrt(np.maximum(1.0 - gx ** 2 - gy ** 2, 0.0))
        r = self.target_radius
        return (target[None, :] + r * (gx[:, None] * e1[None, :]
                                       + gy[:, None] * e2[None, :]
                                       - gz[:, None] * d[None, :]))

    def sense(self, p: np.ndarray, heading: float, target: np.ndarray,
              obstacles: Sequence = ()) -> AttractionCue:
        """Intensity-weighted centroid over both cameras.

        centroid = sum(I_i * u_i) / sum(I_i), with I = 1 for a visible patch of
        the target and 0 where an obstacle blocks it.  For an unoccluded target
        this reduces to the geometric centre; the difference appears exactly
        where it matters here, when clutter hides part of the target and the
        tracker should follow the visible part.
        """
        rig = self.rig
        us: List[float] = []
        vs: List[float] = []
        ws: List[float] = []
        n_cam = 0
        for cam in self.camera_positions(p, heading):
            pts = self._surface_samples(cam, target)
            blocked = _occluded_many(cam, pts, obstacles)
            uu, vv, ii = [], [], []
            for q, b in zip(pts, blocked):
                if b:
                    continue
                uv = rig.project(_to_camera_frame(q, cam, heading))
                if uv is None:
                    continue                    # outside the frame contributes 0
                uu.append(uv[0])
                vv.append(uv[1])
                ii.append(1.0)                  # uniform emitter
            if not ii:
                continue
            w = float(np.sum(ii))
            us.append(float(np.dot(ii, uu) / w))
            vs.append(float(np.dot(ii, vv) / w))
            ws.append(w / len(pts))             # visible fraction, 0..1
            n_cam += 1

        if not us:
            return AttractionCue(0.0, 0.0, 0.0, float("nan"), 0.0, False, 0)

        # combine the cameras by their own visible intensity, so a camera that
        # can only see a sliver counts for a sliver
        wsum = float(np.sum(ws))
        u = float(np.dot(ws, us) / wsum)
        v = float(np.dot(ws, vs) / wsum)
        az, el = rig.unproject(u, v)
        bearing = math.degrees(az)            # + is LEFT, matching CCW yaw
        elevation = math.degrees(el)
        deviation = min(abs(bearing) / (rig.hfov_deg / 2), 1.0)

        # stereo range, only when both cameras have it
        rng = float("nan")
        if len(us) == 2:
            disparity = abs(us[0] - us[1])
            if disparity > 1e-6:
                rng = float(rig.baseline_m * rig.fx / disparity)

        # strength is the integrated intensity: apparent size x how much of the
        # target is actually visible.  Partial occlusion now fades the cue
        # instead of switching it off, which is what a real tracker does.
        dist = float(np.linalg.norm(target - p))
        ang_size = math.degrees(math.atan2(self.target_radius,
                                           max(dist, 1e-6))) * 2
        strength = float(ang_size / rig.hfov_deg * (wsum / 2.0))
        return AttractionCue(bearing, elevation, deviation, rng, strength,
                             True, len(us))


def demo() -> None:
    """Check the cue points at the target, and stops when it cannot see it."""
    import sys
    sys.path.insert(0, "src")
    from environment.target_world import Obstacle

    sensor = AttractionCueSensor()
    p = np.zeros(3)

    # straight ahead -> no bearing
    c = sensor.sense(p, 0.0, np.array([10.0, 0.0, 0.0]))
    assert c.valid and abs(c.bearing_deg) < 1e-6, c
    print("target dead ahead : bearing %+.2f deg, deviation %.3f, range %.2f m"
          % (c.bearing_deg, c.deviation, c.range_m))

    # to the agent's left -> positive bearing (CCW-positive, as the project does)
    left = sensor.sense(p, 0.0, np.array([10.0, 5.0, 0.0]))
    right = sensor.sense(p, 0.0, np.array([10.0, -5.0, 0.0]))
    assert left.valid and right.valid
    assert left.bearing_deg > 0 > right.bearing_deg, (left, right)
    print("target left       : bearing %+.2f deg   target right: %+.2f deg"
          % (left.bearing_deg, right.bearing_deg))
    # and the two must be mirror images, or the sensor is biased
    assert abs(left.bearing_deg + right.bearing_deg) < 1e-9

    # turning to face it must null the bearing -- this is what closes the loop
    turned = sensor.sense(p, math.radians(left.bearing_deg),
                          np.array([10.0, 5.0, 0.0]))
    assert abs(turned.bearing_deg) < abs(left.bearing_deg), turned
    print("after turning by the reported bearing: %+.2f -> %+.2f deg"
          % (left.bearing_deg, turned.bearing_deg))

    # behind, and outside the field of view -> invalid
    behind = sensor.sense(p, 0.0, np.array([-10.0, 0.0, 0.0]))
    assert not behind.valid and behind.n_cameras == 0
    wide = sensor.sense(p, 0.0, np.array([1.0, 10.0, 0.0]))
    assert not wide.valid, wide
    print("behind / outside FOV: valid=%s, %s" % (behind.valid, wide.valid))

    # occlusion must actually hide it
    blocker = [Obstacle(x=5.0, y=0.0, radius=1.5, z=-5.0, height=10.0)]
    hidden = sensor.sense(p, 0.0, np.array([10.0, 0.0, 0.0]), blocker)
    assert not hidden.valid, hidden
    print("target behind an obstacle: valid=%s" % hidden.valid)

    # PARTIAL occlusion is what intensity weighting is for: the centroid must
    # move toward the visible side, and the strength must fade rather than
    # switch off.  A geometric centre-point projection cannot do either.
    tgt = np.array([10.0, 0.0, 0.0])
    full = sensor.sense(p, 0.0, tgt)
    edge = [Obstacle(x=5.0, y=-0.28, radius=0.3, z=-5.0, height=10.0)]
    part = sensor.sense(p, 0.0, tgt, edge)
    print("unoccluded      : bearing %+.3f deg, strength %.4f"
          % (full.bearing_deg, full.strength))
    print("half blocked    : bearing %+.3f deg, strength %.4f"
          % (part.bearing_deg, part.strength))
    assert part.valid, "a partly blocked target is still visible"
    assert part.strength < full.strength, "strength must fade with occlusion"
    assert part.bearing_deg > full.bearing_deg, \
        "blocking the right side must pull the centroid left"

    # strength must rise as the target gets closer
    far = sensor.sense(p, 0.0, np.array([20.0, 0.0, 0.0]))
    near = sensor.sense(p, 0.0, np.array([4.0, 0.0, 0.0]))
    assert near.strength > far.strength, (near.strength, far.strength)
    print("strength at 20 m %.4f  ->  at 4 m %.4f"
          % (far.strength, near.strength))

    # stereo range must be roughly right
    err = abs(far.range_m - 20.0) / 20.0
    print("stereo range at 20 m: %.2f m (%.1f%% error)" % (far.range_m, 100 * err))
    assert err < 0.10, far.range_m

    # the boundary: the cue must carry nothing that is a command
    assert not hasattr(c, "yaw_rate") and not hasattr(c, "command")
    assert len(c.as_vector()) == 5
    print("demo ok -- cue reports where the target is, never what to do")


if __name__ == "__main__":
    demo()


# Same field of view as the compound eye, for like-for-like baselines.
WIDE_RIG = StereoRig(hfov_deg=273.6, vfov_deg=162.6, fisheye=True)
AttractionCueSensor.WIDE = WIDE_RIG
