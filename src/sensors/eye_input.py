"""Fisheye into the lamina, one grid patch per ommatidium.

Each injected lamina cell has its own viewing direction from the MaleCNS eye
lattice, and it reads the fisheye over its ACCEPTANCE CONE rather than at a
single pixel.  A fly ommatidium has an acceptance angle of about 5.8 deg
(flyvis, Lappalainen et al. 2024); this camera is 1.13 deg per azimuth column
and 1.22 deg per elevation row, so the cone is a 5 x 5 block.  Sampling one
pixel instead made every ommatidium an ideal point sensor with no spatial
low-pass at all, which is not what the optic lobe downstream is built to
receive -- and aliasing a 1.13 deg grid straight into a motion detector
manufactures spurious motion wherever the scene has fine structure.

The 6,199 injected cells (L1, L2, L3, L5) share ommatidia, so they look along
only 1,769 distinct directions.  The patch is gathered once per direction and
scattered back.

CONVENTION: `flywire_eye` is right-positive, everything else here is
CCW-positive (+ = LEFT).  The azimuth is negated once, on load.  Feeding a
CCW scene to a right-positive lattice once put every object on the wrong side
and made 48 of 53 glomeruli steer away from the target.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import sensors.flywire_eye as eye
from sensors.fisheye import FisheyeCamera, clearance_ahead

N_SUB = 20                 # sub-steps per control cycle; matches the runner
PROX_R0 = 1.5              # metres, for the optional rangefinder channel


class EyeInput:
    """Drive vector for a lamina-injected subnetwork that keeps the optic lobe.

    `prox_gain` defaults to ZERO, unlike the projection-neuron arm.  That arm
    has no optic lobe, so proximity had to be handed to it: measured, its
    descending common mode tracked 1/tau at -0.15 without a rangefinder and
    +0.77 with one.  This arm exists to ask whether the optic lobe computes
    that itself, and handing it the answer would make the question
    unanswerable.  Turn it on only as a deliberate control.
    """

    def __init__(self, net, ann, info, drive_gain: float = 0.20,
                 orn_gain: float = 200.0,
                 cue_types=("ORN_DA1", "ORN_VM2", "ORN_VA1v", "ORN_DC3",
                            "ORN_VA6"),
                 prox_gain: float = 0.0, fixed_strength=None):
        self.net, self.ann, self.info = net, ann, info
        self.drive_gain, self.orn_gain = drive_gain, orn_gain
        self.prox_gain, self.fixed_strength = prox_gain, fixed_strength
        self.cam = FisheyeCamera()
        self.n_sub = N_SUB

        lat = eye.load_malecns_eye(ann)
        sel = lat.of_type(*eye.MALECNS_INJECT_TYPES)
        ids = np.asarray(info["lam_ids"], dtype=np.int64)
        order = {int(r): i for i, r in enumerate(ids)}
        take = np.array([order.get(int(r), -1) for r in lat.root_id[sel]])
        ok = take >= 0
        if ok.sum() < 0.9 * len(ids):
            raise ValueError("only %d of %d lamina cells matched the lattice; "
                             "`ann` is probably not indexed by root_id"
                             % (int(ok.sum()), len(ids)))
        self.rows = np.asarray(info["lam_rows"])[take[ok]]
        self.eye_az = -lat.azimuth_deg[sel][ok]      # to CCW-positive
        self.eye_el = lat.elevation_deg[sel][ok]

        # one gather per DISTINCT direction, scattered back afterwards
        key = np.stack([np.round(self.eye_az, 4),
                        np.round(self.eye_el, 4)], axis=1)
        _, uidx, self._uinv = np.unique(key, axis=0, return_index=True,
                                        return_inverse=True)
        uaz, uel = self.eye_az[uidx], self.eye_el[uidx]
        self._patch = self._acceptance_patch(uaz, uel)
        self._ucol = np.clip(np.round((uaz - self.cam.a[0]) / self.cam.da)
                             .astype(int), 0, self.cam.n_az - 1)

        ct = ann["cell_type"].astype(str).to_numpy(dtype="<U48")
        side = ann["side"].to_numpy(dtype="<U16")
        is_orn = np.array([any(c.startswith(t) for t in cue_types)
                           for c in ct])
        self.orn_l = np.flatnonzero(is_orn & (side == "left"))
        self.orn_r = np.flatnonzero(is_orn & (side == "right"))
        self.orn_scale_l = 1.0 / max(len(self.orn_l), 1)
        self.orn_scale_r = 1.0 / max(len(self.orn_r), 1)

        self.dt = 0.1
        self.speed = 2.0
        self._prev_val = None
        self._goal_world = None
        self._range = np.inf

    def _acceptance_patch(self, az, el):
        """Flat pixel indices inside each ommatidium's acceptance cone."""
        ha = max(int(round(eye.ACCEPTANCE_DEG / 2 / self.cam.da)), 1)
        he = max(int(round(eye.ACCEPTANCE_DEG / 2 / self.cam.de)), 1)
        ia = np.round((az - self.cam.a[0]) / self.cam.da).astype(int)
        ie = np.round((el - self.cam.e[0]) / self.cam.de).astype(int)
        da, de = np.meshgrid(np.arange(-ha, ha + 1), np.arange(-he, he + 1),
                             indexing="ij")
        aa = np.clip(ia[:, None] + da.ravel()[None, :], 0, self.cam.n_az - 1)
        ee = np.clip(ie[:, None] + de.ravel()[None, :], 0, self.cam.n_el - 1)
        return aa * self.cam.n_el + ee            # (n_unique, k)

    def counts(self) -> dict:
        return {"lamina": len(self.rows),
                "directions": int(self._patch.shape[0]),
                "px_per_ommatidium": int(self._patch.shape[1]),
                "orn_left": len(self.orn_l), "orn_right": len(self.orn_r)}

    def reset(self) -> None:
        self._prev_val = None
        self._goal_world = None
        self._range = np.inf

    def drive(self, world, position, heading_rad):
        r = self.cam.sense(world, position, heading_rad)
        flat = r["unlit"].ravel()                  # never the lit image
        upatch = flat[self._patch].mean(axis=1)    # acceptance-cone average
        lum = upatch[self._uinv]
        lum = lum - lum.mean()                     # L1/L2 transmit contrast
        val = lum * self.drive_gain
        if self.prox_gain > 0.0:
            rng = r["range"][self._ucol][self._uinv]
            prox = np.where(np.isfinite(rng),
                            np.minimum(PROX_R0 / np.maximum(rng, 1e-3), 1.0),
                            0.0)
            val = val + self.prox_gain * prox

        # the drive is a SEQUENCE: prev -> cur across the sub-steps, so a
        # motion-detecting stage has motion to detect.  Holding one column
        # for the whole cycle fed the optic lobe a photograph.
        prev = self._prev_val if self._prev_val is not None else val
        self._prev_val = val
        ramp = np.linspace(1.0 / self.n_sub, 1.0, self.n_sub)
        seq = prev[:, None] * (1.0 - ramp) + val[:, None] * ramp

        d = torch.zeros(self.net.n, self.n_sub)
        d[self.rows] = torch.from_numpy(seq.astype(np.float32))

        scent, mass = r["scent"], r["scent_mass"]
        self._goal_world = heading_rad + math.radians(scent)
        frac = float(np.clip(scent / (self.cam.az_span / 2), -1.0, 1.0))
        s = mass if self.fixed_strength is None else self.fixed_strength
        base = self.orn_gain * s
        d[self.orn_l, :] = base * (1 + frac) * self.orn_scale_l
        d[self.orn_r, :] = base * (1 - frac) * self.orn_scale_r

        rng, brg = clearance_ahead(self.cam, r["range"], 0.0, 60.0)
        self._range = rng
        return d, {"bearing": r["bearing"], "scent": scent,
                   "scent_mass": mass, "blocked": r["blocked"],
                   "weight": r["weight"], "range": rng,
                   "clear_bearing": brg, "profile": r["range"],
                   "rear": r["rear"]}
