// Fisheye render and the quantities derived from it, in C++.
//
// WHY THIS IS NOT PYTHON.  The renderer is a loop over obstacles, each
// touching a small pixel block -- typically 2% of the field.  Measured, that
// is 0.22 ms per obstacle for a block holding a few hundred pixels, i.e.
// almost none of it is arithmetic; it is numpy call overhead paid 161 times
// a frame.  Vectorising the loop over all obstacles at once was tried and is
// WORSE (151 ms against 24.6 ms): the (n_px, n_obstacles) intermediates run
// to 100 MB and memory bandwidth becomes the limit, while the loop keeps
// each obstacle's block in cache.  So the loop is the right structure and
// the per-call overhead is the whole cost -- which is exactly what moving
// the loop into compiled code removes.
//
// The connectome is deliberately NOT here.  torch.sparse.mm already runs at
// 6 GB/s on this machine, which is the memory bandwidth; a hand-written
// kernel would hit the same wall.
//
// FIXES FOLDED IN -- each of these was a real defect in the Python version:
//
//  1. THE HIT POINT HAD THE WRONG SIGN.  The quadratic is set up with
//     b = 2 d.(p-o), root t = (-b - sqrt(disc))/2, so the surface point is
//     p + t*d.  The Python computed p - t*d for both the texture lookup and
//     the surface normal, i.e. the point reflected through the camera.  The
//     texture therefore moved opposite to the agent instead of staying put
//     on the surface -- the exact opposite of the "world-fixed" property it
//     was added for -- and the Lambertian cos term was the cosine of an
//     unrelated normal.  Range inversion survived only because
//     range_profile takes the brightest pixel per column and assumes its
//     shading is ~1, so it was picking whichever pixel the garbled normal
//     happened to favour.
//
//  2. THE LAMP WAS MIXED INTO THE IMAGE WITH NO WAY TO SEPARATE IT.
//     `render` added the lamp to `img` unconditionally and `with_lamp` only
//     controlled whether the lamp buffer was ALSO returned.  Any caller that
//     did not know to subtract got a scene in which a wall at 3 m is
//     brighter than sky and therefore attracts: measured bearing to a wall
//     at +45 deg went -44.2 deg at 5 m to +10.4 at 3 m to +42.5 at 1.5 m,
//     i.e. the avoidance sign inverted at exactly the range that matters.
//     One arm was fixed by hand and the other was not.  This version returns
//     `unlit` and `lamp` as two buffers and never produces the mixture, so
//     the mistake is not available to make.
//
//  3. np.percentile PER COLUMN, EVERY FRAME.  256 partitions of 144 values
//     through numpy's generic percentile machinery.  Here it is one
//     std::nth_element per column on a stack buffer.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Deterministic surface pattern in WORLD coordinates, in [-1, 1].
// Three octaves, so there is something to match at several scales.
inline double texture(double x, double y, double z, double freq) {
    const double k = freq;
    const double n = std::sin(k * x) * std::sin(k * y + 1.3)
                   + 0.5 * std::sin(2.7 * k * x + 0.7) * std::sin(2.3 * k * z)
                   + 0.3 * std::sin(5.1 * k * y) * std::sin(4.3 * k * x + 2.1);
    return n / 1.8;
}

struct Obstacle {
    double x, y, radius, z, height, luminance;
    bool is_sphere() const { return !(height > 0.0); }
};

struct Camera {
    int n_az, n_el;
    double az_span, el_span;
    double a0, e0, da, de;          // first angle and step, degrees
    double lamp, lamp_half, lamp_offset, lamp_ambient;
    double sky, texture_amp, texture_freq;
};

// Pixel block this obstacle can possibly cover.  Returns false if none.
bool span(const Obstacle& o, const Camera& c,
          double px, double py, double pz, double heading_deg,
          int& ia0, int& ia1, int& ie0, int& ie1) {
    const double ox = px - o.x, oy = py - o.y;
    const double dist = std::hypot(ox, oy);
    if (dist <= o.radius) {                 // the camera is inside it
        ia0 = 0; ia1 = c.n_az; ie0 = 0; ie1 = c.n_el;
        return true;
    }
    double centre = std::atan2(-oy, -ox) * 180.0 / M_PI - heading_deg;
    centre = std::fmod(centre + 180.0, 360.0);
    if (centre < 0.0) centre += 360.0;
    centre -= 180.0;
    const double half = std::asin(std::min(o.radius / dist, 1.0)) * 180.0 / M_PI;

    ia0 = std::max(static_cast<int>(std::floor((centre - half - c.a0) / c.da)) - 1, 0);
    ia1 = std::min(static_cast<int>(std::ceil((centre + half - c.a0) / c.da)) + 2, c.n_az);
    if (ia1 <= ia0) return false;

    if (o.is_sphere()) { ie0 = 0; ie1 = c.n_el; return true; }

    const double near = std::max(dist - o.radius, 0.05);
    const double lo = std::atan2(o.z - pz, dist + o.radius) * 180.0 / M_PI;
    const double hi = std::atan2(o.z + o.height - pz, near) * 180.0 / M_PI;
    ie0 = std::max(static_cast<int>(std::floor((lo - c.e0) / c.de)) - 1, 0);
    ie1 = std::min(static_cast<int>(std::ceil((hi - c.e0) / c.de)) + 2, c.n_el);
    return ie1 > ie0;
}

}  // namespace

// Render, and everything read off the render, in one sweep.
//
// `obstacles` is (N, 6): x, y, radius, z, height, luminance.  A height <= 0
// means a sphere.  Targets are passed in the same array as tall cylinders
// with luminance above sky -- the renderer does not know which is which, and
// no world position of a target reaches any consumer of the result.
py::dict render(py::array_t<double, py::array::c_style | py::array::forcecast> obstacles,
                double px, double py_, double pz, double heading_rad,
                int n_az, int n_el, double az_span, double el_span,
                double lamp_power, double lamp_half, double lamp_offset,
                double lamp_ambient, double sky,
                double texture_amp, double texture_freq,
                double band_deg) {
    Camera c;
    c.n_az = n_az; c.n_el = n_el;
    c.az_span = az_span; c.el_span = el_span;
    c.a0 = -az_span / 2.0;
    c.e0 = -el_span / 2.0;
    c.da = az_span / (n_az - 1);
    c.de = el_span / (n_el - 1);
    c.lamp = lamp_power; c.lamp_half = lamp_half;
    c.lamp_offset = lamp_offset; c.lamp_ambient = lamp_ambient;
    c.sky = sky; c.texture_amp = texture_amp; c.texture_freq = texture_freq;

    auto ob = obstacles.unchecked<2>();
    const py::ssize_t n_obs = ob.shape(0);
    const double heading_deg = heading_rad * 180.0 / M_PI;

    // body frame: +y is LEFT, so a positive offset puts the lamp on the RIGHT
    const double lamp_ox = 0.0, lamp_oy = -lamp_offset, lamp_oz = 0.0;

    const py::ssize_t n_px = static_cast<py::ssize_t>(n_az) * n_el;
    py::array_t<double> unlit_arr({static_cast<py::ssize_t>(n_az), static_cast<py::ssize_t>(n_el)});
    py::array_t<double> lamp_arr({static_cast<py::ssize_t>(n_az), static_cast<py::ssize_t>(n_el)});
    double* unlit = unlit_arr.mutable_data();
    double* lampb = lamp_arr.mutable_data();
    std::vector<double> depth(n_px, kInf);
    // SOLID depth: obstacles only, with its own depth test.  This is what a
    // rangefinder returns.  Beacons are not physical here -- the collision
    // test ignores them -- so a sensor that echoed off them would stop the
    // drone 0.8 m short of every beacon it is meant to reach.  Kept apart
    // from `depth` because a beacon in front of a wall must not hide the
    // wall from the rangefinder.
    std::vector<double> solid(n_px, kInf);
    for (py::ssize_t i = 0; i < n_px; ++i) { unlit[i] = sky; lampb[i] = 0.0; }

    // precomputed trig per row/column
    std::vector<double> ca(n_az), sa(n_az), ce(n_el), se(n_el), az_deg(n_az);
    for (int i = 0; i < n_az; ++i) {
        az_deg[i] = c.a0 + c.da * i;
        const double A = (az_deg[i] + heading_deg) * M_PI / 180.0;
        ca[i] = std::cos(A); sa[i] = std::sin(A);
    }
    for (int j = 0; j < n_el; ++j) {
        const double E = (c.e0 + c.de * j) * M_PI / 180.0;
        ce[j] = std::cos(E); se[j] = std::sin(E);
    }

    for (py::ssize_t k = 0; k < n_obs; ++k) {
        Obstacle o{ob(k, 0), ob(k, 1), ob(k, 2), ob(k, 3), ob(k, 4), ob(k, 5)};
        int ia0, ia1, ie0, ie1;
        if (!span(o, c, px, py_, pz, heading_deg, ia0, ia1, ie0, ie1)) continue;

        const double ox = px - o.x, oy = py_ - o.y, oz = pz - o.z;
        const double emits = (o.luminance > sky) ? 1.0 : 0.0;

        for (int i = ia0; i < ia1; ++i) {
            for (int j = ie0; j < ie1; ++j) {
                const double dx = ce[j] * ca[i];
                const double dy = ce[j] * sa[i];
                const double dz = se[j];

                double t = kInf;
                if (o.is_sphere()) {
                    const double b = 2.0 * (dx * ox + dy * oy + dz * oz);
                    const double c2 = ox * ox + oy * oy + oz * oz - o.radius * o.radius;
                    const double disc = b * b - 4.0 * c2;
                    if (disc >= 0.0) {
                        const double tt = (-b - std::sqrt(disc)) / 2.0;
                        if (tt > 0.0) t = tt;
                    }
                } else {
                    const double aa = dx * dx + dy * dy;
                    if (aa > 1e-12) {
                        const double b = 2.0 * (dx * ox + dy * oy);
                        const double c2 = ox * ox + oy * oy - o.radius * o.radius;
                        const double disc = b * b - 4.0 * aa * c2;
                        if (disc >= 0.0) {
                            const double sq = std::sqrt(disc);
                            for (int s = 0; s < 2; ++s) {
                                const double tt = (-b + (s ? sq : -sq)) / (2.0 * aa);
                                if (tt <= 0.0) continue;
                                const double zh = pz + tt * dz;
                                if (zh < o.z || zh > o.z + o.height) continue;
                                if (tt < t) t = tt;
                            }
                        }
                    }
                }
                if (!(t < kInf)) continue;

                const py::ssize_t idx = static_cast<py::ssize_t>(i) * n_el + j;
                if (emits == 0.0 && t < solid[idx]) solid[idx] = t;
                if (!(t < depth[idx])) continue;
                depth[idx] = t;

                if (emits > 0.0) {          // a beacon is an emitter: uniform
                    unlit[idx] = o.luminance;
                    lampb[idx] = 0.0;
                    continue;
                }

                // FIX 1: the surface point is p + t*d.  The Python used
                // p - t*d here, for both texture and normal.
                const double hx = px + t * dx;
                const double hy = py_ + t * dy;
                const double hz = pz + t * dz;

                double shade = o.luminance;
                if (texture_amp > 0.0)
                    shade += texture_amp * texture(hx, hy, hz, texture_freq);
                shade = std::min(std::max(shade, 0.0), sky - 0.02);
                unlit[idx] = shade;

                if (c.lamp > 0.0) {
                    // vector from the LAMP to the surface point
                    const double lx = t * dx - lamp_ox;
                    const double ly = t * dy - lamp_oy;
                    const double lz = t * dz - lamp_oz;
                    const double rr = std::sqrt(lx * lx + ly * ly + lz * lz) + 1e-9;
                    // outward surface normal at the hit point
                    double nx = hx - o.x, ny = hy - o.y, nz;
                    if (o.is_sphere()) nz = hz - o.z; else nz = 0.0;
                    const double nn = std::sqrt(nx * nx + ny * ny + nz * nz) + 1e-9;
                    // light travels FROM the lamp TO the surface, so the
                    // incidence cosine uses the reversed direction
                    double cosang = -(nx * lx + ny * ly + nz * lz) / (nn * rr);
                    cosang = std::min(std::max(cosang, 0.0), 1.0);
                    lampb[idx] = c.lamp
                               * (lamp_ambient + (1.0 - lamp_ambient) * cosang)
                               / (1.0 + (rr / lamp_half) * (rr / lamp_half));
                } else {
                    lampb[idx] = 0.0;
                }
            }
        }
    }

    // ---- quantities read off the buffers, same sweep ------------------
    // FIX 3: one nth_element per column instead of np.percentile.
    py::array_t<double> rng_arr(static_cast<py::ssize_t>(n_az));
    double* rng = rng_arr.mutable_data();
    std::vector<double> col(n_el);
    const int q = std::max(0, static_cast<int>(std::ceil(0.99 * (n_el - 1))));
    for (int i = 0; i < n_az; ++i) {
        for (int j = 0; j < n_el; ++j) col[j] = lampb[static_cast<py::ssize_t>(i) * n_el + j];
        std::nth_element(col.begin(), col.begin() + q, col.end());
        const double b = col[q];
        if (b <= 1e-5) { rng[i] = kInf; continue; }
        const double ratio = c.lamp / b;
        // brighter than the lamp allows at any range: we are on top of it
        rng[i] = (ratio > 1.0) ? lamp_half * std::sqrt(ratio - 1.0) : 0.05;
    }

    // signed centroid (vision: walls push, beacons pull) and rectified
    // centroid (scent: bright things only, walls contribute exactly zero)
    double num_s = 0.0, den_s = 0.0;        // signed
    double num_r = 0.0, den_r = 0.0;        // rectified
    double dark = 0.0; py::ssize_t n_front = 0;
    for (int i = 0; i < n_az; ++i) {
        const bool front = std::fabs(az_deg[i]) < 40.0;
        for (int j = 0; j < n_el; ++j) {
            const double v = unlit[static_cast<py::ssize_t>(i) * n_el + j] - sky;
            num_s += az_deg[i] * v; den_s += std::fabs(v);
            if (v > 0.0) { num_r += az_deg[i] * v; den_r += v; }
            if (front) { dark += (v < 0.0) ? -v : 0.0; ++n_front; }
        }
    }
    // Nearest SOLID surface per azimuth column, over a horizontal band of
    // +-band_deg: the geometry a range sensor's beam actually sees.  Exact,
    // from the same ray intersections the image is made of, so it costs one
    // pass over the band and no extra rays.
    py::array_t<double> srng_arr(static_cast<py::ssize_t>(n_az));
    double* srng = srng_arr.mutable_data();
    for (int i = 0; i < n_az; ++i) {
        double m = kInf;
        for (int j = 0; j < n_el; ++j) {
            if (std::fabs(c.e0 + c.de * j) > band_deg) continue;
            const double d = solid[static_cast<py::ssize_t>(i) * n_el + j];
            if (d < m) m = d;
        }
        srng[i] = m;
    }

    py::dict out;
    out["solid_range"] = srng_arr;
    out["unlit"] = unlit_arr;
    out["lamp"] = lamp_arr;
    out["range"] = rng_arr;
    out["bearing"] = (den_s > 1e-9) ? num_s / den_s : 0.0;
    out["weight"] = (den_s > 1e-9) ? den_s / static_cast<double>(n_px) : 0.0;
    out["scent"] = (den_r > 1e-9) ? num_r / den_r : 0.0;
    out["scent_mass"] = (den_r > 1e-9) ? den_r / static_cast<double>(n_px) : 0.0;
    out["blocked"] = (n_front > 0) ? dark / (static_cast<double>(n_front) * sky) : 0.0;
    return out;
}

PYBIND11_MODULE(_fisheye, m) {
    m.doc() = "Fisheye render and its derived quantities, compiled.";
    m.def("render", &render,
          py::arg("obstacles"), py::arg("px"), py::arg("py"), py::arg("pz"),
          py::arg("heading_rad"), py::arg("n_az"), py::arg("n_el"),
          py::arg("az_span"), py::arg("el_span"), py::arg("lamp_power"),
          py::arg("lamp_half"), py::arg("lamp_offset"), py::arg("lamp_ambient"),
          py::arg("sky"), py::arg("texture_amp"), py::arg("texture_freq"),
          py::arg("band_deg") = 7.5);
}
