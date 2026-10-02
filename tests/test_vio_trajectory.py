"""VIO trajectory generator checks, no renders (pure geometry only).

Scenes: (a) one aprilgrid board at 30 cm squares (the doc's recommended
VIO render), (b) the default respaced three-board scene at 15 cm squares.
For each:
  1. verify_path passes: boards inside the reference camera's image on
     every frame (exact DS projection from DP180IP-30020104.json), all
     four quadrants + both distance bands covered on the driver's KB4
     cameras (camB, camC; camA points away and is not consumed), and the
     per-frame pixel displacement (formula AND measured) under
     TRACKABLE_PX at the default 20 fps.
  2. The stationary lead is truly stationary: every frame before
     STATIONARY_LEAD_S is bit-identical to frame 0.
  3. C2 continuity: finite-difference jerk (translation) and angular
     jerk, sampled at 100 Hz, stay bounded, and the window around the
     stationary->motion boundary is no worse than the rest of the path.
  4. The ground-truth npz has ate_compare's layout (t_ns, p, q_xyzw),
     int64 monotonic stamps matching the mcap stamp formula, unit
     quaternions - and loads through ate_compare.load_npz itself when
     ~/vio_offline/ate_compare.py is present.

Run:  python tests/test_vio_trajectory.py
"""
import math
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

for var in ("VIO_TRAJ", "STATIONARY_LEAD_S", "VIO_MOTION_S", "VIO_FPS",
            "TRACKABLE_PX", "VIO_TRAJ_NPZ", "ORBIT_DIST", "ORBIT_TARGET",
            "ORBIT_FLIP", "ARM_REACH", "VIO_CHECK_CAMS"):
    os.environ.pop(var, None)

from threedgrut_playground.utils.boards import (
    DEFAULT_SCENE, board_half_extents, layout_positions)
from threedgrut_playground.utils import vio_trajectory as vt

CALIB = os.path.join(REPO, "calibration_files", "DP180IP-30020104.json")


def quad_corners(center, hx, hy):
    return np.array([[center[0] + sx * hx, center[1] + sy * hy, center[2]]
                     for sx in (-1, 1) for sy in (-1, 1)])


def scene_single():
    """One 4x7 board, 30 cm squares, drawn at z = -2.75 facing +z."""
    hx, hy = board_half_extents("aprilgrid", 30.0)
    cloud = quad_corners([0.0, 0.0, -2.75], hx, hy)
    return cloud, np.array([0.0, 0.0, -2.75]), np.array([0.0, 0.0, 1.0])


def scene_default():
    """The respaced three-board scene at 15 cm squares (drawn +1.25 in z)."""
    placed = layout_positions(DEFAULT_SCENE, 5.173, 0.2 * 5.173, 0.6)
    pts = []
    for name, pos in placed:
        hx, hy = board_half_extents(name, 15.0)
        pts.append(quad_corners([pos[0], pos[1], pos[2] + 1.25], hx, hy))
    cloud = np.concatenate(pts)
    mn, mx = cloud.min(0), cloud.max(0)
    return cloud, (mn + mx) / 2.0, np.array([0.0, 0.0, 1.0])


def run_scene(tag, cloud, center, normal, rig, ref):
    cfg = vt.VioConfig()
    geom = vt.PathGeometry(cloud, center, normal, rig[ref])
    print(f"\n=== {tag}: bands {geom.d_near:.2f}..{geom.d_far:.2f} m, "
          f"aim budget ({geom.bud_x:.0f}, {geom.bud_y:.0f}) deg")
    times, eyes, R_wc = vt.sample_path(cfg, geom)
    metrics = vt.verify_path(cfg, geom, rig, ref, times, eyes, R_wc)
    print(f"PASS  {tag}: verify_path (in-view, quadrants, bands, "
          f"trackable px {metrics['peak']['px_formula']:.1f}/"
          f"{metrics['peak']['px_measured']:.1f} <= {cfg.trackable_px:.0f})")

    # the scene-plausibility checks are armed for every rig: on the
    # DP180 scenes they must be present and PASS (the design never
    # leaves the fit cone, the speed stays handheld)
    _, checks = vt.run_checks(cfg, geom, rig, ref, times, eyes, R_wc)
    by = {name: ok for name, ok, _msg, _strict in checks}
    assert by["bands within the fit cone"], \
        f"fit-cone row missing or failing ({geom.d_far / geom.d_fit:.1f}x)"
    assert by["peak speed plausible"], \
        f"speed row missing or failing ({metrics['peak']['v']:.2f} m/s)"
    print(f"PASS  {tag}: scene-plausibility rows PASS (bands "
          f"{geom.d_far / geom.d_fit:.1f}x fit, peak "
          f"{metrics['peak']['v']:.2f} m/s)")

    # 2. stationary lead truly stationary
    lead_n = int(math.floor(cfg.lead_s * cfg.fps)) + 1
    assert np.array_equal(eyes[:lead_n], np.repeat(eyes[:1], lead_n, 0)), \
        "positions move during the stationary lead"
    assert np.array_equal(R_wc[:lead_n], np.repeat(R_wc[:1], lead_n, 0)), \
        "orientation moves during the stationary lead"
    assert not np.allclose(eyes[lead_n + int(5 * cfg.fps)], eyes[0]), \
        "path never leaves the initial pose"
    print(f"PASS  {tag}: {lead_n} lead frames bit-identical to frame 0")

    # 3. C2 continuity at 100 Hz: jerk bounded, boundary not an outlier
    hz = 100.0
    tt = np.arange(int(cfg.total_s * hz)) / hz
    _, e2, R2 = vt.sample_path(cfg, geom, times=tt)
    dtf = 1.0 / hz
    acc = np.diff(e2, 2, axis=0) / dtf ** 2
    jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1) / dtf
    wv = np.array([vt._rotvec(R2[i], R2[i + 1]) / dtf
                   for i in range(len(R2) - 1)])
    aacc = np.diff(wv, axis=0) / dtf
    ajerk = np.linalg.norm(np.diff(aacc, axis=0), axis=1) / dtf
    b = int(cfg.lead_s * hz)
    win = slice(max(b - 10, 0), b + 10)
    rest = np.r_[jerk[:max(b - 10, 0)], jerk[b + 10:]]
    arest = np.r_[ajerk[:max(b - 10, 0)], ajerk[b + 10:]]
    assert jerk.max() < 2.0, f"translational jerk {jerk.max():.2f} m/s^3"
    assert ajerk.max() < 2.0, f"angular jerk {ajerk.max():.2f} rad/s^3"
    assert jerk[win].max() <= max(rest.max(), 1e-6) * 1.2 + 1e-6, \
        f"jerk spike at the boundary: {jerk[win].max():.3f} vs {rest.max():.3f}"
    assert ajerk[win].max() <= max(arest.max(), 1e-6) * 1.2 + 1e-6, \
        f"angular jerk spike at the boundary: {ajerk[win].max():.3f}"
    print(f"PASS  {tag}: C2 - jerk max {jerk.max():.3f} m/s^3 "
          f"(boundary {jerk[win].max():.3f}), angular {ajerk.max():.3f} "
          f"rad/s^3 (boundary {ajerk[win].max():.3f})")

    # 4. npz layout
    with tempfile.TemporaryDirectory() as d:
        npz = os.path.join(d, "truth.npz")
        t_ns = vt.export_npz(npz, cfg, eyes, R_wc)
        data = np.load(npz, allow_pickle=True)
        assert set(("t_ns", "p", "q_xyzw")) <= set(data.keys())
        assert data["t_ns"].dtype == np.int64
        assert np.all(np.diff(data["t_ns"]) == cfg.interval_ns)
        assert data["t_ns"][0] == vt.START_TIME_NS
        assert np.allclose(np.linalg.norm(data["q_xyzw"], axis=1), 1.0)
        assert len(data["t_ns"]) == cfg.n_frames
        ate = os.path.expanduser("~/vio_offline/ate_compare.py")
        if os.path.exists(ate):
            sys.path.insert(0, os.path.dirname(ate))
            import ate_compare
            t, p, q = ate_compare.load_npz(npz)
            assert len(t) == cfg.n_frames and p.shape == (cfg.n_frames, 3)
            print(f"PASS  {tag}: npz loads through ate_compare.load_npz")
        else:
            print(f"PASS  {tag}: npz layout ok (ate_compare.py not present)")
        assert np.array_equal(t_ns, data["t_ns"])
    return metrics


NV_CALIB = os.path.join(REPO, "calibration_files", "NV180V2-80.121.0230.json")
NV_CALIB_B = os.path.join(REPO, "calibration_files", "NV180V2-80.120.0101.json")


def run_nv180():
    """NV180 (narrow KB4 reference, camc pitched 20 deg off it):
      5. _reachable_quadrants is exact on both rigs - NV180 camc can only
         reach its upper half, camb its left column, while every DP180
         checked camera keeps all four (so the DP180 checks are untouched);
         the camd fit's fold angle (the starburst polynomial) is honoured.
      6. The default checks derive the tracked pair (camb, camc) from
         the rig: camc passes on its reachable quadrants (the old
         all-four rule failed on [n, n, 0, 0]), camd gets no coverage
         rows, the px formula is bounded by the tracked focal - and the
         reference-framed far design still cannot pass, because the
         scene-plausibility rows fail it.
      7. VIO_CHECK_CAMS=1,2 (the nv180_minimal tracked pair) frames the
         path in camc (select_framing), bringing the bands into the room
         (DP180's range, not 22..35 m): every strict check passes,
         including the armed scene-plausibility checks.
      8. Regression: with the tracked cameras declared, the old
         reference-framed far path fails 'bands within the fit cone' AND
         'peak speed plausible' - a path that leaves the scene can no
         longer pass.
      9. tracked_sockets derives {1,2} on the NV180 and {1,2,3} on the
         DP180 (check rows [camc, camb], the historical order and
         scoping values); plan_framing keeps the DP180's reference
         design bit for bit and gives the NV180 the camc-framed design
         with no knob - identical to the VIO_CHECK_CAMS=1,2 path."""
    rig, ref = vt.load_rig(NV_CALIB)
    assert ref == 3 and rig[ref]["model"] == "kb4"

    th = vt._kb4_fold_theta(rig[3]["params"])
    assert 0.69 < th < 0.71, f"camd fold angle {th:.3f} rad, expected ~0.700"
    masks = {s: vt._reachable_quadrants(rig[s], rig[ref]).tolist()
             for s in sorted(rig)}
    assert masks == {1: [True, False, True, False],
                     2: [True, True, False, False],
                     3: [True, True, True, True]}, masks
    dp_rig, dp_ref = vt.load_rig(CALIB)
    for s in (1, 2):   # the DP180 default check set must stay all-four
        m = vt._reachable_quadrants(dp_rig[s], dp_rig[dp_ref])
        assert m.all(), f"DP180 socket {s} reachable mask {m.tolist()}"
    print("PASS  nv180: reachable-quadrant masks exact on both rigs, "
          f"camd fold angle {math.degrees(th):.1f} deg")

    cfg = vt.VioConfig()
    cloud = vt._default_cloud()
    center = 0.5 * (cloud.min(0) + cloud.max(0))
    geom = vt.PathGeometry(cloud, center, np.array([0.0, 0.0, 1.0]),
                           rig[ref])
    times, eyes, R_wc = vt.sample_path(cfg, geom)

    metrics, checks = vt.run_checks(cfg, geom, rig, ref, times, eyes, R_wc)
    by = {name: ok for name, ok, _msg, _strict in checks}
    q = metrics["coverage"][2]["quad"]
    assert q[0] > 0 and q[1] > 0 and q[2] == 0 and q[3] == 0, q.tolist()
    assert by["quadrant coverage camc"], \
        "camc must pass on its reachable upper half"
    assert sorted(metrics["check_cams"]) == [1, 2], \
        f"derived check set {metrics['check_cams']}: camd is the " \
        f"reference and a navigation lens, camb+camc are tracked"
    assert "quadrant coverage camd" not in by, \
        "the untracked camd must get no coverage rows by default"
    assert by["px formula <= ceiling"], \
        f"tracked focal (camc fx 678) bounds the default run " \
        f"({metrics['peak']['px_formula']:.1f} px)"
    # armed for every rig: the reference-framed far design must fail the
    # scene-plausibility rows even without VIO_CHECK_CAMS
    assert not by["bands within the fit cone"], \
        f"{geom.d_far / geom.d_fit:.1f}x fit must fail the cone check"
    assert not by["peak speed plausible"], \
        f"{metrics['peak']['v']:.2f} m/s must fail the speed ceiling"
    print(f"PASS  nv180: default checks - derived pair {{1,2}}, camc "
          f"passes on quadrants {q.tolist()}, px formula "
          f"{metrics['peak']['px_formula']:.1f} px on the tracked focal, "
          f"and the far design fails both scene-plausibility rows")

    os.environ["VIO_CHECK_CAMS"] = "1,2"
    try:
        fs = vt.select_framing(rig, ref)
        assert fs == 2, f"framing pick {fs}: camc (20.3 deg off the " \
                        f"reference) must beat camb (75.5 deg)"
        geom_f = vt.PathGeometry(cloud, center, np.array([0.0, 0.0, 1.0]),
                                 rig[ref], frame_cam=rig[fs], frame_sock=fs)
        assert geom_f.d_far <= 5.0 * geom_f.d_fit, \
            (geom_f.d_far, geom_f.d_fit)
        assert geom_f.d_far < 4.5, \
            f"far band {geom_f.d_far:.2f} m - outside a room again"
        times_f, eyes_f, R_f = vt.sample_path(cfg, geom_f)
        metrics, checks = vt.run_checks(cfg, geom_f, rig, ref, times_f,
                                        eyes_f, R_f)
        names = [name for name, *_ in checks]
        assert metrics["check_cams"] == [1, 2]
        assert "quadrant coverage camb" in names
        assert "quadrant coverage camd" not in names
        assert "boards inside framing camc" in names
        assert "boards inside reference cam" not in names
        assert "bands within the fit cone" in names
        assert "peak speed plausible" in names
        failed = [name for name, ok, _msg, strict in checks
                  if strict and not ok]
        assert failed == [], f"strict failures with the framed path: {failed}"
        assert metrics["peak"]["v"] <= cfg.max_speed
        assert metrics["peak"]["px_formula"] <= cfg.trackable_px
        assert metrics["peak"]["px_measured"] <= cfg.trackable_px
        print(f"PASS  nv180: framed in camc - bands "
              f"{geom_f.d_near:.2f}..{geom_f.d_far:.2f} m, budget "
              f"({geom_f.bud_x:.0f}, {geom_f.bud_y:.0f}) deg, peak speed "
              f"{metrics['peak']['v']:.2f} m/s, all strict checks pass")

        # 8. the old reference-framed far path cannot pass armed checks
        metrics_b, checks_b = vt.run_checks(cfg, geom, rig, ref, times,
                                            eyes, R_wc)
        failed_b = [name for name, ok, _msg, strict in checks_b
                    if strict and not ok]
        assert "bands within the fit cone" in failed_b, failed_b
        assert "peak speed plausible" in failed_b, failed_b
        print(f"PASS  nv180: the reference-framed far path "
              f"({geom.d_near:.2f}..{geom.d_far:.2f} m, "
              f"{metrics_b['peak']['v']:.2f} m/s) fails the "
              f"scene-plausibility checks: {failed_b}")
    finally:
        del os.environ["VIO_CHECK_CAMS"]

    # 9. derivation and framing with no knob set
    socks, declared = vt.tracked_sockets(rig, ref)
    assert not declared and sorted(socks) == [1, 2], socks
    dp_socks, dp_declared = vt.tracked_sockets(dp_rig, dp_ref)
    assert not dp_declared and sorted(dp_socks) == [1, 2, 3], dp_socks

    up = np.array([0.0, 0.0, 1.0])
    fs_dp, geom_dp = vt.plan_framing(dp_rig, dp_ref, cloud, center, up)
    assert fs_dp is None, "the DP180's wide reference must keep framing"
    g_ref = vt.PathGeometry(cloud, center, up, dp_rig[dp_ref])
    assert (geom_dp.d_near, geom_dp.d_far, geom_dp.bud_x, geom_dp.bud_y) \
        == (g_ref.d_near, g_ref.d_far, g_ref.bud_x, g_ref.bud_y), \
        "DP180 derived design must equal the historical reference design"
    t_dp, e_dp, R_dp = vt.sample_path(cfg, geom_dp)
    m_dp, c_dp = vt.run_checks(cfg, geom_dp, dp_rig, dp_ref, t_dp, e_dp,
                               R_dp)
    assert m_dp["check_cams"] == [2, 1], \
        f"DP180 derived rows {m_dp['check_cams']}: the historical order"
    m_old, _ = vt.run_checks(cfg, geom_dp, dp_rig, dp_ref, t_dp, e_dp,
                             R_dp, check_cams=[2, 1])  # whole-rig scoping
    assert m_dp["peak"]["px_formula"] == m_old["peak"]["px_formula"]
    assert m_dp["peak"]["px_measured"] == m_old["peak"]["px_measured"]
    assert not [n for n, ok, _m, s in c_dp if s and not ok]
    print("PASS  nv180: derived sets {1,2} / {1,2,3}; DP180 keeps the "
          "reference design, rows [camc, camb] and px numbers unchanged")

    # the NV180 with no knob == the VIO_CHECK_CAMS=1,2 framed design
    fs_nv, geom_nv = vt.plan_framing(rig, ref, cloud, center, up)
    assert fs_nv == 2, f"derived framing {fs_nv}: camc must take over"
    t_nv, e_nv, R_nv = vt.sample_path(cfg, geom_nv)
    assert np.array_equal(e_nv, eyes_f) and np.array_equal(R_nv, R_f), \
        "derived design must equal the declared 1,2 design bit for bit"
    _m_nv, c_nv = vt.run_checks(cfg, geom_nv, rig, ref, t_nv, e_nv, R_nv)
    failed_nv = [n for n, ok, _m, s in c_nv if s and not ok]
    assert failed_nv == [], failed_nv
    print("PASS  nv180: derived default framed in camc, bit-identical "
          "to the declared 1,2 path, all strict checks pass")

    # 10. the second converted unit: the same derived design from its
    # own calibration (units share the design, never the values - the
    # camc-camd tilt alone differs by 0.117 deg, ~9 px far away in camd)
    rig_b, ref_b = vt.load_rig(NV_CALIB_B)
    assert ref_b == 3
    assert {s: c["model"] for s, c in rig_b.items()} == \
        {1: "ds", 2: "kb4", 3: "kb4"}
    socks_b, declared_b = vt.tracked_sockets(rig_b, ref_b)
    assert not declared_b and sorted(socks_b) == [1, 2], socks_b
    fs_b, geom_b = vt.plan_framing(rig_b, ref_b, cloud, center, up)
    assert fs_b == 2, f"unit B framing {fs_b}: camc must take over"
    assert geom_b.d_far <= vt.FIT_CONE_MAX * geom_b.d_fit
    th_b = vt._kb4_fold_theta(rig_b[3]["params"])
    assert 0.69 < th_b < 0.72, \
        f"unit B camd fold {th_b:.3f} rad (native cap 40.3 deg)"
    print(f"PASS  nv180: unit 80.120.0101 - same derived design (pair "
          f"{{1,2}}, framed camc, bands {geom_b.d_near:.2f}.."
          f"{geom_b.d_far:.2f} m) from its own calibration (camd fold "
          f"{math.degrees(th_b):.1f} deg)")


def main():
    rig, ref = vt.load_rig(CALIB)
    assert rig[ref]["model"] == "ds", "reference camera should be DS here"
    run_scene("single-board 30cm", *scene_single(), rig, ref)
    run_scene("default 3-board 15cm", *scene_default(), rig, ref)
    run_nv180()
    print("\nall vio trajectory checks passed")


if __name__ == "__main__":
    main()
