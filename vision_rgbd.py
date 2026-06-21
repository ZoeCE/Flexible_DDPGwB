"""OpenCV RGB-D payload pose estimation.

This module keeps the camera-to-lab geometry explicit:
  object(payload) -> OpenCV camera -> lab/world.

The simulator may render RGB-D frames, but payload ground truth must not enter
the policy observation path unless the caller explicitly uses a debug source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None


@dataclass
class CameraFrame:
    name: str
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    K: np.ndarray
    dist: np.ndarray
    T_lab_cam: np.ndarray


@dataclass
class PoseEstimate:
    camera_name: str
    T_lab_payload: np.ndarray
    reprojection_error: float
    depth_rmse: float
    depth_support: int
    marker_ids: Tuple[int, ...]
    weight: float


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for vision.source='opencv_rgbd'. Install "
            "opencv-contrib-python so cv2.aruco/AprilTag dictionaries are "
            "available.")
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is required. Use opencv-contrib-python, not the "
            "minimal opencv-python package.")
    return cv2


def get_aruco_dictionary(name: str, fallback: str = "DICT_4X4_50"):
    cv = require_cv2()
    aruco = cv.aruco
    dict_id = getattr(aruco, str(name), None)
    if dict_id is None:
        dict_id = getattr(aruco, str(fallback))
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def make_camera_matrix(width: int, height: int, fovy_deg: float) -> np.ndarray:
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * float(height) / max(np.tan(0.5 * fovy), 1e-9)
    fx = fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64)


def marker_object_points(marker_cfg: dict) -> np.ndarray:
    length = float(marker_cfg.get("length", marker_cfg.get("marker_length", 0.06)))
    center = np.asarray(marker_cfg.get("center", [0.0, 0.0, 0.10]),
                        dtype=np.float64).reshape(3)
    x_axis = np.asarray(marker_cfg.get("x_axis", [1.0, 0.0, 0.0]),
                       dtype=np.float64).reshape(3)
    y_axis = np.asarray(marker_cfg.get("y_axis", [0.0, 1.0, 0.0]),
                       dtype=np.float64).reshape(3)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-9)
    y_axis = y_axis - float(np.dot(x_axis, y_axis)) * x_axis
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-9)
    hx = 0.5 * length * x_axis
    hy = 0.5 * length * y_axis
    return np.asarray([
        center - hx + hy,
        center + hx + hy,
        center + hx - hy,
        center - hx - hy,
    ], dtype=np.float64)


def deproject_pixel(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    return np.array([
        (float(u) - K[0, 2]) * float(z) / K[0, 0],
        (float(v) - K[1, 2]) * float(z) / K[1, 1],
        float(z),
    ], dtype=np.float64)


def sample_depth_median(depth: np.ndarray, u: float, v: float,
                        radius: int = 2) -> Optional[float]:
    h, w = depth.shape[:2]
    ui = int(round(float(u)))
    vi = int(round(float(v)))
    x0 = max(0, ui - radius)
    x1 = min(w, ui + radius + 1)
    y0 = max(0, vi - radius)
    y1 = min(h, vi + radius + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    patch = patch[np.isfinite(patch) & (patch > 1e-6)]
    if patch.size == 0:
        return None
    return float(np.median(patch))


class OpenCVRGBDPoseEstimator:
    def __init__(self, config: dict):
        self.config = config or {}
        self.dictionary_name = str(self.config.get(
            "dictionary", "DICT_APRILTAG_36h11"))
        self.dictionary_fallback = str(self.config.get(
            "dictionary_fallback", "DICT_4X4_50"))
        self.depth_refine = bool(self.config.get(
            "depth_translation_refine", True))
        self.depth_window = int(self.config.get("depth_window", 2))
        self.max_reproj_error = float(self.config.get(
            "max_reprojection_error_px", 8.0))
        self.max_depth_rmse = float(self.config.get("max_depth_rmse_m", 0.05))
        self.min_markers = int(self.config.get("min_detected_markers", 1))
        self._marker_points = self._build_marker_points(
            self.config.get("markers", []))
        cv = require_cv2()
        aruco = cv.aruco
        self.dictionary = get_aruco_dictionary(
            self.dictionary_name, self.dictionary_fallback)
        if hasattr(aruco, "DetectorParameters"):
            params = aruco.DetectorParameters()
        else:
            params = aruco.DetectorParameters_create()
        if bool(self.config.get("corner_refine", True)):
            refine = getattr(aruco, "CORNER_REFINE_APRILTAG",
                             getattr(aruco, "CORNER_REFINE_SUBPIX", 1))
            try:
                params.cornerRefinementMethod = refine
            except Exception:
                pass
        self.params = params
        self.detector = (aruco.ArucoDetector(self.dictionary, self.params)
                         if hasattr(aruco, "ArucoDetector") else None)

    @staticmethod
    def _build_marker_points(markers: Iterable[dict]) -> Dict[int, np.ndarray]:
        out = {}
        for marker in markers:
            mid = int(marker.get("id", 0))
            out[mid] = marker_object_points(marker)
        return out

    def _detect_markers(self, rgb: np.ndarray):
        cv = require_cv2()
        image = np.asarray(rgb)
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = image
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.params)
        if ids is None:
            return [], np.zeros((0,), dtype=np.int32)
        return corners, ids.reshape(-1).astype(np.int32)

    def estimate_frame(self, frame: CameraFrame) -> Optional[PoseEstimate]:
        corners, ids = self._detect_markers(frame.rgb)
        object_pts: List[np.ndarray] = []
        image_pts: List[np.ndarray] = []
        used_ids: List[int] = []
        for marker_corners, marker_id in zip(corners, ids):
            marker_id = int(marker_id)
            if marker_id not in self._marker_points:
                continue
            pts2 = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            object_pts.append(self._marker_points[marker_id])
            image_pts.append(pts2)
            used_ids.append(marker_id)
        if len(used_ids) < self.min_markers:
            return None

        obj = np.concatenate(object_pts, axis=0).astype(np.float64)
        img = np.concatenate(image_pts, axis=0).astype(np.float64)
        ok, rvec, tvec = self._solve_pnp(obj, img, frame.K, frame.dist)
        if not ok:
            return None
        R_cam_payload, _ = require_cv2().Rodrigues(rvec)
        t_cam_payload = np.asarray(tvec, dtype=np.float64).reshape(3)

        depth_rmse, depth_support, t_cam_payload = self._depth_check_and_refine(
            frame, obj, img, R_cam_payload, t_cam_payload)
        if depth_support > 0 and depth_rmse > self.max_depth_rmse:
            return None

        reproj = self._reprojection_error(
            obj, img, frame.K, frame.dist, rvec, t_cam_payload)
        if reproj > self.max_reproj_error:
            return None

        T_cam_payload = np.eye(4, dtype=np.float64)
        T_cam_payload[:3, :3] = R_cam_payload
        T_cam_payload[:3, 3] = t_cam_payload
        T_lab_payload = np.asarray(frame.T_lab_cam, dtype=np.float64) @ T_cam_payload
        weight = 1.0 / (1e-3 + reproj)
        if depth_support > 0:
            weight *= 1.0 / (1e-3 + depth_rmse)
        return PoseEstimate(
            camera_name=frame.name,
            T_lab_payload=T_lab_payload,
            reprojection_error=float(reproj),
            depth_rmse=float(depth_rmse),
            depth_support=int(depth_support),
            marker_ids=tuple(sorted(set(used_ids))),
            weight=float(weight),
        )

    def _solve_pnp(self, obj, img, K, dist):
        cv = require_cv2()
        if obj.shape[0] == 4:
            return cv.solvePnP(obj, img, K, dist, flags=cv.SOLVEPNP_ITERATIVE)
        flag = getattr(cv, "SOLVEPNP_SQPNP", cv.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec, _ = cv.solvePnPRansac(
            obj, img, K, dist,
            iterationsCount=int(self.config.get("pnp_ransac_iters", 100)),
            reprojectionError=float(self.config.get("pnp_ransac_error_px", 5.0)),
            confidence=float(self.config.get("pnp_ransac_confidence", 0.99)),
            flags=flag)
        if ok and hasattr(cv, "solvePnPRefineLM"):
            try:
                rvec, tvec = cv.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
            except Exception:
                pass
        return ok, rvec, tvec

    def _depth_check_and_refine(self, frame: CameraFrame, obj, img,
                                R_cam_payload, t_cam_payload):
        if frame.depth is None:
            return float("inf"), 0, t_cam_payload
        depth_pts = []
        pred_pts = []
        for p_obj, (u, v) in zip(obj, img):
            z = sample_depth_median(frame.depth, u, v, self.depth_window)
            if z is None:
                continue
            pred = R_cam_payload @ p_obj.reshape(3) + t_cam_payload
            depth_pts.append(deproject_pixel(u, v, z, frame.K))
            pred_pts.append(pred)
        if not depth_pts:
            return float("inf"), 0, t_cam_payload
        depth_arr = np.asarray(depth_pts, dtype=np.float64)
        pred_arr = np.asarray(pred_pts, dtype=np.float64)
        offsets = depth_arr - pred_arr
        refined_t = t_cam_payload
        if self.depth_refine:
            refined_t = t_cam_payload + np.median(offsets, axis=0)
            pred_arr = pred_arr + np.median(offsets, axis=0)
        rmse = float(np.sqrt(np.mean(np.sum((depth_arr - pred_arr) ** 2, axis=1))))
        return rmse, int(depth_arr.shape[0]), refined_t

    @staticmethod
    def _reprojection_error(obj, img, K, dist, rvec, tvec):
        cv = require_cv2()
        proj, _ = cv.projectPoints(obj, rvec, np.asarray(tvec).reshape(3, 1),
                                   K, dist)
        proj = proj.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))

    @staticmethod
    def fuse(estimates: Sequence[PoseEstimate]) -> Optional[dict]:
        if not estimates:
            return None
        weights = np.asarray([max(e.weight, 1e-6) for e in estimates],
                             dtype=np.float64)
        weights = weights / np.sum(weights)
        positions = np.asarray([e.T_lab_payload[:3, 3] for e in estimates],
                               dtype=np.float64)
        pos = np.sum(positions * weights[:, None], axis=0)
        rotations = R.from_matrix([e.T_lab_payload[:3, :3] for e in estimates])
        try:
            R_lab_payload = rotations.mean(weights=weights).as_matrix()
        except Exception:
            quats = rotations.as_quat()
            ref = quats[0].copy()
            for i in range(quats.shape[0]):
                if float(np.dot(ref, quats[i])) < 0.0:
                    quats[i] *= -1.0
            q = np.sum(quats * weights[:, None], axis=0)
            q /= max(float(np.linalg.norm(q)), 1e-9)
            R_lab_payload = R.from_quat(q).as_matrix()
        marker_ids = sorted({mid for est in estimates for mid in est.marker_ids})
        return {
            "pos": pos,
            "R": R_lab_payload,
            "weight": float(np.sum([e.weight for e in estimates])),
            "active_cameras": len(estimates),
            "active_camera_names": [e.camera_name for e in estimates],
            "marker_ids": marker_ids,
            "reprojection_error": float(np.average(
                [e.reprojection_error for e in estimates], weights=weights)),
            "depth_rmse": float(np.average(
                [e.depth_rmse if np.isfinite(e.depth_rmse) else 0.0
                 for e in estimates], weights=weights)),
            "depth_support": int(sum(e.depth_support for e in estimates)),
        }
