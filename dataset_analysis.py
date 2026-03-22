import os
import re
import csv
import json
import glob
import math
import argparse
from collections import Counter, defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class RLDatasetAnalyzer:
    """
    Comprehensive dataset analysis for the BUS tumor-localization roadmap.

    What this script does:
    - integrity audit: image/mask pairing, unreadable files, empty masks, dimension audit
    - multi-focal analysis: connected components, tumor-count distribution, inter-centroid distances
    - bbox extraction: merged-mask bounding box + adaptive margin box
    - geometry: area ratio, aspect ratio, center heatmap, boundary-touch frequency, size threshold flags
    - patch/state sufficiency: local 64x64 visibility, global 32x32 density
    - initial-box difficulty: IoU distribution for random initial boxes (10-20% area)
    - action sensitivity: expected |ΔIoU| for translation / morphology / scaling actions
    - intensity stats: exact dataset mean/std over all pixels + histogram sample
    - overlay suspicion heuristic: flag images likely containing calipers/text markers

    Important:
    - split analysis is only done if an explicit split CSV/TXT/JSON is provided.
    - overlay detection is heuristic. It is for flagging, not ground truth exclusion.
    """

    def __init__(
        self,
        img_dir="Dataset/Breast_US_single_tumor/img",
        mask_dir="Dataset/Breast_US_single_tumor/mask",
        report_dir="Reports/dataset_analysis_2",
        img_size=256,
        patch_size=64,
        global_size=32,
        step_pct=0.10,
        min_tumor_px=8,
        min_tumor_ratio=0.003,
        split_file=None,
        seed=42,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.report_dir = report_dir
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.global_size = int(global_size)
        self.step_pct = float(step_pct)
        self.min_tumor_px = int(min_tumor_px)
        self.min_tumor_ratio = float(min_tumor_ratio)
        self.split_file = split_file
        self.rng = np.random.default_rng(seed)

        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "tables"), exist_ok=True)

        self.split_map = self._load_split_map(split_file) if split_file else {}
        self.rows = []
        self.overlay_flags = []
        self.issues = defaultdict(list)

        self.global_pixel_sum = 0.0
        self.global_pixel_sq_sum = 0.0
        self.global_pixel_count = 0
        self.pixel_hist_sample = []

    # -----------------------------
    # IO and bookkeeping
    # -----------------------------
    def _load_split_map(self, split_file):
        mapping = {}
        ext = os.path.splitext(split_file)[1].lower()

        if ext == ".json":
            with open(split_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    mapping[os.path.basename(k)] = str(v).lower()
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "filename" in item and "split" in item:
                        mapping[os.path.basename(item["filename"])] = str(item["split"]).lower()
        else:
            with open(split_file, "r", encoding="utf-8") as f:
                sample = f.read(4096)
                f.seek(0)
                if "," in sample:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fn = row.get("filename") or row.get("image") or row.get("img") or row.get("path")
                        sp = row.get("split")
                        if fn and sp:
                            mapping[os.path.basename(fn)] = str(sp).lower()
                else:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = re.split(r"\s+", line)
                        if len(parts) >= 2:
                            mapping[os.path.basename(parts[0])] = str(parts[1]).lower()
        return mapping

    @staticmethod
    def _safe_div(a, b):
        return float(a) / float(b) if b else 0.0

    @staticmethod
    def _quantiles(values, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
        if len(values) == 0:
            return {str(q): 0.0 for q in qs}
        arr = np.asarray(values, dtype=float)
        out = np.quantile(arr, qs)
        return {str(q): float(v) for q, v in zip(qs, out)}

    @staticmethod
    def _stats(values):
        if len(values) == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "q05": 0.0,
                "q25": 0.0,
                "median": 0.0,
                "q75": 0.0,
                "q95": 0.0,
            }
        arr = np.asarray(values, dtype=float)
        q = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "q05": float(q[0]),
            "q25": float(q[1]),
            "median": float(q[2]),
            "q75": float(q[3]),
            "q95": float(q[4]),
        }

    def _write_csv(self, path, rows, fieldnames):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # -----------------------------
    # Image and geometry helpers
    # -----------------------------
    """
    def image_to_mask_filename(image_filename: str) -> str:
    
    # Breast_US_single_tumor-style: same filename in img/ and mask/
    if image_filename.startswith(("benign_", "malignant_")):
        return image_filename

    # BUS-style: bus_XXXX -> mask_XXXX
    if image_filename.startswith("bus_"):
        return image_filename.replace("bus_", "mask_", 1)

    # fallback
    return image_filename
    """
    def _resolve_mask_path(self, image_path):
        filename = os.path.basename(image_path)

        # Case 1: same filename exists in mask dir
        same_name = os.path.join(self.mask_dir, filename)
        if os.path.exists(same_name):
            return same_name

        # Case 2: BUS naming rule: bus_xxx -> mask_xxx
        if filename.startswith("bus_"):
            bus_mask = os.path.join(self.mask_dir, filename.replace("bus_", "mask_", 1))
            if os.path.exists(bus_mask):
                return bus_mask

        return None


    def _read_gray(self, path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _resize_pair(self, img, mask):
        img_r = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask_b = (mask_r > 0).astype(np.uint8)
        return img_r, mask_b

    @staticmethod
    def compute_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _clip_box(self, box):
        x1, y1, x2, y2 = box
        x1 = float(np.clip(x1, 0, self.img_size))
        y1 = float(np.clip(y1, 0, self.img_size))
        x2 = float(np.clip(x2, 0, self.img_size))
        y2 = float(np.clip(y2, 0, self.img_size))
        return [x1, y1, x2, y2]

    def _random_initial_box(self):
        target_area = self.rng.uniform(0.10, 0.20) * (self.img_size ** 2)
        aspect = self.rng.uniform(0.5, 2.0)
        w = math.sqrt(target_area * aspect)
        h = math.sqrt(target_area / aspect)
        w = min(w, self.img_size - 1)
        h = min(h, self.img_size - 1)
        x1 = self.rng.uniform(0, self.img_size - w)
        y1 = self.rng.uniform(0, self.img_size - h)
        return [x1, y1, x1 + w, y1 + h]

    def _simulate_actions(self, box, gt):
        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        base = self.compute_iou(box, gt)
        dx = self.step_pct * w
        dy = self.step_pct * h

        translations = [
            [x1 - dx, y1, x2 - dx, y2],
            [x1 + dx, y1, x2 + dx, y2],
            [x1, y1 - dy, x2, y2 - dy],
            [x1, y1 + dy, x2, y2 + dy],
        ]
        morph = [
            [x1 - dx / 2, y1, x2 + dx / 2, y2],
            [x1 + dx / 2, y1, x2 - dx / 2, y2],
            [x1, y1 - dy / 2, x2, y2 + dy / 2],
            [x1, y1 + dy / 2, x2, y2 - dy / 2],
        ]
        scale = [
            [x1 - dx / 2, y1 - dy / 2, x2 + dx / 2, y2 + dy / 2],
            [x1 + dx / 2, y1 + dy / 2, x2 - dx / 2, y2 - dy / 2],
        ]

        def summarize(action_boxes):
            signed = []
            absolute = []
            valid = 0
            for a in action_boxes:
                a = self._clip_box(a)
                if a[2] <= a[0] or a[3] <= a[1]:
                    continue
                new_iou = self.compute_iou(a, gt)
                d = new_iou - base
                signed.append(d)
                absolute.append(abs(d))
                valid += 1
            return {
                "mean_abs": float(np.mean(absolute)) if absolute else 0.0,
                "mean_signed": float(np.mean(signed)) if signed else 0.0,
                "max_abs": float(np.max(absolute)) if absolute else 0.0,
                "valid_actions": int(valid),
            }

        return summarize(translations), summarize(morph), summarize(scale)

    @staticmethod
    def _connected_component_stats(mask_bin):
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
        # exclude background at index 0
        count = max(0, n_labels - 1)
        comps = []
        for idx in range(1, n_labels):
            x, y, w, h, area = stats[idx]
            cx, cy = centroids[idx]
            comps.append({
                "label": idx,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(area),
                "cx": float(cx),
                "cy": float(cy),
            })
        return count, comps

    @staticmethod
    def _bbox_from_mask(mask_bin):
        ys, xs = np.where(mask_bin > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        xmin = int(xs.min())
        xmax = int(xs.max())
        ymin = int(ys.min())
        ymax = int(ys.max())
        return xmin, ymin, xmax, ymax

    def _overlay_suspicion_score(self, img_u8):
        """
        Heuristic only.
        Flags possible measurement markers / text overlays using edge-heavy, bright, thin structures,
        with extra weight near image borders where overlays often sit.
        """
        h, w = img_u8.shape
        _, bright = cv2.threshold(img_u8, 235, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(img_u8, 120, 220)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=12, maxLineGap=2)

        border = np.zeros_like(img_u8, dtype=np.uint8)
        bw = max(8, int(0.08 * min(h, w)))
        border[:bw, :] = 1
        border[-bw:, :] = 1
        border[:, :bw] = 1
        border[:, -bw:] = 1

        bright_border_ratio = self._safe_div(np.sum((bright > 0) & (border > 0)), np.sum(border))
        edge_border_ratio = self._safe_div(np.sum((edges > 0) & (border > 0)), np.sum(border))

        line_count = 0
        plus_like = 0
        if lines is not None:
            lines = lines[:, 0, :]
            line_count = int(lines.shape[0])
            horizontal = []
            vertical = []
            for l in lines:
                x1, y1, x2, y2 = map(int, l)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx >= 8 and dy <= 2:
                    horizontal.append((x1, y1, x2, y2))
                if dy >= 8 and dx <= 2:
                    vertical.append((x1, y1, x2, y2))
            for hx1, hy1, hx2, hy2 in horizontal:
                hy = (hy1 + hy2) / 2.0
                for vx1, vy1, vx2, vy2 in vertical:
                    vx = (vx1 + vx2) / 2.0
                    if min(hx1, hx2) <= vx <= max(hx1, hx2) and min(vy1, vy2) <= hy <= max(vy1, vy2):
                        plus_like += 1
                        break

        score = 4.0 * bright_border_ratio + 3.0 * edge_border_ratio + 0.03 * line_count + 0.20 * plus_like
        suspicious = bool(score >= 0.35 or plus_like >= 1 or line_count >= 18)
        return {
            "overlay_score": float(score),
            "overlay_suspected": suspicious,
            "bright_border_ratio": float(bright_border_ratio),
            "edge_border_ratio": float(edge_border_ratio),
            "line_count": int(line_count),
            "plus_like_count": int(plus_like),
        }

    # -----------------------------
    # Main analysis
    # -----------------------------
    def analyze(self):
        image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        if not image_paths:
            raise RuntimeError(f"No images found in {self.img_dir}")

        for image_path in tqdm(image_paths, desc="Analyzing dataset"):
            image_filename = os.path.basename(image_path)
            mask_path = self._resolve_mask_path(image_path)
            mask_filename = os.path.basename(mask_path) if mask_path is not None else None

            row = {
                "image_filename": image_filename,
                "mask_filename": mask_filename,
                "status": "ok",
                "split": self.split_map.get(image_filename, "unknown"),
            }

            if mask_path is None or not os.path.exists(mask_path):
                row["status"] = "missing_mask"
                self.issues["missing_mask"].append(image_filename)
                self.rows.append(row)
                continue

            img0 = self._read_gray(image_path)
            mask0 = self._read_gray(mask_path)
            if img0 is None or mask0 is None:
                row["status"] = "corrupt_or_unreadable"
                self.issues["corrupt_or_unreadable"].append(image_filename)
                self.rows.append(row)
                continue

            row["orig_h"] = int(img0.shape[0])
            row["orig_w"] = int(img0.shape[1])
            row["mask_orig_h"] = int(mask0.shape[0])
            row["mask_orig_w"] = int(mask0.shape[1])
            row["nonstandard_image_dims"] = int((img0.shape[0] != self.img_size) or (img0.shape[1] != self.img_size))
            row["image_mask_dim_mismatch"] = int(img0.shape[:2] != mask0.shape[:2])

            overlay = self._overlay_suspicion_score(img0)
            row.update(overlay)
            if overlay["overlay_suspected"]:
                self.overlay_flags.append(image_filename)

            img, mask_bin = self._resize_pair(img0, mask0)
            if int(mask_bin.sum()) == 0:
                row["status"] = "empty_mask"
                self.issues["empty_mask"].append(image_filename)
                self.rows.append(row)
                continue

            self.global_pixel_sum += float(img.sum())
            self.global_pixel_sq_sum += float((img.astype(np.float64) ** 2).sum())
            self.global_pixel_count += int(img.size)
            flat = img.reshape(-1)
            sample_n = min(500, flat.size)
            sample_idx = self.rng.choice(flat.size, size=sample_n, replace=False)
            self.pixel_hist_sample.extend((flat[sample_idx] / 255.0).tolist())

            tumor_count, components = self._connected_component_stats(mask_bin)
            bbox = self._bbox_from_mask(mask_bin)
            if bbox is None:
                row["status"] = "empty_mask_after_resize"
                self.issues["empty_mask_after_resize"].append(image_filename)
                self.rows.append(row)
                continue

            xmin, ymin, xmax, ymax = bbox
            w = int(xmax - xmin)
            h = int(ymax - ymin)
            if w <= 0 or h <= 0:
                row["status"] = "degenerate_bbox"
                self.issues["degenerate_bbox"].append(image_filename)
                self.rows.append(row)
                continue

            gt = [float(xmin), float(ymin), float(xmax), float(ymax)]
            area = int(w * h)
            ratio = area / float(self.img_size ** 2)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            margin = 0.05 * min(w, h)
            margin_box = self._clip_box([xmin - margin, ymin - margin, xmax + margin, ymax + margin])
            boundary_touch = int(xmin <= 0 or ymin <= 0 or xmax >= self.img_size - 1 or ymax >= self.img_size - 1)
            micro_tumor = int(w < self.min_tumor_px or h < self.min_tumor_px or ratio < self.min_tumor_ratio)

            inter_dist_mean = 0.0
            inter_dist_min = 0.0
            if len(components) > 1:
                centers = np.array([[c["cx"], c["cy"]] for c in components], dtype=float)
                dmat = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
                dists = dmat[np.triu_indices(len(centers), k=1)]
                if dists.size > 0:
                    inter_dist_mean = float(dists.mean())
                    inter_dist_min = float(dists.min())

            init_box = self._random_initial_box()
            initial_iou = self.compute_iou(init_box, gt)
            margin_iou = self.compute_iou(init_box, margin_box)
            margin_reward_shift = margin_iou - initial_iou

            px = int((init_box[0] + init_box[2]) / 2.0)
            py = int((init_box[1] + init_box[3]) / 2.0)
            half = self.patch_size // 2
            x1 = max(0, px - half)
            x2 = min(self.img_size, px + half)
            y1 = max(0, py - half)
            y2 = min(self.img_size, py + half)
            visible_ratio = self._safe_div(mask_bin[y1:y2, x1:x2].sum(), mask_bin.sum())

            global_mask = cv2.resize(mask_bin, (self.global_size, self.global_size), interpolation=cv2.INTER_NEAREST)
            global_density = float(global_mask.mean())

            trans_stats, morph_stats, scale_stats = self._simulate_actions(init_box, gt)

            row.update({
                "status": "valid",
                "tumor_count": int(tumor_count),
                "component_areas": ";".join(str(c["area"]) for c in components),
                "inter_dist_mean": float(inter_dist_mean),
                "inter_dist_min": float(inter_dist_min),
                "bbox_xmin": int(xmin),
                "bbox_ymin": int(ymin),
                "bbox_xmax": int(xmax),
                "bbox_ymax": int(ymax),
                "bbox_w": int(w),
                "bbox_h": int(h),
                "bbox_area": int(area),
                "size_ratio": float(ratio),
                "aspect_ratio": float(w / float(h)),
                "center_x_px": float(cx),
                "center_y_px": float(cy),
                "center_x_norm": float(cx / self.img_size),
                "center_y_norm": float(cy / self.img_size),
                "boundary_touch": int(boundary_touch),
                "micro_tumor": int(micro_tumor),
                "margin_px": float(margin),
                "margin_box_xmin": float(margin_box[0]),
                "margin_box_ymin": float(margin_box[1]),
                "margin_box_xmax": float(margin_box[2]),
                "margin_box_ymax": float(margin_box[3]),
                "init_xmin": float(init_box[0]),
                "init_ymin": float(init_box[1]),
                "init_xmax": float(init_box[2]),
                "init_ymax": float(init_box[3]),
                "initial_iou": float(initial_iou),
                "margin_iou": float(margin_iou),
                "margin_reward_shift": float(margin_reward_shift),
                "local_visible_ratio": float(visible_ratio),
                "global_density_32": float(global_density),
                "mean_intensity_image": float(img.mean() / 255.0),
                "std_intensity_image": float(img.std() / 255.0),
                "delta_translation_abs": float(trans_stats["mean_abs"]),
                "delta_translation_signed": float(trans_stats["mean_signed"]),
                "delta_translation_max_abs": float(trans_stats["max_abs"]),
                "delta_morphology_abs": float(morph_stats["mean_abs"]),
                "delta_morphology_signed": float(morph_stats["mean_signed"]),
                "delta_morphology_max_abs": float(morph_stats["max_abs"]),
                "delta_scaling_abs": float(scale_stats["mean_abs"]),
                "delta_scaling_signed": float(scale_stats["mean_signed"]),
                "delta_scaling_max_abs": float(scale_stats["max_abs"]),
            })
            self.rows.append(row)

    # -----------------------------
    # Summary and reporting
    # -----------------------------
    def _valid_rows(self):
        return [r for r in self.rows if r.get("status") == "valid"]

    def _split_summary(self, rows):
        out = {}
        grouped = defaultdict(list)
        for r in rows:
            grouped[r.get("split", "unknown")].append(r)
        for split, split_rows in grouped.items():
            out[split] = {
                "count": len(split_rows),
                "size_ratio_mean": float(np.mean([r["size_ratio"] for r in split_rows])) if split_rows else 0.0,
                "aspect_ratio_mean": float(np.mean([r["aspect_ratio"] for r in split_rows])) if split_rows else 0.0,
                "center_x_mean": float(np.mean([r["center_x_norm"] for r in split_rows])) if split_rows else 0.0,
                "center_y_mean": float(np.mean([r["center_y_norm"] for r in split_rows])) if split_rows else 0.0,
            }
        return out

    def build_summary(self):
        valid = self._valid_rows()
        total = len(self.rows)
        valid_n = len(valid)

        dataset_mean = self._safe_div(self.global_pixel_sum, self.global_pixel_count) / 255.0
        dataset_var = self._safe_div(self.global_pixel_sq_sum, self.global_pixel_count) / (255.0 ** 2) - dataset_mean ** 2
        dataset_std = float(max(dataset_var, 0.0) ** 0.5)

        tumor_count_counter = Counter(int(r["tumor_count"]) for r in valid)

        action_reward_strength = 5.0 * np.array([
            np.mean([r["delta_translation_abs"] for r in valid]) if valid else 0.0,
            np.mean([r["delta_morphology_abs"] for r in valid]) if valid else 0.0,
            np.mean([r["delta_scaling_abs"] for r in valid]) if valid else 0.0,
        ])

        summary = {
            "dataset": {
                "image_dir": self.img_dir,
                "mask_dir": self.mask_dir,
                "target_resolution": [self.img_size, self.img_size],
                "total_pairs_seen": total,
                "valid_cases": valid_n,
                "missing_mask": len(self.issues["missing_mask"]),
                "corrupt_or_unreadable": len(self.issues["corrupt_or_unreadable"]),
                "empty_mask": len(self.issues["empty_mask"]),
                "empty_mask_after_resize": len(self.issues["empty_mask_after_resize"]),
                "degenerate_bbox": len(self.issues["degenerate_bbox"]),
                "overlay_suspected": len([r for r in self.rows if r.get("overlay_suspected")]),
                "nonstandard_image_dims": len([r for r in self.rows if r.get("nonstandard_image_dims") == 1]),
                "image_mask_dim_mismatch": len([r for r in self.rows if r.get("image_mask_dim_mismatch") == 1]),
            },
            "quality_checks": {
                "boundary_touch_frequency": self._safe_div(sum(r["boundary_touch"] for r in valid), valid_n),
                "micro_tumor_frequency": self._safe_div(sum(r["micro_tumor"] for r in valid), valid_n),
                "overlay_flag_frequency": self._safe_div(len([r for r in self.rows if r.get("overlay_suspected")]), total),
            },
            "multi_focal": {
                "tumor_count_distribution": {str(k): int(v) for k, v in sorted(tumor_count_counter.items())},
                "fraction_multi_focal": self._safe_div(sum(1 for r in valid if int(r["tumor_count"]) > 1), valid_n),
                "inter_dist_mean_stats": self._stats([r["inter_dist_mean"] for r in valid if r["inter_dist_mean"] > 0]),
                "inter_dist_min_stats": self._stats([r["inter_dist_min"] for r in valid if r["inter_dist_min"] > 0]),
            },
            "geometry": {
                "bbox_width": self._stats([r["bbox_w"] for r in valid]),
                "bbox_height": self._stats([r["bbox_h"] for r in valid]),
                "bbox_area": self._stats([r["bbox_area"] for r in valid]),
                "size_ratio": self._stats([r["size_ratio"] for r in valid]),
                "aspect_ratio": self._stats([r["aspect_ratio"] for r in valid]),
                "center_x_norm": self._stats([r["center_x_norm"] for r in valid]),
                "center_y_norm": self._stats([r["center_y_norm"] for r in valid]),
                "margin_px": self._stats([r["margin_px"] for r in valid]),
            },
            "state_sufficiency": {
                "local_visible_ratio": self._stats([r["local_visible_ratio"] for r in valid]),
                "global_density_32": self._stats([r["global_density_32"] for r in valid]),
                "fraction_local_visibility_lt_0_25": self._safe_div(sum(1 for r in valid if r["local_visible_ratio"] < 0.25), valid_n),
                "fraction_local_visibility_lt_0_50": self._safe_div(sum(1 for r in valid if r["local_visible_ratio"] < 0.50), valid_n),
            },
            "initial_box": {
                "initial_iou": self._stats([r["initial_iou"] for r in valid]),
                "margin_reward_shift": self._stats([r["margin_reward_shift"] for r in valid]),
            },
            "intensity": {
                "dataset_mean": float(dataset_mean),
                "dataset_std": float(dataset_std),
                "per_image_mean": self._stats([r["mean_intensity_image"] for r in valid]),
                "per_image_std": self._stats([r["std_intensity_image"] for r in valid]),
            },
            "action_sensitivity": {
                "translation_abs": self._stats([r["delta_translation_abs"] for r in valid]),
                "morphology_abs": self._stats([r["delta_morphology_abs"] for r in valid]),
                "scaling_abs": self._stats([r["delta_scaling_abs"] for r in valid]),
                "scaled_reward_strength": {
                    "translation": float(action_reward_strength[0]),
                    "morphology": float(action_reward_strength[1]),
                    "scaling": float(action_reward_strength[2]),
                    "step_penalty_reference": 0.02,
                },
            },
            "splits": {
                "split_file": self.split_file,
                "summary": self._split_summary(valid) if self.split_map else {},
                "note": "No split analysis was performed because no split file was provided." if not self.split_map else "Split analysis uses the provided split file only.",
            },
        }
        return summary

    def write_reports(self):
        valid = self._valid_rows()
        summary = self.build_summary()

        table_path = os.path.join(self.report_dir, "tables", "per_image_analysis.csv")
        fieldnames = sorted({k for r in self.rows for k in r.keys()})
        self._write_csv(table_path, self.rows, fieldnames)

        overlay_rows = [
            {"image_filename": r["image_filename"], "overlay_score": r.get("overlay_score", 0.0), "line_count": r.get("line_count", 0), "plus_like_count": r.get("plus_like_count", 0)}
            for r in self.rows if r.get("overlay_suspected")
        ]
        overlay_rows.sort(key=lambda x: x["overlay_score"], reverse=True)
        self._write_csv(
            os.path.join(self.report_dir, "tables", "overlay_flags.csv"),
            overlay_rows,
            ["image_filename", "overlay_score", "line_count", "plus_like_count"],
        )

        with open(os.path.join(self.report_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        lines = []
        lines.append("DATASET ANALYSIS REPORT")
        lines.append("=" * 80)
        d = summary["dataset"]
        lines.append(f"Total files seen: {d['total_pairs_seen']}")
        lines.append(f"Valid cases: {d['valid_cases']}")
        lines.append(f"Missing masks: {d['missing_mask']}")
        lines.append(f"Unreadable/corrupt files: {d['corrupt_or_unreadable']}")
        lines.append(f"Empty masks: {d['empty_mask']} | Empty after resize: {d['empty_mask_after_resize']}")
        lines.append(f"Degenerate boxes: {d['degenerate_bbox']}")
        lines.append(f"Non-standard image dims: {d['nonstandard_image_dims']}")
        lines.append(f"Image/mask dim mismatch: {d['image_mask_dim_mismatch']}")
        lines.append(f"Overlay-suspected images: {d['overlay_suspected']}")
        lines.append("")

        q = summary["quality_checks"]
        lines.append("QUALITY CHECKS")
        lines.append("-" * 80)
        lines.append(f"Boundary-touch frequency: {100.0 * q['boundary_touch_frequency']:.2f}%")
        lines.append(f"Micro-tumor frequency: {100.0 * q['micro_tumor_frequency']:.2f}%")
        lines.append(f"Overlay-flag frequency: {100.0 * q['overlay_flag_frequency']:.2f}%")
        lines.append("")

        mf = summary["multi_focal"]
        lines.append("MULTI-FOCAL ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Tumor-count distribution: {mf['tumor_count_distribution']}")
        lines.append(f"Fraction with >1 tumor region: {100.0 * mf['fraction_multi_focal']:.2f}%")
        lines.append("")

        g = summary["geometry"]
        lines.append("GEOMETRY")
        lines.append("-" * 80)
        lines.append(f"Size ratio mean: {g['size_ratio']['mean']:.5f} | median: {g['size_ratio']['median']:.5f} | range: [{g['size_ratio']['min']:.5f}, {g['size_ratio']['max']:.5f}]")
        lines.append(f"Aspect ratio mean: {g['aspect_ratio']['mean']:.5f} | median: {g['aspect_ratio']['median']:.5f} | range: [{g['aspect_ratio']['min']:.5f}, {g['aspect_ratio']['max']:.5f}]")
        lines.append(f"Width px mean: {g['bbox_width']['mean']:.2f} | Height px mean: {g['bbox_height']['mean']:.2f}")
        lines.append(f"Adaptive margin px mean: {g['margin_px']['mean']:.2f}")
        lines.append("")

        ss = summary["state_sufficiency"]
        lines.append("STATE SUFFICIENCY")
        lines.append("-" * 80)
        lines.append(f"Local visible ratio mean: {ss['local_visible_ratio']['mean']:.5f}")
        lines.append(f"Fraction local visibility < 0.25: {100.0 * ss['fraction_local_visibility_lt_0_25']:.2f}%")
        lines.append(f"Fraction local visibility < 0.50: {100.0 * ss['fraction_local_visibility_lt_0_50']:.2f}%")
        lines.append(f"Global 32x32 density mean: {ss['global_density_32']['mean']:.5f}")
        lines.append("")

        ib = summary["initial_box"]
        lines.append("INITIAL BOX DIFFICULTY")
        lines.append("-" * 80)
        lines.append(f"Initial IoU mean: {ib['initial_iou']['mean']:.5f} | median: {ib['initial_iou']['median']:.5f} | q95: {ib['initial_iou']['q95']:.5f}")
        lines.append(f"Margin reward shift mean: {ib['margin_reward_shift']['mean']:.5f}")
        lines.append("")

        inten = summary["intensity"]
        lines.append("INTENSITY")
        lines.append("-" * 80)
        lines.append(f"Dataset mean: {inten['dataset_mean']:.6f}")
        lines.append(f"Dataset std: {inten['dataset_std']:.6f}")
        lines.append("")

        act = summary["action_sensitivity"]
        lines.append("ACTION SENSITIVITY")
        lines.append("-" * 80)
        lines.append(f"Mean |ΔIoU| translation: {act['translation_abs']['mean']:.6f}")
        lines.append(f"Mean |ΔIoU| morphology: {act['morphology_abs']['mean']:.6f}")
        lines.append(f"Mean |ΔIoU| scaling: {act['scaling_abs']['mean']:.6f}")
        lines.append(
            "Scaled reward strength [translation, morphology, scaling]: "
            f"[{act['scaled_reward_strength']['translation']:.6f}, "
            f"{act['scaled_reward_strength']['morphology']:.6f}, "
            f"{act['scaled_reward_strength']['scaling']:.6f}] vs step penalty 0.02"
        )
        lines.append("")

        split_note = summary["splits"]["note"]
        lines.append("SPLIT ANALYSIS")
        lines.append("-" * 80)
        lines.append(split_note)
        if self.split_map:
            lines.append(json.dumps(summary["splits"]["summary"], indent=2))
        lines.append("")

        warnings = []
        if valid and act["scaled_reward_strength"]["translation"] < 0.02 and act["scaled_reward_strength"]["morphology"] < 0.02 and act["scaled_reward_strength"]["scaling"] < 0.02:
            warnings.append("All three action groups have mean scaled reward below the time penalty. Step size may be too small.")
        if valid and ss["local_visible_ratio"]["mean"] < 0.25:
            warnings.append("Average local patch visibility is very low. The agent may be partially blind at initialization.")
        if valid and ib["initial_iou"]["mean"] > 0.30:
            warnings.append("Initial boxes may be too easy. The search problem could be weak.")
        if d["overlay_suspected"] > 0:
            warnings.append("Some images are flagged as possible measurement-marker/text-overlay cases. Review overlay_flags.csv manually.")

        lines.append("WARNINGS")
        lines.append("-" * 80)
        if warnings:
            lines.extend([f"- {w}" for w in warnings])
        else:
            lines.append("- None")

        with open(os.path.join(self.report_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return summary

    # -----------------------------
    # Plotting
    # -----------------------------
    def _save_hist(self, values, title, xlabel, filename, bins=40, logy=False):
        plt.figure(figsize=(8, 5))
        if len(values) > 0:
            plt.hist(values, bins=bins, edgecolor="black", alpha=0.8)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        if logy:
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "plots", filename), dpi=160)
        plt.close()

    def make_plots(self):
        valid = self._valid_rows()
        if not valid:
            return

        self._save_hist([r["tumor_count"] for r in valid], "Tumor Count per Image", "Connected tumor regions", "01_tumor_count_distribution.png", bins=np.arange(1, max(3, max(r["tumor_count"] for r in valid) + 2)) - 0.5)
        self._save_hist([r["size_ratio"] for r in valid], "Tumor Size Ratio", "bbox area / image area", "02_size_ratio.png")
        self._save_hist([r["aspect_ratio"] for r in valid], "Aspect Ratio Distribution", "w / h", "03_aspect_ratio.png")
        self._save_hist([r["local_visible_ratio"] for r in valid], "Local Patch Visibility", "visible tumor fraction in 64x64 patch", "04_local_visibility.png")
        self._save_hist([r["initial_iou"] for r in valid], "Initial IoU Distribution", "IoU(init, GT)", "05_initial_iou.png")
        self._save_hist([r["margin_reward_shift"] for r in valid], "Margin Reward Shift", "IoU(init, margin_box) - IoU(init, GT)", "06_margin_reward_shift.png")
        self._save_hist(self.pixel_hist_sample, "Global Pixel Intensity Histogram", "pixel intensity in [0,1]", "07_pixel_histogram.png", bins=60)
        self._save_hist([r["overlay_score"] for r in self.rows if "overlay_score" in r], "Overlay Suspicion Score", "heuristic score", "08_overlay_score.png")

        # center heatmap
        plt.figure(figsize=(6, 6))
        xs = [r["center_x_norm"] for r in valid]
        ys = [r["center_y_norm"] for r in valid]
        plt.hist2d(xs, ys, bins=25)
        plt.gca().invert_yaxis()
        plt.title("Tumor Center Heatmap")
        plt.xlabel("x / 256")
        plt.ylabel("y / 256")
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "plots", "09_center_heatmap.png"), dpi=160)
        plt.close()

        # action sensitivity
        plt.figure(figsize=(8, 5))
        data = [
            [r["delta_translation_abs"] for r in valid],
            [r["delta_morphology_abs"] for r in valid],
            [r["delta_scaling_abs"] for r in valid],
        ]
        bp = plt.boxplot(data)
        plt.xticks([1, 2, 3], ["Translate", "Morph", "Scale"])
        plt.axhline(y=0.02 / 5.0, linestyle="--")
        plt.title("Action Sensitivity: Mean |ΔIoU|")
        plt.ylabel("|ΔIoU|")
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "plots", "10_action_sensitivity_boxplot.png"), dpi=160)
        plt.close()

        # split heatmaps only if real split info exists
        if self.split_map:
            split_values = sorted(set(r.get("split", "unknown") for r in valid))
            n = len(split_values)
            cols = min(3, n)
            rows_n = int(math.ceil(n / cols))
            plt.figure(figsize=(5 * cols, 5 * rows_n))
            for idx, split in enumerate(split_values, start=1):
                subset = [r for r in valid if r.get("split") == split]
                plt.subplot(rows_n, cols, idx)
                plt.hist2d([r["center_x_norm"] for r in subset], [r["center_y_norm"] for r in subset], bins=20)
                plt.gca().invert_yaxis()
                plt.title(f"{split} center heatmap")
                plt.xlabel("x / 256")
                plt.ylabel("y / 256")
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_dir, "plots", "11_split_heatmaps.png"), dpi=160)
            plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Comprehensive RL dataset analysis for BUS tumor localization.")
    parser.add_argument("--img_dir", type=str, default="Dataset/Breast_US_single_tumor/img")
    parser.add_argument("--mask_dir", type=str, default="Dataset/Breast_US_single_tumor/mask")
    parser.add_argument("--report_dir", type=str, default="Reports/dataset_analysis_2")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--global_size", type=int, default=32)
    parser.add_argument("--step_pct", type=float, default=0.10)
    parser.add_argument("--min_tumor_px", type=int, default=8)
    parser.add_argument("--min_tumor_ratio", type=float, default=0.003)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = RLDatasetAnalyzer(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        report_dir=args.report_dir,
        img_size=args.img_size,
        patch_size=args.patch_size,
        global_size=args.global_size,
        step_pct=args.step_pct,
        min_tumor_px=args.min_tumor_px,
        min_tumor_ratio=args.min_tumor_ratio,
        split_file=args.split_file,
        seed=args.seed,
    )
    analyzer.analyze()
    analyzer.write_reports()
    analyzer.make_plots()
    print(f"Analysis complete. Outputs written to: {analyzer.report_dir}")


if __name__ == "__main__":
    main()
