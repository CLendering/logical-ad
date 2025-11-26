import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import timm

from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from torchvision import transforms
import matplotlib.pyplot as plt

Tensor = torch.Tensor
ImageArray = np.ndarray

import segment_anything.automatic_mask_generator
from torchvision.ops import batched_nms


# 1. Define the patched function
def batched_nms_patched(boxes, scores, idxs, iou_threshold):
    # FORCE indices to match the device of the boxes (e.g., CUDA)
    if boxes.device != idxs.device:
        idxs = idxs.to(boxes.device)
    return batched_nms(boxes, scores, idxs, iou_threshold)


# 2. Apply the patch to the library
segment_anything.automatic_mask_generator.batched_nms = batched_nms_patched

print(
    "[Fix] Applied patch to segment_anything.automatic_mask_generator.batched_nms"
)

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------
@dataclass
class VisualObject:
    id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    embedding: Tensor
    area: float
    centroid: Tuple[float, float]
    prototype_id: int = -1

@dataclass
class SceneGraph:
    image_id: str
    nodes: List[VisualObject]
    # Map: src_id -> list of (dst_id, relative_dx_norm, relative_dy_norm)
    edges: Dict[int, List[Tuple[int, float, float]]]

@dataclass
class Violation:
    rule_name: str
    description: str
    severity: float


# -------------------------------------------------------------------------
# 1. CompositionMapExtractor
# -------------------------------------------------------------------------


class CompositionMapExtractor:
    """
    Exact replication of SALAD (ICCV 2025) composition map generation.

    Pipeline:
    1. DINO ViT-B/8 on 512x512 image.
    2. Interpolate features to 256x256 (dense).
    3. Global KMeans (K=6) on foreground features.
    4. SAM Automatic Mask Generator on 256x256 image.
    5. Refine: Vote for DINO label within each SAM mask.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_clusters: int = 6,  # SALAD default
        sam_checkpoint: str | None = "./sam_vit_h_4b8939.pth",
    ):
        self.device = device
        self.n_clusters = n_clusters

        # SALAD uses DINO ViT-Base/8
        print("[CompositionMapExtractor] Loading DINO ViT-Base/8 (SALAD backbone)...")
        self.dino = timm.create_model(
            "vit_base_patch8_224.dino",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        ).to(device)
        self.dino.requires_grad_(False)
        self.dino.eval()

        # Transform for DINO (512x512 input as per SALAD dataset config)
        self.transform_dino = transforms.Compose(
            [
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        # Clusterer
        self.kmeans = None

        # SAM Setup
        self.mask_generator = None
        if sam_checkpoint and os.path.exists(sam_checkpoint):
            try:
                from segment_anything import (
                    sam_model_registry,
                    SamAutomaticMaskGenerator,
                )

                sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)

                # Exact SALAD parameters for SAM
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=64,
                    pred_iou_thresh=0.8,
                    min_mask_region_area=50,
                    crop_n_layers=0,
                )
                print(
                    "[CompositionMapExtractor] SAM (ViT-H) initialized with SALAD params."
                )
            except Exception as e:
                print(f"[CompositionMapExtractor] SAM init failed: {e}")
        else:
            print(
                "[CompositionMapExtractor] WARNING: SAM checkpoint not found. Refinement will be skipped."
            )

    def _get_dino_features_256(
        self, image: ImageArray
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        1. Resize img to 512 (DINO input).
        2. Extract features.
        3. Resize features to 256x256 (SALAD logic).
        Returns:
             flat_tokens: [65536, 768] (numpy)
             feat_map_256: [768, 256, 256] (torch)
        """
        pil_img = Image.fromarray(image)
        # Input to DINO is 512x512
        img_t = self.transform_dino(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # forward_features usually returns [B, N, C]
            # For patch size 8 and img 512, N = (512/8)^2 = 64^2 = 4096 tokens + CLS
            out = self.dino.forward_features(img_t)

            # Remove CLS token if present
            if self.dino.num_prefix_tokens > 0:
                out = out[:, self.dino.num_prefix_tokens :, :]

            # Reshape to [B, H, W, C] -> [1, 64, 64, 768]
            B, N, C = out.shape
            H_grid = W_grid = int(np.sqrt(N))
            feat = out.reshape(B, H_grid, W_grid, C).permute(
                0, 3, 1, 2
            )  # [1, 768, 64, 64]

            # SALAD explicit resize to 256x256 using bilinear interpolation
            # Note: create_pseudo_labels.py uses torchvision.transforms.Resize((256,256))
            # which expects [..., H, W]
            feat_256 = F.interpolate(
                feat, size=(256, 256), mode="bilinear", align_corners=False
            )

            # Prepare flattened tokens for clustering: [256*256, 768]
            feat_flat = (
                feat_256.squeeze(0).permute(1, 2, 0).reshape(-1, C).cpu().numpy()
            )

            return feat_flat, feat_256.squeeze(0)

    def fit_global_clusters(self, normal_images: List[ImageArray]):
        """
        Fits KMeans (K=6) on features sampled from normal images.
        Mimics SALAD sampling: sample_size = 256*256 // 500 per image (approx).
        """
        print(
            f"[CompositionMapExtractor] Fitting Global KMeans (K={self.n_clusters})..."
        )
        all_feats = []

        # Approximate sampling rate to match SALAD memory constraints
        # SALAD: sample_size = 256 * 256 // 500  (~130 pixels per image)
        sample_per_img = 1000

        print(
            f"[CompositionMapExtractor] Sampling {sample_per_img} features per image for KMeans training."  
        )

        for img in normal_images:
            feats_flat, _ = self._get_dino_features_256(img)  # [65536, 768]

            # Random sample
            if feats_flat.shape[0] > sample_per_img:
                idx = np.random.choice(
                    feats_flat.shape[0], sample_per_img, replace=False
                )
                sampled = feats_flat[idx]
            else:
                sampled = feats_flat
            all_feats.append(sampled)

        training_data = np.concatenate(all_feats, axis=0)

        self.kmeans = KMeans(
            n_clusters=self.n_clusters, init="k-means++", random_state=42, n_init=10
        )
        self.kmeans.fit(training_data)
        print("[CompositionMapExtractor] KMeans fit complete.")

    def _transform_to_mask_array(self, sam_masks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Replicates SALAD's `transform_to_mask_array`.
        Sorts masks by area (Large -> Small).
        Paints them into a stack, allowing smaller masks to 'punch holes' in larger ones
        by clearing the pixel in previous layers.

        Result: [H, W, N] boolean stack where pixels belong to exactly one mask (the smallest one covering it).
        """
        if not sam_masks:
            return np.zeros((256, 256, 0))

        # Sort largest to smallest
        sorted_anns = sorted(sam_masks, key=(lambda x: x["area"]), reverse=True)

        # Initialize stack
        H, W = sorted_anns[0]["segmentation"].shape
        mask_array = np.zeros((H, W, len(sorted_anns)), dtype=bool)

        for i, ann in enumerate(sorted_anns):
            if ann["area"] < 50:  # SALAD threshold
                mask_array = mask_array[:, :, :i]
                break

            m = ann["segmentation"]
            # Clear this region in ALL channels (hole punching)
            # "mask_array[m, :] = 0" in python means where m is True, set all depth to 0
            mask_array[m, :] = False

            # Set current channel
            mask_array[m, i] = True

        return mask_array

    def build_composition_map(
        self, image: ImageArray
    ) -> Tuple[np.ndarray, torch.Tensor]:
        if self.kmeans is None:
            raise RuntimeError("Call fit_global_clusters first.")

        # 1. Get Dense DINO Features (256x256)
        feats_flat, feat_map_256 = self._get_dino_features_256(
            image
        )  # feats_flat is numpy

        # 2. Raw Clustering (mask_dino)
        # Predict on all pixels
        labels_raw = self.kmeans.predict(feats_flat)
        mask_dino = labels_raw.reshape(256, 256)  # [256, 256]

        # 3. SAM Generation
        if self.mask_generator:
            # Resize image to 256x256 for SAM (as per SALAD refine_masks_with_sam)
            img_256_pil = Image.fromarray(image).resize(
                (256, 256), resample=Image.BILINEAR
            )
            img_256 = np.array(img_256_pil)

            sam_anns = self.mask_generator.generate(img_256)

            # 4. Refinement / Voting
            # Create disjoint mask stack (Large -> Small, Small wins)
            sam_stack = self._transform_to_mask_array(sam_anns)  # [256, 256, N_masks]

            refined_map = np.zeros((256, 256), dtype=np.int32) - 1  # Default -1

            # For each disjoint region defined by SAM
            # Note: SALAD removes 'background' if bg mask is provided.
            # We assume BG is just another cluster here for simplicity unless provided.

            for i in range(sam_stack.shape[2]):
                curr_mask = sam_stack[:, :, i]
                if not np.any(curr_mask):
                    continue

                # Vote using underlying DINO labels
                dino_votes = mask_dino[curr_mask]
                if dino_votes.size == 0:
                    continue

                counts = np.bincount(dino_votes.astype(np.int64))
                most_common = np.argmax(counts)

                refined_map[curr_mask] = most_common

            final_map = refined_map
        else:
            final_map = mask_dino

        # Upscale final map to original image size if needed (usually 512)
        H_orig, W_orig = image.shape[:2]
        if H_orig != 256:
            # Nearest neighbor upscale
            final_map_t = torch.from_numpy(final_map.astype(np.float32))[
                None, None, ...
            ]
            final_map_hi = (
                F.interpolate(final_map_t, size=(H_orig, W_orig), mode="nearest")[0, 0]
                .numpy()
                .astype(np.int32)
            )

            # Also upscale feature map for the NS-AD object extractor
            feat_map_hi = F.interpolate(
                feat_map_256.unsqueeze(0), size=(H_orig, W_orig), mode="bilinear"
            )[0]
        else:
            final_map_hi = final_map
            feat_map_hi = feat_map_256

        return final_map_hi, feat_map_hi

    def save_composition_viz(self, comp_map: np.ndarray, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # -1 = Black. 0..K-1 = Random Colors.
        labels = np.unique(comp_map)
        valid_labels = labels[labels >= 0]

        rgb = np.zeros((*comp_map.shape, 3), dtype=np.uint8)

        # Consistent palette
        rng = np.random.RandomState(42)
        palette = rng.randint(50, 255, size=(self.n_clusters, 3), dtype=np.uint8)

        for l in valid_labels:
            rgb[comp_map == l] = palette[l % self.n_clusters]

        Image.fromarray(rgb).save(out_path)


# -------------------------------------------------------------------------
# 2. Composition map -> VisualObjects
# -------------------------------------------------------------------------
def composition_map_to_objects(
    comp_map: np.ndarray,
    feat_map: torch.Tensor,
    min_cc_area: int = 10,
) -> List[VisualObject]:
    """
    Convert composition map + DINO feature map into VisualObjects.

    comp_map: [H, W] cluster labels, -1 = background, >=0 = foreground prototypes.
    feat_map: [C, Hf, Wf] patch feature map.
    """
    H, W = comp_map.shape
    C, Hf, Wf = feat_map.shape
    objects: List[VisualObject] = []
    obj_id = 0

    # dynamic area threshold (ignore tiny CCs)
    min_area_px = max(min_cc_area, int(0.01 * H * W))

    # Patch centers in image coordinates
    yy, xx = np.meshgrid(np.arange(Hf), np.arange(Wf), indexing="ij")
    yy_img = (yy.astype(np.float32) + 0.5) * (H / Hf)
    xx_img = (xx.astype(np.float32) + 0.5) * (W / Wf)
    yy_img = yy_img.astype(int).clip(0, H - 1)
    xx_img = xx_img.astype(int).clip(0, W - 1)

    feat_map_flat = feat_map.view(C, -1)  # [C, Hf*Wf]

    labels = np.unique(comp_map)
    for k in labels:
        if k < 0:
            # skip background
            continue

        mask_k = comp_map == k
        if not mask_k.any():
            continue

        labeled, num = ndi.label(mask_k)
        for cc_idx in range(1, num + 1):
            cc_mask = labeled == cc_idx
            area = float(cc_mask.sum())
            if area < min_area_px:
                continue

            ys, xs = np.where(cc_mask)
            cy, cx = float(ys.mean()), float(xs.mean())
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bbox = (x0, y0, x1, y1)

            touches_border = (
                ys.min() == 0 or ys.max() == H - 1 or xs.min() == 0 or xs.max() == W - 1
            )
            if touches_border:
                continue
            # which patches fall into this CC?
            inside = cc_mask[yy_img, xx_img]  # [Hf, Wf]
            if inside.sum() == 0:
                # fallback: use all patches of this label k
                label_mask = comp_map[yy_img, xx_img] == k
                if label_mask.sum() == 0:
                    # ultimate fallback: global mean
                    emb_vec = feat_map_flat.mean(dim=1, keepdim=True).T  # [1, C]
                else:
                    idx_flat = torch.from_numpy(label_mask.reshape(-1)).to(
                        feat_map.device
                    )
                    patches = feat_map_flat[:, idx_flat].T  # [N_k, C]
                    emb_vec = patches.mean(dim=0, keepdim=True)  # [1, C]
            else:
                idx_flat = torch.from_numpy(inside.reshape(-1)).to(feat_map.device)
                patches = feat_map_flat[:, idx_flat].T  # [N_cc, C]
                emb_vec = patches.mean(dim=0, keepdim=True)  # [1, C]

            objects.append(
                VisualObject(
                    id=obj_id,
                    mask=cc_mask,
                    bbox=bbox,
                    embedding=emb_vec.cpu(),
                    area=area,
                    centroid=(cx, cy),
                    prototype_id=int(k),
                )
            )
            obj_id += 1

    return objects


# -------------------------------------------------------------------------
# 3. Symbolic Graph Builder (KNN-based local neighborhood)
# -------------------------------------------------------------------------


class SymbolicGraphBuilder:
    """
    Builds a local scene graph:
    - Nodes: objects with prototype_id, centroid, etc.
    - Edges: K nearest neighbors in normalized coordinate space.
    """

    def __init__(self, max_neighbors: int = 6, max_radius: float = 0.5):
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius  # in normalized coords

    def build(
        self,
        image_id: str,
        objects: List[VisualObject],
        image_shape: Tuple[int, int],
    ) -> SceneGraph:
        N = len(objects)
        edges: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

        if N == 0:
            return SceneGraph(image_id, [], {})

        H, W = image_shape
        centroids = np.array([o.centroid for o in objects], dtype=np.float32)
        centroids_norm = np.stack(
            [centroids[:, 0] / W, centroids[:, 1] / H], axis=1
        )  # [N, 2]

        k = min(self.max_neighbors + 1, N)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(centroids_norm)
        dists, indices = nn.kneighbors(centroids_norm)

        for i in range(N):
            src_id = objects[i].id
            for dist, j in zip(dists[i][1:], indices[i][1:]):  # skip itself
                if dist > self.max_radius:
                    continue
                dx_norm = centroids_norm[j, 0] - centroids_norm[i, 0]
                dy_norm = centroids_norm[j, 1] - centroids_norm[i, 1]
                edges[src_id].append((objects[j].id, dx_norm, dy_norm))

        return SceneGraph(image_id=image_id, nodes=objects, edges=dict(edges))


# -------------------------------------------------------------------------
# 4. Probabilistic Logic Engine
# -------------------------------------------------------------------------


class Theorem:
    def check(self, graph: SceneGraph) -> List[Violation]:
        raise NotImplementedError


class RangeCountInvariant(Theorem):
    """
    Count rule with soft tolerance:
      - expected_count: typical count (e.g., mode).
      - tolerance: ignore deviations within +/- tolerance.
      - sigma: scale factor for severity (rough std of counts).
    """

    def __init__(
        self, prototype_id: int, expected_count: float, tolerance: float, sigma: float
    ):
        self.prototype_id = prototype_id
        self.expected_count = expected_count
        self.tolerance = tolerance
        self.sigma = max(sigma, 1e-3)

    def check(self, graph: SceneGraph) -> List[Violation]:
        count = sum(1 for node in graph.nodes if node.prototype_id == self.prototype_id)
        diff = float(count - self.expected_count)
        if abs(diff) <= self.tolerance:
            return []
        severity = abs(diff) / self.sigma
        return [
            Violation(
                "CountInvariant",
                f"Count mismatch for Type {self.prototype_id}: "
                f"expected ≈ {self.expected_count:.1f}±{self.tolerance:.1f}, found {count}",
                severity,
            )
        ]


class AreaShareInvariant(Theorem):
    """
    Area-share rule:
      - expected_share: typical area fraction of this prototype among all parts.
      - tolerance: ignore small relative deviations.
      - sigma: scale factor for severity (rough std of shares).
    """

    def __init__(
        self, prototype_id: int, expected_share: float, tolerance: float, sigma: float
    ):
        self.prototype_id = prototype_id
        self.expected_share = expected_share
        self.tolerance = tolerance
        self.sigma = max(sigma, 1e-6)

    def check(self, graph: SceneGraph) -> List[Violation]:
        total_area = sum(n.area for n in graph.nodes) or 1.0
        area_p = sum(n.area for n in graph.nodes if n.prototype_id == self.prototype_id)
        share = area_p / total_area
        diff = share - self.expected_share
        if abs(diff) <= self.tolerance:
            return []
        severity = abs(diff) / self.sigma
        return [
            Violation(
                "AreaShareInvariant",
                f"Area share mismatch for Type {self.prototype_id}: "
                f"expected ≈ {self.expected_share:.3f}±{self.tolerance:.3f}, found {share:.3f}",
                severity,
            )
        ]


class GaussianSpatialInvariant(Theorem):
    """
    'Type A relates to Type B with vector distribution N(μ, Σ)'
    - Full 2D covariance Σ.
    - Mahalanobis distance + data-driven threshold.
    """

    def __init__(
        self,
        subject_proto: int,
        object_proto: int,
        mu: np.ndarray,
        cov: np.ndarray,
        threshold: float,
    ):
        self.subject = subject_proto
        self.object = object_proto
        self.mu = np.asarray(mu, dtype=np.float64).reshape(2)
        cov = np.asarray(cov, dtype=np.float64).reshape(2, 2)
        cov = cov + np.eye(2) * 1e-6  # regularize
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)
        self.threshold = float(threshold)

    def _mahalanobis(self, vec: np.ndarray) -> float:
        diff = vec - self.mu
        return float(np.sqrt(diff.T @ self.cov_inv @ diff))

    def check(self, graph: SceneGraph) -> List[Violation]:
        id_to_proto = {n.id: n.prototype_id for n in graph.nodes}
        violations: List[Violation] = []

        for src_id, neighbors in graph.edges.items():
            if id_to_proto.get(src_id) != self.subject:
                continue

            for dst_id, dx, dy in neighbors:
                if id_to_proto.get(dst_id) != self.object:
                    continue

                vec = np.array([dx, dy], dtype=np.float64)
                maha = self._mahalanobis(vec)

                if maha > self.threshold:
                    violations.append(
                        Violation(
                            "SpatialAnomaly",
                            f"Misplaced Component: Type {self.subject} -> {self.object}. "
                            f"Offset ({dx:.3f},{dy:.3f}) has Mahalanobis d={maha:.2f} "
                            f"(thr={self.threshold:.2f}).",
                            maha,
                        )
                    )
        return violations


class ProbabilisticInductiveEngine:
    """
    Learns:
    - Count invariants for stable prototypes.
    - Area-share invariants for stable composition ratios.
    - Spatial Gaussian invariants for stable, frequent relations.

    Provides:
      - evaluate(graph): list of Violations
      - score(violations): calibrated scalar logical anomaly score
    """

    def __init__(
        self,
        min_graph_support: float = 0.5,
        max_spatial_std: float = 0.15,
        min_samples_spatial: int = 10,
        spatial_quantile: float = 0.99,
        min_count_support_frac: float = 0.7,
        max_area_share_std: float = 0.05,
        debug_dir: str = "debug",
    ):
        self.theorems: List[Theorem] = []
        self.min_graph_support = min_graph_support
        self.max_spatial_std = max_spatial_std
        self.min_samples_spatial = min_samples_spatial
        self.spatial_quantile = spatial_quantile
        self.min_count_support_frac = min_count_support_frac
        self.max_area_share_std = max_area_share_std
        self.debug_dir = debug_dir

        # calibration on normal graphs
        self._calib_mu: float = 0.0
        self._calib_sigma: float = 1.0

    def fit(self, graphs: List[SceneGraph]):
        print("[*] Mining probabilistic logic rules...")
        self.theorems = []

        if not graphs:
            print("[!] No graphs provided for logic induction.")
            return

        os.makedirs(self.debug_dir, exist_ok=True)

        self._fit_count_and_area_invariants(graphs)
        self._fit_spatial_invariants(graphs)

        # --- calibration on normal graphs ---
        train_scores = []
        for g in graphs:
            v = self.evaluate(g)
            s = self.raw_score(v)
            train_scores.append(s)

        if train_scores:
            mu = float(np.mean(train_scores))
            sigma = float(np.std(train_scores))
            if sigma < 1e-6:
                sigma = 1.0
            self._calib_mu = mu
            self._calib_sigma = sigma
            print(f"[calib] logical scores on normals: mu={mu:.3f}, sigma={sigma:.3f}")

    # --------------------- count & area invariants ----------------------

    def _fit_count_and_area_invariants(self, graphs: List[SceneGraph]):
        counts_map: Dict[int, List[int]] = defaultdict(list)
        share_map: Dict[int, List[float]] = defaultdict(list)

        for g in graphs:
            local_counts: Dict[int, int] = defaultdict(int)
            total_area = sum(n.area for n in g.nodes) or 1.0
            area_per_proto: Dict[int, float] = defaultdict(float)

            for n in g.nodes:
                local_counts[n.prototype_id] += 1
                area_per_proto[n.prototype_id] += n.area

            for p_id, c in local_counts.items():
                counts_map[p_id].append(c)
            for p_id, a in area_per_proto.items():
                share_map[p_id].append(a / total_area)

        num_graphs = len(graphs)
        min_support = self.min_count_support_frac * num_graphs

        # debug: plot count histograms
        self._plot_count_histograms(counts_map)

        for p_id, counts in counts_map.items():
            if not counts:
                continue

            hist = Counter(counts)
            mode_count, freq = hist.most_common(1)[0]
            if freq < min_support:
                # counts too variable -> skip hard count rule
                continue

            mu = float(np.mean(counts))
            std = float(np.std(counts)) or 1.0

            # tolerance based on how peaked the distribution is
            support_frac = freq / num_graphs
            if support_frac > 0.95:
                tol = 0.0
            elif support_frac > 0.8:
                tol = 0.5
            else:
                tol = 1.0

            self.theorems.append(
                RangeCountInvariant(
                    prototype_id=p_id,
                    expected_count=mu,
                    tolerance=tol,
                    sigma=std,
                )
            )
            print(
                f"    -> Count law: Type_{p_id} count ≈ {mu:.2f} "
                f"(mode={mode_count}, support={freq}/{num_graphs}, tol={tol})"
            )

        # area-share invariants (for composition ratios)
        for p_id, shares in share_map.items():
            if len(shares) < min_support:
                continue
            mu = float(np.mean(shares))
            std = float(np.std(shares))
            if std > self.max_area_share_std:
                # area share too unstable, skip
                continue
            tol = 2.0 * std  # ignore small fluctuations
            self.theorems.append(
                AreaShareInvariant(
                    prototype_id=p_id,
                    expected_share=mu,
                    tolerance=tol,
                    sigma=std if std > 0 else 1e-3,
                )
            )
            print(
                f"    -> Area law: Type_{p_id} share ≈ {mu:.3f} "
                f"(std={std:.3f}, tol={tol:.3f})"
            )

    # --------------------- spatial invariants ---------------------------

    def _fit_spatial_invariants(self, graphs: List[SceneGraph]):
        # Map: (proto_A, proto_B) -> list of [dx, dy]
        spatial_data: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)

        for g in graphs:
            id_to_proto = {n.id: n.prototype_id for n in g.nodes}
            for src_id, neighbors in g.edges.items():
                src_p = id_to_proto[src_id]
                for dst_id, dx, dy in neighbors:
                    dst_p = id_to_proto[dst_id]
                    spatial_data[(src_p, dst_p)].append([dx, dy])

        num_graphs = len(graphs)

        for (p_A, p_B), vectors in spatial_data.items():
            if len(vectors) < max(
                self.min_samples_spatial, int(self.min_graph_support * num_graphs)
            ):
                continue

            vecs = np.array(vectors, dtype=np.float64)  # [M, 2]
            mus = np.mean(vecs, axis=0)
            cov = np.cov(vecs, rowvar=False)
            stds = np.sqrt(np.diag(cov))

            if float(np.mean(stds)) >= self.max_spatial_std:
                # Too loose; not a rigid relation
                continue

            # Compute Mahalanobis distances on training vectors
            cov_reg = cov + np.eye(2) * 1e-6
            cov_inv = np.linalg.inv(cov_reg)

            def mahal(v):
                d = v - mus
                return float(np.sqrt(d.T @ cov_inv @ d))

            maha_vals = np.array([mahal(v) for v in vecs], dtype=np.float64)

            if len(maha_vals) < 5:
                thr = float(np.max(maha_vals) + 3.0)
            else:
                thr = float(np.quantile(maha_vals, self.spatial_quantile))
                thr = max(thr, 3.0)

            self.theorems.append(
                GaussianSpatialInvariant(
                    subject_proto=p_A,
                    object_proto=p_B,
                    mu=mus,
                    cov=cov,
                    threshold=thr,
                )
            )
            print(
                f"    -> Spatial law: Type_{p_A}->{p_B} ~ "
                f"N(mu=[{mus[0]:.2f},{mus[1]:.2f}], "
                f"std=[{stds[0]:.2f},{stds[1]:.2f}]), thr={thr:.2f}, M={len(maha_vals)}"
            )

    # --------------------- evaluation & scoring -------------------------

    def evaluate(self, graph: SceneGraph) -> List[Violation]:
        all_violations: List[Violation] = []
        for rule in self.theorems:
            all_violations.extend(rule.check(graph))
        return all_violations

    def raw_score(self, violations: List[Violation]) -> float:
        """
        Uncalibrated logical anomaly score from a list of violations.
        """
        if not violations:
            return 0.0

        max_sev = max(v.severity for v in violations)
        strong = sum(1 for v in violations if v.severity > 1.5)
        bonus = 0.3 * np.log1p(strong)

        return float(max_sev + bonus)

    def score(self, violations: List[Violation]) -> float:
        """
        Calibrated logical anomaly score (non-negative z-score).
        """
        raw = self.raw_score(violations)
        if self._calib_sigma <= 1e-6:
            return max(raw - self._calib_mu, 0.0)
        z = (raw - self._calib_mu) / self._calib_sigma
        return float(max(z, 0.0))

    # --------------------- debug plots ---------------------------------

    def _plot_count_histograms(
        self,
        counts_map: Dict[int, List[int]],
    ):
        out_dir = os.path.join(self.debug_dir, "count_histograms")
        os.makedirs(out_dir, exist_ok=True)
        for p_id, counts in counts_map.items():
            if not counts:
                continue
            plt.figure()
            plt.hist(
                counts,
                bins=range(min(counts), max(counts) + 2),
                align="left",
                rwidth=0.8,
            )
            plt.xlabel("Count per image")
            plt.ylabel("Frequency")
            plt.title(f"Prototype {p_id} count distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"proto_{p_id}_counts.png"))
            plt.close()


# -------------------------------------------------------------------------
# 5. Orchestrator
# -------------------------------------------------------------------------


class NeuroSymbolicInductor:
    """
    High-level API:
    - train(normal_images): builds composition clusters + logic rules from normal LOCO images.
    - test(image): returns scene graph, list of violations, and scalar logical anomaly score.
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        n_clusters: int = 6,
    ):
        self.comp_extractor = CompositionMapExtractor(
            device=device,
            n_clusters=n_clusters,
            sam_checkpoint="./sam_vit_h_4b8939.pth",
        )
        self.builder = SymbolicGraphBuilder(
            max_neighbors=6,
            max_radius=0.5,
        )
        self.logic_engine = ProbabilisticInductiveEngine(debug_dir="debug")
        self.vocab_ready = False

    def train(self, normal_images: List[np.ndarray], category):
        print("\n=== Phase 0: Learn global composition clusters ===")
        self.comp_extractor.fit_global_clusters(normal_images)
        self.vocab_ready = True

        print("\n=== Phase 1: Build graphs from composition maps ===")
        reference_graphs: List[SceneGraph] = []
        os.makedirs(f"debug/{category}/comp_maps", exist_ok=True)

        for i, img in enumerate(normal_images):
            comp_map, feat_map = self.comp_extractor.build_composition_map(img)
            self.comp_extractor.save_composition_viz(
                comp_map,
                out_path=f"debug/comp_maps/train_{i}.png",
            )
            objs = composition_map_to_objects(comp_map, feat_map)
            print(
                f"    [train] image {i}: {len(objs)} parts "
                f"(clusters: {np.unique(comp_map).size})"
            )

            graph = self.builder.build(f"norm_{i}", objs, img.shape[:2])
            reference_graphs.append(graph)

        print("\n=== Phase 2: Probabilistic Logic Induction ===")
        self.logic_engine.fit(reference_graphs)
        return reference_graphs

    def test(
        self, query_image: np.ndarray, query_id: str = "test"
    ) -> Tuple[SceneGraph, List[Violation], float]:
        if not self.vocab_ready:
            raise RuntimeError("NeuroSymbolicInductor.train() must be called first.")

        comp_map, feat_map = self.comp_extractor.build_composition_map(query_image)
        objs = composition_map_to_objects(comp_map, feat_map)
        graph = self.builder.build(query_id, objs, query_image.shape[:2])
        violations = self.logic_engine.evaluate(graph)
        score = self.logic_engine.score(violations)
        return graph, violations, score


# -------------------------------------------------------------------------
# 6. Simple synthetic sanity check
# -------------------------------------------------------------------------


def generate_synthetic_image(num_objects: int = 2, anomaly: bool = False) -> np.ndarray:
    """Toy example: blue square + red circle left/right."""
    w, h = 512, 512
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Anchor (Blue Square) at Center
    draw.rectangle((236, 236, 276, 276), fill=(0, 0, 255))

    # Satellite (Red Circle)
    offset = 100 if not anomaly else -100
    x = 256 + offset
    y = 256
    draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=(255, 0, 0))

    return np.array(img)