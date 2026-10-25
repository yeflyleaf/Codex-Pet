from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage

# Codex-Pet V2 硬性规格常量 (8列 x 11行, 单帧 192x208, 总尺寸 1536x2288)
CODEX_V2_COLS: int = 8
CODEX_V2_ROWS: int = 11
CODEX_V2_FRAME_W: int = 192
CODEX_V2_FRAME_H: int = 208


def _get_supported_extensions() -> set[str]:
    """动态获取当前 Pillow 环境支持读取的所有图片扩展名。"""
    Image.init()
    return {
        ext.lower()
        for ext, fmt in Image.registered_extensions().items()
        if fmt in Image.OPEN
    }


SUPPORTED_IMAGE_EXTS: set[str] = _get_supported_extensions()


def _is_valid_image_file(file_path: Path) -> bool:
    """校验文件是否为受支持且可正常解码的图像文件（不局限于扩展名）。"""
    if not file_path.is_file():
        return False
    try:
        with Image.open(file_path) as test_img:
            test_img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _resolve_input_image(
    input_path: str | Path,
    exclude_output: Path | None = None,
) -> Path:
    """智能解析输入图片路径，支持任意图片格式。"""
    raw_path = Path(input_path)
    exclude_resolved = (
        exclude_output.resolve()
        if exclude_output and exclude_output.exists()
        else None
    )

    if raw_path.is_file():
        if _is_valid_image_file(raw_path):
            return raw_path
        raise ValueError(f"文件存在但不是有效的图像格式: {raw_path.resolve()}")

    if raw_path.is_dir():
        candidates = sorted(
            p
            for p in raw_path.iterdir()
            if p.is_file()
            and (exclude_resolved is None or p.resolve() != exclude_resolved)
            and (p.suffix.lower() in SUPPORTED_IMAGE_EXTS or _is_valid_image_file(p))
        )
        for candidate in candidates:
            if _is_valid_image_file(candidate):
                return candidate
        raise FileNotFoundError(f"目录中未找到任何可用的图像文件: {raw_path.resolve()}")

    parent_dir = raw_path.parent
    target_stem = raw_path.stem
    if parent_dir.is_dir():
        same_stem_candidates = sorted(
            p
            for p in parent_dir.glob(f"{target_stem}.*")
            if p.is_file()
            and (exclude_resolved is None or p.resolve() != exclude_resolved)
        )
        for candidate in same_stem_candidates:
            if (
                candidate.suffix.lower() in SUPPORTED_IMAGE_EXTS
                or _is_valid_image_file(candidate)
            ):
                if _is_valid_image_file(candidate):
                    return candidate

    raise FileNotFoundError(
        f"未找到匹配的输入图像文件: {raw_path.resolve()} "
        f"(已尝试自动检索同目录下 '{target_stem}.*' 的所有图片类型)"
    )


def _estimate_global_bg_color(
    rgb: np.ndarray,
    cols: int,
    rows: int,
) -> np.ndarray:
    """从整图四周边缘与网格分割线上采样，鲁棒估计全局背景基准色 (R, G, B)。"""
    h, w, _ = rgb.shape
    samples = []

    pad = max(2, min(h, w) // 200)
    samples.append(rgb[:pad, :, :].reshape(-1, 3))
    samples.append(rgb[h - pad :, :, :].reshape(-1, 3))
    samples.append(rgb[:, :pad, :].reshape(-1, 3))
    samples.append(rgb[:, w - pad :, :].reshape(-1, 3))

    for r in range(1, rows):
        y = min(h - 1, max(0, round(r * h / rows)))
        samples.append(rgb[max(0, y - 1) : min(h, y + 2), :, :].reshape(-1, 3))
    for c in range(1, cols):
        x = min(w - 1, max(0, round(c * w / cols)))
        samples.append(rgb[:, max(0, x - 1) : min(w, x + 2), :].reshape(-1, 3))

    all_samples = np.concatenate(samples, axis=0)
    coarse_median = np.median(all_samples, axis=0)
    dists = np.linalg.norm(all_samples - coarse_median, axis=1)
    inliers = all_samples[dists <= 35.0]
    if len(inliers) > 0:
        return np.median(inliers, axis=0)
    return coarse_median


def _estimate_cell_bg_color(
    cell_rgb: np.ndarray,
    global_bg: np.ndarray,
) -> np.ndarray:
    """估计单个网格帧的局部背景色，自适应消除生成图可能存在的轻微光照渐变。"""
    ch, cw, _ = cell_rgb.shape
    b = max(2, min(ch, cw) // 25)
    border_pixels = np.concatenate(
        [
            cell_rgb[:b, :, :].reshape(-1, 3),
            cell_rgb[ch - b :, :, :].reshape(-1, 3),
            cell_rgb[:, :b, :].reshape(-1, 3),
            cell_rgb[:, cw - b :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    dists = np.linalg.norm(border_pixels - global_bg, axis=1)
    valid = border_pixels[dists <= 42.0]
    if len(valid) >= 12:
        return np.median(valid, axis=0)
    return global_bg


def _compute_color_distance(
    rgb: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """计算每个像素与背景色的综合感知色差（融合欧氏距离与最大单通道偏差）。"""
    diff = rgb - bg_color.reshape(1, 1, 3)
    euclid = np.linalg.norm(diff, axis=2)
    max_ch = np.max(np.abs(diff), axis=2)
    return 0.7 * euclid + 0.3 * max_ch


def _unmultiply_color(
    rgb: np.ndarray,
    alpha: np.ndarray,
    bg_color: np.ndarray,
) -> np.ndarray:
    """对半透明边缘像素执行背景色反混（Color Decontamination），彻底剥离底色残留。"""
    cleaned = rgb.copy()
    semi_mask = (alpha > 0.0) & (alpha < 254.0)
    if np.any(semi_mask):
        a_norm = (alpha[semi_mask] / 255.0)[:, None]
        bg_vec = bg_color.reshape(1, 3)
        fg_est = (cleaned[semi_mask] - (1.0 - a_norm) * bg_vec) / np.maximum(
            a_norm, 0.20
        )
        cleaned[semi_mask] = np.clip(fg_est, 0.0, 255.0)
    cleaned[alpha <= 0.0] = 0.0
    return cleaned


def _find_best_1d_gutter_split(
    fg_strip: np.ndarray,
    row_fg_profile: np.ndarray,
    nom_pos: int,
    min_pos: int,
    max_pos: int,
) -> int:
    """在 [min_pos, max_pos] 搜索窗口内寻找能够完整避开上行角色脚底与下行角色头顶的最佳分割线。"""
    if max_pos <= min_pos:
        return nom_pos

    local_counts = np.sum(fg_strip[min_pos : max_pos + 1], axis=1).astype(np.float32)
    global_counts = row_fg_profile[min_pos : max_pos + 1].astype(np.float32)
    combined_activity = local_counts + 0.15 * global_counts

    # 寻找完全无前景像素的空白行（或局部极小值行）
    zero_indices = np.where(local_counts == 0)[0]
    if len(zero_indices) > 0:
        # 将连续空白段聚类，优先选择包含或最接近名义分割线且宽度充足的空白段
        splits = np.where(np.diff(zero_indices) > 1)[0] + 1
        runs = np.split(zero_indices, splits)
        best_run = None
        best_score = float("inf")
        nom_rel = nom_pos - min_pos
        for run in runs:
            run_start, run_end = int(run[0]), int(run[-1])
            run_len = run_end - run_start + 1
            dist_to_nom = (
                0.0
                if (run_start <= nom_rel <= run_end)
                else min(abs(nom_rel - run_start), abs(nom_rel - run_end))
            )
            score = dist_to_nom - 0.35 * run_len
            if score < best_score:
                best_score = score
                best_run = (run_start, run_end)

        if best_run is not None:
            r_start, r_end = best_run
            # 分割位置放在上行脚底下方偏中后位置，确保脚尖拥有充足缓冲
            offset = min(r_end, r_start + max(1, (r_end - r_start + 1) // 2))
            return min_pos + offset

    # 若窗口内无绝对零值行，选择综合前景响应最小且靠近名义位置的分割点
    positions = np.arange(min_pos, max_pos + 1, dtype=np.float32)
    dist_penalty = 0.08 * np.abs(positions - float(nom_pos))
    best_idx = int(np.argmin(combined_activity + dist_penalty))
    return min_pos + best_idx


def _compute_adaptive_cell_boxes(
    full_rgba: np.ndarray,
    global_bg: np.ndarray,
    cols: int,
    rows: int,
    fg_detect_tol: float = 24.0,
) -> list[list[tuple[int, int, int, int]]]:
    """计算自适应网格切分边界 (x0, y0, x1, y1)，防止等分网格硬切越过分割线的角色脚底或发梢。"""
    src_h, src_w, _ = full_rgba.shape
    rgb = full_rgba[:, :, :3]
    alpha = full_rgba[:, :, 3]

    if np.mean(alpha == 0) > 0.20:
        fg_map = alpha > 16
    else:
        color_dist = _compute_color_distance(rgb, global_bg)
        fg_map = (color_dist > fg_detect_tol) & (alpha > 128)
        # 过滤单像素孤立噪点，避免干扰分割线搜索
        fg_map = ndimage.binary_opening(fg_map, structure=np.ones((2, 2), dtype=bool))

    nom_x = [round(c * src_w / cols) for c in range(cols + 1)]
    nom_y = [round(r * src_h / rows) for r in range(rows + 1)]

    max_dx = max(4, round((src_w / cols) * 0.14))
    max_dy = max(6, round((src_h / rows) * 0.22))

    row_fg_profile = np.sum(fg_map, axis=1)
    col_fg_profile = np.sum(fg_map, axis=0)

    # 先求每一列在各行交界处的自适应水平分割线 y_splits[r][c]
    y_splits = np.zeros((rows + 1, cols), dtype=int)
    y_splits[0, :] = 0
    y_splits[rows, :] = src_h

    for r in range(1, rows):
        y_nom = nom_y[r]
        y_min = max(1, y_nom - max_dy)
        y_max = min(src_h - 2, y_nom + max_dy)
        for c in range(cols):
            cx0 = nom_x[c]
            cx1 = nom_x[c + 1]
            y_splits[r, c] = _find_best_1d_gutter_split(
                fg_strip=fg_map[:, cx0:cx1],
                row_fg_profile=row_fg_profile,
                nom_pos=y_nom,
                min_pos=y_min,
                max_pos=y_max,
            )

    # 再求每一行在各列交界处的自适应垂直分割线 x_splits[r][c]
    x_splits = np.zeros((rows, cols + 1), dtype=int)
    x_splits[:, 0] = 0
    x_splits[:, cols] = src_w

    for c in range(1, cols):
        x_nom = nom_x[c]
        x_min = max(1, x_nom - max_dx)
        x_max = min(src_w - 2, x_nom + max_dx)
        for r in range(rows):
            ry0 = min(y_splits[r, c - 1], y_splits[r, c])
            ry1 = max(y_splits[r + 1, c - 1], y_splits[r + 1, c])
            x_splits[r, c] = _find_best_1d_gutter_split(
                fg_strip=fg_map[ry0:ry1, :].T,
                row_fg_profile=col_fg_profile,
                nom_pos=x_nom,
                min_pos=x_min,
                max_pos=x_max,
            )

    boxes: list[list[tuple[int, int, int, int]]] = []
    for r in range(rows):
        row_boxes: list[tuple[int, int, int, int]] = []
        for c in range(cols):
            x0 = int(x_splits[r, c])
            x1 = int(x_splits[r, c + 1])
            y0 = int(y_splits[r, c])
            y1 = int(y_splits[r + 1, c])
            row_boxes.append((x0, y0, x1, y1))
        boxes.append(row_boxes)

    return boxes


def _remove_cell_background(
    cell_rgba: np.ndarray,
    global_bg: np.ndarray,
    flood_tolerance: float = 30.0,
    enclosed_core_tolerance: float = 24.0,
    enclosed_expand_tolerance: float = 38.0,
    edge_softness: float = 44.0,
    min_enclosed_hole_area: int = 6,
    max_dark_eye_hole_area: int = 160,
    min_fg_island_area: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """对单帧图像执行 100% 纯透明背景剥离，同时严格保护角色脚底与周身纯黑外描边不被腐蚀。"""
    out = cell_rgba.copy().astype(np.float32)
    rgb = out[:, :, :3]
    orig_alpha = out[:, :, 3]
    ch, cw = orig_alpha.shape

    local_bg = _estimate_cell_bg_color(rgb, global_bg)

    if np.mean(orig_alpha == 0) > 0.20:
        zero_m = orig_alpha <= 10.0
        orig_alpha[zero_m] = 0.0
        rgb = _unmultiply_color(rgb, orig_alpha, local_bg)
        out[:, :, :3] = rgb
        out[:, :, 3] = orig_alpha
        return out.astype(np.uint8), local_bg

    color_dist = _compute_color_distance(rgb, local_bg)
    channel_span = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    # 计算感知亮度，用于区分灰色背景与角色的纯黑/深褐外描边
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    bg_lum = float(
        0.299 * local_bg[0] + 0.587 * local_bg[1] + 0.114 * local_bg[2]
    )

    bg_span = float(np.max(local_bg) - np.min(local_bg))
    is_neutral_bg = bg_span <= 15.0
    is_dark_bg = float(np.max(local_bg)) < 35.0

    # 深色描边保护掩码：明显比灰底更暗的像素属于角色外描边（尤其脚尖轮廓），严禁归入背景
    dark_outline_protect = (
        (bg_lum - lum > 18.0) if not is_dark_bg else np.zeros((ch, cw), dtype=bool)
    )

    struct_8 = np.ones((3, 3), dtype=bool)
    struct_4 = ndimage.generate_binary_structure(2, 1)

    # 步骤 1：从单元格四周边框种子执行 8-邻域连通泛洪
    eff_flood_tol = flood_tolerance * (0.75 if is_dark_bg else 1.0)
    bg_candidate = (
        (color_dist <= eff_flood_tol)
        & (orig_alpha > 128)
        & (~dark_outline_protect)
    )

    # 仅允许亮度与背景接近的低饱和度像素作为灰底压缩噪点，绝不吞噬深色脚尖描边
    if is_neutral_bg and not is_dark_bg:
        neutral_gray_noise = (
            (channel_span <= 12.0)
            & (np.abs(lum - bg_lum) <= 16.0)
            & (color_dist <= eff_flood_tol * 1.12)
        )
        bg_candidate |= neutral_gray_noise

    border_seeds = np.zeros((ch, cw), dtype=bool)
    border_seeds[0, :] = True
    border_seeds[-1, :] = True
    border_seeds[:, 0] = True
    border_seeds[:, -1] = True
    border_seeds &= bg_candidate

    exterior_bg = ndimage.binary_propagation(
        border_seeds, mask=bg_candidate, structure=struct_8
    )

    # 步骤 2：穿透角色发丝/双腿间的闭合背景死角
    interior_region = ~exterior_bg
    enclosed_bg_mask = np.zeros((ch, cw), dtype=bool)

    if is_dark_bg:
        inner_dark = interior_region & (color_dist <= enclosed_core_tolerance)
        labeled, num_features = ndimage.label(inner_dark, structure=struct_8)
        if num_features > 0:
            sizes = ndimage.sum(inner_dark, labeled, range(1, num_features + 1))
            large_labels = np.where(sizes > max_dark_eye_hole_area)[0] + 1
            enclosed_bg_mask = np.isin(labeled, large_labels)
    else:
        inner_core = (
            interior_region
            & (color_dist <= enclosed_core_tolerance)
            & (~dark_outline_protect)
        )
        labeled, num_features = ndimage.label(inner_core, structure=struct_8)
        if num_features > 0:
            sizes = ndimage.sum(inner_core, labeled, range(1, num_features + 1))
            valid_labels = np.where(sizes >= min_enclosed_hole_area)[0] + 1
            valid_core = np.isin(labeled, valid_labels)

            expand_mask = (
                interior_region
                & (color_dist <= enclosed_expand_tolerance)
                & (~dark_outline_protect)
            )
            if is_neutral_bg:
                expand_mask &= channel_span <= 20.0
            enclosed_bg_mask = ndimage.binary_propagation(
                valid_core, mask=expand_mask, structure=struct_8
            )

    full_bg_mask = exterior_bg | enclosed_bg_mask
    fg_mask = ~full_bg_mask

    # 步骤 3：清除远离角色主体的孤立背景杂色斑块，但严格保护紧邻主体的脚尖与特效
    fg_labeled, fg_num = ndimage.label(fg_mask, structure=struct_8)
    if fg_num > 1:
        fg_sizes = ndimage.sum(fg_mask, fg_labeled, range(1, fg_num + 1))
        main_label = int(np.argmax(fg_sizes)) + 1
        main_body_mask = fg_labeled == main_label
        # 主体向外膨胀 5px 作为近邻保护区（保护因抗锯齿产生的细小脚尖或发梢）
        near_body_zone = ndimage.binary_dilation(
            main_body_mask, structure=struct_8, iterations=5
        )

        for lbl in range(1, fg_num + 1):
            if lbl == main_label:
                continue
            comp_mask = fg_labeled == lbl
            comp_area = fg_sizes[lbl - 1]
            comp_mean_dist = float(np.mean(color_dist[comp_mask]))
            is_near_body = bool(np.any(comp_mask & near_body_zone))

            if is_near_body and comp_mean_dist >= 28.0:
                continue
            if comp_area < min_fg_island_area and comp_mean_dist < 42.0:
                fg_mask[comp_mask] = False
                full_bg_mask[comp_mask] = True

    # 步骤 4：精细边缘抗锯齿计算，保护深色外描边不发生向内缩水
    fg_inner_core = ndimage.binary_erosion(fg_mask, structure=struct_4, iterations=1)
    fg_inner_edge = fg_mask & (~fg_inner_core)

    alpha = np.zeros((ch, cw), dtype=np.float32)
    alpha[fg_inner_core] = 255.0

    edge_dists = color_dist[fg_inner_edge]
    min_edge_tol = flood_tolerance * 0.45
    eff_soft = max(edge_softness, min_edge_tol + 10.0)
    edge_alpha = np.clip(
        (edge_dists - min_edge_tol) / (eff_soft - min_edge_tol), 0.0, 1.0
    )

    # 若边缘像素本身属于深色描边，赋予更高不透明度，防止脚尖黑边在缩放后变淡消失
    edge_dark_boost = dark_outline_protect[fg_inner_edge]
    edge_alpha = np.where(edge_dark_boost, np.maximum(edge_alpha, 0.85), edge_alpha)
    alpha[fg_inner_edge] = edge_alpha * 255.0

    alpha[full_bg_mask] = 0.0
    alpha[alpha < 14.0] = 0.0
    alpha = np.minimum(alpha, orig_alpha)

    # 步骤 5：背景色反混
    rgb = _unmultiply_color(rgb, alpha, local_bg)

    out[:, :, :3] = rgb
    out[:, :, 3] = np.clip(alpha, 0.0, 255.0)
    return out.astype(np.uint8), local_bg


def _post_resize_clean(
    resized_cell: Image.Image,
    local_bg: np.ndarray,
    alpha_cutoff: int = 12,
) -> Image.Image:
    """消除 Lanczos 重采样在透明背景上产生的微弱振铃残影，同时保留脚底抗锯齿细节。"""
    arr = np.array(resized_cell, dtype=np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    ghost_mask = alpha <= float(alpha_cutoff)
    alpha[ghost_mask] = 0.0
    rgb[ghost_mask] = 0.0

    rgb = _unmultiply_color(rgb, alpha, local_bg)

    arr[:, :, :3] = rgb
    arr[:, :, 3] = np.clip(alpha, 0.0, 255.0)
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def _process_and_align_to_codex_grid(
    img: Image.Image,
    cols: int,
    rows: int,
    frame_w: int,
    frame_h: int,
    pad_x: int = 3,
    pad_top: int = 3,
    pad_bottom: int = 4,
    flood_tolerance: float = 30.0,
    enclosed_core_tolerance: float = 24.0,
    enclosed_expand_tolerance: float = 38.0,
    soft_threshold: float = 44.0,
    max_hole_area: int = 160,
) -> tuple[Image.Image, np.ndarray]:
    """结合自适应网格切分、前景包围盒检测与行级统一基线对齐，确保脚底完整且留有安全边距。"""
    src_w, src_h = img.size
    full_data = np.array(img, dtype=np.float32)
    global_bg = _estimate_global_bg_color(full_data[:, :, :3], cols=cols, rows=rows)

    cell_boxes = _compute_adaptive_cell_boxes(
        full_rgba=full_data,
        global_bg=global_bg,
        cols=cols,
        rows=rows,
    )

    target_total_w = cols * frame_w
    target_total_h = rows * frame_h
    canvas = Image.new("RGBA", (target_total_w, target_total_h), (0, 0, 0, 0))

    inner_max_w = frame_w - (pad_x * 2)
    inner_max_h = frame_h - (pad_top + pad_bottom)

    nom_cell_w = src_w / cols
    nom_cell_h = src_h / rows
    base_scale = min(inner_max_w / nom_cell_w, inner_max_h / nom_cell_h)

    for r in range(rows):
        nom_row_y0 = r * nom_cell_h
        row_cells: list[dict] = []
        row_rel_y_min = float("inf")
        row_rel_y_max = float("-inf")
        max_fg_w = 1
        max_fg_h = 1

        # 第一遍：提取当前行所有帧的纯净前景并统计行级垂直跨度
        for c in range(cols):
            x0, y0, x1, y1 = cell_boxes[r][c]
            cell_rgba = full_data[y0:y1, x0:x1, :]
            ch, cw, _ = cell_rgba.shape
            if cw == 0 or ch == 0:
                continue

            clean_cell_arr, local_bg = _remove_cell_background(
                cell_rgba=cell_rgba,
                global_bg=global_bg,
                flood_tolerance=flood_tolerance,
                enclosed_core_tolerance=enclosed_core_tolerance,
                enclosed_expand_tolerance=enclosed_expand_tolerance,
                edge_softness=soft_threshold,
                max_dark_eye_hole_area=max_hole_area,
            )

            alpha = clean_cell_arr[:, :, 3]
            ys, xs = np.where(alpha > 16)
            if len(xs) == 0 or len(ys) == 0:
                continue

            bx0, bx1 = int(np.min(xs)), int(np.max(xs)) + 1
            by0, by1 = int(np.min(ys)), int(np.max(ys)) + 1

            fg_crop = clean_cell_arr[by0:by1, bx0:bx1, :]
            fg_h, fg_w, _ = fg_crop.shape

            # 记录相对于名义行顶部的全局垂直坐标，以保持同行各帧原有的地面高度关系
            rel_y0 = (y0 + by0) - nom_row_y0
            rel_y1 = (y0 + by1) - nom_row_y0
            row_rel_y_min = min(row_rel_y_min, rel_y0)
            row_rel_y_max = max(row_rel_y_max, rel_y1)

            nom_col_x0 = c * nom_cell_w
            rel_x_center = (x0 + (bx0 + bx1) * 0.5) - nom_col_x0

            max_fg_w = max(max_fg_w, fg_w)
            max_fg_h = max(max_fg_h, fg_h)

            row_cells.append(
                {
                    "c": c,
                    "fg_crop": fg_crop,
                    "fg_w": fg_w,
                    "fg_h": fg_h,
                    "local_bg": local_bg,
                    "rel_y0": rel_y0,
                    "rel_x_center": rel_x_center,
                }
            )

        if not row_cells:
            continue

        # 计算行级统一缩放比例：确保全行垂直跨度与最宽单帧均严格落在安全边距内
        row_span_h = max(1.0, row_rel_y_max - row_rel_y_min)
        row_scale = min(
            base_scale,
            inner_max_w / max_fg_w,
            inner_max_h / max(row_span_h, float(max_fg_h)),
        )

        scaled_row_span_h = row_span_h * row_scale
        # 将全行前景垂直居中于 [pad_top, frame_h - pad_bottom] 安全区内
        target_row_top = pad_top + (inner_max_h - scaled_row_span_h) * 0.5

        # 第二遍：按行统一比例缩放并精准贴入 192x208 单元格
        for item in row_cells:
            c = item["c"]
            fg_img = Image.fromarray(item["fg_crop"], mode="RGBA")
            new_w = max(1, round(item["fg_w"] * row_scale))
            new_h = max(1, round(item["fg_h"] * row_scale))

            resized_fg = fg_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            purified_fg = _post_resize_clean(resized_fg, local_bg=item["local_bg"])

            # 水平方向：围绕原名义中心对齐并限制在左右安全边距内
            offset_x = round(
                (item["rel_x_center"] - nom_cell_w * 0.5) * row_scale
                + (frame_w - new_w) * 0.5
            )
            local_x = int(np.clip(offset_x, pad_x, frame_w - pad_x - new_w))

            # 垂直方向：保持同行动画帧相对高度，并严格保证底部留有 >= pad_bottom 的安全边距
            offset_y = round(
                target_row_top + (item["rel_y0"] - row_rel_y_min) * row_scale
            )
            local_y = int(
                np.clip(offset_y, pad_top, frame_h - pad_bottom - new_h)
            )

            paste_x = c * frame_w + local_x
            paste_y = r * frame_h + local_y
            canvas.paste(purified_fg, (paste_x, paste_y), purified_fg)

    return canvas, global_bg


def remove_black_bg_for_spritesheet(
    input_path: str,
    output_path: str,
    cols: int = CODEX_V2_COLS,
    rows: int = CODEX_V2_ROWS,
    frame_w: int = CODEX_V2_FRAME_W,
    frame_h: int = CODEX_V2_FRAME_H,
    flood_tolerance: int = 30,
    enclosed_core_tolerance: int = 24,
    enclosed_expand_tolerance: int = 38,
    soft_threshold: int = 44,
    max_hole_area: int = 160,
) -> None:
    """兼容灰底、黑底、白底及任意纯色背景的精灵图 100% 透明化与防裁脚规格重构入口函数。"""
    dst_file = Path(output_path)
    src_file = _resolve_input_image(input_path, exclude_output=dst_file)

    with Image.open(src_file) as raw_img:
        img_format = raw_img.format or src_file.suffix.upper()
        if getattr(raw_img, "is_animated", False):
            raw_img.seek(0)
        oriented_img = ImageOps.exif_transpose(raw_img)
        img = oriented_img.convert("RGBA")

    final_img, detected_bg = _process_and_align_to_codex_grid(
        img=img,
        cols=cols,
        rows=rows,
        frame_w=frame_w,
        frame_h=frame_h,
        pad_x=3,
        pad_top=3,
        pad_bottom=4,
        flood_tolerance=float(flood_tolerance),
        enclosed_core_tolerance=float(enclosed_core_tolerance),
        enclosed_expand_tolerance=float(enclosed_expand_tolerance),
        soft_threshold=float(soft_threshold),
        max_hole_area=max_hole_area,
    )

    expected_size = (cols * frame_w, rows * frame_h)
    assert final_img.size == expected_size, (
        f"尺寸校验失败: 当前 {final_img.size}, 预期 {expected_size}"
    )

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    out_ext = dst_file.suffix.lower()
    if out_ext == ".png":
        final_img.save(dst_file, format="PNG", optimize=True)
    else:
        final_img.save(dst_file, format="WEBP", lossless=True, quality=100, method=6)

    final_rgba = np.array(final_img)
    final_alpha = final_rgba[:, :, 3]

    bg_pixels_mask = final_alpha == 0
    bg_clean_rate = (
        float(np.mean(np.all(final_rgba[bg_pixels_mask] == 0, axis=1)) * 100.0)
        if np.any(bg_pixels_mask)
        else 0.0
    )
    canvas_empty_ratio = float(np.mean(bg_pixels_mask) * 100.0)
    fg_opaque_ratio = float(np.mean(final_alpha > 0) * 100.0)
    bg_rgb_tuple = tuple(int(round(v)) for v in detected_bg)

    print(
        f"处理完成: 输入 [{src_file.name} ({img_format})] -> 输出 [{dst_file}]\n"
        f"  - 自动识别原图背景色 RGB: {bg_rgb_tuple}\n"
        f"  - 背景区域 RGBA=(0,0,0,0) 纯净透明率: {bg_clean_rate:.2f}%\n"
        f"  - 画布面积构成: 透明背景占 {canvas_empty_ratio:.1f}% | 角色本体占 {fg_opaque_ratio:.1f}%\n"
        f"  - 最终尺寸: {expected_size[0]}x{expected_size[1]} "
        f"(网格: {cols}列x{rows}行, 单帧: {frame_w}x{frame_h})"
    )


if __name__ == "__main__":
    remove_black_bg_for_spritesheet(
        input_path="./assets/1",
        output_path="./assets/2.webp",
        cols=8,
        rows=11,
    )
