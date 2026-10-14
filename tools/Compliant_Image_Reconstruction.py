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
    """智能解析输入图片路径，支持任意图片格式：
    1. 若传入路径精确存在且为有效图片，直接使用；
    2. 若传入 './assets/1.png' 或 './assets/1' 但实际文件是 '1.jpg'/'1.webp' 等，自动匹配同名任意格式图片；
    3. 若传入的是目录（如 './assets'），自动查找该目录下第一个有效图片文件（排除输出文件）。
    """
    raw_path = Path(input_path)
    exclude_resolved = exclude_output.resolve() if exclude_output and exclude_output.exists() else None

    # 情况 1：路径直接存在且是文件
    if raw_path.is_file():
        if _is_valid_image_file(raw_path):
            return raw_path
        raise ValueError(f"文件存在但不是有效的图像格式: {raw_path.resolve()}")

    # 情况 2：传入的是目录，扫描目录内所有有效图片
    if raw_path.is_dir():
        candidates = sorted(
            p for p in raw_path.iterdir()
            if p.is_file()
            and (exclude_resolved is None or p.resolve() != exclude_resolved)
            and (p.suffix.lower() in SUPPORTED_IMAGE_EXTS or _is_valid_image_file(p))
        )
        for candidate in candidates:
            if _is_valid_image_file(candidate):
                return candidate
        raise FileNotFoundError(f"目录中未找到任何可用的图像文件: {raw_path.resolve()}")

    # 情况 3：文件不存在（例如写死 1.png 或无后缀 1，实际存放的是 1.jpg / 1.bmp 等）
    parent_dir = raw_path.parent
    target_stem = raw_path.stem
    if parent_dir.is_dir():
        # 优先匹配同主文件名（如 1.*）的任意图像格式
        same_stem_candidates = sorted(
            p for p in parent_dir.glob(f"{target_stem}.*")
            if p.is_file()
            and (exclude_resolved is None or p.resolve() != exclude_resolved)
        )
        for candidate in same_stem_candidates:
            if candidate.suffix.lower() in SUPPORTED_IMAGE_EXTS or _is_valid_image_file(candidate):
                if _is_valid_image_file(candidate):
                    return candidate

    raise FileNotFoundError(
        f"未找到匹配的输入图像文件: {raw_path.resolve()} "
        f"(已尝试自动检索同目录下 '{target_stem}.*' 的所有图片类型)"
    )


def _build_internal_dark_mask(
    opaque_body: np.ndarray,
    max_hole_area: int = 160,
) -> np.ndarray:
    """仅保留角色内部极小闭合深色区域（如眼瞳），防止双腿或发丝间的闭合黑底被误留。"""
    solid_interior = ndimage.binary_fill_holes(opaque_body)
    all_holes = solid_interior & (~opaque_body)

    labeled_holes, num_features = ndimage.label(all_holes)
    if num_features == 0:
        return np.zeros_like(opaque_body, dtype=bool)

    hole_sizes = ndimage.sum(all_holes, labeled_holes, range(1, num_features + 1))
    small_hole_labels = np.where(hole_sizes <= max_hole_area)[0] + 1
    return np.isin(labeled_holes, small_hole_labels)


def _align_to_codex_grid(
    img: Image.Image,
    cols: int,
    rows: int,
    frame_w: int,
    frame_h: int,
    cell_padding: int = 1,
) -> Image.Image:
    """将图像按网格逐帧居中重采样至严格的 192x208 单元格，并清理边界防止串帧。"""
    src_w, src_h = img.size
    target_total_w = cols * frame_w
    target_total_h = rows * frame_h

    canvas = Image.new("RGBA", (target_total_w, target_total_h), (0, 0, 0, 0))
    inner_max_w = frame_w - (cell_padding * 2)
    inner_max_h = frame_h - (cell_padding * 2)

    for r in range(rows):
        for c in range(cols):
            # 计算原图对应单元格的精确浮点边界并切片
            x0 = round(c * src_w / cols)
            y0 = round(r * src_h / rows)
            x1 = round((c + 1) * src_w / cols)
            y1 = round((r + 1) * src_h / rows)

            cell = img.crop((x0, y0, x1, y1))
            cw, ch = cell.size
            if cw == 0 or ch == 0:
                continue

            # 保持原单元格比例缩放并居中放入 192x208 帧内
            scale = min(inner_max_w / cw, inner_max_h / ch)
            new_w = max(1, round(cw * scale))
            new_h = max(1, round(ch * scale))
            resized_cell = cell.resize((new_w, new_h), Image.Resampling.LANCZOS)

            paste_x = c * frame_w + (frame_w - new_w) // 2
            paste_y = r * frame_h + (frame_h - new_h) // 2
            canvas.paste(resized_cell, (paste_x, paste_y), resized_cell)

    return canvas


def remove_black_bg_for_spritesheet(
    input_path: str,
    output_path: str,
    cols: int = CODEX_V2_COLS,
    rows: int = CODEX_V2_ROWS,
    frame_w: int = CODEX_V2_FRAME_W,
    frame_h: int = CODEX_V2_FRAME_H,
    noise_floor: int = 12,
    soft_threshold: int = 55,
    max_hole_area: int = 160,
) -> None:
    dst_file = Path(output_path)
    src_file = _resolve_input_image(input_path, exclude_output=dst_file)

    # 1. 读取任意格式图像，处理 EXIF 旋转信息（针对 JPG/WebP 等），并统一转换为 RGBA 浮点数组
    with Image.open(src_file) as raw_img:
        img_format = raw_img.format or src_file.suffix.upper()
        # 若为 GIF/WebP/APNG 等多帧图像，默认取第一帧（序列图本身应为单张大图）
        if getattr(raw_img, "is_animated", False):
            raw_img.seek(0)
        oriented_img = ImageOps.exif_transpose(raw_img)
        img = oriented_img.convert("RGBA")

    data = np.array(img, dtype=np.float32)
    rgb = data[:, :, :3]
    orig_alpha = data[:, :, 3]

    # 计算每个像素的最大通道亮度 (0 ~ 255)
    max_rgb = np.max(rgb, axis=2)

    # 2. 保护角色内部细小深色区域（如眼瞳、闭合小线条），排除肢体间大块闭合黑底
    opaque_body = (max_rgb >= soft_threshold) & (orig_alpha > 128)
    internal_dark_mask = _build_internal_dark_mask(
        opaque_body=opaque_body,
        max_hole_area=max_hole_area,
    )

    # 3. 计算平滑 Alpha 通道（消除底噪与边缘黑边）
    alpha = np.ones_like(max_rgb, dtype=np.float32) * 255.0

    # 纯黑底噪区直接设为全透明 (Alpha = 0)
    bg_zero_mask = (max_rgb <= noise_floor) & (~internal_dark_mask)
    alpha[bg_zero_mask] = 0.0

    # 边缘过渡与发光特效区（如第9行星光）按亮度线性映射 Alpha
    transition_mask = (
        (max_rgb > noise_floor)
        & (max_rgb < soft_threshold)
        & (~internal_dark_mask)
    )
    alpha[transition_mask] = (
        (max_rgb[transition_mask] - noise_floor) / (soft_threshold - noise_floor)
    ) * 255.0

    # 结合原图自带的 Alpha 通道（兼容带透明度的 PNG/WebP 及无透明度的 JPG/BMP）
    alpha = np.minimum(alpha, orig_alpha)

    # 4. 颜色反乘（Unmultiply）：去除半透明边缘混入的黑色底色，恢复原本鲜艳色彩
    alpha_norm = alpha / 255.0
    valid_alpha = alpha_norm > 0.01
    for c in range(3):
        channel = rgb[:, :, c]
        channel[valid_alpha] = np.clip(
            channel[valid_alpha] / alpha_norm[valid_alpha], 0.0, 255.0
        )
        channel[~valid_alpha] = 0.0
        data[:, :, c] = channel

    data[:, :, 3] = np.clip(alpha, 0.0, 255.0)
    clean_img = Image.fromarray(data.astype(np.uint8), mode="RGBA")

    # 5. 严格对齐至 Codex V2 标准网格尺寸：单帧 192x208，总分辨率 1536x2288
    final_img = _align_to_codex_grid(
        img=clean_img,
        cols=cols,
        rows=rows,
        frame_w=frame_w,
        frame_h=frame_h,
        cell_padding=1,
    )

    # 6. 校验最终输出尺寸并导出为无损透明 WebP 格式
    expected_size = (cols * frame_w, rows * frame_h)
    assert final_img.size == expected_size, (
        f"尺寸校验失败: 当前 {final_img.size}, 预期 {expected_size}"
    )

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(dst_file, format="WEBP", lossless=True, quality=100, method=6)

    print(
        f"处理完成: 输入 [{src_file.name} ({img_format})] -> 输出 [{dst_file}] | "
        f"最终尺寸: {expected_size[0]}x{expected_size[1]} "
        f"(网格: {cols}列x{rows}行, 单帧: {frame_w}x{frame_h})"
    )


if __name__ == "__main__":
    # 支持以下任意传参方式：
    # 1. 传 "./assets/1.png" 或 "./assets/1"：会自动匹配 1.png、1.jpg、1.jpeg、1.webp、1.bmp 等任意格式
    # 2. 传 "./assets"：会自动识别该目录下第一个有效的源图片文件并处理
    remove_black_bg_for_spritesheet(
        input_path="./assets/1",
        output_path="./assets/2.webp",
        cols=8,
        rows=11,
    )
