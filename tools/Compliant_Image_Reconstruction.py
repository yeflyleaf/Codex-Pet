from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from scipy import ndimage

# Codex-Pet V2 硬性规格常量 (8列 x 11行, 单帧 192x208, 总尺寸 1536x2288)
CODEX_V2_COLS: int = 8
CODEX_V2_ROWS: int = 11
CODEX_V2_FRAME_W: int = 192
CODEX_V2_FRAME_H: int = 208


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
    src_file = Path(input_path)
    if not src_file.exists():
        raise FileNotFoundError(f"未找到输入图像文件: {src_file.resolve()}")

    # 1. 读取图像并转换为 RGBA 浮点数组
    img = Image.open(src_file).convert("RGBA")
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

    # 结合原图自带的 Alpha 通道（若输入本身为带透明度的 PNG）
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

    dst_file = Path(output_path)
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    final_img.save(dst_file, format="WEBP", lossless=True, quality=100, method=6)

    print(
        f"处理完成: {dst_file} | "
        f"最终尺寸: {expected_size[0]}x{expected_size[1]} "
        f"(网格: {cols}列x{rows}行, 单帧: {frame_w}x{frame_h})"
    )


if __name__ == "__main__":
    # 将输入文件名替换为保存的图片路径，输出直接命名为 pet.json 所需的 WebP 文件
    remove_black_bg_for_spritesheet(
        input_path="./assets/demo.png",
        output_path="./assets/demo.webp",
        cols=8,
        rows=11,
    )
    