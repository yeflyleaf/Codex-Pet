# 安装 Pets

1. 找到 Codex 数据目录；默认是 `%USERPROFILE%\.codex`，如果你设置了 `CODEX_HOME`，使用对应的数据目录。
2. 将本项目的整个 `pets/aimisi` 文件夹复制到数据目录的 `pets` 文件夹，使 `pet.json` 和 `spritesheet.webp` 位于同一个 `aimisi` 目录。
3. 如果已有同名目录，先备份，不直接覆盖。在 Codex 的桌宠选择器中选择 **爱弥斯**。

### 图集为 1536×2288 的透明 WebP，8 列 × 11 行，每帧 192×208；`pet.json` 使用 `spriteVersionNumber: 2`。

### 移除时，仅移走自己安装的 `pets/aimisi` 目录；不要删除整个 Codex 数据目录。

# 开发环境配置

## 创建虚拟环境

```bash
python -m venv .venv
```

## 激活虚拟环境

```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 重构精灵图以符合 Codex Pet 规范

```bash
python tools/Compliant_Image_Reconstruction.py
```
