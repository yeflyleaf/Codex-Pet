# 安装 Pets

1. 找到 Codex 数据目录；默认是 `%USERPROFILE%\.codex`，如果你设置了 `CODEX_HOME`，使用对应的数据目录。
2. 将本项目的整个 `pets/aimisi` 文件夹复制到数据目录的 `pets` 文件夹，使 `pet.json` 和 `aimisi.webp` 位于同一个目录。
3. 如果已有同名目录，先备份，不直接覆盖。
4. 在 Codex 的桌宠选择器中选择对应的角色。

#### 移除时，仅移走自己安装的 `pets/aimisi` 目录；不要删除整个 Codex 数据目录！

# 开发环境配置 (统一开发环境)

## 安装 uv

```PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 无论当前电脑全局安装的是什么版本，直接指定 Python 3.12 创建虚拟环境 (uv 检测到本地没有 3.12 时，会自动从官方源拉取独立的 3.12 解释器并生成 .venv)

```PowerShell
uv venv --python 3.12
```

## 安装依赖

```PowerShell
uv pip install -r requirements.txt
```

## 重构精灵图以符合 Codex Pet 规范

#### 前提：需要确保目标精灵图存放在 assets 文件夹下。

```PowerShell
uv run python tools/Compliant_Image_Reconstruction.py
```
