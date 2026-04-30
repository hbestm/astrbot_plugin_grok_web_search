"""Sarasa Gothic 字体下载/解压辅助。

封装从镜像（清华 TUNA）/ GitHub Releases 自动获取最新版本字体压缩包并解压
到本地字体目录的逻辑。card_render 通过 ``init_fonts(font_dir)`` 取得字体路径。

设计要点：
- 启动时通过 GitHub Releases API 探测最新版本（失败回退到 ``FALLBACK_VERSION``），
  避免硬编码版本随上游更新而失效。
- 下载源按"清华 TUNA → GitHub 直链"顺序尝试，任一成功即停止。
- 解压优先调用系统 ``7z`` / ``7za``，否则回退 py7zr。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── 常量 ────────────────────────────────────────────────────

REPO = "be5invis/Sarasa-Gothic"
FALLBACK_VERSION = "1.0.37"
ARCHIVE_TEMPLATE = "SarasaTermSlabSC-TTF-{version}.7z"
DEFAULT_FONT_REGULAR = "SarasaTermSlabSC-Regular.ttf"
DEFAULT_FONT_BOLD = "SarasaTermSlabSC-Bold.ttf"
GITHUB_API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
TUNA_BASE = f"https://mirrors.tuna.tsinghua.edu.cn/github-release/{REPO}"
GITHUB_DOWNLOAD_BASE = f"https://github.com/{REPO}/releases/download"

_NUM_THREADS = 4
_SEVEN_Z_MAGIC = b"7z\xbc\xaf\x27\x1c"
_VERSION_RE = re.compile(r"^v?(\d+\.\d+\.\d+)$")

_log = logging.getLogger(__name__)


def set_logger(logger: logging.Logger) -> None:
    """允许调用方注入自定义 logger。"""
    global _log
    _log = logger


# ─── 版本发现 ────────────────────────────────────────────────


def discover_latest_version(timeout: float = 8.0) -> str:
    """探测 Sarasa Gothic 最新 release 版本，失败回退 FALLBACK_VERSION。"""
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={
                "User-Agent": "astrbot_plugin_grok_web_search font-loader",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        tag = str(data.get("tag_name", "")).strip()
        m = _VERSION_RE.match(tag)
        if m:
            ver = m.group(1)
            _log.info(f"检测到 Sarasa Gothic 最新版本: {ver}")
            return ver
        _log.warning(f"无法解析 release tag: {tag!r}, 使用回退版本")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        _log.info(f"探测最新版本失败 ({e})，使用回退版本 {FALLBACK_VERSION}")
    return FALLBACK_VERSION


def build_download_urls(version: str) -> tuple[str, ...]:
    """根据版本号生成候选下载 URL（按优先级排列）。"""
    archive = ARCHIVE_TEMPLATE.format(version=version)
    archive_q = urllib.parse.quote(archive)
    # 清华 TUNA 镜像采用 "Sarasa Gothic, Version X.Y.Z" 作为子目录名
    tuna_dir = urllib.parse.quote(f"Sarasa Gothic, Version {version}")
    return (
        f"{TUNA_BASE}/{tuna_dir}/{archive_q}",
        f"{TUNA_BASE}/LatestRelease/{archive_q}",
        f"{GITHUB_DOWNLOAD_BASE}/v{version}/{archive_q}",
    )


# ─── 本地字体探测 ────────────────────────────────────────────


def find_fonts_in_dir(font_dir: str) -> tuple[str, str] | None:
    """在目录中查找可用字体对 (regular, bold)。"""
    if not os.path.isdir(font_dir):
        return None
    ttf_files = [f for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
    if not ttf_files:
        return None

    regular = bold = None
    for f in sorted(ttf_files):
        fl = f.lower()
        if "bold" in fl:
            bold = bold or os.path.join(font_dir, f)
        elif any(k in fl for k in ("regular", "normal", "medium")):
            regular = regular or os.path.join(font_dir, f)

    if not regular and not bold and ttf_files:
        regular = bold = os.path.join(font_dir, ttf_files[0])
    elif regular and not bold:
        bold = regular
    elif bold and not regular:
        regular = bold

    return (regular, bold) if regular and bold else None


# ─── 解压 ─────────────────────────────────────────────────────


def _extract_7z(archive_path: str, output_dir: str) -> None:
    """优先用系统 7z / 7za 解压，回退 py7zr。"""
    for cmd in ("7z", "7za"):
        try:
            result = subprocess.run(
                [cmd, "x", archive_path, f"-o{output_dir}", "-y"],
                capture_output=True,
                timeout=120,
            )
            if result.returncode == 0:
                _log.info(f"使用 {cmd} 解压成功")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    _log.info("系统未安装 7z，尝试使用 py7zr 解压 ...")
    try:
        import py7zr
    except ImportError as e:
        raise RuntimeError(
            "解压字体压缩包需要系统 7z 工具或 py7zr Python 包，"
            "请安装之后重试：pip install py7zr"
        ) from e

    with py7zr.SevenZipFile(archive_path, "r") as z:
        z.extractall(path=output_dir)


# ─── 下载 ─────────────────────────────────────────────────────


def _fetch_to_file(url: str, dest_archive: str, font_dir: str) -> None:
    """从 url 下载字体压缩包到 dest_archive，按需多线程分段。"""
    _log.info(f"正在下载字体: {url}")

    supports_range = False
    total_size = 0
    try:
        probe = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        probe_resp = urllib.request.urlopen(probe, timeout=30)
        if probe_resp.status == 206:
            content_range = probe_resp.headers.get("Content-Range", "")
            if "/" in content_range:
                total_size = int(content_range.split("/")[-1])
                supports_range = True
        probe_resp.close()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass

    if supports_range and total_size > 1024 * 1024:
        _log.info(
            f"文件大小: {total_size / 1024 / 1024:.1f}MB, 使用 {_NUM_THREADS} 线程下载"
        )
        chunk_sz = total_size // _NUM_THREADS
        chunk_ranges = []
        for i in range(_NUM_THREADS):
            start = i * chunk_sz
            end = (total_size - 1) if i == _NUM_THREADS - 1 else (start + chunk_sz - 1)
            chunk_ranges.append((i, start, end))

        downloaded_lock = threading.Lock()
        downloaded_bytes = [0]
        last_logged = [-1]

        def _download_chunk(idx: int, byte_start: int, byte_end: int) -> str:
            part_file = os.path.join(font_dir, f"_chunk_{idx}.part")
            r = urllib.request.Request(
                url, headers={"Range": f"bytes={byte_start}-{byte_end}"}
            )
            response = urllib.request.urlopen(r, timeout=180)
            with open(part_file, "wb") as f:
                while True:
                    buf = response.read(64 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    with downloaded_lock:
                        downloaded_bytes[0] += len(buf)
                        pct = int(downloaded_bytes[0] / total_size * 100)
                        if pct // 10 > last_logged[0] // 10:
                            dl_mb = downloaded_bytes[0] / 1024 / 1024
                            tot_mb = total_size / 1024 / 1024
                            _log.info(
                                f"字体下载进度: {pct}% ({dl_mb:.1f}/{tot_mb:.1f}MB)"
                            )
                            last_logged[0] = pct
            return part_file

        part_files: list[str | None] = [None] * _NUM_THREADS
        try:
            with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
                futures = {
                    pool.submit(_download_chunk, idx, s, e): idx
                    for idx, s, e in chunk_ranges
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    part_files[idx] = future.result()

            with open(dest_archive, "wb") as out:
                for pf in part_files:
                    if pf is None:
                        raise RuntimeError("分段下载缺失")
                    with open(pf, "rb") as inp:
                        shutil.copyfileobj(inp, out)
            _log.info("字体下载完成，正在校验 ...")
        finally:
            for pf in part_files:
                if pf and os.path.exists(pf):
                    try:
                        os.remove(pf)
                    except OSError:
                        pass
        return

    if not supports_range:
        _log.info("服务器不支持分段下载，使用单线程模式")
    total_mb = total_size / 1024 / 1024 if total_size > 0 else 0
    if total_mb:
        _log.info(f"文件大小: {total_mb:.1f}MB")

    resp = urllib.request.urlopen(urllib.request.Request(url), timeout=180)
    downloaded = 0
    last_logged_pct = -1
    part_path = dest_archive + ".part"

    with open(part_path, "wb") as f:
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = int(min(downloaded / total_size, 1.0) * 100)
                if pct // 10 > last_logged_pct // 10:
                    f.flush()
                    os.fsync(f.fileno())
                    _log.info(
                        f"字体下载进度: {pct}% "
                        f"({downloaded / 1024 / 1024:.1f}/{total_mb:.1f}MB)"
                    )
                    last_logged_pct = pct

    os.rename(part_path, dest_archive)
    _log.info("字体下载完成")


def _cleanup_partial(font_dir: str, archive_path: str) -> None:
    candidates = [archive_path, archive_path + ".part"]
    candidates.extend(
        os.path.join(font_dir, f"_chunk_{i}.part") for i in range(_NUM_THREADS)
    )
    for stale in candidates:
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass


def download_and_install(font_dir: str, version: str | None = None) -> None:
    """下载并解压字体到 font_dir，留下 Regular/Bold 两个 ttf。"""
    os.makedirs(font_dir, exist_ok=True)
    if version is None:
        version = discover_latest_version()

    archive_path = os.path.join(font_dir, "_font_download.7z")
    urls = build_download_urls(version)

    if os.path.exists(archive_path):
        _log.info("检测到已下载的字体包，正在校验 ...")
    else:
        last_err: Exception | None = None
        downloaded_ok = False
        for url in urls:
            try:
                _fetch_to_file(url, archive_path, font_dir)
                downloaded_ok = True
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                _log.warning(f"从 {url} 下载失败: {e}")
                _cleanup_partial(font_dir, archive_path)
        if not downloaded_ok:
            raise RuntimeError(
                f"所有镜像均下载失败，最后错误: {last_err}"
            ) from last_err
        _log.info("正在校验下载的字体包 ...")

    with open(archive_path, "rb") as f:
        header = f.read(6)
    if header != _SEVEN_Z_MAGIC:
        _log.warning("下载的文件不是有效的 7z 压缩包，已删除，请重试")
        os.remove(archive_path)
        raise ValueError("Downloaded file is not a valid 7z archive")

    expected = {DEFAULT_FONT_REGULAR, DEFAULT_FONT_BOLD}
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _extract_7z(archive_path, tmp_dir)
            kept = 0
            for root, _dirs, files in os.walk(tmp_dir):
                for fname in files:
                    if fname in expected:
                        shutil.copy2(os.path.join(root, fname), font_dir)
                        kept += 1
            _log.info(f"字体安装完成 ({kept} 个文件)")
    finally:
        if os.path.exists(archive_path):
            try:
                os.remove(archive_path)
            except OSError:
                pass


def init_fonts(font_dir: str) -> tuple[str, str] | None:
    """探测/下载字体，返回 (regular_path, bold_path) 或 None。"""
    found = find_fonts_in_dir(font_dir)
    if found:
        return found
    try:
        download_and_install(font_dir)
    except Exception as e:  # noqa: BLE001
        _log.warning(f"字体下载失败: {e}")
        return None
    return find_fonts_in_dir(font_dir)
