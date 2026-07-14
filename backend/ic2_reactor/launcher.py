from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path


FRONTEND_SOURCE_DIRS = ("app", "src", "public")
FRONTEND_SOURCE_FILES = (
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "vite.config.ts",
    "eslint.config.mjs",
)
FRONTEND_EXTENSIONS = {
    ".css", ".gif", ".html", ".ico", ".jpeg", ".jpg", ".js", ".json",
    ".jsx", ".png", ".svg", ".ts", ".tsx", ".webp",
}


class FrontendBuildError(RuntimeError):
    """Raised when automatic frontend preparation cannot finish."""


def _frontend_sources(root: Path) -> list[Path]:
    sources = [root / name for name in FRONTEND_SOURCE_FILES]
    for directory_name in FRONTEND_SOURCE_DIRS:
        directory = root / directory_name
        if directory.exists():
            sources.extend(
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in FRONTEND_EXTENSIONS
            )
    return [path for path in sources if path.exists()]


def frontend_build_required(root: Path) -> bool:
    output = root / "dist" / "index.html"
    if not output.exists():
        return True
    built_at = output.stat().st_mtime_ns
    return any(path.stat().st_mtime_ns > built_at for path in _frontend_sources(root))


def _dependency_fingerprint(root: Path) -> str:
    dependency_file = root / "package-lock.json"
    if not dependency_file.exists():
        dependency_file = root / "package.json"
    return hashlib.sha256(dependency_file.read_bytes()).hexdigest()


def _run_npm(npm: str, arguments: list[str], root: Path) -> None:
    print(f"[前端] 执行：npm {' '.join(arguments)}", flush=True)
    try:
        subprocess.run([npm, *arguments], cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise FrontendBuildError(f"npm {' '.join(arguments)} 失败，退出码 {exc.returncode}") from exc


def prepare_frontend(root: Path, *, force: bool = False) -> bool:
    """Install dependencies when needed and produce a current frontend build."""
    if not force and not frontend_build_required(root):
        print("[前端] 构建已是最新，跳过 npm。", flush=True)
        return False

    npm = shutil.which("npm")
    if npm is None:
        raise FrontendBuildError("需要构建前端，但未找到 npm。请先安装 Node.js（包含 npm）。")

    node_modules = root / "node_modules"
    fingerprint = _dependency_fingerprint(root)
    marker = node_modules / ".ic2-package-lock.sha256"
    installed_fingerprint = marker.read_text(encoding="ascii").strip() if marker.exists() else ""
    if not node_modules.is_dir():
        install_command = ["ci"] if (root / "package-lock.json").exists() else ["install"]
        _run_npm(npm, install_command, root)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_dependency_fingerprint(root), encoding="ascii")
    elif installed_fingerprint != fingerprint:
        # ``npm ci`` deletes node_modules first and is prone to EPERM failures
        # on Windows when a native Vite/Rolldown module is briefly locked.
        _run_npm(npm, ["install"], root)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_dependency_fingerprint(root), encoding="ascii")

    _run_npm(npm, ["run", "build"], root)
    if not (root / "dist" / "index.html").exists():
        raise FrontendBuildError("npm 构建完成，但没有生成 dist/index.html。")
    return True
