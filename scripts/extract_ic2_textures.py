from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

VERSION = "2.8.221-ex112"
OFFICIAL_DOWNLOAD = "https://edge.forgecdn.net/files/3078/604/industrialcraft-2-2.8.221-ex112.jar"
PREFIX = "assets/ic2/textures/items/reactor/"


def main() -> None:
    parser = argparse.ArgumentParser(description="将 IC2 2.8.221 反应堆贴图提取到本地 Git 忽略目录")
    parser.add_argument("jar", nargs="?", type=Path, help="本地 industrialcraft-2-2.8.221-ex112.jar")
    parser.add_argument("--output", type=Path, default=Path("public/ic2-textures"))
    args = parser.parse_args()
    jar = args.jar or Path("public/ic2-2.8.221.jar")
    if not jar.exists():
        jar.parent.mkdir(parents=True, exist_ok=True)
        print(f"正在从官方 CurseForge CDN 下载 IC2 {VERSION}…")
        urllib.request.urlretrieve(OFFICIAL_DOWNLOAD, jar)
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(jar) as archive:
        for info in archive.infolist():
            if not info.filename.startswith(PREFIX) or not info.filename.endswith(".png"):
                continue
            relative = Path(info.filename[len(PREFIX):])
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            count += 1
    print(f"已提取 {count} 张反应堆贴图到 {args.output.resolve()}（该目录不会提交到 Git）")


if __name__ == "__main__":
    main()
