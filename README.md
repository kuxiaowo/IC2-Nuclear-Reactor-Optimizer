# IC2 2.8.221 核反应堆模拟与优化器

面向 Minecraft 1.12.2 / IndustrialCraft² Experimental 2.8.221 的中文本地 Web 工具。第一阶段仅模拟 EU 模式、铀单/双/四联燃料棒、IC2 原版热管理组件和三种中子反射板。

## 已实现

- 6×3 至 6×9 布局编辑，点击/拖动放置、右键擦除、清空、复制和组件计数。
- 每秒一次反应堆结算，并无损展开为 20 个 game ticks；播放 `tick/s` 与物理计算解耦。
- 暂停、继续、逐 tick、拖动跳转、温度/EU 曲线、事件列表、组件热量/耐久覆盖层。
- 融毁或 tick 上限停止；支持原位自动续棒和周期稳态检测。
- HDF5 分块压缩轨迹、分页 tick 查询、按像素采样图表、摘要/完整组件 CSV 导出。
- 库存约束、两种燃料约束、Mark I–V 独立榜单、固定种子、取消与继续改进。
- 多进程遗传/局部变异启发式搜索，以及 1–64 工作进程的库存感知并行穷举与全局最优证明；穷举开始前会显示完整枚举方案数并要求确认。
- CUDA 加速：启发式模式由 GPU 批量筛选候选池；穷举可选择“固定点证书 + CPU 回退”或实验性的“一个 CUDA 线程完整模拟一个布局”，完整计数和全局最优证明不变。
- FastAPI 托管 Vite 生产构建；只监听本机地址。

当前明确不支持流体反应堆、MOX、红石脉冲、自动换散热件、外部冷却和 Reactor Coolant Injector。

## 环境与启动

普通 Python 虚拟环境（推荐，Python 3.12 或更高版本）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

最后一个命令会自动检查前端：首次运行或前端源码、依赖清单发生变化时，自动执行 npm 依赖安装与生产构建；已有最新构建时直接启动。随后打开 `http://127.0.0.1:8000`。它不依赖 conda、PowerShell 启动脚本或预先安装本项目的软件包，但首次构建仍要求系统已安装 Node.js/npm。

可用 `python main.py --no-browser` 禁止自动打开浏览器，通过 `--host` 和 `--port` 修改监听地址；`--rebuild` 强制重新构建，`--no-build` 跳过自动构建检查。

如果更习惯 conda，也可以继续使用：

```powershell
conda env create -f environment.yml
conda activate ic2-reactor-optimizer
python main.py
```

`environment.yml` 会安装 CUDA 13 的 `numba-cuda` 工具链，适配本机 RTX 5070 Ti（计算能力 12.0）。已存在的环境可执行：

```powershell
conda install -n ic2-reactor-optimizer -c conda-forge numba-cuda "cuda-version=13"
```

普通 `requirements.txt` 保持 CPU 轻量安装。若使用 pip 配置 NVIDIA GPU，可安装 `.[cuda]`；仍需兼容的 NVIDIA 驱动。界面中“自动”优先使用 CUDA 固定点证书，“仅 CPU”完全跳过 GPU 探测。穷举有两种 CUDA 后端：

- `CUDA 固定点证书 + CPU 回退`：GPU 只接受可以精确证明的短期固定点；换热器、有限热容/耐久等复杂状态回退 CPU。
- `CUDA 完整模拟（实验）`：每个 CUDA 线程保存一个布局的堆热、组件热量、耐久和事件状态，执行与 CPU 相同的行优先热循环；kernel 按默认 256 个反应堆周期分段启动，以降低 Windows WDDM 超时风险。周期判定先用 64 位哈希筛选，再逐字段比较完整状态，哈希碰撞不会被当作证明。

两者都使用单一 GPU 上下文和大批次；CPU-only 穷举继续使用多进程。完整模拟的设计、适用边界和复现基准见 [docs/GPU_FULL_SIMULATION.md](docs/GPU_FULL_SIMULATION.md)。

开发时可分别运行：

```powershell
npm run api
npm run dev
```

Vite 开发地址为 `http://127.0.0.1:3000`，`/api` 自动代理到 FastAPI。

## 原版贴图

```powershell
python scripts/extract_ic2_textures.py
```

脚本在缺少 JAR 时从 CurseForge CDN 下载 `industrialcraft-2-2.8.221-ex112.jar`，只提取反应堆物品贴图到 `public/ic2-textures/`。JAR 和贴图目录均被 Git 忽略，不会随仓库提交；缺失时界面显示中文缩写。

## 验证

```powershell
pytest -q
npm run build
npm audit
python scripts/benchmark_gpu.py
python scripts/benchmark_gpu_exhaustive.py
```

测试覆盖官方字节码燃料金标准、组件参数、容量与即时移除边界、槽位顺序、反射板脉冲损耗、耗尽/续棒、临界/融毁、Mark 边界、HDF5 game-tick 展开、库存约束和完整有标签空间枚举。

规则、参数出处与已知边界见 [docs/RULES.md](docs/RULES.md)。Mark I 固定点快进、二层枚举完整性、最优解不遗漏和性能边界的结构化证明见 [Markdown 证明文档](docs/MARK_I_MATHEMATICAL_ANALYSIS.md)，可编译的正式公式稿见 [LaTeX 源文件](docs/MARK_I_OPTIMIZATION_PROOFS.tex)。长轨迹写入 Git 忽略的 `.data/traces/`。

## 重要说明

“启发式最优”只表示当前预算内找到的最佳解。排行榜严格按平均 EU/t 排序。全局穷举不使用时间、代数、种群或随机种子预算，只选择计算后端、GPU 批大小或 CPU 工作进程数；镜像方向会分别检查，仅在排行榜展示时归组。Mark I 搜索先计算燃料与反射板组成的发电骨架，低于当前榜单功率下界的完整布局由数学上界证明跳过热学模拟；其余布局由 CPU、GPU 精确证书或 GPU 完整状态机验证。取消后只保留当前最佳，只有完整有标签空间全部完成精确验证或上界证明后才显示“已证明全局最优”。本项目暂不附带开源许可证；这不等于默认允许复制、修改或再发布。
