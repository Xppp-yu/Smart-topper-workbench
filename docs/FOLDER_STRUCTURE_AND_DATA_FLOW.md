# Windows工作台文件夹结构与功能目的

## 1. E:\TeamProjects顶层

```text
E:\TeamProjects\
├── README.md                         整个Windows工作区导航
├── datasets\                        跨工程共享的公共原始数据
└── smarttopper-team-workbench\      压力感知算法研究工程
```

## 2. datasets数据层

```text
datasets\
└── smart-topper\
    ├── DATASET_COPY_RECORD.md        数据来源、复制记录和数量证据
    ├── popu\PoPu_data\              PoPu完整Windows研究副本
    └── pmd\                          PMD完整Windows研究副本
```

原则：这里只保存来源数据与数据版本记录，不存放清洗结果、特征、模型或图表。

## 3. smarttopper-team-workbench项目层

```text
smarttopper-team-workbench\
├── .git\                    Windows研究代码版本历史
├── .venv\                   本项目Python 3.12环境，不提交Git
├── .vscode\                 VS Code解释器和pytest设置
├── configs\                 标签、路径、阈值、实验和模型参数
├── data\
│   ├── public_raw\          临时接收入口或轻量manifest，不放公共数据主副本
│   └── processed\           Inventory、清洗数据、特征表等派生数据
├── demo\                    固定输入下的可演示最小入口
├── docs\                    方向、结构、验收、口径和阶段结论
├── models\                  经过人工挑选、准备晋升的候选模型
├── notebooks\              探索性分析，不作为最终唯一运行入口
├── outputs\
│   ├── figures\             热力图、叠加图、混淆矩阵、错误样本图
│   ├── metrics\             指标、逐受试者结果、逐样本预测
│   ├── models\              每次实验自动生成的模型产物
│   └── reports\             run manifest、汇总JSON和实验结论
├── scripts\                 可直接运行的Inventory、Viewer、训练、评估入口
├── src\topper_perception\   可复用的读取、质量、几何、特征和模型代码
├── tests\                   单元测试、异常测试和真实数据烟雾测试
├── .gitignore               环境、数据和生成结果排除规则
├── .python-version          约定Python 3.12
├── pyproject.toml           依赖和测试配置
├── uv.lock                  精确依赖锁
└── README.md                项目入口与常用命令
```

## 4. src正式研究代码层

目前已经存在：

```text
src\topper_perception\
├── healthcheck.py
├── io\
│   └── popu.py
└── visualization\
    └── pressure_heatmap.py
```

后续按实际阶段逐步增加，不提前制造空壳：

```text
quality\
preprocessing\
geometry\
features\
posture\
evaluation\
robustness\
regions\
```

## 5. scripts运行入口层

`src`保存可复用函数，`scripts`负责把函数按一次任务串起来。

例如：

```text
scripts\preview_popu.py
    ↓ 调用
src\topper_perception\io\popu.py
    ↓ 调用
src\topper_perception\visualization\pressure_heatmap.py
    ↓ 输出
outputs\figures\*.png + *.json
```

稳定实验必须有脚本入口；Notebook中的代码如果被确定采用，应移入`src`并由`scripts`调用。

## 6. outputs与data/processed的区别

- `data/processed/`：下一阶段程序还会继续读取的数据，例如Inventory表、清洗结果和特征表。
- `outputs/`：供人验收或交付的结果，例如图片、指标、模型、报告。

不要把所有文件都堆到`outputs/`，也不要把图和模型放进原始数据目录。

## 7. models与outputs/models的区别

- `outputs/models/`：每次实验自动生成，可能很多，默认不是正式模型。
- `models/`：经过评价和人工选择、准备形成WSL晋升包的候选模型。

模型从`outputs/models/`进入`models/`必须附带metrics、config、split和limitations。

## 8. 数据流

```text
E:\TeamProjects\datasets\smart-topper
    ↓ configs/paths.local.json
src/io Adapter
    ↓
data/processed Inventory/Features
    ↓
src/quality/geometry/features/posture
    ↓
outputs/metrics + figures + models + reports
    ↓
研究决策
    ↓
handoff冻结包
    ↓
WSL正式工程实现
```

