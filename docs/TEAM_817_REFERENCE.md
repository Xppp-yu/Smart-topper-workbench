# 817论文数据探索接收记录

## 状态

`REFERENCE-ONLY / NOT-INTEGRATED`

来源目录：

```text
E:\创业\团队文件接收与项目推进存档\817论文数据探索
```

本工作区只读取该目录进行审阅，没有修改原件，也没有复制其中的 `.venv`。

## 核心文件哈希

| 文件 | SHA-256 |
|---|---|
| `README.md` | `C80E8C573C83F86355FA467F16BE628D7F1F3FECBE6D48456A47EBBF4BB52BEC` |
| `run_exploration.py` | `881ED6DE004F22ADCCA6E6D986E0CE088F0C26AF5D754CF2CD6518EC512E6E5A` |
| `run_pmd_exploration.py` | `ECC3E9CA9A494BFE1915D62B9732500EBE30E9B4035E43C14E046310E59B1FA8` |

## 已确认内容

- PoPu 脚本将每个记录中的 10 帧取均值，再进行受试者分组训练、验证和测试。
- 输出记录为 5100 条、60 名受试者，训练/验证/测试受试者无交叉。
- 已生成压力示例、混淆矩阵、分辨率指标和报告。
- PMD 脚本包含 Experiment I LOSO 和 Experiment II 跨床垫比较。

## 当前阻塞

- README 依赖相邻目录中的 `PoPu_data.zip` 和 `pmd-1.0.0.zip`，接收目录中不存在该相邻数据目录，因此不能原样复跑。
- 压缩包携带的 `.venv` 是 macOS Python 3.13 环境，不能在 Windows 或 WSL 复用。
- 截图中的 `schema.py`、`popu.py`、`inventory_popu.py`、`preview_popu.py` 不在本次接收目录中。
- 当前只有两份整段式探索脚本，没有统一 Sample、Loader、质量门、模型产物、置信度、REJECT 和正式自动测试。

## 方法修正要求

- 分辨率和模型选择必须使用验证集，不能根据测试集选择“最佳”方案。
- `empty` 只有 60 条记录，且幅值归一化可能放大空床噪声，应单独建立空床质量门。
- 单次 GroupShuffleSplit 应升级为重复分组评估或 GroupKFold/LOSO。
- 软件区域平均降采样只属于算法消融，不代表真实传感器硬件验证。
- 数组形状统一写作 `(rows=64, columns=27)`，显示布局可写作 `27 columns × 64 rows`。

## 后续接入顺序

1. 获取截图中真正的团队工程源码及依赖说明。
2. 在独立 Git 分支中进行只读审阅和路径适配。
3. 先接入统一 `PressureSample` 与 PoPu Loader。
4. 再接入 Quality、Downsample、Grouped Evaluation 和模型模块。
5. 运行本工作区测试，生成可追溯结果后再形成团队交付。

