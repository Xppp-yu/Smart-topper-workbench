# 私密 GitHub 与多 Agent 交接清单 v0.1

## 1. 当前本地状态

- Git 仓库已有历史，当前分支 `main`。
- 当前没有配置 Git remote。
- `configs/paths.local.json`、`data/processed/*`、`outputs/*` 已被 `.gitignore` 排除。
- 原始 PoPu/PMD/SLP 数据位于仓库外部，不应加入 Git。
- 当前 SLP 路线和代码仍在工作区，上传前需先审查、测试、提交。
- 当前候选上传集合为 181 个文件、1,333,100 bytes，最大单文件 108,691 bytes，未发现误纳入的大数据/模型文件。
- 对当前候选工作区做了高信号 secret pattern 扫描，未发现私钥、GitHub token、AWS key、OpenAI key 或显式 credential assignment；`gitleaks` 当前未安装，Git 历史级扫描仍是上传前待办。

## 2. 建议上传内容

- `src/`、`scripts/`、`tests/`；
- 不含秘密的 `configs/` 模板和冻结协议；
- `README.md`、`AGENTS.md`、`CLAUDE.md`；
- `docs/PROJECT_STATUS.md`、路线、任务清单、阶段报告；
- `docs/evidence/` 中脱敏且体积小的证据摘要。

禁止上传：

- SLP/PoPu/PMD 原始数据和压缩包；
- RGB、IR、Depth、压力逐帧原图，除非许可与隐私均明确且已脱敏；
- `configs/paths.local.json`、`.env`、GitHub/PAT/API tokens；
- AutoDL 登录信息、SSH 私钥；
- 大型 checkpoint、逐帧预测和完整 OOF 表；
- 含真实本地绝对路径或人员隐私的临时日志。

## 3. 上传前 Gate

```powershell
git status --short --branch
git diff --check
uv run pytest -q
git ls-files
git remote -v
```

另外完成：

1. 检查当前 diff，只提交本任务文件；
2. 对当前文件和 Git 历史做 secret scan；
3. 检查 tracked 文件大小，避免误提交模型/数据；
4. 确认 `docs/evidence/` 不含原始样本、绝对路径或个人信息；
5. 用明确 commit message 提交；
6. 创建 GitHub private repository；
7. 只给 ChatGPT/协作者授权这个仓库，不给 account-wide 不必要权限；
8. 配置 branch protection 或至少使用任务分支和 PR。

## 4. ChatGPT 读取方式

根据 OpenAI 官方说明，ChatGPT 可以通过 GitHub App 读取、搜索和引用已授权仓库中的代码和文档；可在 ChatGPT `Settings → Apps → GitHub` 连接并选择具体仓库。GitHub App 在 ChatGPT 中主要是读取/分析，不用于直接推送代码；写代码和推送应使用 Codex 或本地开发工具。

官方说明：[Connecting GitHub to ChatGPT](https://help.openai.com/en/articles/11145903-connecting-github-to-chatgpt)

私有或新仓库可能需要等待索引；官方文档给出的手动触发方式是在 GitHub 搜索：

```text
repo:<username>/<repository> import
```

GitHub App 是否出现在标准 ChatGPT、Deep Research 或 Agent Mode 中可能随账户方案和产品入口不同。

## 5. 推荐权限模型

```text
网页 GPT：读取仓库、讨论方案、审阅代码/报告
Claude Code/Codex：在本地 clone/worktree 中定点修改
Experiment Runner：只运行冻结配置并上传小型结果摘要
Reviewer：独立复算、抽查证据、批准 Gate
Owner：决定合并、远程 GPU、数据上传和最终路线
```

不要让任一 Agent 获得整个 GitHub 账号的无边界修改权限。授权对象应是一个明确的私有仓库；写入通过本地凭据、任务分支和 PR 管理。

## 6. 每轮连续开发流程

1. Owner/Controller 从 `docs/SLP_AGENT_TASK_BACKLOG_v0.1.md` 选择一个 TASK-ID。
2. 网页 GPT 阅读仓库并讨论方案，不直接把讨论称为完成。
3. Claude Code/Codex 在任务分支实现代码和测试。
4. 本地 Smoke 通过后停止；昂贵实验等待冻结协议和授权。
5. Experiment Runner 执行并保留 DONE/FAILED 证据。
6. Reviewer 独立复核。
7. 更新阶段报告和 `PROJECT_STATUS.md`。
8. 每 3–5 个任务、每个关键 Gate 或 Full 完成后进行一次总审计。

## 7. 网页 GPT 首轮提示模板

```text
请只读取并分析这个私有仓库。先阅读 README.md、AGENTS.md、
docs/PROJECT_STATUS.md、docs/SLP_TWO_PHASE_CONTINUOUS_DEVELOPMENT_PLAN_v0.2.md
和 docs/SLP_AGENT_TASK_BACKLOG_v0.1.md。

当前目标 TASK-ID = TASK-SLP-XXX。
请先总结已完成证据、依赖、风险和验收标准，再提出实现方案。
不要把代码计划当作已运行结果；不要把 R0/R1 OpenCV 伪标签称为真值；
不要建议上传原始数据或启动未授权 Full。
```
