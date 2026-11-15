# AGX FPS Agent

用于 AGX「多人枪战：爆破模式」的 Python 起始策略与实验记录。先建立可复现基线，再依据本人比赛日志逐项改进。

**当前版本：v2，已通过 45 项本地测试，并于 2026-11-15 10:56:48 上传、随后通过 AGX 准入并自动启用参赛（指纹 `1667e3dea501`）；尚无正式比赛成绩。** 平台状态与版本身份记录在[迭代日志](experiments/iteration-log.md)。不把单元测试、历史日志重放或近似模型结果当作胜率。

## 快速开始

本项目只使用 Python 标准库，推荐本地 Python 3.10+。

```bash
python3 -m unittest discover -s tests -v
python3 scripts/pack.py
```

第二条命令生成 `dist/agx-fps-agent.zip`，ZIP 根目录只有 `agent.py`，并输出包的 SHA256。以后修改源码后必须重新打包。

在 [FPS 比赛页面](https://agx.cool/#/competitions/2a97f94f-816a-52b7-b4b8-5925db160245) 使用课程登记的评分账号报名，上传 ZIP，等待准入通过。2026-11-15 网页说明最新通过版本自动参赛，仍须确认「当前代码版本」已经更新；若界面提供版本切换入口，可主动选定已接受版本。上传、准入、生效以及完成排名赛是不同状态。

完成比赛后下载本人 `training.jsonl`，存到已忽略的 `data/private/`：

```bash
python3 scripts/replay_check.py data/private/training.jsonl --seat 0
```

`--seat` 必须使用该日志中本人实际席号。工具只在历史合法观察上检查代码运行和动作输出，不推进游戏，也不估计新策略胜率。

## 策略与接口

```python
def init(context):
    ...  # 初始化单个席位的状态

def act(observation):
    return {}  # 返回合法动作；空对象是中性动作
```

每个 agent 控制 5v5 中的一名玩家。策略基于官方公开 starter：缓存首次下发地图，按攻守购买装备，处理可见敌人的瞄准、开火和换弹，以及安包、拆包和脱困。

v1 使用按几何距离计价的 A* 寻路；到达任一包点的保守内圈即可安包；已知近处炸弹可在缺地图时拆除；安拆包保持当前武器槽。主武器耗尽后切换可用副武器，开火与装填检查自身时钟，购买顺序改为主武器、护甲、拆弹钳。保留 v0 的缺地图保护和地图身份变化清理。设计依据见[策略参考](docs/strategy-references.md)。

v2 在这些修复上选择性吸收朋友策略的思路：按自身速度刹停后射击，按实际请求控制手枪释放与步枪短连发，交战时切换已装弹副枪，并保持已开始的装填；当前可见目标之间采用迟滞选择，合法观察到的敌人位置仅用于短时预瞄，失去可见性就停止射击。攻方按回合选择共同包点，守方按席位分点，已知安包位置保留到回合或地图失效，安拆包先降速再连续交互。来源核查、明确拒绝的机制和朋友版本中的问题见[审查记录](docs/friend-agent-review.md)。

这仍是待实战验证的候选策略：未优化梯子导航、复杂团队协作、经济分配或投掷物。公开导航边也不保证实际可通行。

比赛策略运行时不调用 LLM；LLM 用于理解规则、设计实验、分析失败和辅助修改代码。课程材料没有要求运行时调用 LLM，当前提交环境和 CPU 限制也适合轻量策略。

## 迭代顺序

1. **v0 已完成：** 建立官方示例基线与本地工具，未参加排名赛。
2. **v1 已实现：** 修正可离线复现的目标、武器和预算问题，并加入距离寻路；完成首次平台准入验证。
3. **v2 已实现：** 审查朋友策略，选择性加入交战控制、短期记忆与分工；暂未取得正式对局证据。
4. **下一轮：** 用本人日志定位 faults、导航停滞、绕路及交火问题，逐项消融 v2 改动。
5. **后续：** 比较射击节奏、守包/回防、拆包时间和经济策略；跨地图与攻守验证，保留未用于调参的对局。

每次只验证一个明确假设，记录 Git commit、AGX 版本 ID、地图/阵营、完成状态、faults 和比赛结果。胜负受队友及对手影响，少量单局不代表改进有效。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `agent.py` | 上传策略 |
| `scripts/pack.py` | 生成可复现的最小 ZIP |
| `scripts/replay_check.py` | 本人历史观察的离线运行检查 |
| `scripts/benchmark_agent.py` | 合成网格上的寻路 CPU 计时，不代表平台总预算 |
| `tests/` | 协议和工具的合成用例 |
| `docs/competition.md` | 当前规则、预算和提交流程 |
| `docs/course-checklist.md` | 课程交付检查与报告结构 |
| `experiments/iteration-log.md` | 假设、改动、结果、下一步 |
| `experiments/ai-use-log.md` | AI 使用与人工核验记录 |
| `experiments/contributions.md` | 团队贡献记录 |

原始个人日志、账号凭据、课程原 PDF 和报名表不放入公开仓库。公开实验摘要只保留必要技术信息。

## 来源

规则核对日期：2026-11-15，当前官方 FPS 合同标为 `rsi-fps-5.5`。赛事页面及对应冻结版本是最终依据。

- [官方 FPS 开发指南](https://agx.cool/#/docs/fps)
- [官方 Python 提交指南](https://agx.cool/#/docs/python-submissions)
- [官方示例 agent.py](https://agx.cool/docs/schema/fps.example.agent.py)
- [官方最小示例 ZIP](https://agx.cool/docs/schema/fps.example.zip)
- [官方本地工具包](https://agx.cool/docs/schema/kits/agx-fps-kit.zip)：地图与分析工具，**不是完整模拟器**。
- [代码来源说明](NOTICE.md)
