# AGX FPS Agent

用于 AGX「多人枪战：爆破模式」的 Python 起始策略与实验记录。先建立可复现基线，再依据本人比赛日志逐项改进。

**当前状态：本地起始版本，尚未通过 AGX 准入，也没有正式比赛成绩。** 不把单元测试、历史日志重放或近似模型结果当作胜率。

## 快速开始

本项目只使用 Python 标准库，推荐本地 Python 3.10+。

```bash
python3 -m unittest discover -s tests -v
python3 scripts/pack.py
```

第二条命令生成 `dist/agx-fps-agent.zip`，ZIP 根目录只有 `agent.py`，并输出包的 SHA256。以后修改源码后必须重新打包。

在 [FPS 比赛页面](https://agx.cool/#/competitions/2a97f94f-816a-52b7-b4b8-5925db160245) 使用课程登记的评分账号报名，上传 ZIP，等待准入通过，再选择接受的版本并 **Use this version in competition / 激活参赛**。上传、准入、激活以及完成排名赛是不同步骤。

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

每个 agent 控制 5v5 中的一名玩家。当前策略基于官方公开 starter：缓存首次下发的地图，用 BFS 导航；按攻守购买装备；处理可见敌人的瞄准、开火和换弹；尝试安包、拆包和脱困。本项目增加缺地图保护、地图身份变化时清理状态、初始化清理，以及目标交互时避免换弹冲突。

这仍是弱基线：未优化梯子导航、团队协作、经济分配或投掷物。公开导航边也不保证实际可通行。

比赛策略运行时不调用 LLM；LLM 用于理解规则、设计实验、分析失败和辅助修改代码。课程材料没有要求运行时调用 LLM，当前提交环境和 CPU 限制也适合轻量策略。

## 迭代顺序

1. **v0：建立基线。** 准入通过、激活、完成至少一场 ranked match；记录版本和故障。
2. **v1：导航。** 用本人日志定位卡住、绕路、高低差和目标不可达，先降低无效移动。
3. **v2：交战。** 比较停步射击、短点射、瞄准阈值和换弹时机。
4. **v3：目标与经济。** 检查携包路线、守包/回防、拆包时间、购买优先级。
5. **v4：泛化。** 跨五张地图、攻守两侧和不同对手验证；保留未用于调参的对局。

每次只验证一个明确假设，记录 Git commit、AGX 版本 ID、地图/阵营、完成状态、faults 和比赛结果。胜负受队友及对手影响，少量单局不代表改进有效。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `agent.py` | 上传策略 |
| `scripts/pack.py` | 生成可复现的最小 ZIP |
| `scripts/replay_check.py` | 本人历史观察的离线运行检查 |
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
