# 策略参考与实验边界

核查日期：2026-11-15。适用合同：`rsi-fps-5.5`；运行约束与字段以 [AGX 官方 FPS 指南](https://agx.cool/#/docs/fps) 为准。

## 搜索结果

网页检索 `agx.cool FPS GitHub`、`rsi-fps-5.5`、`rsi-fps`，并搜索 GitHub 仓库 `agx fps`、`"agx.cool"`，未找到可确认的其他 AGX 参赛者公开策略代码。搜索结果包括本项目和不相关项目；这不代表所有参赛代码都不存在，也不能据此声称参考了榜首策略。

本项目直接基于 [AGX 官方 starter](https://agx.cool/docs/schema/fps.example.agent.py) 修改。以下外部项目只提供设计启发，没有复制其源码，也没有将其运行时引入提交包。

## 已核查的一手来源

| 来源 | 实际查阅内容 | 许可与适用限制 |
| --- | --- | --- |
| [YaPB `tasks.cpp`](https://github.com/yapb/yapb/blob/master/src/tasks.cpp) | 任务优先级、保留当前目标；安包、拆包分别处理；比较拆包剩余时间和爆炸时钟 | 当前仓库为 [Unlicense](https://github.com/yapb/yapb/blob/master/UNLICENSE)。面向 Counter-Strike / GoldSrc；引擎状态访问、地图单位和拆包时长不能直接用于 AGX。 |
| [YaPB `combat.cpp`](https://github.com/yapb/yapb/blob/master/src/combat.cpp) | 按可见部位、距离及后坐力选择瞄准位置，保持瞄准选择稳定 | 同上。AGX 只能使用合法 observation；不能照搬对全局实体、敌人隐藏状态的读取。 |
| [Recast / Detour `DetourNavMeshQuery.cpp`](https://github.com/recastnavigation/recastnavigation/blob/main/Detour/Source/DetourNavMeshQuery.cpp) | 路径成本按空间距离和区域成本计算 | [zlib 许可](https://github.com/recastnavigation/recastnavigation/blob/main/License.txt)。借鉴距离成本原则；AGX 的导航候选不等同权威碰撞，本项目独立实现 Python A*。 |

若后续直接使用外部源码，应同时记录具体版本、保留适用的许可声明，并标明修改；目前这些链接用于解释设计依据。

## v1 已采用

- **几何 A\*：** 用节点间距离替代 BFS 的边数成本，保留有向边语义并复用当前路径。地图或目标变化时重新规划。距离成本参考 Detour；可达性仍需通过实际对局验证。
- **目标交互：** 到达任一合法包点即可安包；已知炸弹位置时，拆包不依赖 `sites` 是否提供；交互保持当前 `active_slot`，避免策略主动切枪中断动作。这是结合 AGX 规则进行的修正，与 YaPB 将目标动作独立处理的思路一致。
- **装备与本人时钟：** 购买顺序为主武器、护甲、拆弹钳，避免 CT 恰有 2900 时先买钳导致买不起 M4；主武器弹匣与备弹均耗尽后选择可用副武器；开火检查本人换枪、装填和射击时钟。这些修正来自本项目行为与 AGX 接口核查。

这些变更不表示已经提高线上胜率。合成观察和本地计时可以验证分支、输出及耗时，不能代替真实排名对局。

## 后续实验，尚未实现

- **目标迟滞：** 保持仍可见的交火目标，避免两名敌人距离交替时来回转枪；测试目标消失及明显更优目标的切换。
- **射击节奏：** 用本人速度、精度和后坐力调整中远距射击；在真实日志中比较命中、耗弹与存活，避免直接套用 CS 参数。
- **拆弹决策：** 结合本人拆弹进度、工具包和公共爆炸时钟判断剩余时间；只使用合法可见包点，不猜测隐藏位置。
- **投掷物：** 先验证弹道、友伤和动作中断，再单独比较策略收益；v1 没有实现投掷物战术。

实验需同时满足 20 Hz 决策、单次后续调用 50 ms、整场累计 64 CPU 秒预算。建议一次只改一类策略，记录版本、地图、攻守方、故障和排名结果；不要把旧日志上的动作回放当作新策略的胜率证明。
