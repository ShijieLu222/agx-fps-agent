# v11 地图初始化修复与验证

修复可选地图字段为 null 时初始化失败的风险，并把建图与守点计划放到首次回调的 1 秒预算内，避免推迟到第二次回调的 50 毫秒预算。缺失或空的 bot_waypoints、单独守点/观察点列表、boxes、route_landmarks、出生点组均有回归覆盖。地图缓存继续复用，换图重建。

发布后已取得 9669 的本人完整原始日志并校验导出指纹：冻结版本确为 V10，真实策略输入包含 `bot_waypoints: null`。已归档 V10 对第二次原始输入在 `hints.get(...)` 复现 AttributeError；已发布 V11 对相同两次输入均无异常、动作合法。该空值故障已修复。导出只有 `fault: true`，没有服务器异常文本或子类型；两次输入复现也不等于 V11 整场实战通过。见 [真实输入故障复现](../experiments/v11-9669-fault-reproduction.json)。

## 已完成验证

- 163 项单元测试通过，新增 8 项地图契约/初始化回归。
- 五张地图、双方、七种字段场景，共 70 个开局检查通过。Mirage 使用已校验公开回放地图配合合成开局观察；其他四图取自本人历史输入。
- 首次回调含 JSON 的本地最大耗时 12.462 ms，后续开局回调最大 1.517 ms。不是服务器 CPU 或准入证明。
- 六场 169,907 次完整历史输入检查通过，无异常、非法动作或警告，本地预算检查通过。
- 对同样的历史输入，v11 与已归档 v10 的动作差异为 0；原有突击、支援、回防策略保持一致。历史输入不会随新动作改变，不代表新的胜率。

复现：

```bash
python3 -B -m unittest discover -s tests -v
python3 -B scripts/verify_v11_startup.py
python3 -B scripts/verify_9669_fault.py
python3 -B scripts/pack.py --output dist/agx-fps-agent-v11.zip
```

原始比赛数据保留在忽略的 data/private。公开验证结果见 experiments/v11-validation.json。

源码 SHA256：`e044b26e3de850493d231a753df71d4435277030468ebb43b00fa8c0d97f7163`。

ZIP SHA256：`adcccc500476286c4fda1ab6187dcb7d35b878ef330ec7bb5bd09beeb08657de`，20447 字节，根目录仅 agent.py，包内源码与受测文件一致。

发布状态：2026-11-16 17:23:12（北京时间）平台收到提交，准入 passed。刷新赛事页面确认当前代码版本为 `v11 · adcccc500476`；包和源码指纹均与本地受测候选一致。发布回执见 [platform-v11.json](../experiments/platform-v11.json)。
