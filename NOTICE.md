# 来源说明

`agent.py` 基于 AGX 官方提供给参赛者的公开 FPS starter：

- 源码：https://agx.cool/docs/schema/fps.example.agent.py
- 文档：https://agx.cool/#/docs/fps
- 获取日期：2026-11-15

初始策略来自官方示例。本项目增加健壮性保护、几何 A*、目标与装备决策修正、测试及开发工具；变化记录在 README 和实验日志中。当前没有真实比赛结果或性能提升声明。

v1 查阅 YaPB 的任务/交战控制及 Recast/Detour 的路径成本实现，独立借鉴设计思想，没有复制这些项目的源码或引入其运行时。具体来源、许可证和适用限制见 [策略参考](docs/strategy-references.md)。

下载的官方示例没有附带明确的开源许可证。本仓库保留来源归属，不将第三方代码重新声明为 MIT 或其他授权；公开可见不自动授予额外的再许可权。课程原 PDF 和报名资料未收录。
