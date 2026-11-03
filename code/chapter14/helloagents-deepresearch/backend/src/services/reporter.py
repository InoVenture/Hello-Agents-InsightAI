"""Service that consolidates task results into the final report."""

from __future__ import annotations

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState
from config import Configuration
from prompts import get_current_date
from utils import strip_thinking_tokens
from services.text_processing import strip_tool_calls

# 仅 Report Writer 生效的输出上限：DeepSeek 端点已验证可接受并实际产出 12000+ token；
# 报告输入（任务总结）约 2.8 万 token，加上 1.6 万输出仍在上下文窗口内。
REPORT_MAX_TOKENS = 16384


class ReportingService:
    """Generates the final structured report."""

    def __init__(self, report_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = report_agent
        self._config = config

    def generate_report(self, state: SummaryState) -> str:
        """Generate a structured report based on completed tasks."""

        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "暂无可用信息"
            sources_block = task.sources_summary or "暂无来源"
            tasks_block.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        note_references = []
        for task in state.todo_items:
            if task.note_id:
                note_references.append(
                    f"- 任务 {task.id}《{task.title}》：note_id={task.note_id}"
                )

        notes_section = "\n".join(note_references) if note_references else "- 暂无可用任务笔记"

        prompt = (
            f"当前日期：{get_current_date()}\n"
            f"研究主题：{state.research_topic}\n"
            f"任务概览：\n{''.join(tasks_block)}\n"
            f"任务笔记参考：\n{notes_section}\n"
            f"以上各任务总结已由上游分析完成，请直接基于任务概览撰写最终商业分析报告，"
            f"无需读取笔记，也不要调用任何工具。\n"
            f"请直接输出完整 Markdown 报告正文：严格遵循系统提示中的七段报告模板与可靠性要求。\n"
            f"不要输出任何工具调用指令（如 [TOOL_CALL:...]）、DSML 语法（如 <｜DSML｜ ...>）、"
            f"JSON 工具参数或中间推理过程。"
        )

        response = self._agent.run(prompt, max_tokens=REPORT_MAX_TOKENS)
        self._agent.clear_history()

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()

        return report_text or "报告生成失败，请检查输入。"

