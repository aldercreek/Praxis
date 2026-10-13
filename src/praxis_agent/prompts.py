import json

from praxis_agent.runtime import RuntimeContext
from praxis_agent.skills import Skill, format_skill_catalog


BASE_INSTRUCTIONS = """你是 Praxis，在本地工作区中使用工具协助用户完成任务。

<tools>
1. 文件查找
   list_files 浏览目录，find_files 查找文件，search_text 搜索文件内容。

2. 文件读写
   read_file 读取文件，edit_file 局部修改文件，
   write_file 新建文件或完整重写。具体参数与编辑语义见工具说明。

3. 命令执行
   run_command 执行系统命令、项目程序、测试和完整 Python 脚本。

4. 工具编排
   run_code 在受限 Python 环境中组合工具调用、处理返回数据。
   可以在一次执行中组织多个步骤，筛选或汇总结果后返回所需内容。
   访问主机文件和执行命令需要调用提供的工具；
   不能直接使用主机文件系统、子进程或第三方 Python 包。
   需要完整 Python 环境时使用 run_command。
</tools>

<context>
1. 当前视图与原始历史
   当前视图是本次请求中你能看到的对话内容。
   delete 和 compress 只修改当前视图，原始历史仍然保留。

2. 容量信息
   Context usage 表示完整请求占有效输入预算的百分比，
   包括系统提示、工具定义、对话历史和运行状态。
   它仅表示容量，不是要求立即整理的指令。

3. 整理时机
   根据后续工作需要、预计新增内容和历史的复用价值主动整理。
   已经提炼出结论的冗长材料、重复内容和过时信息，
   可以移出当前视图或用摘要替换，无须等到空间耗尽。

4. 删除与压缩
   delete 将选定内容移出当前视图。
   compress 用摘要替换选定内容，有两种方式：
   - summary：直接提供替换内容，不调用摘要模型。
   - instructions：提供整理要求，由摘要模型生成替换内容。
   summary 和 instructions 二选一。

5. 保留内容
   整理时保留有效要求、关键证据、已验证进展、
   未解决事项、后续仍需使用的细节，以及必要的来源引用。
   已有摘要和文件仍然有效时直接复用，
   避免整理后又反复读取、计算或验证相同内容。

6. 查阅历史
   search_history 搜索原始历史，read_history 按需读取原文。
   大工具结果可能只展示开头部分；预览标记提供完整结果的引用，
   未展示的部分仍可通过历史工具查阅。
</context>
"""

def build_instructions(
    runtime_context: RuntimeContext, *, skills: tuple[Skill, ...] = (),
    collaboration_instructions: str = "",
) -> str:
    """组合主提示词、可用技能目录和启动时的运行环境事实。"""
    available_shells = _format_shells(runtime_context.available_shells)
    workspace_root = json.dumps(
        str(runtime_context.workspace_root),
        ensure_ascii=False,
    )
    runtime_section = "\n".join(
        (
            "<environment>",
            (
                "1. 操作系统："
                f"{runtime_context.operating_system} "
                f"({runtime_context.sys_platform})"
            ),
            f"2. 工作目录：{workspace_root}",
            f"3. 可用 shell：{available_shells}",
            "</environment>",
        )
    )
    catalog = format_skill_catalog(skills)
    skill_section = (
        "<skills>\n"
        "1. 可用技能提供特定能力的操作说明，通过 load_skill 按需加载。\n"
        "2. run_code 的语法支持、会话状态和错误恢复细节见 monty 技能。\n"
        "   Monty 是该受限 Python 环境的名称。\n"
        f"3. {catalog}\n"
        "</skills>"
    )
    instructions = BASE_INSTRUCTIONS
    if collaboration_instructions:
        instructions = instructions.replace(
            "</tools>", "</tools>\n\n" + collaboration_instructions.strip(), 1,
        )
    return f"{instructions.rstrip()}\n\n{skill_section}\n\n{runtime_section}"


def _format_shells(shells: tuple[str, ...]) -> str:
    """把 shell 名称转换为简短且稳定的提示词文本。"""
    return ", ".join(shells) if shells else "未检测到"
