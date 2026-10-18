# Praxis

Praxis 是一个本地 Python coding agent，当前入口是基于 Textual 的终端 UI。它可以持续对话、调用本地和 MCP 工具、执行命令、接收运行中的补充输入，并通过子 Agent 并行处理任务。

当前 UI 和 Agent 运行在同一个进程，通过线程和队列交互；尚未提供桌面壳、Web 服务或供桌面前端调用的 stdio app-server。MCP 的 stdio 是连接外部工具服务的通道，与 UI 接入协议是两回事。

## 安装与快速开始

需要 Python 3.12+。文件搜索依赖 `ripgrep`，请确保 `rg` 在 PATH 中可用；`find_files` 和 `search_text` 没有纯 Python 回退实现。

在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
rg --version
```

配置模型凭据：

```powershell
$env:ADA_API_KEY = "your-coding-plan-token"
# 可选；以下是当前默认值
$env:ADA_MODEL = "auto(free)"
$env:ADA_BASE_URL = "https://ada-cli-go.ctripcorp.com/coding-plan/openai/v1"
```

模型客户端使用流式 Responses API。主 Agent、子 Agent 和独立摘要客户端使用同一份模型配置；摘要客户端设置 `reasoning_effort="none"`。恢复已有会话时使用会话保存的模型，容量相关环境变量仍可覆盖默认值。

切换到要处理的项目目录后启动，当前目录就是主 Agent 的工作目录：

```powershell
praxis
praxis "检查这个项目的任务执行流程"
praxis "运行测试，分析并修复失败"
```

传入初始任务也会打开终端 UI，任务完成后继续保留输入框。

## 终端交互

界面显示流式回答、工具活动、命令输出预览、计划、上下文容量和运行时长。多 Agent 各有独立执行页面；工具完整结果和完整参数在详情页查看。

| 操作 | 行为 |
| --- | --- |
| Enter | 空闲时开始任务；运行中将补充输入排队 |
| Ctrl+T | 打开或关闭当前 Agent 的完整记录；审批时查看完整参数 |
| Alt+← / Alt+→ | 切换主 Agent / 子 Agent 页面 |
| 审批面板 y / 1 | 允许本次操作 |
| 审批面板 n / 2 / Esc | 拒绝本次操作 |
| 审批面板 ↑ / ↓ + Enter | 选择并提交审批决定 |
| /clear | 空闲时保留旧会话，切换到新的空会话并重置运行时 |
| /exit、Ctrl+Q、Ctrl+C | 请求关闭整个应用及其 Agent、命令和连接 |

`run_command` 默认需要审批。MCP 工具也默认需要审批，具体配置见后文。可以在本次进程内跳过审批：

```powershell
praxis --bypass
praxis --bypass "运行测试并修复失败"
praxis --bypass --continue
```

`--bypass` 不会保存到会话，也不会取消工具本身的参数校验。

### 运行中补充输入（steer）

提交任务后可以继续输入，例如：

```text
› 检查这个项目
› 重点看配置加载流程
› 先只分析，不要修改文件
```

后两条可以在第一条任务执行期间发送。输入始终交给主 Agent，包括正在查看子 Agent 页面时。底部会显示尚未处理的消息数量。

实现只有一个线程安全的 `user_input` 队列：

1. UI 接收输入并入队，不并发调用同一个 Agent 的 `run()`。
2. Agent 在每次模型循环开始时按 FIFO 顺序取出消息，以 `role=user` 加入历史，然后构造下一次模型请求。
3. 模型准备给出最终回答时再次检查队列；有补充就继续当前任务的模型循环。
4. 如果输入晚于最后一次检查到达，UI 在本轮结束后自动启动同一会话的下一轮，不丢弃这部分消息。

补充不会中断正在进行的模型请求，也不会取消它已经产生的工具调用。当前请求及其工具批次结束后，模型才能看到补充。消息被 Agent 接收后会显示在对话记录并写入会话；仍在队列内的消息仅保存在内存中。

运行中 `/clear` 会被拒绝，不会作为补充发给模型。审批期间输入框暂时被审批面板替代，处理审批后可继续输入。当前没有“仅停止本轮、保留应用继续提问”的独立 UI 操作，退出操作会关闭应用。

## 会话与恢复

会话数据库默认位于 `~/.praxis/state.sqlite3`，按工作目录组织。一个会话包含多次任务运行、按顺序保存的用户/模型消息、工具结果和上下文操作。

```powershell
praxis --sessions                 # 列出当前工作目录最近 20 个会话，无需模型凭据
praxis --continue                 # 也可写 praxis -c
praxis --continue "接着检查工具调用"
praxis --resume a3f21c8d          # 完整 ID 或可唯一匹配的前缀
```

`--continue`、`--resume`、`--sessions` 互斥。可用 CLI 参数以 `praxis --help` 为准；没有 `--plain` 模式。

恢复时加载模型历史和已保存的上下文操作，模型会记得之前的内容。终端执行页面仅显示本次启动后的事件，不重放旧页面。若上次进程在任务中途退出，恢复会将该任务标记为失败，并为缺少结果的工具调用补充中断说明；不会自动重跑工具。

子 Agent 的运行时、Monty 变量和 MCP 已激活工具集合不随数据库恢复。`/clear` 会重置子 Agent、主 Agent 的 Monty 会话和对话上下文；MCP 连接继续由当前进程持有。

## 一次任务的执行流程

```text
终端输入
  → 空闲：后台线程调用 Agent.run(task)
  → 忙碌：写入 user_input 队列

Agent 主循环
  → 接收协作消息和用户补充
  → 构建上下文视图与工具定义，必要时压缩历史
  → 发起流式模型请求，保存模型输出
  → 有工具调用：校验、审批、执行、按调用顺序保存结果，再循环
  → 有文本且无工具调用：检查子 Agent 和待处理消息
  → 满足收尾条件：完成本轮，UI 继续接收下一次输入
```

`Agent.run()` 用锁防止同一个 Agent 同时执行两轮。Textual UI 线程负责输入、显示和审批；主 Agent 的后台线程负责模型循环和历史写入。工具执行器只把允许并行的处理函数交给线程池，结果仍按原调用顺序加入历史，保持工具调用与结果的配对。

每次请求还会附带当前计划、文件工具记录的已修改文件、协作状态和当前 Agent 可用的 Monty 会话名。这些运行状态用于当前请求，不是对历史消息的替换。

## 内置工具

| 类别 | 工具 | 用途 |
| --- | --- | --- |
| 文件读取与搜索 | `read_file`、`list_files`、`find_files`、`search_text` | 读 UTF-8 文本、列目录、按 glob 找文件、按字面文本搜索 |
| 文件修改 | `edit_file`、`write_file` | 精确替换，或创建/完整覆写文件 |
| 命令 | `run_command` | 执行项目代码、测试或系统命令 |
| 计划 | `update_plan` | 更新当前任务计划 |
| 上下文 | `delete`、`compress`、`search_history`、`read_history` | 管理模型视图并检索完整历史 |
| 技能 | `load_skill` | 按名称分页加载技能正文 |
| 程序化编排 | `run_code` | 在 Monty Python 子集中组合允许的工具调用 |

### 文件与命令的实际行为

- 读工具支持工作目录内相对路径，也支持外部文件/目录的绝对路径。写工具要求路径位于所分配的工作目录内。
- `read_file` 从 `start_line=1`、`start_offset=0` 开始；省略结束位置表示读到文件末尾，不再默认只读 200 行。行号从 1 开始，字符偏移从 0 开始。仅设置 `end_line` 时包含该整行；同时设置 `end_offset` 时结束位置不包含在结果内。
- 直接工具读取受字符预算限制，返回 `text`、`truncated`、`next_position` 和说明；按 `next_position` 继续，保持原来的结束位置。Monty 程序化读取返回完整的指定范围。
- `list_files` 列出一层内容；`find_files` 与 `search_text` 包含隐藏和 gitignore 忽略的文件，不自动排除目录，也没有结果条数上限。用 `relative_path`、`file_pattern` 和 `ignore_patterns` 控制范围。
- `search_text` 是区分大小写的单行字面搜索，返回文件、行号和字符偏移，不返回源码正文；再用 `read_file` 读取。
- `edit_file` 的每个旧文本必须唯一匹配，多个替换按顺序应用，全部成功才写回文件。
- `run_command` 接收 argv 数组，不隐式解析管道或 shell 语法。需要这些语法时显式调用 PowerShell、bash 等。`workdir` 接受已存在的绝对目录或相对于 Agent 工作目录的目录，可以位于工作区外；省略时使用 Agent 工作目录。默认超时 60 秒，允许 1–120 秒。
- 命令以前台方式运行，stdin 不交互，输出可以实时显示。结束或取消时清理其后代进程，不适合作为后台常驻服务启动器。

读文件、列目录、搜索和历史读取等标记为 `parallel_safe` 的连续调用可并行。默认每组最多 4 个处理函数，`PRAXIS_MAX_PARALLEL_TOOLS=1` 可关闭并行。其他调用是顺序屏障，不会跨过它们重排。

## 多 Agent

默认创建主 Agent 时启用协作。主 Agent 有 `spawn_agent`、`send_messages`、`wait_agents`、`close_agent`；子 Agent 通过 `send_message` 向主 Agent 发消息，默认没有继续创建子 Agent 的工具。

`spawn_agent` 接收任务、简短任务描述和已存在的 `working_directory`。目录可为相对路径或绝对路径，必须事先准备；Praxis 不自动创建 worktree，也不复制、同步或合并文件。主 Agent 需要明确交付位置和整合方式。

子 Agent 有自己的历史、计划、上下文和 Monty 会话，使用相同模型配置，具备文件和命令工具，并沿用审批设置。MCP 连接由进程共享，但各 Agent 独立激活工具。

- `send_messages` 批量投递，每条单独返回结果；空闲的子 Agent 会重新开始工作，运行中的子 Agent 在循环边界接收消息。
- `wait_agents` 等待新消息、所选运行结束或超时，单次最多 60 秒；取出的消息只消费一次。运行结束不等于工作已经验收完成。
- 主 Agent 收尾前会检查消息与子 Agent 状态；仍有 queued、running 或 closing 的子 Agent 时必须继续协调或等待。
- `close_agent` 请求取消并释放该子 Agent 的上下文；忙碌时可能先返回 closing，目录和文件不会删除。

`PRAXIS_AGENT_MAX_WORKERS` 控制同时执行的子 Agent 数，默认 8。线程池满时任务排队。终端可切换各 Agent 页面查看进度，输入仍发给主 Agent。

## MCP 外部工具

配置文件为 `~/.praxis/mcp.json`。没有配置文件时可以只使用内置工具。支持本地 stdio 和远程 Streamable HTTP，以下是需要替换路径/地址的示例：

```json
{
  "servers": {
    "local-example": {
      "command": "python",
      "args": ["D:/tools/example_mcp_server.py"],
      "cwd": "D:/tools",
      "env_vars": ["EXAMPLE_API_KEY"],
      "startup_timeout_sec": 30,
      "tool_timeout_sec": 60
    },
    "remote-example": {
      "url": "https://example.com/mcp",
      "bearer_token_env_var": "EXAMPLE_MCP_TOKEN",
      "enabled": false
    }
  },
  "search": {"limit": 5}
}
```

顶层也兼容 `mcp_servers`。stdio 可配置 `env` 和要转发的 `env_vars`，未指定 `cwd` 时使用配置文件所在目录。HTTP 可配置 `http_headers`、`env_http_headers` 或 `bearer_token_env_var`。默认启动超时 30 秒、工具超时 60 秒，可设为大于 0 且不超过 3600 秒。

工具按需发现和激活：

1. 启动时连接服务并建立本地工具索引，模型先看到服务目录。
2. `search_tools` 按关键词/BM25 或名称检索，`list_tools` 按服务分页浏览；两者都不自动激活。
3. `activate_tools` 加载完整工具定义；不传 `arguments` 表示仅激活，传入参数（包括 `{}`）则同时执行。
4. 激活后可以直接调用对应工具，也可通过 `run_code` 调用；`deactivate_tools` 从当前 Agent 的工具集合移除它。

服务支持 `enabled_tools`、`disabled_tools` 以及 `tools.<原始工具名>.enabled` 过滤。MCP 工具默认要求审批；`tools.<原始工具名>.approval_mode="approve"` 表示该工具自动允许。单个服务连接失败会记录不可用状态，其他服务仍可使用；本地配置错误可能阻止启动。

## 技能（Skills）

启动时扫描以下目录，后者覆盖前者的同名技能：

1. 包内 `src/praxis_agent/builtin_skills/`，当前包含 `monty`。
2. `~/.agents/skills/`。
3. 当前项目的 `.agents/skills/`。

每个技能放在 `<技能名>/SKILL.md`，以 YAML frontmatter 开始，至少包含与目录同名的 `name`（不超过 64 字符）和 `description`（不超过 1024 字符）。无效条目跳过并记录警告。

启动只加载名称和描述。模型使用 `load_skill(name, offset=0, version="")` 读取正文；若未读完，继续使用返回的 `next_offset` 和 `version`。文件中途变化需要从头加载。引用的资源按返回的 `resource_directory` 定位，不会自动全部加载。子 Agent 使用主会话项目发现的技能目录。

## Monty 程序化工具调用

`run_code(session_id, code)` 提供私有、具名的 Python 子集会话，用于循环、过滤、汇总和组合工具调用。代码通过 `call_tool(name, arguments)` 使用文件工具、历史读取和已激活的 MCP 工具；不能调用命令、管理工具或嵌套 `run_code`。

它不执行宿主项目模块，也不是完整 Python 环境。运行项目导入、第三方库或测试时使用 `run_command`。内置 `monty` 技能说明支持的语法和恢复方式。

成功执行会保存变量/函数，下次使用同名会话继续；失败保留上次成功状态，但已经完成的文件或外部工具操作不会回滚。会话状态仅在当前进程内保留，关闭 Agent、`/clear` 或重启后不恢复。

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `PRAXIS_CODE_COMPUTE_SECONDS` | 5 | 每次解释器推进的计算秒数，不包含宿主工具等待 |
| `PRAXIS_CODE_MEMORY_BYTES` | 536870912 | 内存预算，512 MiB |
| `PRAXIS_CODE_MAX_PROCESSES` | 8 | 共享 Monty 进程池上限 |
| `PRAXIS_CODE_QUEUE_SECONDS` | 30 | 等待可用执行槽的秒数 |
| `PRAXIS_CODE_MAX_SUSPENSIONS` | 5000 | 单次执行允许的挂起次数 |

## 上下文与历史管理

Praxis 区分三层数据：完整会话历史、持久化的上下文操作，以及本次发送给模型的临时视图。隐藏或压缩消息不会删除 SQLite 中的原始记录。

普通消息和工具结果带有 `[message_id=mN]` 标记，ID 来自历史位置。`delete(message_ids)` 将选中内容替换为占位消息；`compress(message_ids, summary=...)` 直接使用给定摘要，或用 `instructions=...` 请求独立摘要模型生成，二者必须且只能选一个。

上下文操作保存 `operation_id` 和新的替代消息 ID，在下一次请求生效。只能选择当前视图中可处理的消息，同一轮已改动的目标不能再次修改。工具结果被隐藏时，对应调用也从视图移除，避免协议配对断裂；原始调用和结果仍保留。系统/developer、协议专用和多模态内容不能直接作为压缩目标。

检索完整历史：

- `search_history(query, max_results)`：不区分大小写的字面匹配，默认 10 条、最多 20 条；返回消息 ID、片段和字符偏移，查询最多 200 字符。
- `read_history(message_id, start_offset=0, end_offset=null)`：读取半开字符区间，省略终点表示到正文末尾。返回 `text`、`truncated`、`next_offset` 和说明，按返回游标继续并保持原结束位置。偏移按 Python 字符计算，不是字节或 token。

检索先查替代消息，再查原始历史，各组从新到旧。直接历史读取按配置的字符预算分页，程序化调用可以读取完整指定范围。检索结果作为新的工具结果进入对话。

### 容量与自动压缩

每次请求会估算指令、消息和工具定义的总输入量。首次使用启发式估算，后续结合模型返回的实际 input tokens 校准；上下文变更后重新估算。容量优先使用显式配置，其次使用内置模型目录，未识别模型回退到 128,000 token 窗口。

工具结果在模型视图中按字符预算预览，完整工具输出保留在历史中。超出有效输入预算时，自动压缩较早的历史、保留能容纳的近期交互，工具调用与结果一起处理。摘要放不进一次请求时，按时间顺序分块更新同一份摘要；成功后保存一个上下文操作，恢复会话时重放。

当前没有独立摘要目标长度或摘要输出上限配置，也没有给摘要调用传入专属 `max_output_tokens`。超长输入不会自动转成附件；固定内容、近期消息或压缩后的结果仍无法容纳时会报错。当前处理本地估算溢出，尚无针对服务端上下文溢出的专门重试流程。

| 环境变量 | 默认行为 |
| --- | --- |
| `ADA_CONTEXT_WINDOW_TOKENS` | 覆盖模型窗口大小 |
| `ADA_MAX_INPUT_TOKENS` | 覆盖模型最大输入量 |
| `ADA_MAX_OUTPUT_TOKENS` | 覆盖容量元数据中的最大输出量，用于限制输出预留 |
| `ADA_CONTEXT_SAFETY_RATIO` | 0.95 |
| `ADA_DESIRED_OUTPUT_TOKENS` | 按窗口分档预留输出空间，不直接设置 API 生成上限 |
| `ADA_MAX_TOOL_OUTPUT_CHARS` | 单条工具结果的模型预览预算 |
| `ADA_MAX_FILE_READ_CHARS` | 直接文件读取预算 |
| `ADA_MAX_HISTORY_READ_CHARS` | 直接历史读取预算 |
| `PRAXIS_MAX_PARALLEL_TOOLS` | 4 |
| `PRAXIS_AGENT_MAX_WORKERS` | 8 |

分档默认值由 `context/budgets.py` 定义，k 按十进制计数；显式环境变量优先：

| 窗口上界 | 输出预留 token | 工具/文件字符 | 历史字符 |
| --- | ---: | ---: | ---: |
| ≤32k | 4,000 | 16,000 | 8,000 |
| ≤64k | 8,000 | 24,000 | 12,000 |
| ≤128k | 16,000 | 32,000 | 16,000 |
| ≤256k | 24,000 | 48,000 | 24,000 |
| ≤512k | 32,000 | 64,000 | 32,000 |
| >512k | 32,000 | 96,000 | 48,000 |

旧文档里的 `ADA_SUMMARY_TARGET_TOKENS`、`ADA_SUMMARY_MAX_OUTPUT_TOKENS`、`ADA_MAX_FILE_READ_LINES` 已不被当前配置读取，不能继续用它们调节行为。

## 可选 Langfuse 接入

没有配置时正常使用本地功能。配置两个项目 key 后启用：

```powershell
$env:LANGFUSE_PUBLIC_KEY = "your-project-public-key"
$env:LANGFUSE_SECRET_KEY = "your-project-secret-key"
# 可选；以下是默认值
$env:LANGFUSE_BASE_URL = "https://langfuse.flight.ctripcorp.com"
$env:LANGFUSE_TRACING_ENVIRONMENT = "development"
```

每个 CLI 进程关联一个 Langfuse session，每次任务产生一个 trace，包含模型、工具、子 Agent 和上下文压缩事件。关闭时刷新待发送数据；Windows 使用系统证书存储。

## 开发与验证

```powershell
python -m pytest
# 仅验证 steer、主循环、协作收尾与 CLI
python -m pytest tests/test_user_input.py tests/test_agent.py tests/test_multi_agent_finalization.py tests/test_cli.py -q
```

`tests/test_user_input.py` 使用模型替身和 Textual 测试界面，验证 FIFO、运行中补充、工具结果配对、持久化恢复，以及最终检查之后到达的消息。它不调用真实模型或外部 MCP 服务。

## 代码结构

```text
src/praxis_agent/
  cli.py                CLI 参数、会话选择、UI 和 Agent 组装
  agent.py              Agent 生命周期、模型循环、用户输入队列消费
  model.py              流式 Responses API 客户端
  config.py             模型和 Langfuse 配置
  tool_execution.py     工具校验、审批、并行执行与事件
  events.py             UI / 观测消费者共享的事件定义
  runtime.py            操作系统、工作目录和 shell 探测
  prompts.py            模型指令
  terminal/             Textual UI、各 Agent 视图和输出渲染
  tools/                文件、命令、进程管理与 Monty
  multi_agent/          子 Agent 调度、消息、状态与协作工具
  planning/             任务计划、运行状态和模型状态快照
  context/              历史投影、容量、压缩和历史检索
  sessions/             SQLite 持久化与中断恢复
  mcp.py                MCP 连接、工具注册与按需激活
  tool_search.py        MCP 工具索引和搜索
  skills.py             技能发现和分页加载
  builtin_skills/       随包提供的技能
  observability.py      Langfuse 事件消费
  deepseek.py            DeepSeek 响应兼容处理
tests/                  自动化测试
pyproject.toml          依赖、打包和 praxis 命令入口
```
