---
name: monty
description: Praxis run_code 的官方 Monty 1.2.0 使用指南。在编排多步文件/MCP/历史工具、解析 CSV/JSON/TOML、复用会话或处理执行失败时加载；包含实际工具返回格式、资源限制和恢复示例。
---

# 在 Praxis 中使用 Monty

适用版本：PyPI 官方 `pydantic-monty==1.2.0` 及配套 client/runtime。以下说明针对 Praxis 的接入方式。官方支持某个 API，不代表 Praxis 已经为它开放宿主资源。

## 先选择执行方式

| 任务 | 使用方式 |
| --- | --- |
| 一次简单读取、搜索或修改 | 直接调用对应工具 |
| 根据上一步结果调用下一工具；循环处理多份数据；只把汇总交回模型 | `run_code` + `call_tool` |
| 解析已经取得的 JSON、CSV、TOML 文本 | `run_code` 中的内置模块 |
| 运行项目代码、项目测试、第三方包、系统命令或完整 Python 脚本 | 单独调用 `run_command`，不经过 Monty 桥接 |

如果当前角色没有 `run_command`，说明缺少什么能力；在 Monty 中重写一份算法不能证明实际项目代码已运行或测试通过。

## 执行过程与输入输出

把 Python 源码 `code` 交给 `run_code`，没有 `session_id` 参数。每个 Agent 首次使用时创建自己的 Monty worker，后续调用直接使用同一份内存中的变量和函数；不同 Agent 相互隔离。

Praxis 的执行过程：找到当前 Agent 的 worker（首次使用才创建）→ 类型检查本次代码 → 在已有环境中计算 → 遇到 `call_tool` 时由 Praxis 执行工具并把结果交回 → 返回结果，保留 worker。没有快照加载、保存或回滚。

`call_tool` 和 `call_parallel_tools` 是 Praxis 提供的函数，不需要导入。文件、网络服务等宿主操作由工具完成，Monty 主要负责循环、判断、解析和汇总。单个 `call_tool` 同步等待；独立调用用 `call_parallel_tools` 交给宿主并发执行，代码等待整批结果，不需要管理事件循环。同一 Agent 的 `run_code` 调用仍串行执行，不同 Agent 可以并发。

`run_code` 的响应是 JSON 对象，重点查看：

| 字段 | 含义 |
| --- | --- |
| `result` | 最后一条表达式的结果，通常用小对象或少量样本 |
| `stdout` / `stderr` | 分别返回输出预览；每条流最多保留开头 6,000 和结尾 24,000 个 Python 字符，超限时附省略提示 |
| `output_files` | 仅超限流具有条目，包含完整日志的 `path`、`total_chars`、`omitted_chars` 和 `complete`；用 `read_file` 读取原文 |
| `output_errors` | 落盘失败时的诊断；`complete=false` 的日志可能不完整 |
| `calls` | 已返回的工具调用记录，包括工具名、成功与否及失败信息；不含完整业务返回值 |
| `worker_created` | 本次是否创建了新的空执行环境 |
| `state_lost` | 本次是否废弃了执行环境；为 true 时原变量已丢失，下次需要重新初始化 |
| `error` / `error_kind` / `error_type` / `phase` | 失败时的诊断、类别、异常类型和阶段 |
| `effects` | 提醒已发生的外部操作不会回滚 |

只输出完成任务需要的汇总和样本。超限打印会自动保存完整原文到 `~/.praxis/outputs/code/` 下的 UTF-8 文件，包括已经展示的开头和结尾；历史保留的是预览与文件引用，`read_history` 不会展开日志文件，需要用 `read_file` 读取 `output_files` 中的绝对路径。失败或取消时保留已收到的输出。`result` 不受打印预览额度影响，仍走通用工具预览/历史流程。需要长期保留的交付物用文件工具写出。

## 正确使用工具桥接

调用形式为 `call_tool(name, arguments)`，`arguments` 是符合当前工具 schema 的字典。参数名、分页字段和坐标规则以工具定义为准。

互不依赖的操作可用 `call_parallel_tools([{"name": "实际工具名", "arguments": {...}}, ...])`。列表可动态构造、混合不同工具；空列表返回 `[]`。放入列表即表示调用方明确要求并发，不检查 `parallel_safe`；仍受当前角色、程序化调用白名单、参数检查和审批约束。不要把依赖前项返回值的操作或会互相覆盖的文件修改放入同一批。

每批默认最多同时执行 16 项，宿主可用 `PRAXIS_CODE_MAX_PARALLEL_TOOLS` 调整；剩余项排队，不限制总项数。结果按输入顺序返回，每项为 `{"success": True, "output": 数据}` 或 `{"success": False, "error": 说明}`。成功项的 `output` 使用下表相同的结构化优先规则。单项失败不会抛出整个批次错误或自动重试；必须逐项检查 `success`。调用列表格式错误会在执行任何工具前报错。取消会停止派发、取消支持取消的运行中工具并等待清理；已完成的外部操作不会回滚。

- 可以调用当前 Agent 允许的文件工具、历史搜索/读取和已激活 MCP 工具。角色限制、参数检查、审批和事件记录仍由原工具执行器处理。
- MCP 工具必须先通过普通工具调用完成发现和激活；使用激活后注册的可调用名称，不猜测服务名与工具名的拼接方式。
- 不可调用 `run_command`、管理工具或嵌套 `run_code`。需要命令时结束本次计算，再单独调用命令工具。
- 用普通循环调用工具。不要在 `sorted(key=...)`、`map`、`filter` 等内置回调里调用工具；先取得数据，再用纯计算回调处理。

### 返回值不是统一的 JSON 字符串

桥接优先返回工具的结构化对象；工具没有结构化结果时才返回输出字符串。

| 工具 | 在 Monty 中取得的值 | 推荐处理 |
| --- | --- | --- |
| `read_file` | 字典，含 `text`、`truncated`、`next_position`、`note` | 直接取 `reply["text"]`，不要对整个字典 `json.loads` |
| 其他文件工具 | 结构化字典，具体字段见工具定义 | 例如写入返回 `path`，查找返回 `files` |
| `read_history` | 结构化字典 | 按工具 schema 读取正文/分页字段 |
| `search_history` | JSON 字符串 | `json.loads` 一次后处理 |
| MCP（有 `structuredContent`） | 服务端提供的结构化对象，空字典也有效 | 直接按该工具的结果结构读取，不要对整个对象 `json.loads` |
| MCP（没有 `structuredContent`） | MCP 返回包的 JSON 字符串 | 先 `json.loads` 得到返回包，再检查 `content` |

MCP 的 `content` 可以有多个块。`type == "text"` 的 `text` 可能是普通文本，只有确认其内容是 JSON 时才再解码；不要假设第一块一定是 JSON。不要把图片、链接和正文混为一谈。不熟悉的工具先取小样本确认结构，后续复用该样本和解析逻辑。

程序化 `read_file` 和 `read_history` 没有模型预览的字符截断，但只返回请求的范围。完整范围可能只是文件的一部分。远程分页、服务限制仍然有效；不要根据一次调用成功就认定数据完整。JSONL 应逐个非空行解码，不是对整份内容调用一次 `json.loads`。

### 示例：读取 CSV 和 TOML，再保存汇总

假设工作区有 `input.csv`，列为 `category,count,note`，其中 `count` 是整数字符串；`settings.toml` 包含 `[report]` 下的整数 `min_count`。以下代码作为第一次 `run_code` 的 `code`：

```python
import csv
import tomllib

source = call_tool("read_file", {"relative_path": "input.csv"})
settings = tomllib.loads(
    call_tool("read_file", {"relative_path": "settings.toml"})["text"]
)
totals = {}
for row in csv.DictReader(source["text"].splitlines(keepends=True)):
    category = row["category"]
    totals[category] = totals.get(category, 0) + int(row["count"])

report = [
    {"category": category, "count": count}
    for category, count in totals.items()
    if count >= settings["report"]["min_count"]
]
report = sorted(report, key=lambda item: item["count"], reverse=True)
{"groups": len(report), "preview": report[:10]}
```

`keepends=True` 保留 CSV 带引号字段内部的换行；不要用 `split(",")` 或不保留换行的拆分方式代替 CSV 解析。`DictReader` 的值通常是字符串，按字段含义转换，非法数据应报告或按明确规则处理。

确认结果后，同一 Agent 再调用 `run_code` 执行以下代码即可写出内存中的 `report`。根据任务授权选择输出路径，已有文件应先检查是否适合完整覆写：

```python
import json

saved = call_tool("write_file", {
    "relative_path": "out/report.json",
    "content": json.dumps(report, ensure_ascii=False, indent=2),
})
saved
```

这里不必重读 CSV，也不必把完整 `report` 发回模型再传入。局部修改现有文件用 `edit_file`，新建或完整重写用 `write_file`。

## 常驻环境、内存与失败处理

每个 Agent 拥有独立子进程，变量、函数和导入的模块直接保留在执行环境中。Agent 回复后进入空闲仍保留环境；关闭 Agent、`/clear` 或应用退出会释放子进程，恢复对话历史不会恢复变量。

检查变量是否存在可用 `'report' in locals()`。`locals()` 返回快照字典，修改它不会给真实作用域赋值。不要用 `globals()`、`dir()` 或 `vars()` 探索环境。

分多次执行不意味着内存自动下降：保留所有原文、拆分后的行、解析对象和汇总，会同时占用空间。完成核验后可重新绑定不再需要的变量，如 `source = {}`、`rows = []`、`raw_text = ""`；注意原对象是否还有其他引用。`clear()` 会原地清空容器，影响所有别名；`pop()` 可移除具体元素。

**执行不是事务。** 例如已有 `count = 1`，本次先写入文件，再改成 `count = 2`，随后抛出普通异常：下次仍是 `count = 2`，文件也已经写入。先检查变量、`calls` 和实际产物，再决定补做哪一步；调用未返回时，`calls` 也不能证明远端没有完成操作。

返回结果必须能编码成 JSON；例如最终字典含 tuple/bytes 键导致编码失败，会返回 `error_kind=serialize`，但此前执行的变量修改和工具操作已经生效。输出日志保存失败也不会回滚变量。`call_parallel_tools` 中某一项参数无法编码成 JSON 时，只返回该项失败，其他合法项继续执行；完整列表结构错误仍会在执行前拒绝整批。

可把预期的单项失败捕获后记录到失败清单，再返回已完成数和未完成项；普通异常之前的内存改动也会保留。响应中明确标记部分完成，不能把遗漏项当作全部成功。执行中取消、资源耗尽或 worker 故障会废弃整个环境，并返回 `state_lost=true`。下一次调用创建空环境；需要从文件或工具重新初始化数据，不能自动重放可能已有外部效果的代码。排队时取消不影响正在执行的调用；执行已完成后取消，不会撤销已完成的修改。

### 有限重试：只重试可安全重复的读取

单个 `call_tool` 失败会抛 `RuntimeError`，内容包含工具名和失败说明；`call_parallel_tools` 的单项失败通过结果中的 `success` / `error` 返回。参数错误、权限问题、业务拒绝不能靠原样重试解决。超时的写操作可能已在远端完成，先核查状态或使用该服务明确提供的幂等机制。

下面是可保存到会话中的辅助函数，适用于已经确认无副作用的 MCP 查询；调用时传入实际激活的工具名和其参数。它只对包含 `TimeoutError` 的失败做最多三次尝试，其他错误立即交回调用者：

```python
import json
import time

def read_mcp_with_retry(tool_name: str, arguments: dict):
    for attempt in range(3):
        try:
            raw = call_tool(tool_name, arguments)
        except RuntimeError as error:
            if "TimeoutError" not in str(error) or attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))
        else:
            return json.loads(raw) if isinstance(raw, str) else raw
```

解析失败不会触发重试。调用者应按已确认的返回包结构提取内容。在同一批次累计结果时，可捕获最终的读取失败并记录失败项，便于继续处理剩余项目。

## 官方 1.2.0 的可用能力与边界

优先使用普通容器、函数、循环、推导式、`try/except` 和小型纯计算 helper。以下是常用能力，不是完整 CPython 兼容性承诺。

| 能力 | 在 Praxis 中怎么用 / 有什么限制 |
| --- | --- |
| `json` | `loads` / `dumps` 处理工具返回的文本；`dumps` 支持 `indent`、`sort_keys`、`ensure_ascii`、`allow_nan`、`separators`、`skipkeys` |
| `csv` | `reader` / `DictReader` 解析文本行；也有 writer，但没有 `io.StringIO`。不支持 `Sniffer` 或注册 dialect，单字段上限固定为 131072 字符 |
| `tomllib` | `loads` 解析文本；没有 `parse_float`。日期时间会变成对应对象，写 JSON 前应显式转字符串 |
| `bisect` | 有序列表的二分查找/插入 |
| `match/case`、列表切片赋值 | 可用于数据分支和局部替换，无需为这些写法退回宿主 Python |
| 简单类、`dataclasses` | 支持 `field(default=...)`、`field(default_factory=...)` 和 `__post_init__`；factory/post-init 中不能调用外部工具。无类继承、`slots`、`asdict` 等完整 dataclass 能力 |
| `math`、`re`、`collections`、`datetime`、`itertools`、`hashlib` 等 | 内置的标准库子集；具体函数以对应官方限制文档为准，不等于已安装所有 Python 标准库 |
| `os.path` / `pathlib` 的纯路径运算 | 使用 POSIX 路径语义，不自动遵循 Windows 宿主盘符和反斜杠规则 |
| `open`、文件版 `json.load/dump`、`tomllib.load`、walk/glob | 官方有这些接口，但 Praxis 未配置宿主文件系统挂载；工作区文件仍走文件工具，再用文本版解析接口 |
| `external_modules` | 官方提供宿主模块接入能力，但 Praxis 没有注册；不能 `import tools`、导入项目包或随意导入宿主模块 |

## 类型检查和故障定位

Praxis 在执行前启用官方类型检查，并为 `call_tool(name: str, arguments: dict[str, Any]) -> Any` 提供声明。检查器会带上此前成功执行的声明，但工具返回值是动态值，检查通过不能保证字段存在、数据正确或所有 API 都能运行。不要用 `eval` / `exec` 绕过检查。

| 诊断 | 处理方式 |
| --- | --- |
| `phase=preflight`，`error_kind=type_check` / `syntax` | 本次脚本未执行；修正报告的类型或语法，不需要为这次被拒绝的脚本补偿工具操作 |
| `error_kind=tool` | 查看失败工具和 `calls`，修正参数/权限/服务问题；只按工具语义决定是否重试 |
| `error_kind=runtime` | 查看异常及源位置，区分缺字段、类型不匹配和不支持的 API；检查之前是否已有工具成功 |
| `error_kind=resource` | 缩小批次、减少保留数据或改用合适工具；不要直接重复同样的大计算 |
| `queue`、`timeout`、`worker_crash`，或 `phase=startup` | 查看阶段和 `state_lost`，核查外部操作；环境丢失时从持久化输入重新初始化 |

修改辅助函数时尽量保持签名稳定。如果跨次类型诊断仍引用旧签名，用新的 helper 名称并更新调用者，避免为修复一个函数而丢弃整个已有数据会话。

## 默认预算与工作拆分

| Praxis 环境变量 | 默认值 | 实际含义 |
| --- | --- | --- |
| `PRAXIS_CODE_COMPUTE_SECONDS` | 5 秒 | 一次 feed 的解释器计算预算，暂停等待工具和 sleep 的时间不计入 |
| `PRAXIS_CODE_QUEUE_SECONDS` | 30 秒 | 等待同一 Agent 前一次代码调用结束的预算 |

Praxis 不设置 Monty 内存预算或全局 worker 名额。每个使用过 `run_code` 且未关闭的 Agent 持有一个独立子进程。Monty 1.2.0 无法关闭挂起次数计数，因此使用其支持的最大值 `2^64 - 1`。原文、解析结果、`stdout` 和宿主工具数据仍会占用实际内存。

大任务用有意义的批次：处理一批 → 只返回进度/必要样本 → 同一 Agent 继续。保留已完成条目的 ID 和必要结果，避免重复读取或写入；需要跨进程保留的中间结果用文件工具写出，完成后释放无用大对象。

## 依据与核查入口

官方资料固定到 v1.2.0，避免把尚未发布的 main 分支能力写成已支持：

- [v1.2.0 发布说明](https://github.com/pydantic/monty/releases/tag/v1.2.0)
- [标准库清单及各模块限制](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/modules.md)、[语法限制](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/language.md)、[内置函数差异](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/builtins.md)
- [宿主函数](https://github.com/pydantic/monty/blob/v1.2.0/docs/host-functions.md)、[类型检查](https://github.com/pydantic/monty/blob/v1.2.0/docs/type-checking.md)
- [常驻执行环境](https://github.com/pydantic/monty/blob/v1.2.0/docs/quickstart/python.md#sessions-keep-state)、[资源限制](https://github.com/pydantic/monty/blob/v1.2.0/docs/resource-limits.md)

Praxis 接入事实来自 `src/praxis_agent/tools/code.py`、`tool_execution.py`、`tools/registry.py`、`context/tools.py` 和 `mcp.py`。升级版本或改变桥接时，同时核对依赖、这些接入点与本技能；不要只更改版本号或照搬官方所有可选能力。
