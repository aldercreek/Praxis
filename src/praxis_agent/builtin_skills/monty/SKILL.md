---
name: monty
description: Praxis run_code 的官方 Monty 1.2.0 使用指南。在编排多步文件/MCP/历史工具、解析 CSV/JSON/TOML、复用会话、处理执行失败或不确定 Python 兼容性时加载；包含实际工具返回格式、资源限制和恢复示例。
---

# 在 Praxis 中使用 Monty

适用版本：PyPI 官方 `pydantic-monty==1.2.0` 及配套 client/runtime。以下说明针对 Praxis 的接入方式，不依赖本地 Monty 补丁。官方支持某个 API，不代表 Praxis 已经为它开放宿主资源。

## 先选择执行方式

| 任务 | 使用方式 |
| --- | --- |
| 一次简单读取、搜索或修改 | 直接调用对应工具 |
| 根据上一步结果调用下一工具；循环处理多份数据；只把汇总交回模型 | `run_code` + `call_tool` |
| 解析已经取得的 JSON、CSV、TOML 文本 | `run_code` 中的内置模块 |
| 运行项目代码、项目测试、第三方包、系统命令或完整 Python 脚本 | 单独调用 `run_command`，不经过 Monty 桥接 |

如果当前角色没有 `run_command`，说明缺少什么能力；在 Monty 中重写一份算法不能证明实际项目代码已运行或测试通过。

## 执行过程与输入输出

把有意义的 `session_id` 和 Python 源码 `code` 交给 `run_code`。同一个 Agent 内，同名会话继续已有计算；新名字从空状态开始。会话名不是文件路径。

Praxis 的执行过程：恢复上次成功的快照 → 类型检查本次代码 → 在 Monty worker 中计算 → 遇到 `call_tool` 时由 Praxis 执行工具并把结果交回 → 整段成功后保存新快照。

`call_tool` 是 Praxis 提供的函数，不需要导入。文件、网络服务等宿主操作由工具完成，Monty 主要负责循环、判断、解析和汇总。当前桥接是同步的；`asyncio.gather` 不会让同步 `call_tool` 并发。同一 Agent 的 `run_code` 调用也会串行执行，即使使用不同会话名。

`run_code` 的响应是 JSON 对象，重点查看：

| 字段 | 含义 |
| --- | --- |
| `result` | 最后一条表达式的结果，通常用小对象或少量样本 |
| `stdout` | 本次所有 `print()` 文本 |
| `calls` | 已返回的工具调用记录，包括工具名、成功与否及失败信息；不含完整业务返回值 |
| `session_id` / `session_created` | 会话名，以及本次是否从空状态开始 |
| `state_saved` | 是否成功保存了本次计算状态 |
| `error` / `error_kind` / `error_type` / `phase` | 失败时的诊断、类别、异常类型和阶段 |
| `effects` | 提醒已发生的外部操作不会回滚 |

只输出完成任务需要的汇总和样本，不要 `print` 整份原始数据。大输出仍会走工具预览/历史流程，不会自动变成文件。需要长期保留的交付物用文件工具写出。

## 正确使用工具桥接

调用形式为 `call_tool(name, arguments)`，`arguments` 是符合当前工具 schema 的字典。参数名、分页字段和坐标规则以工具定义为准。

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
| 当前 MCP 桥接 | MCP 返回包的 JSON 字符串 | 先 `json.loads` 得到返回包，再检查 `structuredContent` 或 `content` |

MCP 的 `content` 可以有多个块。`type == "text"` 的 `text` 可能是普通文本，只有确认其内容是 JSON 时才再解码；不要假设第一块一定是 JSON。不要把图片、链接和正文混为一谈。不熟悉的工具先取小样本确认结构，后续复用该样本和解析逻辑。

程序化 `read_file` 和 `read_history` 没有模型预览的字符截断，但只返回请求的范围。完整范围可能只是文件的一部分。远程分页、服务限制仍然有效；不要根据一次调用成功就认定数据完整。JSONL 应逐个非空行解码，不是对整份内容调用一次 `json.loads`。

### 示例：读取 CSV 和 TOML，再保存汇总

假设工作区有 `input.csv`，列为 `category,count,note`，其中 `count` 是整数字符串；`settings.toml` 包含 `[report]` 下的整数 `min_count`。以下代码作为第一次 `run_code` 的 `code`，会话名使用 `csv-report`：

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

确认结果后，同一 Agent 再用 `session_id="csv-report"` 执行以下代码即可写出已保存的 `report`。根据任务授权选择输出路径，已有文件应先检查是否适合完整覆写：

```python
import json

saved = call_tool("write_file", {
    "relative_path": "out/report.json",
    "content": json.dumps(report, ensure_ascii=False, indent=2),
})
saved
```

这里不必重读 CSV，也不必把完整 `report` 发回模型再传入。局部修改现有文件用 `edit_file`，新建或完整重写用 `write_file`。

## 会话保存、内存与失败恢复

每个 Agent 拥有独立的会话。变量、函数和导入的模块会在成功调用后保存，可以在同一进程的空闲/继续执行间复用；其他 Agent 不能直接访问这些变量。关闭 Agent、`/clear` 或应用重启会丢失快照，恢复对话历史不会恢复 Monty 状态。

检查变量是否存在可用 `'report' in locals()`。`locals()` 返回快照字典，修改它不会给真实作用域赋值。不要用 `globals()`、`dir()` 或 `vars()` 探索环境。

每次成功调用保存整个保留状态。分多次执行不意味着内存自动下降：保留所有原文、拆分后的行、解析对象和汇总，会同时占用空间。完成核验后可重新绑定不再需要的变量，如 `source = {}`、`rows = []`、`raw_text = ""`；注意原对象是否还有其他引用。`clear()` 会原地清空容器，影响所有别名；`pop()` 可移除具体元素。官方版本不支持 `del`。

**快照保存与外部操作不是同一个事务。** 例如上次保存 `count = 1`，本次先写入文件，再改成 `count = 2`，随后抛出异常：下次恢复仍是 `count = 1`，但文件已经写入。先检查 `calls` 和实际产物，再决定补做哪一步；调用未返回时，`calls` 也不能证明远端没有完成操作。

如果循环需要保留部分成功结果，可把预期的单项失败捕获后记录到失败清单，再正常返回已完成数和未完成项；正常返回才会保存这些变量。响应中明确标记部分完成，不能把遗漏项当作全部成功。遇到内存/计算时间耗尽，不尝试在脚本内部捕获并继续运行；让 Praxis 结束本次 checkout，下次从最后成功的快照恢复。

### 有限重试：只重试可安全重复的读取

工具失败会抛 `RuntimeError`，内容包含工具名和失败说明。参数错误、权限问题、业务拒绝不能靠原样重试解决。超时的写操作可能已在远端完成，先核查状态或使用该服务明确提供的幂等机制。

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
            return json.loads(raw)
```

解析失败不会触发重试。调用者应按已确认的返回包结构提取内容。若在同一批次累计结果，要在外层捕获最终的读取失败、记录失败项并正常返回，才能保存之前累计的结果；未捕获异常会让本次全部内存改动丢失。

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

### 从旧本地补丁迁移时尤其注意

- 不支持 `json.dumps(default=...)`、`cls` 或 `check_circular` 参数。先将 set、日期等转换成可序列化的列表/字符串/基础字典，再调用 `dumps`；也没有 `JSONEncoder` / `JSONDecoder` 可供自定义。
- `json.loads` 不接受 `object_hook`、`parse_float` 等解码回调参数。若需要 Decimal、特殊精度或第三方解析器，使用完整 Python 环境。
- 内置方法只能直接调用，不能写 `append = items.append`、`sorted(keys, key=mapping.get)` 或 `map(str.lower, values)`。改为 `items.append(value)`、`sorted(keys, key=lambda key: mapping[key])` 或纯 lambda；缺失键的行为应按数据要求选择。
- `del name` 和 `del data[key]` 都不支持。使用上面的重新绑定或容器操作。
- 无 `yield` / `yield from` 生成器。生成器表达式会物化成列表，一些 iterator 风格内置函数也会立即计算；不要把它们当作流式内存优化。
- 无 `io`、`decimal`、`zlib`、`subprocess`、`socket` 或第三方包，也没有可在其中安装依赖的 pip 环境。

## 类型检查和故障定位

Praxis 在执行前启用官方类型检查，并为 `call_tool(name: str, arguments: dict[str, Any]) -> Any` 提供声明。检查器会带上已成功保存的会话声明，但工具返回值是动态值，检查通过不能保证字段存在、数据正确或所有 API 都能运行。不要用 `eval` / `exec` 绕过检查。

| 诊断 | 处理方式 |
| --- | --- |
| `phase=preflight`，`error_kind=type_check` / `syntax` | 本次脚本未执行；修正报告的类型或语法，不需要为这次被拒绝的脚本补偿工具操作 |
| `error_kind=tool` | 查看失败工具和 `calls`，修正参数/权限/服务问题；只按工具语义决定是否重试 |
| `error_kind=runtime` | 查看异常及源位置，区分缺字段、类型不匹配和不支持的 API；检查之前是否已有工具成功 |
| `error_kind=resource` | 缩小批次、减少保留数据或改用合适工具；不要直接重复同样的大计算 |
| `queue`、`timeout`、`worker_crash`，或 `phase=restore/save` | 查看阶段和 `state_saved`，核查外部操作；必要时从持久化输入重建新会话 |

有些不支持的语法（例如 `del`）会由官方编译器抛 `NotImplementedError`，Praxis 当前将其标为 `runtime`；不能仅凭类别推断脚本已经执行，结合诊断和调用记录判断。

修改辅助函数时尽量保持签名稳定。如果跨次类型诊断仍引用旧签名，用新的 helper 名称并更新调用者，避免为修复一个函数而丢弃整个已有数据会话。

## 默认预算与工作拆分

| Praxis 环境变量 | 默认值 | 实际含义 |
| --- | --- | --- |
| `PRAXIS_CODE_COMPUTE_SECONDS` | 5 秒 | 一次 feed 的解释器计算预算，暂停等待工具和 sleep 的时间不计入 |
| `PRAXIS_CODE_MEMORY_BYTES` | 536870912（512 MiB） | Monty worker 分配内存预算；不是宿主总内存或进程 RSS 上限 |
| `PRAXIS_CODE_MAX_SUSPENSIONS` | 5000 | 每次 checkout 的宿主交互预算，工具调用、名称解析等都可能计数 |
| `PRAXIS_CODE_QUEUE_SECONDS` | 30 秒 | 等待同一 Agent 执行锁和可用执行槽的总预算 |
| `PRAXIS_CODE_MAX_PROCESSES` | 8 | 共享 Monty worker 进程池上限，不表示一段脚本自动并发 |

保存/恢复快照、原文和解析结果同时存在都需要内存余量。Praxis 自己收集的 `stdout` 和宿主工具数据也占内存，不能用 512 MiB 预算推断它们受同一限制。

大任务用有意义的批次：成功保存一批 → 只返回进度/必要样本 → 同会话继续。保留已完成条目的 ID 和必要结果，避免恢复时重复读取或写入；完成后释放无用大对象。1.2.0 恢复快照时会保留快照与目标 checkout 中较严格的限制，不能假定改环境变量就能放宽已有快照。

## 依据与核查入口

官方资料固定到 v1.2.0，避免把尚未发布的 main 分支能力写成已支持：

- [v1.2.0 发布说明](https://github.com/pydantic/monty/releases/tag/v1.2.0)
- [标准库清单及各模块限制](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/modules.md)、[语法限制](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/language.md)、[内置函数差异](https://github.com/pydantic/monty/blob/v1.2.0/docs/limitations/builtins.md)
- [宿主函数](https://github.com/pydantic/monty/blob/v1.2.0/docs/host-functions.md)、[类型检查](https://github.com/pydantic/monty/blob/v1.2.0/docs/type-checking.md)
- [快照](https://github.com/pydantic/monty/blob/v1.2.0/docs/snapshots.md)、[资源限制](https://github.com/pydantic/monty/blob/v1.2.0/docs/resource-limits.md)

Praxis 接入事实来自 `src/praxis_agent/tools/code.py`、`tool_execution.py`、`tools/registry.py`、`context/tools.py` 和 `mcp.py`。升级版本或改变桥接时，同时核对依赖、这些接入点与本技能；不要只更改版本号或照搬官方所有可选能力。
