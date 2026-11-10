"""Deterministic complex task fixtures and independent acceptance checks.

Only invoked by run_complex_code_tasks.py; no model calls during pytest.
Expected answers stay outside the agent workspace.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import random
import sqlite3
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

CASES = ("reconcile", "repair_project", "sqlite_sync", "mcp_paging",
         "large_logs", "state_recovery", "cancel_resume", "offline_package")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def money(value):
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def csv_bytes(rows, encoding="utf-8-sig"):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode(encoding)


def prepare(case, root, destination):
    expected = {}
    if case == "reconcile":
        rates = {"CNY": "1", "USD": "7.1235", "EUR": "7.8612"}
        products = {f"P{i}": ["图书", "食品", "电子"][i % 3] for i in range(9)}
        rows = []
        for i in range(900):
            rows.append(dict(row_id=f"R{len(rows):04}", order_id=f"O{i:04}", product=f"P{i % 9}",
                             price=f"{(i * 37 % 12000) / 100:.2f}", qty=str(i % 7 + 1),
                             currency=list(rates)[i % 3], status=["paid", "paid", "refunded", "cancelled"][i % 4],
                             version="2", note='中文备注，含逗号,和\n换行' if i % 31 == 0 else "普通"))
        for i in range(150):
            row = dict(rows[i], row_id=f"R{len(rows):04}", version="3" if i < 90 else "1", price="19.995")
            rows.append(row)
        rejected = []
        for field, value, reason, count in [("price", "bad", "bad_price", 12), ("currency", "ZZZ", "bad_currency", 10),
                                            ("status", "pending", "bad_status", 8)]:
            for i in range(count):
                row = dict(rows[i], row_id=f"R{len(rows):04}", version="99")
                row[field] = value
                rows.append(row)
                rejected.append({"row_id": row["row_id"], "reason": reason})
        latest = {}
        invalid = {r["row_id"] for r in rejected}
        for row in rows:
            if row["row_id"] not in invalid and int(row["version"]) > int(latest.get(row["order_id"], {}).get("version", -1)):
                latest[row["order_id"]] = row
        totals = {category: Decimal(0) for category in products.values()}
        booked = 0
        for row in latest.values():
            if row["status"] == "cancelled":
                continue
            booked += 1
            amount = Decimal(money(Decimal(row["price"]) * int(row["qty"]) * Decimal(rates[row["currency"]])))
            totals[products[row["product"]]] += -amount if row["status"] == "refunded" else amount
        expected = {"summary": dict(input_rows=len(rows), rejected_rows=len(rejected), dedup_removed=150,
                                    latest_orders=900, booked_orders=booked, net_cny=money(sum(totals.values())),
                                    by_category={k: money(v) for k, v in totals.items()}), "rejected": rejected}
        random.Random(20261110).shuffle(rows)
        with zipfile.ZipFile(root / "财务输入.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for i, encoding in enumerate(["utf-8-sig", "gb18030", "utf-8-sig"]):
                archive.writestr(f"订单{i + 1}.csv", csv_bytes(rows[i::3], encoding))
            archive.writestr("config.json", json.dumps({"rates": rates, "products": products,
                "encodings": {"订单1.csv": "utf-8-sig", "订单2.csv": "gb18030", "订单3.csv": "utf-8-sig"}}, ensure_ascii=False))
        prompts = ["完成财务输入.zip 的跨文件对账。先按 config.json 指定编码读取全部 CSV（含引号内换行），"
                   "再校验：price 必须是有限非负 Decimal，否则 bad_price；currency 必须在 rates，否则 bad_currency；"
                   "status 必须是 paid/refunded/cancelled，否则 bad_status。无效行按此顺序判定，只输出一个原因。"
                   "先排除无效行，再按 order_id 保留整数 version 最大的一条。cancelled 不计金额；其他按 price*qty*汇率，"
                   "每订单 ROUND_HALF_UP 到两位小数，refunded 取负，然后分类汇总。输出 summary.json："
                   "input_rows,rejected_rows,dedup_removed（有效行数减去去重后订单数）,latest_orders,booked_orders（非cancelled订单数）,"
                   "net_cny（两位小数字符串）,by_category（分类到两位小数字符串）。另输出 rejected.csv，字段 row_id,reason，按 row_id 排序。"
                   "保留可重新执行的 reconcile.py，并实际运行它、检查结果。"]
    elif case == "repair_project":
        (root / "billing").mkdir()
        (root / "billing/__init__.py").write_text("", encoding="utf-8")
        (root / "billing/calc.py").write_text(
            "def invoice(lines, coupon='0', shipping='0', tax_rate='0'):\n"
            "    subtotal = sum(float(x['price']) * x['qty'] for x in lines)\n"
            "    net = subtotal - float(coupon) + float(shipping)\n"
            "    return {'subtotal': str(round(subtotal, 2)), 'tax': str(round(net * float(tax_rate), 2)), 'total': str(round(net * (1 + float(tax_rate)), 2))}\n", encoding="utf-8")
        (root / "test_visible.py").write_text(
            "from billing.calc import invoice\n"
            "def test_rounding():\n"
            "    assert invoice([{'price':'1.005','qty':1}]) == {'subtotal':'1.01','tax':'0.00','total':'1.01'}\n"
            "def test_coupon():\n"
            "    assert invoice([{'price':'10','qty':1}], coupon='20', shipping='3', tax_rate='0.1')['total'] == '3.30'\n", encoding="utf-8")
        vectors = [dict(lines=[dict(price=p, qty=q)], coupon=c, shipping=s, tax_rate=t)
                   for p, q, c, s, t in [("1.005", 1, "0", "0", "0"), ("19.995", 7, "2.50", "3.20", "0.13"),
                                          ("0", 2, "5", "4", "0.07"), ("100.005", 3, "999", "0.005", "1"),
                                          ("999999.99", 1000, "0.01", "0.01", "0.17")]]
        vectors.append(dict(lines=[dict(price="0.005", qty=1)] * 3, coupon="0", shipping="0", tax_rate="0"))
        answers = []
        for args in vectors:
            subtotal = sum(Decimal(money(Decimal(x["price"]) * x["qty"])) for x in args["lines"])
            base = max(Decimal(0), subtotal - Decimal(args["coupon"])) + Decimal(args["shipping"])
            tax = Decimal(money(base * Decimal(args["tax_rate"])))
            answers.append(dict(subtotal=money(subtotal), tax=money(tax), total=money(base + tax)))
        expected = {"vectors": vectors, "answers": answers}
        prompts = ["修复 billing/calc.py 的 invoice，并补充边界测试，实际运行 pytest，保存 junit.xml。"
                   "不能修改 test_visible.py 来绕过失败。契约：lines 是含 price 字符串和 qty 正整数的列表（bool不是整数）；"
                   "价格必须有限且非负；coupon、shipping 是有限非负金额字符串；tax_rate 是0到1的有限字符串；"
                   "不合格输入统一 ValueError。允许空lines。用 Decimal：每行 price*qty 单独 HALF_UP 两位再相加为 subtotal；"
                   "优惠券最多抵扣 subtotal，再加运费为应税基数；税额 HALF_UP 两位；总价为基数加税再 HALF_UP 两位。"
                   "返回且只返回 subtotal,tax,total 三个两位小数字符串。保存一份 verification.json，包含 pytest_exit_code 和你增加的测试用例数。"]
    elif case == "sqlite_sync":
        db = sqlite3.connect(root / "warehouse.sqlite")
        db.executescript("PRAGMA foreign_keys=ON; CREATE TABLE customers(id TEXT PRIMARY KEY);"
                         "CREATE TABLE orders(id TEXT PRIMARY KEY,customer_id TEXT REFERENCES customers(id),amount_cents INTEGER NOT NULL,version INTEGER NOT NULL);"
                         "CREATE TABLE applied(id TEXT,version INTEGER,PRIMARY KEY(id,version));")
        db.executemany("INSERT INTO customers VALUES (?)", [(f"C{i}",) for i in range(12)])
        base = {f"O{i:04}": [f"O{i:04}", f"C{i%12}", i * 100, 2] for i in range(200)}
        db.executemany("INSERT INTO orders VALUES (?,?,?,?)", base.values())
        db.commit()
        db.close()
        rows = []
        for i in range(350):
            rows.append(dict(id=f"O{i:04}", customer_id=f"C{i%12}", amount_cents=str(i * 137), version="3"))
            if i % 4 == 0:
                rows.append(dict(rows[-1], amount_cents=str(i * 151), version="4"))
            if i % 7 == 0:
                rows.append(dict(rows[-1], version="1", amount_cents="1"))
        for i in range(15):
            rows.append(dict(id=f"O{i:04}", customer_id="missing", amount_cents="5", version="99"))
        random.Random(24).shuffle(rows)
        (root / "batch.csv").write_bytes(csv_bytes(rows))
        latest = {}
        for row in rows:
            if row["customer_id"] == "missing":
                continue
            if int(row["version"]) > int(latest.get(row["id"], {}).get("version", -1)):
                latest[row["id"]] = row
        final, applied = dict(base), []
        for key, row in latest.items():
            if key not in final or int(row["version"]) > final[key][3]:
                final[key] = [key, row["customer_id"], int(row["amount_cents"]), int(row["version"])]
                applied.append([key, int(row["version"])])
        expected = {"orders": sorted(final.values()), "applied": sorted(applied)}
        prompts = ["给 warehouse.sqlite 实现 pipeline.py 的 sync(db_path,batch_path)。batch.csv 是 UTF-8 BOM，"
                   "先排除不存在的 customer_id 和不合法数值（amount_cents非负整数、version正整数），再按 id 选最大 version；"
                   "仅当 version 比数据库中大时更新，不存在则插入；相同或更小版本不变。每个真正发生变更的(id,version)写 applied 一次。"
                   "整个批次事务提交，开启外键；不要逐条中间版本都写applied。对同一文件执行两次，第二次不应有变化。"
                   "保存 sync-report.json 为 {first_changed:整数,second_changed:整数,invalid_rows:整数}。保留 sync 函数供后续调用。"]
    elif case == "mcp_paging":
        orders = [dict(id=f"O{i:03}", amount=f"{(i * 93 + 17) / 100:.2f}", currency=["CNY", "USD", "EUR"][i % 3]) for i in range(127)]
        rates = {"CNY": "1", "USD": "7.1235", "EUR": "7.8612"}
        expected = {"count": len(orders), "total_cny": money(sum(Decimal(money(Decimal(r["amount"]) * Decimal(rates[r["currency"]]))) for r in orders)),
                    "ids": [r["id"] for r in orders]}
        pages = {"": {"items": orders[:35], "next_cursor": "a_opaque"},
                 "a_opaque": {"items": [], "next_cursor": "b_opaque"},
                 "b_opaque": {"items": orders[30:70], "next_cursor": "c_opaque"},
                 "c_opaque": {"items": orders[65:105], "next_cursor": "d_opaque"},
                 "d_opaque": {"items": orders[100:], "next_cursor": None}}
        dump(destination / "server-data.json", {"pages": pages, "rates": rates})
        (destination / "mcp_server.py").write_text('''from mcp.server import MCPServer
from pathlib import Path
from typing import Any
import json, threading
root = Path(__file__).parent
data = json.loads((root / "server-data.json").read_text(encoding="utf-8"))
lock = threading.Lock()
attempts = {}
server = MCPServer("benchmark")
def record(name, args):
    with lock:
        with (root / "server-requests.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"name":name,"arguments":args})+"\\n")
@server.tool(structured_output=True)
def orders_page(cursor: str = "") -> dict[str, Any]:
    """Return one page. Follow next_cursor, including when items is empty. Overlapping IDs are possible."""
    record("orders_page", {"cursor":cursor})
    return data["pages"][cursor]
@server.tool(structured_output=True)
def rate(currency: str) -> dict[str, str]:
    """Get CNY exchange rate. Temporary failures may be retried at most twice."""
    record("rate", {"currency":currency})
    with lock:
        attempts[currency] = attempts.get(currency, 0) + 1
        fail = currency == "EUR" and attempts[currency] == 1
    if fail:
        raise RuntimeError("INJECTED_TRANSIENT: temporary EUR rate failure; retry")
    return {"currency": currency, "cny_rate": data["rates"][currency]}
server.run()
''', encoding="utf-8")
        prompts = ["使用 benchmark MCP 的 orders_page 获取所有订单（第一页cursor为空字符串），按 next_cursor 翻页直到null，"
                   "不能因空页提前结束；跨页按id去重。发现并激活工具后，在 run_code 里用 call_parallel_tools 并行查询各币种 rate。"
                   "汇率会有一次临时失败：只重试失败的币种，最多两次，保留成功结果。不要访问服务器内部文件。"
                   "每订单用 Decimal 转换CNY并HALF_UP两位再求和。输出 mcp-result.json，包含 count,total_cny（两位字符串）,ids（排序的唯一ID列表）。"]
    elif case == "large_logs":
        valid, malformed, anomalies = [], [], []
        with gzip.open(root / "服务日志.jsonl.gz", "wt", encoding="utf-8") as stream:
            for i in range(12000):
                if i % 211 == 0:
                    stream.write('{"broken":\n')
                    malformed.append(i + 1)
                    continue
                row = dict(id=i, hour=f"2026-11-09T{i%24:02}", endpoint=f"/api/{i%11}",
                           latency_ms=(i * 37) % 2300, status=500 if i % 17 == 0 else 200,
                           message="处理中文请求" + "x" * 180)
                valid.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row["status"] >= 500 or row["latency_ms"] >= 2000:
                    anomalies.append({k: row[k] for k in ("id", "endpoint", "status", "latency_ms")})
        by_hour = defaultdict(list)
        for row in valid:
            by_hour[row["hour"]].append(row)
        metrics = {}
        for hour, records in by_hour.items():
            latencies = sorted(r["latency_ms"] for r in records)
            metrics[hour] = dict(count=len(records), errors=sum(r["status"] >= 500 for r in records),
                                 p95_ms=latencies[math.ceil(len(records) * .95) - 1])
        top = sorted(Counter(r["endpoint"] for r in valid).items(), key=lambda x: (-x[1], x[0]))[:5]
        expected = {"summary": dict(valid_rows=len(valid), malformed_lines=malformed, anomaly_rows=len(anomalies),
                                     hourly=metrics, top_endpoints=[dict(endpoint=k, count=v) for k, v in top]), "anomalies": anomalies}
        prompts = ["流式分析 服务日志.jsonl.gz（UTF-8 JSONL），跳过无效JSON，记录从1开始的物理行号。"
                   "输出 logs-summary.json：valid_rows,malformed_lines（行号数组）,anomaly_rows,hourly,top_endpoints。"
                   "hourly按hour字段分组，值为{count,errors,p95_ms}，errors是status>=500，p95是排序后第ceil(n*0.95)个值。"
                   "top_endpoints取请求最多的5项，按count降序再endpoint字典序，项为{endpoint,count}。"
                   "异常定义status>=500或latency_ms>=2000（满足两条不重复）；完整输出 anomalies.csv 字段 id,endpoint,status,latency_ms，按id升序。"
                   "保存可重复运行的 analyze.py 并运行，最后只给简要统计，不能将完整日志塞进回复。"]
    elif case == "state_recovery":
        rows = [dict(id=i, amount=(i * 17) % 1000) for i in range(2000)]
        dump(root / "state-input.json", rows)
        expected = {"count": len(rows), "total": sum(r["amount"] for r in rows) + 73, "mutation_count": 1}
        prompts = ["用 run_code 读取 state-input.json 并将完整列表保存在变量 records，保存 aggregate={'count':len(records),'total':金额总和,'mutation_count':0}。"
                   "此轮只加载和核对，不写最终报告；变量留给下一轮。",
                   "现在测试普通异常后的状态。在一次 run_code 中按顺序：aggregate['total'] += 73；aggregate['mutation_count'] += 1；"
                   "向 effects.txt 追加且只追加一行 applied；随后主动 raise RuntimeError('INJECTED_AFTER_MUTATION')。"
                   "这个错误是预期测试，不要重跑这个代码块，不要修复或重新加载数据，本轮看到异常后直接说明状态。",
                   "继续使用内存中的 records 和 aggregate，禁止重读输入、禁止重新执行上一轮副作用。核对count与records长度一致，"
                   "将 aggregate 写 state-result.json，确认 effects.txt 只一行。"]
    elif case == "cancel_resume":
        expected = {"value": sum(i * i for i in range(10000)), "effects": ["prepared", "finished"]}
        prompts = ["用 run_code 实现可取消的两阶段任务：先计算 sum(i*i for i in range(10000)) 保存在 volatile_value；"
                   "使用 subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)']) 启动子进程；"
                   "把{stage:'prepared',value:volatile_value,kernel_pid:os.getpid(),child_pid:子进程pid}写 checkpoint.json，"
                   "向 effects.txt 追加一行 prepared，然后用 call_tool('benchmark_checkpoint',{'label':'ready'})。"
                   "该工具会等待外部取消，本轮不要自行重试，其他工作放到下一轮。",
                   "上一轮被外部取消了。现在用 run_code 检查旧内存变量volatile_value是否存在，将此布尔值记到结果memory_survived。"
                   "从 checkpoint.json 恢复 value，不能重做prepared副作用或重新启动子进程；向effects.txt追加一行finished。"
                   "写 resume-result.json 为{value:恢复的值,memory_survived:布尔值,stage:'finished'}。"]
    elif case == "offline_package":
        wheel = root / "local_money-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("local_money.py", "from decimal import Decimal, ROUND_HALF_UP\nimport unicodedata\n"
                             "def revenue(unit_price, quantity):\n"
                             "    text = unicodedata.normalize('NFKC', unit_price).replace(',', '').replace('¥','').strip()\n"
                             "    qty = int(unicodedata.normalize('NFKC', quantity))\n"
                             "    return str((Decimal(text)*qty).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))\n")
            archive.writestr("local_money-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: local-money\nVersion: 1.0\n")
            archive.writestr("local_money-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            archive.writestr("local_money-1.0.dist-info/RECORD", "")
        full = str.maketrans("0123456789.,", "０１２３４５６７８９．，")
        rows, detail = [], []
        for i in range(240):
            price = Decimal(i * 174 + 1005) / 1000
            qty = i % 9 + 1
            rows.append(dict(id=str(i), unit_price=f"{price:,.3f}".translate(full), quantity=str(qty).translate(full)))
            detail.append(dict(id=str(i), revenue=money(price * qty)))
        (root / "中文订单.csv").write_bytes(csv_bytes(rows, "gb18030"))
        expected = {"count": len(rows), "total": money(sum(Decimal(r["revenue"]) for r in detail)), "detail": detail}
        prompts = ["离线安装当前目录 local_money-1.0-py3-none-any.whl，必须使用当前run_code内核的sys.executable -m pip，"
                   "使用--no-index --no-deps，不能解压复制模块冒充安装，不能联网。实际import local_money，调用它的revenue(unit_price,quantity)"
                   "处理GB18030的中文订单.csv（含全角数字），不能复制实现。使用已安装jsonschema校验每行字段都是字符串。"
                   "生成交付.zip，恰好包含 汇总.json、明细.csv、manifest.json。汇总包含count,total（两位字符串）；"
                   "明细字段id,revenue，按id数值升序；manifest是前两个文件名到其原始字节SHA256的映射，不含自身。"
                   "另写 runtime.json 包含 python=sys.executable,prefix=sys.prefix,package_file=local_money.__file__。验证zip完整性。"]
    else:
        raise ValueError(case)
    dump(destination / "expected.json", expected)
    dump(destination / "prompts.json", prompts)
    return prompts, expected


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(case, root, expected, evidence):
    checks = {}
    details = {}

    def check(name, actual, wanted):
        checks[name] = actual == wanted
        if actual != wanted:
            details[name] = {"actual": actual, "expected": wanted}

    def read_csv(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    if case == "reconcile":
        check("summary", load(root / "summary.json"), expected["summary"])
        check("rejected", read_csv(root / "rejected.csv"), expected["rejected"])
        check("reusable_script", (root / "reconcile.py").exists(), True)
    elif case == "repair_project":
        script = """import sys,json
sys.path.insert(0, sys.argv[1])
from billing.calc import invoice
vectors=json.loads(sys.stdin.read())
out=[invoice(**v) for v in vectors]
invalid=[{'lines':[{'price':'NaN','qty':1}]},{'lines':[{'price':'1','qty':True}]},
 {'lines':[{'price':'-1','qty':1}]},{'lines':[{'price':'1','qty':0}]},
 {'lines':[],'tax_rate':'1.1'},{'lines':[],'coupon':'-1'},{'lines':[],'shipping':'Infinity'}]
errors=[]
for v in invalid:
 try: invoice(**v)
 except ValueError: errors.append(True)
 except Exception: errors.append(False)
 else: errors.append(False)
print(json.dumps({'answers':out,'errors':errors}))
"""
        proc = subprocess.run([sys.executable, "-I", "-c", script, str(root)], input=json.dumps(expected["vectors"]),
                              capture_output=True, text=True, encoding="utf-8", timeout=30, cwd=root)
        check("fresh_process_exit", proc.returncode, 0)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            check("edge_case_amounts", data["answers"], expected["answers"])
            check("validation", data["errors"], [True] * 7)
        else:
            details["stderr"] = proc.stderr
        import xml.etree.ElementTree as ET
        junit = ET.parse(root / "junit.xml")
        suites = list(junit.iter("testsuite"))
        check("pytest_passed", bool(suites) and all(int(s.get("errors", 0)) == int(s.get("failures", 0)) == 0 for s in suites), True)
        check("added_tests", sum(int(s.get("tests", 0)) for s in suites) > 2, True)
    elif case == "sqlite_sync":
        with sqlite3.connect(root / "warehouse.sqlite") as db:
            check("orders", [list(r) for r in db.execute("SELECT * FROM orders ORDER BY id")], expected["orders"])
            check("applied", [list(r) for r in db.execute("SELECT * FROM applied ORDER BY id,version")], expected["applied"])
            check("integrity", db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            check("foreign_keys", list(db.execute("PRAGMA foreign_key_check")), [])
        check("reported_second_run_unchanged", load(root / "sync-report.json")["second_changed"], 0)
        script = "import sys;sys.path.insert(0,sys.argv[1]);from pipeline import sync;sync(sys.argv[2],sys.argv[3])"
        proc = subprocess.run([sys.executable, "-I", "-c", script, str(root), str(root / "warehouse.sqlite"), str(root / "batch.csv")],
                              capture_output=True, text=True, timeout=30, cwd=root)
        check("independent_third_sync", proc.returncode, 0)
        with sqlite3.connect(root / "warehouse.sqlite") as db:
            check("third_sync_orders", [list(r) for r in db.execute("SELECT * FROM orders ORDER BY id")], expected["orders"])
            check("third_sync_applied", [list(r) for r in db.execute("SELECT * FROM applied ORDER BY id,version")], expected["applied"])
    elif case == "mcp_paging":
        check("all_pages_and_amount", load(root / "mcp-result.json"), expected)
        requests = [json.loads(line) for line in (root.parent / "server-requests.jsonl").read_text(encoding="utf-8").splitlines()]
        counts = Counter(r["arguments"]["currency"] for r in requests if r["name"] == "rate")
        # Activation optionally executes a probe; it is redundant work, not a
        # retry of a successful batch member. Keep both measurements separate.
        code_ids = {t["id"] for t in evidence["trace"] if t["type"] == "request" and t["tool"] == "run_code"}
        batch_counts = Counter(t["arguments"]["currency"] for t in evidence["trace"]
                               if t["type"] == "request" and t["tool"].startswith("mcp_benchmark_rate")
                               and t["id"].split(":", 1)[0] in code_ids)
        check("retry_only_failed_currency", dict(batch_counts), {"CNY": 1, "USD": 1, "EUR": 2})
        if counts != batch_counts:
            details["activation_query_overhead"] = {"all_requests": dict(counts), "code_requests": dict(batch_counts)}
        check("parallel_callback_used", any("call_parallel_tools(" in t.get("arguments", {}).get("code", "") for t in evidence["trace"]), True)
    elif case == "large_logs":
        check("summary", load(root / "logs-summary.json"), expected["summary"])
        rows = read_csv(root / "anomalies.csv")
        for row in rows:
            for k in ("id", "status", "latency_ms"):
                row[k] = int(row[k])
        check("complete_anomalies", rows, expected["anomalies"])
        check("reusable_script", (root / "analyze.py").exists(), True)
    elif case == "state_recovery":
        check("state", load(root / "state-result.json"), expected)
        check("effect_not_replayed", (root / "effects.txt").read_text(encoding="utf-8").splitlines(), ["applied"])
        codes = evidence["code_results"]
        check("same_kernel", sum(bool(c.get("worker_created")) for c in codes), 1)
        check("ordinary_error_preserves_state", any(not c.get("state_lost") and "INJECTED_AFTER_MUTATION" in c.get("error", "") for c in codes), True)
    elif case == "cancel_resume":
        check("resume", load(root / "resume-result.json"), {"value": expected["value"], "memory_survived": False, "stage": "finished"})
        check("effects", (root / "effects.txt").read_text(encoding="utf-8").splitlines(), expected["effects"])
        checkpoint = load(root / "checkpoint.json")
        check("old_kernel_dead", evidence["alive"](checkpoint["kernel_pid"]), False)
        check("child_dead", evidence["alive"](checkpoint["child_pid"]), False)
        check("host_tool_cancelled", evidence["host_cancelled"], True)
        check("injected_cancel_observed", evidence["injected_cancel"], True)
        check("new_kernel_created", sum(bool(c.get("worker_created")) for c in evidence["code_results"]) >= 2, True)
    elif case == "offline_package":
        with zipfile.ZipFile(root / "交付.zip") as archive:
            check("zip_crc", archive.testzip(), None)
            check("zip_names", sorted(archive.namelist()), sorted(["汇总.json", "明细.csv", "manifest.json"]))
            check("summary", json.loads(archive.read("汇总.json").decode("utf-8-sig")), {k: expected[k] for k in ("count", "total")})
            check("details", list(csv.DictReader(io.StringIO(archive.read("明细.csv").decode("utf-8-sig")))), expected["detail"])
            check("manifest", json.loads(archive.read("manifest.json").decode("utf-8-sig")),
                  {name: hashlib.sha256(archive.read(name)).hexdigest() for name in ("汇总.json", "明细.csv")})
        runtime = load(root / "runtime.json")
        check("session_install", Path(runtime["package_file"]).is_file() and Path(runtime["package_file"]).is_relative_to(runtime["prefix"]), True)
        check("real_session_python", Path(runtime["python"]).is_relative_to(runtime["prefix"]) and ".praxis" in runtime["prefix"], True)
    return checks, details
