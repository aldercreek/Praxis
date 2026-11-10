"""Generated complex tasks and independent, deterministic outcome checks.

Answers and graders stay outside model workspaces. No external services or data.
"""
import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data, indent=2), encoding="utf-8")


@dataclass
class Scenario:
    task: str
    check: object
    followup: str = ""


def incident(root):
    rng = random.Random(714)
    requests, cache, ledger = [], [], []
    contaminated, impacted, loss = [], [], 0
    for i in range(240):
        request = f"r{i:04}"
        tenant = f"tenant-{i % 7}"
        price = rng.randrange(1000, 9000)
        true_time = 1000 + i * 3
        version = "v2" if 1110 <= true_time < 1540 else "v1"
        cross = version == "v2" and i % 5 in (1, 3)
        charged = price + (rng.choice([-400, 250, 700]) if cross else 0)
        cache_tenant = f"tenant-{(i + 1) % 7}" if cross else tenant
        requests.append(dict(id=request, tenant=tenant, observed_at=true_time + 17,
                             expected_cents=price, cache_trace=f"c{i:04}"))
        cache.append(dict(trace=f"c{i:04}", observed_at=true_time - 11,
                          key_version=version, payload_tenant=cache_tenant,
                          amount_cents=charged))
        posted = cross and i % 4 != 0
        ledger.append(dict(event_id=f"e{i}", request_id=request, observed_at=true_time + 29,
                           type="capture" if posted else "void", amount_cents=charged))
        if i % 6 == 0:
            ledger.append(dict(ledger[-1]))  # Delivery replay, not an extra charge.
        if cross:
            contaminated.append(request)
        if posted:
            impacted.append(request)
            loss += charged - price
    for rows in (requests, cache, ledger):
        rng.shuffle(rows)
    write(root, "requests.json", requests)
    write(root, "cache.json", cache)
    write(root, "ledger.json", ledger)
    write(root, "policy.json", dict(clock_offsets=dict(requests=17, cache=-11, ledger=29),
          rule="Join request.cache_trace to cache.trace. Cross-tenant payloads are contaminated. Only captures are financial impact; deduplicate event_id. Net delta is capture amount minus expected amount, including negative deltas.",
          deployments=[dict(version="v2", real_start=1110, real_end=1540)],
          v1_key="tenant|route|date", v2_key="route|date"))
    expected = dict(contaminated_ids=sorted(contaminated), financially_affected_ids=sorted(impacted),
                    net_delta_cents=loss, defective_version="v2", faulty_key_field="tenant")
    return Scenario(
        "Investigate a cross-tenant pricing incident using requests.json, cache.json, ledger.json and policy.json. "
        "Delegate independent cache, financial and timeline investigations to at least three workers. "
        "Have specialists directly exchange join evidence, not only final summaries. Resolve clock offsets, "
        "duplicate deliveries, negative differences and captures versus voids. Identify the defective key. "
        "Write result.json with contaminated_ids, financially_affected_ids (sorted ID lists), net_delta_cents, "
        "defective_version and faulty_key_field. Write evidence.md explaining causal timeline and remediation. "
        "Use inherited context for at least one worker, which must report the earlier CONTEXT_CODE back to you.",
        lambda data: {key: data.get(key) == value for key, value in expected.items()})


def settlement(root):
    rng = random.Random(2718)
    events, totals = [], {f"t{i}": 0 for i in range(6)}
    # Include half-cent ties as well as repeating thirds, so half-even rounding
    # is checked by the data instead of appearing only in the written rules.
    rates = {"USD": (100, 1), "EUR": (219, 2), "JPY": (2, 3)}
    for i in range(350):
        tenant, currency = f"t{i % 6}", rng.choice(list(rates))
        amount = rng.randint(1, 4000)
        kind = "refund" if i % 5 == 0 else "capture"
        revision = 2 if i % 7 == 0 else 1
        final_amount = amount + 37 if revision == 2 else amount
        event = dict(event_id=f"e{i:04}", revision=1, tenant=tenant, currency=currency,
                     amount_minor=amount, kind=kind)
        events.append(event)
        if revision == 2:
            events.append({**event, "revision": 2, "amount_minor": final_amount})
        if i % 3 == 0:
            events.append(dict(event))
        numerator, denominator = rates[currency]
        gross = int((Decimal(final_amount) * numerator / denominator).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))
        fee = max(7, (gross * 23 + 999) // 1000) if kind == "capture" else 0
        totals[tenant] += gross - fee if kind == "capture" else -gross
    rng.shuffle(events)
    bank = {tenant: amount + (i - 2) * 13 for i, (tenant, amount) in enumerate(totals.items())}
    write(root, "events.json", events)
    write(root, "fx.json", rates)
    write(root, "bank.json", bank)
    write(root, "rules.txt", "Deduplicate by event_id, using greatest revision regardless of arrival order. "
          "Convert each retained event using the numerator/denominator in fx.json and round half to even to integer settlement cents. "
          "Capture fee = max(7, ceil(converted_cents*23/1000)); refund has no fee. Net capture=gross-fee; refund=-gross. "
          "bank difference = computed net minus bank posted. Do not aggregate before per-event rounding.")
    expected = dict(tenant_net_cents=totals, bank_difference_cents={k: totals[k] - bank[k] for k in totals},
                    unique_events=350, raw_events=len(events))
    return Scenario(
        "Reconcile a multi-currency payment ledger with out-of-order revisions and replayed events. "
        "Use three or more workers for dedup/revision analysis, FX/fee calculation and bank reconciliation. "
        "Workers must directly exchange intermediate validation totals and challenge each other's rounding assumptions. "
        "Follow rules.txt exactly. Write result.json with tenant_net_cents and bank_difference_cents BOTH as objects "
        "mapping every tenant ID to its integer cents value (not a scalar grand total), plus integer unique_events and raw_events. "
        "Write audit.md describing invariant checks and how independent totals were reconciled. "
        "At least one worker must inherit context and tell you the earlier CONTEXT_CODE.",
        lambda data: {key: data.get(key) == value for key, value in expected.items()})


def routes(root):
    rng = random.Random(31415)
    edges = []
    for origin in range(7):
        for dest in range(origin + 1, 8):
            for option in range(2):
                edges.append(dict(id=f"f{origin}{dest}{option}", origin=origin, dest=dest,
                                  depart=origin * 100 + option * 20 + 10,
                                  arrive=dest * 100 - 10 + option * 15,
                                  cost=rng.randrange(30, 180), carbon=rng.randrange(10, 80),
                                  capacity=rng.choice([1, 1, 2])))
    orders = [dict(id="cargo-A", start=0, end=7, deadline=710, max_legs=3, max_carbon=150),
              dict(id="cargo-B", start=0, end=6, deadline=620, max_legs=3, max_carbon=160),
              dict(id="cargo-C", start=1, end=7, deadline=710, max_legs=3, max_carbon=150)]
    def enumerate_paths(order, banned):
        result = []
        def visit(node, arrival, path, cost, carbon):
            if node == order["end"]:
                result.append((cost, tuple(path)))
                return
            for edge in edges:
                if (edge["origin"] == node and edge["dest"] <= order["end"]
                        and edge["id"] not in banned and edge["depart"] >= arrival + (15 if path else 0)
                        and edge["arrive"] <= order["deadline"] and len(path) < order["max_legs"]
                        and carbon + edge["carbon"] <= order["max_carbon"]):
                    visit(edge["dest"], edge["arrive"], path + [edge["id"]], cost + edge["cost"], carbon + edge["carbon"])
        visit(order["start"], 0, [], 0, 0)
        return sorted(result)
    def optimum(banned):
        options = [enumerate_paths(order, banned) for order in orders]
        best = None
        capacities = {edge["id"]: edge["capacity"] for edge in edges}
        for combination in itertools.product(*options):
            total = sum(path[0] for path in combination)
            if best is not None and total > best[0]:
                continue
            used = {}
            for _, path in combination:
                for edge in path:
                    used[edge] = used.get(edge, 0) + 1
            if any(count > capacities[edge] for edge, count in used.items()):
                continue
            candidate = total, tuple(path[1] for path in combination)
            if best is None or candidate < best:
                best = candidate
        return best
    original = optimum(set())
    banned = original[1][0][0]
    final = optimum({banned})
    write(root, "flights.json", edges)
    write(root, "cargo.json", orders)
    write(root, "rules.txt", "Connection minimum 15 minutes. Cost is per cargo using a flight. Each flight's capacity is shared across all cargo. "
          "Honor max_legs, max_carbon, deadlines. Find the global minimum cost, not independent minima. "
          "Ties: lexicographically smallest tuple of flight-ID paths ordered cargo-A,B,C. Await a live user update before finalizing.")
    return Scenario(
        "Solve a globally coupled cargo-routing problem from flights.json, cargo.json and rules.txt. "
        "Use at least three workers for route enumeration, shared-capacity optimization and independent verification. "
        "Have workers directly exchange candidates and capacity conflicts. Parent must call wait_agents so a live disruption update can arrive. "
        "After the update, recompute optimality. Write result.json with total_cost, paths (cargo-ID to flight-ID list), banned_flight. "
        "Write proof.md with feasibility and global optimality evidence. Inherit context into a worker and recover the earlier CONTEXT_CODE.",
        lambda data: dict(total_cost=data.get("total_cost") == final[0],
                          paths=data.get("paths") == {order["id"]: list(path) for order, path in zip(orders, final[1])},
                          banned_flight=data.get("banned_flight") == banned),
        f"LIVE DISRUPTION: flight {banned} is cancelled and must not appear in any cargo path. "
        "Propagate this change to workers and recompute the global optimum. Set banned_flight in result.json to this flight ID.")


def schedule(root):
    jobs = [dict(id="design", duration=2, deps=[], machine="cpu"),
            dict(id="schema", duration=3, deps=["design"], machine="cpu"),
            dict(id="adapter", duration=3, deps=["design"], machine="cpu"),
            dict(id="fixture", duration=2, deps=[], machine="lab"),
            dict(id="migrate", duration=2, deps=["schema", "fixture"], machine="lab"),
            dict(id="integrate", duration=2, deps=["adapter", "schema"], machine="cpu"),
            dict(id="test", duration=3, deps=["migrate", "integrate"], machine="lab"),
            dict(id="release", duration=1, deps=["test"], machine="cpu")]
    by_id = {job["id"]: job for job in jobs}
    # Exhaustive discrete-time state search: all possible sets of ready jobs start
    # at each integer time, including deliberate idle choices.
    states = {(frozenset(), ()): {}}
    optimum = None
    for time in range(30):
        next_states = {}
        for (completed, running), plan in states.items():
            completed = completed | {name for name, end in running if end <= time}
            running = tuple((name, end) for name, end in running if end > time)
            if len(completed) == len(jobs):
                optimum = time
                break
            active = {name for name, _ in running}
            ready = [job for job in jobs if job["id"] not in completed | active
                     and set(job["deps"]) <= completed]
            for count in range(min(2 - len(running), len(ready)) + 1):
                for chosen in itertools.combinations(ready, count):
                    occupied = [by_id[name]["machine"] for name, _ in running] + [j["machine"] for j in chosen]
                    if occupied.count("lab") > 1 or occupied.count("cpu") > 2:
                        continue
                    if any(j["machine"] == "lab" and time < 7 and time + j["duration"] > 5 for j in chosen):
                        continue
                    new_running = tuple(sorted((*running, *((j["id"], time + j["duration"]) for j in chosen))))
                    new_plan = {**plan, **{j["id"]: time for j in chosen}}
                    next_states.setdefault((completed, new_running), new_plan)
        if optimum is not None:
            break
        states = next_states
    write(root, "jobs.json", jobs)
    write(root, "resources.json", dict(workers=2, cpu_slots=2, lab_slots=1, lab_unavailable=[[5, 7]],
                                     intervals="half-open; integer starts; non-preemptive"))
    def check(data):
        starts = data.get("starts", {})
        valid = set(starts) == set(by_id) and all(type(v) is int and v >= 0 for v in starts.values())
        if not valid:
            return dict(feasible=False, optimal=False)
        ends = {j["id"]: starts[j["id"]] + j["duration"] for j in jobs}
        valid = all(ends[dep] <= starts[j["id"]] for j in jobs for dep in j["deps"])
        valid &= all(not (starts[j["id"]] < 7 and ends[j["id"]] > 5) for j in jobs if j["machine"] == "lab")
        for time in range(max(ends.values())):
            running = [j for j in jobs if starts[j["id"]] <= time < ends[j["id"]]]
            valid &= len(running) <= 2 and sum(j["machine"] == "lab" for j in running) <= 1
        return dict(feasible=bool(valid), optimal=valid and max(ends.values()) == optimum,
                    reported_makespan=data.get("makespan") == max(ends.values()))
    return Scenario(
        "Build a globally optimal project schedule from jobs.json and resources.json. "
        "Use three workers for precedence/lower bounds, resource/maintenance feasibility and exact optimization. "
        "They must directly exchange constraints and counterexamples, not optimize in isolation. "
        "Find minimum makespan under dependencies, two workers, shared lab, maintenance and non-preemption. "
        "Write result.json with starts (all job IDs mapped to integer start times) and makespan. "
        "Write optimality.md with independent feasibility checks and proof no shorter schedule exists. "
        "At least one worker inherits context and reports the earlier CONTEXT_CODE. "
        "After one specialist finishes and its result is received, close it and create a new independent verifier; "
        "other workers should see the updated team.", check)


def inventory(root):
    write(root, "inventory.py", '''"""Public API: Inventory(capacity), apply(event), available(tenant, sku, now)."""
from threading import RLock

class Inventory:
    def __init__(self, capacity):
        self.capacity = dict(capacity)
        self.reservations = {}
        self.seen = set()
        self.lock = RLock()

    def apply(self, event):
        key = event['event_id']
        if key in self.seen:
            return False
        self.seen.add(key)
        reservation = event['reservation_id']
        if event['kind'] == 'reserve':
            if self.available(event['tenant'], event['sku'], event['now']) < event['qty']:
                return False
            self.reservations[reservation] = dict(event)
            return True
        old = self.reservations.pop(reservation, None)
        return old is not None

    def available(self, tenant, sku, now):
        used = sum(r['qty'] for r in self.reservations.values()
                   if r['sku'] == sku and r['expires'] >= now)
        return self.capacity[(tenant, sku)] - used
''')
    write(root, "spec.md", "Capacity is per (tenant,sku). apply must be linearizable under concurrent calls, including available reads. "
          "Event identity is (tenant,event_id); reservation identity is (tenant,reservation_id). Duplicate events return their original boolean result. "
          "An event with the same identity but conflicting payload raises ValueError. Failed events are remembered too. "
          "Reserve qty must be a positive integer (bool excluded), expires > now, and not collide with any live reservation of same identity. "
          "Invalid fields raise ValueError without mutating state. kind is reserve or release. Required base fields: tenant,event_id,reservation_id,kind,now. "
          "Reserve also requires sku,qty,expires; sku must exist for this tenant. At expires <= now reservation is inactive. "
          "Release returns true only for an active reservation and may not affect other tenants. Time can be read independently; "
          "assume calls use nondecreasing now values. Keep the public API unchanged. Standard library only.")
    def check(_):
        # Check via a fresh interpreter, so the grader imports the generated artifact.
        code = r'''
import concurrent.futures, importlib.util, json, sys
spec=importlib.util.spec_from_file_location('target',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
checks={}
def event(t,e,r,q=1,now=1,expires=10,kind='reserve'):
 return dict(tenant=t,event_id=e,reservation_id=r,kind=kind,now=now,sku='s',qty=q,expires=expires)
def run(name, fn):
 try: checks[name]=bool(fn())
 except Exception: checks[name]=False
def tenant():
 x=m.Inventory({('a','s'):3,('b','s'):3}); a=event('a','same','same',2); b=event('b','same','same',1)
 return x.apply(a) and x.apply(b) and x.available('a','s',1)==1 and x.available('b','s',1)==2
def replay():
 x=m.Inventory({('a','s'):3}); e=event('a','e','r',2); return x.apply(e) is True and x.apply(e) is True and x.available('a','s',1)==1
def failed():
 x=m.Inventory({('a','s'):1}); e=event('a','e','r',2); return x.apply(e) is False and x.apply(e) is False and x.available('a','s',1)==1
def conflict():
 x=m.Inventory({('a','s'):3}); x.apply(event('a','e','r'))
 try: x.apply(event('a','e','r',2))
 except ValueError: return x.available('a','s',1)==2
 return False
def expiry():
 x=m.Inventory({('a','s'):3}); x.apply(event('a','e','r',2,expires=3)); return x.available('a','s',3)==3
def release():
 x=m.Inventory({('a','s'):3,('b','s'):3}); x.apply(event('a','a','r',2)); x.apply(event('b','b','r',1)); rel=event('a','rel','r',kind='release')
 return x.apply(rel) is True and x.apply(rel) is True and x.available('a','s',1)==3 and x.available('b','s',1)==2
def invalid():
 x=m.Inventory({('a','s'):3})
 for q in (0,-1,True,1.5):
  try: x.apply(event('a',str(q),'r',q))
  except ValueError: pass
  else: return False
 return x.available('a','s',1)==3 and x.apply(event('a','0','r',1)) is True
def collision():
 x=m.Inventory({('a','s'):3}); x.apply(event('a','e','r',2)); return x.apply(event('a','e2','r',1)) is False and x.available('a','s',1)==1
def concurrency():
 x=m.Inventory({('a','s'):7})
 with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
  values=list(pool.map(lambda i:x.apply(event('a',str(i),str(i))),range(150)))
 return sum(v is True for v in values)==7 and x.available('a','s',1)==0
for name,fn in [('tenant_isolation',tenant),('successful_replay',replay),('failed_replay',failed),('payload_conflict',conflict),('expiry_boundary',expiry),('release_isolation',release),('validation_atomicity',invalid),('reservation_collision',collision),('concurrent_capacity',concurrency)]: run(name,fn)
print(json.dumps(checks))
'''
        result = subprocess.run([sys.executable, "-c", code, str(root / "inventory.py")],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            return {"artifact_import": False}
        return json.loads(result.stdout)
    return Scenario(
        "Repair the event-driven multi-tenant inventory engine in inventory.py to satisfy spec.md. "
        "Use three workers with clear non-overlapping ownership: protocol/idempotency analysis, implementation, and adversarial tests. "
        "They must directly discuss conflicts such as expired identities and failed replays. Have only the implementation worker edit inventory.py. "
        "Create regression tests and run them, then independently review the final artifact. "
        "Write result.json with a brief status, and verification.md explaining commands and unresolved risks. "
        "At least one worker must inherit context and report the earlier CONTEXT_CODE. "
        "Use standard library only and preserve public API.", check)


SCENARIOS = dict(incident=incident, settlement=settlement, routes=routes, schedule=schedule, inventory=inventory)
