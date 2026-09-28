"""送达站序规划(出库后多站配送路径)。

以生成方案时的输入快照(plan_input,与 /api/plan 请求体相同)重新规划得到
**全员可运输装箱方案**,在 **箱号、批次与各批件数** 经核对与方案一致的前提下,
按用户为每个货物批次指定的送达站点,搜索全部可行站序:

- 以出库时刻为 0,行车时长取自有向行车时长矩阵(仓库→站、站→仓库、
  站→站,行列均按 仓库,站点1,站点2,… 排列,允许双向不对称);
- 抵站时刻早于该站可签收时间窗开始时必须等待,等待后开始卸货;
- 卸货必须在时间窗结束前(恰在边界完成允许)完成,否则该站序不可行;
- 最后一站卸毕时刻不得超过方案的运输总时长,否则该站序不可行。

在全部可行站序(1~8 站的全排列,空站也须抵达)中,取 **最终卸毕时刻最早**
者;并列时取 **按输入站点序号组成的字典序最小** 者。无可行站序时抛出
ValueError(HTTP 400),不返回任何部分路线。

响应携带输入快照指纹(SHA-256),输入变化后前端据此将旧结果标记为已过期。
"""
from __future__ import annotations

import math
from itertools import permutations
from typing import Any

from .planner import EPS, _r, input_fingerprint, plan

# 站点数量硬上限:超过后全排列不可接受
MAX_STATIONS = 8


def _num(raw: Any, label: str) -> float:
    """解析非负有限数值;非法抛 ValueError(400,消息直接展示)。"""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label}必须为非负数值,实际为 {raw!r},已拒绝")
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label}必须为非负有限数,实际为 {raw!r},已拒绝")
    return value


def _parse_stations(raw: Any) -> list[dict[str, Any]]:
    """解析站点列表:1~8 个,每个站点名称(可选)、时间窗、卸货耗时。"""
    if not isinstance(raw, list):
        raise ValueError("站点列表(stations)必须为数组,实际已拒绝")
    n = len(raw)
    if not 1 <= n <= MAX_STATIONS:
        raise ValueError(
            f"送达站点数量须为 1~{MAX_STATIONS} 个,当前为 {n} 个,已拒绝"
        )
    stations: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        ctx = f"站点 {i + 1}"
        if not isinstance(row, dict):
            raise ValueError(f"{ctx}的信息必须为对象,实际为 {row!r},已拒绝")
        name = row.get("name")
        name = str(name).strip() if name not in (None, "") else f"站点 {i + 1}"
        win_start = _num(row.get("window_start"), f"{ctx}「{name}」的时间窗开始")
        win_end = _num(row.get("window_end"), f"{ctx}「{name}」的时间窗结束")
        if win_end < win_start - EPS:
            raise ValueError(
                f"{ctx}「{name}」的时间窗结束({_r(win_end)})不得早于开始"
                f"({_r(win_start)}),已拒绝"
            )
        unload = _num(row.get("unload_hours"), f"{ctx}「{name}」的卸货耗时")
        stations.append(
            {
                "no": i + 1,
                "name": name,
                "window_start": win_start,
                "window_end": win_end,
                "unload_hours": unload,
            }
        )
    return stations


def _parse_assignments(
    raw: Any, batch_count: int, station_count: int
) -> list[int]:
    """解析批次→站点归属:每个输入批次恰好出现一次,站号有效。

    返回按批号(1 起)排列的站号列表。
    """
    if not isinstance(raw, list):
        raise ValueError("批次到站归属(batch_assignments)必须为数组,实际已拒绝")
    if len(raw) != batch_count:
        raise ValueError(
            f"批次到站归属须覆盖全部 {batch_count} 个货物批次(每批恰好归属一站),"
            f"当前收到 {len(raw)} 条,已拒绝"
        )
    assignment_by_batch: dict[int, int] = {}
    for i, row in enumerate(raw):
        ctx = f"批次归属第 {i + 1} 行"
        if not isinstance(row, dict):
            raise ValueError(f"{ctx}必须为对象,实际为 {row!r},已拒绝")
        b_raw = row.get("batch_no")
        s_raw = row.get("station_no")
        if isinstance(b_raw, bool) or not isinstance(b_raw, int):
            raise ValueError(f"{ctx}的批号(batch_no)须为正整数,实际为 {b_raw!r}")
        if not 1 <= b_raw <= batch_count:
            raise ValueError(
                f"{ctx}的批号 {b_raw} 超出范围:规划快照共 {batch_count} 个批次,"
                f"批号须为 1~{batch_count}"
            )
        if isinstance(s_raw, bool) or not isinstance(s_raw, int):
            raise ValueError(
                f"{ctx}的站号(station_no)须为正整数,实际为 {s_raw!r}"
            )
        if not 1 <= s_raw <= station_count:
            raise ValueError(
                f"{ctx}批号 {b_raw} 指定的站号 {s_raw} 超出范围:"
                f"当前共 {station_count} 个站点,站号须为 1~{station_count}"
            )
        if b_raw in assignment_by_batch:
            raise ValueError(
                f"批号 {b_raw} 被重复指定送达站点(第 "
                f"{assignment_by_batch[b_raw]} 站与第 {s_raw} 站);"
                f"每个批次须恰好归属一个站点"
            )
        assignment_by_batch[b_raw] = s_raw
    missing = sorted(set(range(1, batch_count + 1)) - set(assignment_by_batch))
    if missing:
        raise ValueError(
            f"以下批次未指定送达站点:批号 {', '.join(map(str, missing))};"
            f"全部批次须恰好归属一个站点"
        )
    return [assignment_by_batch[b] for b in range(1, batch_count + 1)]


def _parse_travel_matrix(raw: Any, n: int) -> list[list[float]]:
    """解析有向行车时长矩阵:(n+1)×(n+1),行列序为 仓库,站点1…n。

    所有单元格均须为非负有限数(对角线不使用,也参与数值合法性校验)。
    """
    size = n + 1
    if not isinstance(raw, list):
        raise ValueError("有向行车时长矩阵(travel_times)必须为数组,实际已拒绝")
    if len(raw) != size:
        raise ValueError(
            f"有向行车时长矩阵须为 {size}×{size}(仓库 + {n} 个站点),"
            f"当前为 {len(raw)} 行,已拒绝"
        )
    labels = ["仓库"] + [f"站点 {i + 1}" for i in range(n)]
    matrix: list[list[float]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, list):
            raise ValueError(f"行车时长矩阵第 {i + 1} 行必须为数组,已拒绝")
        if len(row) != size:
            raise ValueError(
                f"行车时长矩阵第 {i + 1} 行须有 {size} 个单元格"
                f"(仓库 + {n} 个站点),当前为 {len(row)} 个,已拒绝"
            )
        parsed: list[float] = []
        for j, v in enumerate(row):
            if i == j:
                # 对角线(同一地点)不参与计算,仅要求是有限数,负值/缺失忽略为 0
                try:
                    diag = float(v)
                except (TypeError, ValueError):
                    diag = 0.0
                parsed.append(diag if math.isfinite(diag) else 0.0)
            else:
                parsed.append(
                    _num(v, f"行车时长「{labels[i]}」→「{labels[j]}」")
                )
        matrix.append(parsed)
    return matrix


def _evaluate_permutation(
    order: tuple[int, ...],
    stations: list[dict[str, Any]],
    travel: list[list[float]],
) -> dict[str, Any] | None:
    """模拟一个站序(站号 1 起),返回时刻明细;不满足时间窗/总时长则 None。"""
    t = 0.0  # 出库时刻为 0
    cur = 0  # 0 = 仓库
    stops: list[dict[str, Any]] = []
    feasible = True
    for no in order:
        st = stations[no - 1]
        t = _r(t + travel[cur][no])
        arrival = t
        wait = max(0.0, _r(st["window_start"] - arrival))
        if wait > EPS:
            t = _r(arrival + wait)
        else:
            wait = 0.0
        unload_start = t
        finish = _r(unload_start + st["unload_hours"])
        window_ok = finish <= st["window_end"] + EPS
        stops.append(
            {
                "station_no": no,
                "arrival_hours": _r(arrival),
                "wait_hours": _r(wait),
                "unload_start_hours": _r(unload_start),
                "finish_hours": _r(finish),
                "window_ok": window_ok,
            }
        )
        if not window_ok:
            feasible = False
        t = finish
        cur = no
    return {"stops": stops, "final_finish": t, "feasible": feasible}


def plan_delivery(payload: dict[str, Any]) -> dict[str, Any]:
    """送达站序规划入口;输入不合法或无可行站序时抛 ValueError。"""
    plan_input = payload.get("plan_input")
    if not isinstance(plan_input, dict):
        raise ValueError("请提供生成方案时的规划输入快照(plan_input)")
    stations = _parse_stations(payload.get("stations"))
    n = len(stations)

    # 绑定规划输入快照:按快照重新规划,逐箱逐批复核箱号/批次/件数
    baseline = plan(plan_input)
    cargo_rows = plan_input.get("cargoes") or []
    batch_count = len(cargo_rows)
    if batch_count == 0:  # 理论上 Pydantic/plan 已拦截,兜底
        raise ValueError("规划快照中没有任何货物批次,无法规划送达站序")

    assigned = _parse_assignments(
        payload.get("batch_assignments"), batch_count, n
    )
    travel = _parse_travel_matrix(payload.get("travel_times"), n)

    total_hours = float(baseline["summary"]["transport_hours"])
    boxes = baseline["boxes"]
    if not boxes:
        raise ValueError("当前方案没有任何已装箱货物,无法规划送达站序")
    if not baseline["summary"].get("all_transportable"):
        raise ValueError(
            "当前方案不是全员可运输装箱方案(存在未装载或全程不可运输的货物),"
            "请先调整规划输入并重新生成全员可运输方案后再规划送达站序"
        )

    # 快照复核:汇总方案中各箱各批件数,同时给出批次名称与总件数
    batch_names = {i + 1: str((c.get("name") or f"批次 {i + 1}"))
                   for i, c in enumerate(cargo_rows)}
    box_batches: dict[tuple[int, int], int] = {}
    batch_units_planned: dict[int, int] = {}
    box_nos: list[int] = []
    for box in boxes:
        box_no = int(box["box_no"])
        box_nos.append(box_no)
        for it in box["items"]:
            key = (box_no, int(it["batch_no"]))
            box_batches[key] = box_batches.get(key, 0) + int(it["units"])
            batch_units_planned[int(it["batch_no"])] = (
                batch_units_planned.get(int(it["batch_no"]), 0) + int(it["units"])
            )

    # 按站点归集箱号、批次与件数(箱号与批号均按方案序号升序)
    station_items: dict[int, list[dict[str, Any]]] = {i + 1: [] for i in range(n)}
    station_batch_units: dict[int, dict[int, int]] = {i + 1: {} for i in range(n)}
    station_box_set: dict[int, set[int]] = {i + 1: set() for i in range(n)}
    for (box_no, b_no), units in sorted(box_batches.items()):
        s_no = assigned[b_no - 1]
        station_items[s_no].append(
            {
                "box_no": box_no,
                "batch_no": b_no,
                "name": batch_names[b_no],
                "units": units,
            }
        )
        station_batch_units[s_no][b_no] = (
            station_batch_units[s_no].get(b_no, 0) + units
        )
        station_box_set[s_no].add(box_no)

    assignment_rows = []
    for b_no in range(1, batch_count + 1):
        s_no = assigned[b_no - 1]
        assignment_rows.append(
            {
                "batch_no": b_no,
                "name": batch_names[b_no],
                "units": batch_units_planned[b_no],
                "station_no": s_no,
                "station_name": stations[s_no - 1]["name"],
            }
        )

    # 穷举全部站序;itertools.permutations 按输入序号字典序产出
    candidate_count = math.factorial(n)
    best: dict[str, Any] | None = None
    feasible_count = 0
    window_fail_count = 0
    overtime_count = 0
    for order in permutations(range(1, n + 1)):
        ev = _evaluate_permutation(order, stations, travel)
        if not ev["feasible"]:
            window_fail_count += 1
            continue
        if ev["final_finish"] > total_hours + EPS:
            overtime_count += 1
            continue
        feasible_count += 1
        if (
            best is None
            or ev["final_finish"] < best["final_finish"] - EPS
        ):
            best = {"order": order, **ev}
        # 最终卸毕时刻相同时,先产出的排列即字典序最小,保留不替换

    if best is None:
        bits = [f"已穷举全部 {candidate_count} 种站序,无可行站序"]
        if window_fail_count:
            bits.append(
                f"{window_fail_count} 种站序存在卸货无法在时间窗结束前完成的站点"
            )
        if overtime_count:
            bits.append(
                f"{overtime_count} 种站序虽能在时间窗内卸完,但最后一站卸毕时刻"
                f"超过方案运输总时长 {_r(total_hours)}h"
            )
        raise ValueError(
            "送达站序无可行解:" + "、".join(bits) + ";请放宽时间窗、调整卸货耗时、"
            "行车时长或批次归属后重试,不返回部分路线"
        )

    # 逐站输出:时刻 + 该站箱号/批次/件数(与方案明细一致)
    stops_out: list[dict[str, Any]] = []
    for idx, sp in enumerate(best["stops"]):
        s_no = sp["station_no"]
        st = stations[s_no - 1]
        batches_out = [
            {
                "batch_no": b_no,
                "name": batch_names[b_no],
                "units": station_batch_units[s_no][b_no],
            }
            for b_no in sorted(station_batch_units[s_no])
        ]
        stops_out.append(
            {
                "stop_no": idx + 1,
                "station_no": s_no,
                "name": st["name"],
                "window_start": _r(st["window_start"]),
                "window_end": _r(st["window_end"]),
                "unload_hours": _r(st["unload_hours"]),
                "arrival_hours": sp["arrival_hours"],
                "wait_hours": sp["wait_hours"],
                "unload_start_hours": sp["unload_start_hours"],
                "finish_hours": sp["finish_hours"],
                "box_nos": sorted(station_box_set[s_no]),
                "boxes": [
                    {
                        "box_no": box_no,
                        "items": [
                            it for it in station_items[s_no] if it["box_no"] == box_no
                        ],
                    }
                    for box_no in sorted(station_box_set[s_no])
                ],
                "batches": batches_out,
                "total_units": sum(b["units"] for b in batches_out),
            }
        )

    final_finish = _r(best["final_finish"])
    total_travel = _r(
        sum(
            travel[prev][cur_no]
            for prev, cur_no in zip((0,) + best["order"][:-1], best["order"])
        )
    )
    total_wait = _r(sum(s["wait_hours"] for s in stops_out))
    total_unload = _r(sum(s["unload_hours"] for s in stops_out))
    verdict = (
        f"最优站序 {'→'.join(str(x) for x in best['order'])},"
        f"最后一站于 {final_finish}h 卸毕"
        f"(方案运输总时长 {_r(total_hours)}h),共评估 {candidate_count} 种站序,"
        f"其中 {feasible_count} 种可行"
    )

    return {
        "input_fingerprint": input_fingerprint(plan_input),
        "transport_hours": _r(total_hours),
        "stations": [
            {
                "station_no": s["no"],
                "name": s["name"],
                "window_start": _r(s["window_start"]),
                "window_end": _r(s["window_end"]),
                "unload_hours": _r(s["unload_hours"]),
            }
            for s in stations
        ],
        "travel_times": [[_r(v) for v in row] for row in travel],
        "batch_assignments": assignment_rows,
        "route": {
            "station_sequence": list(best["order"]),
            "stops": stops_out,
            "depart_hours": 0.0,
            "final_finish_hours": final_finish,
            "total_travel_hours": total_travel,
            "total_wait_hours": total_wait,
            "total_unload_hours": total_unload,
        },
        "summary": {
            "station_count": n,
            "box_count": len(box_nos),
            "batch_count": batch_count,
            "total_units": sum(batch_units_planned.values()),
            "candidate_permutations": candidate_count,
            "feasible_permutations": feasible_count,
            "infeasible_window_permutations": window_fail_count,
            "infeasible_overtime_permutations": overtime_count,
            "final_finish_hours": final_finish,
            "within_transport_hours": True,
            "verdict": verdict,
        },
        "verdict": verdict,
    }
