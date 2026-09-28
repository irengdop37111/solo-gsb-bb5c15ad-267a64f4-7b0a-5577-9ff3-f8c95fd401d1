"""送达站序规划 API 测试(stdlib unittest,无需额外数据库)。

运行:python tests/test_delivery.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # 未安装 Web 依赖时仍可运行核心规则用例,API 用例自动跳过
    from fastapi.testclient import TestClient  # noqa: E402
    from app.main import app  # noqa: E402

    _APP_AVAILABLE = True
except ImportError:  # pragma: no cover - 仅极简环境触发
    TestClient = None  # type: ignore[assignment,misc]
    app = None  # type: ignore[assignment]
    _APP_AVAILABLE = False

from app.delivery import plan_delivery  # noqa: E402

# 单一温区、3 批货物全部可装(总时长 12h)
PLAN = {
    "transport_hours": 12,
    "box_capacity": 100,
    "zones": [{"name": "冷藏区", "t_min": 0, "t_max": 8}],
    "cargoes": [
        {"name": "鲜奶", "units": 10, "unit_volume": 0.2, "t_min": 2, "t_max": 6},
        {"name": "三文鱼", "units": 5, "unit_volume": 0.4, "t_min": 0, "t_max": 4},
        {"name": "奶酪", "units": 4, "unit_volume": 0.5, "t_min": 2, "t_max": 8},
    ],
}

# 零行车时长矩阵(仓库 + 2 站)
ZERO_T2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
S2 = [
    {"name": "甲站", "window_start": 0, "window_end": 24, "unload_hours": 1},
    {"name": "乙站", "window_start": 0, "window_end": 24, "unload_hours": 1},
]
ASSIGN2 = [
    {"batch_no": 1, "station_no": 1},
    {"batch_no": 2, "station_no": 2},
    {"batch_no": 3, "station_no": 1},
]


def _payload(plan=PLAN, stations=None, assignments=None, travel=None):
    return {
        "plan_input": plan,
        "stations": S2 if stations is None else stations,
        "batch_assignments": ASSIGN2 if assignments is None else assignments,
        "travel_times": ZERO_T2 if travel is None else travel,
    }


class DeliveryDirectTestCase(unittest.TestCase):
    """不依赖 HTTP 层的核心规则测试。"""

    def test_zero_travel_basic_timeline(self):
        r = plan_delivery(_payload())
        self.assertEqual(r["route"]["station_sequence"], [1, 2])
        stops = r["route"]["stops"]
        # 零等待:甲站 0 抵达 0 卸毕 1;乙站 1 抵达 1 卸毕 2
        self.assertEqual(
            [(s["arrival_hours"], s["wait_hours"],
              s["unload_start_hours"], s["finish_hours"]) for s in stops],
            [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 2.0)],
        )
        self.assertEqual(r["route"]["final_finish_hours"], 2.0)
        self.assertEqual(r["summary"]["candidate_permutations"], 2)
        self.assertEqual(r["summary"]["feasible_permutations"], 2)

    def test_wait_until_window_start(self):
        # 仓库→甲站 1h,甲站窗口 5 点开始:等待 4h
        travel = [[0, 1, 4], [0, 0, 1], [0, 10, 0]]
        stations = [
            {"name": "甲站", "window_start": 5, "window_end": 24, "unload_hours": 1},
            {"name": "乙站", "window_start": 0, "window_end": 24, "unload_hours": 1},
        ]
        r = plan_delivery(_payload(stations=stations, travel=travel))
        first = r["route"]["stops"][0]
        self.assertEqual(first["arrival_hours"], 1.0)
        self.assertEqual(first["wait_hours"], 4.0)
        self.assertEqual(first["unload_start_hours"], 5.0)
        self.assertEqual(first["finish_hours"], 6.0)
        # 1→2 行驶 1h,乙站 7 抵达 7 卸毕 8;站序 2,1 最后 16 点卸毕
        self.assertEqual(r["route"]["stops"][1]["arrival_hours"], 7.0)
        self.assertEqual(r["route"]["final_finish_hours"], 8.0)
        self.assertEqual(r["summary"]["feasible_permutations"], 1)
        self.assertEqual(r["summary"]["infeasible_overtime_permutations"], 1)
        self.assertEqual(r["route"]["total_wait_hours"], 4.0)
        self.assertEqual(r["route"]["total_travel_hours"], 2.0)

    def test_prefers_earliest_finish_over_lexicographic(self):
        # 1→3、2→3 与 2 相关的部分边很长;最优为 1→3→2,3 点卸毕
        travel = [
            [0, 1, 10, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 10],
            [1, 1, 1, 0],
        ]
        stations = [
            {"window_start": 0, "window_end": 99, "unload_hours": 0}
        ] * 3
        assignments = [
            {"batch_no": 1, "station_no": 1},
            {"batch_no": 2, "station_no": 2},
            {"batch_no": 3, "station_no": 3},
        ]
        p = _payload(stations=stations, assignments=assignments, travel=travel)
        r = plan_delivery(p)
        self.assertEqual(r["route"]["station_sequence"], [1, 3, 2])
        self.assertEqual(r["route"]["final_finish_hours"], 3.0)

    def test_tie_breaks_leksicographically(self):
        # 零行驶/零卸货:6 种站序全部 0 点卸毕,取字典序最小 [1,2,3]
        stations = [
            {"window_start": 0, "window_end": 99, "unload_hours": 0}
        ] * 3
        assignments = [
            {"batch_no": 1, "station_no": 1},
            {"batch_no": 2, "station_no": 2},
            {"batch_no": 3, "station_no": 3},
        ]
        travel = [[0] * 4 for _ in range(4)]
        r = plan_delivery(
            _payload(stations=stations, assignments=assignments, travel=travel)
        )
        self.assertEqual(r["route"]["station_sequence"], [1, 2, 3])
        self.assertEqual(r["summary"]["feasible_permutations"], 6)

    def test_finish_exactly_at_boundaries_is_feasible(self):
        # 恰在窗口结束/运输总时长时刻卸毕允许
        stations = [{"window_start": 0, "window_end": 6, "unload_hours": 1}]
        assignments = [
            {"batch_no": 1, "station_no": 1},
            {"batch_no": 2, "station_no": 1},
            {"batch_no": 3, "station_no": 1},
        ]
        travel = [[0, 5], [0, 0]]
        plan = {**PLAN, "transport_hours": 6}
        r = plan_delivery(
            _payload(
                plan=plan, stations=stations,
                assignments=assignments, travel=travel,
            )
        )
        stop = r["route"]["stops"][0]
        self.assertEqual(stop["finish_hours"], 6.0)
        self.assertTrue(r["summary"]["within_transport_hours"])

    def test_no_feasible_window(self):
        stations = [
            {"window_start": 0, "window_end": 0.5, "unload_hours": 1},
            {"window_start": 0, "window_end": 0.5, "unload_hours": 1},
        ]
        with self.assertRaisesRegex(ValueError, "无可行站序"):
            plan_delivery(_payload(stations=stations))

    def test_no_feasible_overtime(self):
        # 每段行驶 8h、卸货 1h:两站至少 17 点才能全部卸毕,超过总时长 12h
        travel = [
            [0, 8, 8],
            [8, 0, 8],
            [8, 8, 0],
        ]
        with self.assertRaisesRegex(ValueError, "超过方案运输总时长"):
            plan_delivery(_payload(travel=travel))

    def test_station_manifest_and_units(self):
        r = plan_delivery(_payload())
        by_station = {s["station_no"]: s for s in r["route"]["stops"]}
        # 顺序为 1,2:甲站有批 1(10)与批 3(4),乙站有批 2(5)
        first = by_station[1]
        self.assertEqual(first["total_units"], 14)
        self.assertEqual(
            [(b["batch_no"], b["units"]) for b in first["batches"]],
            [(1, 10), (3, 4)],
        )
        self.assertEqual(by_station[2]["total_units"], 5)
        # 批次归属回显件数与方案一致
        assign = {a["batch_no"]: a for a in r["batch_assignments"]}
        self.assertEqual(assign[1]["units"], 10)
        self.assertEqual(assign[2]["units"], 5)
        self.assertEqual(assign[3]["units"], 4)
        # 每站明细都带箱号且箱号来自方案
        for stop in r["route"]["stops"]:
            self.assertTrue(stop["box_nos"])
            for box in stop["boxes"]:
                self.assertIn(box["box_no"], stop["box_nos"])
                self.assertTrue(box["items"])

    def test_split_batch_manifest(self):
        # 同批分拆到多箱时,站序明细里该站件数为各箱件数之和
        plan = {
            "transport_hours": 10,
            "box_capacity": 0.5,
            "max_boxes": 10,
            "zones": [{"name": "冷藏区", "t_min": 0, "t_max": 8}],
            "cargoes": [
                {"name": "鲜奶", "units": 5, "unit_volume": 0.2,
                 "t_min": 2, "t_max": 6},
            ],
        }
        stations = [{"window_start": 0, "window_end": 20, "unload_hours": 1}]
        assignments = [{"batch_no": 1, "station_no": 1}]
        travel = [[0, 0], [0, 0]]
        r = plan_delivery(
            _payload(
                plan=plan, stations=stations,
                assignments=assignments, travel=travel,
            )
        )
        stop = r["route"]["stops"][0]
        self.assertEqual(stop["total_units"], 5)
        self.assertEqual(len(stop["box_nos"]), 3)  # 每箱 0.4 体积,装 2 件,末箱 1 件
        self.assertEqual(
            sum(it["units"] for box in stop["boxes"] for it in box["items"]),
            5,
        )

    def test_empty_station_is_still_visited(self):
        # 两批都归 1 站:2 站为空站(0 货、卸货耗时仍计入),仍须在站序中出现
        assignments = [
            {"batch_no": 1, "station_no": 1},
            {"batch_no": 2, "station_no": 1},
            {"batch_no": 3, "station_no": 1},
        ]
        r = plan_delivery(_payload(assignments=assignments))
        self.assertEqual(len(r["route"]["stops"]), 2)
        empties = [s for s in r["route"]["stops"] if s["total_units"] == 0]
        self.assertEqual(len(empties), 1)
        self.assertEqual(empties[0]["station_no"], 2)
        self.assertEqual(empties[0]["box_nos"], [])


class DeliveryValidationTestCase(unittest.TestCase):
    """输入校验:非法输入一律 ValueError(HTTP 层映射为 400)。"""

    def _assert_400_text(self, payload, fragment):
        with self.assertRaises(ValueError) as cm:
            plan_delivery(payload)
        self.assertIn(fragment, str(cm.exception))

    def test_station_count_bounds(self):
        self._assert_400_text(
            _payload(stations=[]), "1~8"
        )
        self._assert_400_text(
            _payload(stations=[{} for _ in range(9)]), "1~8"
        )

    def test_window_end_before_start(self):
        bad = [
            {"window_start": 5, "window_end": 4, "unload_hours": 1},
            S2[1],
        ]
        self._assert_400_text(_payload(stations=bad), "结束")

    def test_negative_and_non_finite_values(self):
        self._assert_400_text(
            _payload(stations=[
                {"window_start": -1, "window_end": 5, "unload_hours": 1},
                S2[1],
            ]),
            "非负有限数",
        )
        self._assert_400_text(
            _payload(stations=[
                {"window_start": 0, "window_end": 5, "unload_hours": -1},
                S2[1],
            ]),
            "非负有限数",
        )
        for bad_v in (float("nan"), float("inf"), -0.1):
            travel = [row[:] for row in ZERO_T2]
            travel[0][1] = bad_v
            if math.isnan(bad_v):
                with self.assertRaises(ValueError):
                    plan_delivery(_payload(travel=travel))
            else:
                self._assert_400_text(
                    _payload(travel=travel), "非负有限数"
                )

    def test_matrix_shape(self):
        self._assert_400_text(
            _payload(travel=[[0, 0], [0, 0]]), "3×3"
        )
        self._assert_400_text(
            _payload(travel=ZERO_T2[:2]), "3×3"
        )

    def test_assignment_coverage(self):
        self._assert_400_text(
            _payload(assignments=ASSIGN2[:2]), "恰好归属一站"
        )
        self._assert_400_text(
            _payload(
                assignments=[
                    {"batch_no": 1, "station_no": 1},
                    {"batch_no": 1, "station_no": 2},
                    {"batch_no": 3, "station_no": 1},
                ]
            ),
            "重复指定",
        )
        self._assert_400_text(
            _payload(
                assignments=[
                    {"batch_no": 1, "station_no": 3},
                    {"batch_no": 2, "station_no": 1},
                    {"batch_no": 3, "station_no": 1},
                ]
            ),
            "站号 3 超出范围",
        )
        self._assert_400_text(
            _payload(
                assignments=[
                    {"batch_no": 0, "station_no": 1},
                    {"batch_no": 2, "station_no": 1},
                    {"batch_no": 3, "station_no": 1},
                ]
            ),
            "批号",
        )

    def test_non_transportable_plan_rejected(self):
        # 容量只够装部分货物 -> 存在未装载批次
        plan = {**PLAN, "box_capacity": 0.3, "max_boxes": 1}
        self._assert_400_text(
            _payload(plan=plan), "全员可运输"
        )

    def test_missing_plan_input(self):
        with self.assertRaises(ValueError):
            plan_delivery({"stations": S2, "batch_assignments": ASSIGN2,
                           "travel_times": ZERO_T2})


@unittest.skipUnless(_APP_AVAILABLE, "未安装 FastAPI 依赖,跳过 API 用例")
class DeliveryApiTestCase(unittest.TestCase):
    """HTTP 层:200 / 400 / 422 状态码与错误体。"""

    def setUp(self):
        self.client = TestClient(app)

    def test_api_success(self):
        resp = self.client.post("/api/delivery-route", json=_payload())
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["route"]["station_sequence"], [1, 2])
        self.assertIn("input_fingerprint", data)
        self.assertEqual(len(data["route"]["stops"]), 2)

    def test_api_infeasible_returns_400_no_partial_route(self):
        bad_stations = [
            {"window_start": 0, "window_end": 0.5, "unload_hours": 1},
            {"window_start": 0, "window_end": 0.5, "unload_hours": 1},
        ]
        resp = self.client.post(
            "/api/delivery-route", json=_payload(stations=bad_stations)
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("无可行站序", resp.json()["detail"])
        self.assertNotIn("route", resp.json())

    def test_api_validation_400(self):
        # 窗口结束早于开始 -> 业务校验 400
        resp = self.client.post(
            "/api/delivery-route",
            json=_payload(stations=[
                {"window_start": 5, "window_end": 4, "unload_hours": 1},
                S2[1],
            ]),
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("结束", resp.json()["detail"])

    def test_api_schema_422(self):
        # 缺少必填字段 -> Pydantic 422
        resp = self.client.post(
            "/api/delivery-route", json={"plan_input": PLAN}
        )
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
