"""概念episodeと排他的な経過窓の研究専用検査。"""

import unittest
from copy import deepcopy

from tools.research_s3_b_replay.transitions import group, partition


def fixture():
    base = [
        dict(
            client_id=c,
            sample_index=s,
            concept=0 if s < 5 or s >= 1105 else 1,
            input_hash=f"{c}-{s}",
            label=0,
            predicted_label=0,
            correct=True,
        )
        for c in range(10)
        for s in range(1210)
    ]
    candidate = deepcopy(base)
    base[0].update(predicted_label=1, correct=False)
    for sample in (5, 55, 1105):
        candidate[sample].update(predicted_label=1, correct=False)
    return base, candidate


class TransitionChecks(unittest.TestCase):
    def test_exclusive_bounds_and_recurrence(self):
        base, candidate = fixture()
        segments, clients = partition(base, candidate, 1210)
        own = [s for s in segments if s["client"] == 0]
        by_index = {}
        for row in own:
            for index in range(row["start"], row["end"]):
                self.assertNotIn(index, by_index)
                by_index[index] = row
        self.assertEqual(set(by_index), set(range(1210)))
        self.assertEqual(by_index[0]["phase"], "initial")
        self.assertIsNone(by_index[0]["age_start"])
        for index, age_bin in (
            (5, "0:50"),
            (54, "0:50"),
            (55, "50:100"),
            (105, "100:250"),
            (255, "250:500"),
            (505, "500:1000"),
            (1005, "1000:+"),
        ):
            self.assertEqual(by_index[index]["age_bin"], age_bin)
            self.assertEqual(by_index[index]["phase"], "first_visit")
        self.assertEqual(by_index[1105]["phase"], "return")
        self.assertEqual(by_index[1105]["previous_concept"], 1)
        self.assertEqual(by_index[1105]["age_bin"], "0:50")
        self.assertEqual(sum(r["observations"] for r in segments), 12100)
        self.assertEqual(sum(r["net_correct"] for r in segments), -2)
        self.assertEqual(sum(r["net_correct"] for r in clients), -2)

    def test_rejects_input_mismatch_and_missing_tail(self):
        base, candidate = fixture()
        candidate[0]["input_hash"] = "different"
        with self.assertRaises(ValueError):
            partition(base, candidate, 1210)
        with self.assertRaises(ValueError):
            partition(base[:-1], base[:-1], 1210)

    def test_grouping_is_pooled_and_tracks_coverage(self):
        base, candidate = fixture()
        segments, _ = partition(base, list(reversed(candidate)), 1210)
        rows = [dict(interval=100, seed=0, **r) for r in segments]
        result = group(rows, ("interval", "phase"))
        phases = {r["phase"]: r for r in result}
        self.assertEqual(phases["initial"]["net_correct"], 1)
        self.assertEqual(phases["first_visit"]["net_correct"], -2)
        self.assertEqual(phases["return"]["net_correct"], -1)
        self.assertEqual(phases["initial"]["pooled_delta_pp"], 2)
        self.assertEqual(phases["initial"]["seeds"], 1)
        self.assertEqual(phases["initial"]["client_runs"], 10)
        self.assertEqual(phases["initial"]["episodes"], 10)


if __name__ == "__main__":
    unittest.main()
