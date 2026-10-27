#!/usr/bin/env python3
"""Tests for scripts/publish.py. Run: python3 scripts/test_publish.py"""
import unittest
from datetime import datetime, timezone

import publish as p


def meta(*pkgs):
    return {"packages": [
        {"name": n, "version": "0.1.0", "publish": pub,
         "dependencies": [{"name": d, "req": req, "kind": kind} for d, req, kind in deps]}
        for n, pub, deps in pkgs
    ]}


class Order(unittest.TestCase):
    def test_dependencies_come_first_and_ties_are_by_name(self):
        m = meta(
            ("facade", None, [("core", "^0.1.0", None), ("providers", "^0.1.0", None)]),
            ("providers", None, [("openai", "^0.1.0", None)]),
            ("openai", None, [("core", "^0.1.0", None)]),
            ("core", None, []),
            ("macros", None, []),
        )
        names = [c.name for c in p.publish_order(p.crates_from_metadata(m))]
        self.assertEqual(names, ["core", "macros", "openai", "providers", "facade"])

    def test_dev_dependencies_and_unversioned_path_deps_are_not_edges(self):
        m = meta(
            ("teams", None, [("testing", "^0.1.0", "dev"), ("core", "*", None)]),
            ("testing", None, [("core", "^0.1.0", None)]),
            ("core", None, []),
        )
        crates = {c.name: c for c in p.crates_from_metadata(m)}
        self.assertEqual(crates["teams"].deps, set())
        self.assertEqual(crates["testing"].deps, {"core"})

    def test_unpublishable_packages_are_left_out(self):
        m = meta(("core", None, []), ("demo", [], [("core", "^0.1.0", None)]))
        self.assertEqual([c.name for c in p.crates_from_metadata(m)], ["core"])

    def test_a_cycle_is_refused(self):
        m = meta(("a", None, [("b", "^0.1.0", None)]), ("b", None, [("a", "^0.1.0", None)]))
        with self.assertRaises(p.PublishError):
            p.publish_order(p.crates_from_metadata(m))


class Index(unittest.TestCase):
    def test_paths(self):
        self.assertEqual(p.index_path("a"), "1/a")
        self.assertEqual(p.index_path("ab"), "2/ab")
        self.assertEqual(p.index_path("abc"), "3/a/abc")
        self.assertEqual(p.index_path("opai-agents-core"), "op/ai/opai-agents-core")

    def test_parse(self):
        body = '{"vers":"0.1.0","yanked":false}\n{"vers":"0.1.1","yanked":true}\n\n'
        self.assertEqual(p.parse_index(body), {"0.1.0": False, "0.1.1": True})


class Pacing(unittest.TestCase):
    def test_burst_then_one_per_interval(self):
        b = p.Bucket(5, 600.0)
        t = 0.0
        waits = []
        for _ in range(8):
            w = b.wait_for_token(t)
            t += w
            b.take(t)
            waits.append(w)
        self.assertEqual(waits, [0, 0, 0, 0, 0, 600, 600, 600])

    def test_estimate_for_the_twelve_crate_first_release(self):
        # 5 at once, then 7 more at 10 minutes each: 70 minutes.
        self.assertEqual(p.estimate_seconds(12), 7 * 600)
        self.assertEqual(p.estimate_seconds(5), 0)

    def test_a_drained_bucket_waits_for_the_first_token(self):
        self.assertEqual(p.estimate_seconds(1, tokens=0), 600)

    def test_tokens_come_back_over_time(self):
        b = p.Bucket(5, 600.0, tokens=0.0)
        self.assertEqual(b.wait_for_token(0.0), 600.0)
        self.assertEqual(b.wait_for_token(600.0), 0.0)


class RateLimitText(unittest.TestCase):
    def test_reads_the_time_in_the_message(self):
        text = ("error: failed to publish\n429 Too Many Requests: You have published too many new "
                "crates in a short period of time. Please try again after Tue, 21 Oct 2026 "
                "02:48:00 GMT or email help@crates.io to have your limit increased.")
        now = datetime(2026, 10, 21, 2, 37, 0, tzinfo=timezone.utc)
        self.assertEqual(p.retry_after_seconds(text, now), 660.0)
        self.assertTrue(p.is_rate_limited(text))

    def test_a_time_in_the_past_means_no_wait(self):
        text = "try again after Tue, 21 Oct 2026 02:48:00 GMT"
        now = datetime(2026, 10, 22, tzinfo=timezone.utc)
        self.assertEqual(p.retry_after_seconds(text, now), 0.0)

    def test_retry_after_header(self):
        self.assertEqual(p.retry_after_seconds("Retry-After: 120"), 120.0)

    def test_unknown_text(self):
        self.assertIsNone(p.retry_after_seconds("something else"))
        self.assertFalse(p.is_rate_limited("something else"))

    def test_already_published(self):
        self.assertTrue(p.is_already_published("crate version `0.1.0` is already uploaded"))
        self.assertTrue(p.is_already_published("error: crate opai-agents@0.1.0 already exists"))


class Secrets(unittest.TestCase):
    def test_the_token_is_redacted(self):
        import os
        os.environ["CARGO_REGISTRY_TOKEN"] = "cio-secret-value"
        try:
            self.assertEqual(p.redact("x cio-secret-value y"), "x *** y")
        finally:
            del os.environ["CARGO_REGISTRY_TOKEN"]


if __name__ == "__main__":
    unittest.main()
