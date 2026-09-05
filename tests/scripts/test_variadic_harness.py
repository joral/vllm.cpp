#!/usr/bin/env python3
"""Controls for the variadic-load harness, against a mock OpenAI stream.

`.agents/specs/bench-qwen38-exl3-variadic.md`, #2970. The harness measures two
engines on a leased GPU. Every property it claims is a property of the CLIENT
and the REPORT, not of the GPU, so all of them are checkable here with no
lease, no toolchain, no weights and no network beyond a loopback socket.

The mock server streams `tokens_per_chunk` tokens in each SSE chunk and reports
its own `usage`, which is what lets these tests separate a token count from a
chunk count. That separation is the defect the harness exists to avoid: one
engine here streams 4.46 tokens per chunk and the other 1.08, so a client that
counted chunks would compare two different quantities and call it a result.

What each case pins:

  PercentileContract      linear interpolation, matching numpy's default and
                          `vllm/benchmarks/serve.py:739`
  ClosedLoopConcurrency   c workers really overlap, and c=1 really does not
  WarmupIsExcluded        the measured window contains no warmup request, and
                          the with-warmup view still sees them
  ChunkingIsRecovered     tokens per chunk is read back from usage, and the
                          TTFT correction moves in the right direction
  UsageIsMandatory        a server that reports no usage is marked `chunks`,
                          never silently counted
  AcceptanceFromMetrics   the /metrics delta becomes an acceptance rate, and an
                          engine with no /metrics reports absent rather than 0
  CorpusIsDeterministic   the corpus is a function of (sources, seed, weights)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "benchmarks" / "variadic"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


client = _load("variadic_client", HARNESS / "client.py")
report = _load("variadic_report", HARNESS / "report.py")


class MockHandler(BaseHTTPRequestHandler):
    """One OpenAI streaming endpoint, with the two knobs that matter."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):            # silence the default stderr spam
        return

    def do_GET(self):                                       # noqa: N802
        cfg = self.server.cfg
        if self.path != "/metrics" or not cfg["metrics"]:
            self.send_error(404)
            return
        with self.server.lock:
            drafted = self.server.drafted
            accepted = self.server.accepted
        body = (
            "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
            f'vllm:spec_decode_num_draft_tokens_total{{model="m"}} {drafted}\n'
            "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
            f'vllm:spec_decode_num_accepted_tokens_total{{model="m"}} {accepted}\n'
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                      # noqa: N802
        cfg = self.server.cfg
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        with self.server.lock:
            self.server.inflight += 1
            self.server.peak = max(self.server.peak, self.server.inflight)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            n_tok = cfg["tokens"]
            per = cfg["tokens_per_chunk"]
            time.sleep(cfg["prefill_s"])
            emitted = 0
            while emitted < n_tok:
                k = min(per, n_tok - emitted)
                time.sleep(cfg["per_token_s"] * k)
                emitted += k
                self._sse({"choices": [{"delta": {"content": "ab " * k}}]})
            usage = {"prompt_tokens": cfg["prompt_tokens"],
                     "completion_tokens": n_tok}
            if cfg["accepted"]:
                usage["accepted_draft_tokens"] = cfg["accepted"]
            self._sse({"choices": [], "usage": usage} if cfg["usage"]
                      else {"choices": []})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            with self.server.lock:
                self.server.drafted += cfg["drafted"]
                self.server.accepted += cfg["accepted"]
        finally:
            with self.server.lock:
                self.server.inflight -= 1

    def _sse(self, obj):
        payload = ("data: " + json.dumps(obj) + "\n\n").encode()
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()


def _stop(srv):
    srv.shutdown()
    srv.server_close()


def start_mock(**over):
    cfg = {"tokens": 8, "tokens_per_chunk": 1, "per_token_s": 0.01,
           "prefill_s": 0.02, "prompt_tokens": 100, "usage": True,
           "metrics": True, "drafted": 10, "accepted": 6}
    cfg.update(over)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    srv.cfg = cfg
    srv.lock = threading.Lock()
    srv.inflight = 0
    srv.peak = 0
    srv.drafted = 0
    srv.accepted = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def run_leg(url, tmp, label, concurrency, num_prompts, warmup, arm="OURS",
            corpus=None):
    corpus_path = corpus or make_corpus(tmp, num_prompts + warmup + 2)
    out = Path(tmp) / f"{label}.json"
    rc = subprocess.run(
        [sys.executable, str(HARNESS / "client.py"), "--url", url,
         "--dataset", str(corpus_path), "--num-prompts", str(num_prompts),
         "--warmup", str(warmup), "--concurrency", str(concurrency),
         "--max-tokens", "8", "--label", label, "--arm", arm,
         "--round", "1", "--out", str(out)],
        capture_output=True, text=True, timeout=300)
    return rc, json.loads(out.read_text()) if out.exists() else None


def make_corpus(tmp, n):
    path = Path(tmp) / "corpus.json"
    path.write_text(json.dumps([
        {"id": f"X-{i:04d}", "band": "S",
         "conversations": [{"from": "human", "value": f"question {i}"},
                           {"from": "gpt", "value": ""}]}
        for i in range(n)]))
    return path


class PercentileContract(unittest.TestCase):
    """Linear interpolation between order statistics, numpy's default."""

    def test_matches_hand_computed_linear_interpolation(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        # idx = (n-1)*p/100 = 3*0.5 = 1.5 -> 2.0 + (3.0-2.0)*0.5
        self.assertAlmostEqual(report.percentile(xs, 50), 2.5)
        self.assertAlmostEqual(report.percentile(xs, 0), 1.0)
        self.assertAlmostEqual(report.percentile(xs, 100), 4.0)
        # idx = 3*0.9 = 2.7 -> 3.0 + (4.0-3.0)*0.7
        self.assertAlmostEqual(report.percentile(xs, 90), 3.7)

    def test_a_nearest_rank_implementation_would_fail_this(self):
        """The mutation this case exists to catch."""
        xs = [1.0, 2.0, 3.0, 4.0]
        nearest_rank = sorted(xs)[min(len(xs) - 1, int(len(xs) * 0.9))]
        self.assertNotAlmostEqual(report.percentile(xs, 90), nearest_rank)

    def test_single_sample_and_empty(self):
        self.assertEqual(report.percentile([7.0], 99), 7.0)
        self.assertTrue(report.percentile([], 50) != report.percentile([], 50))


class ClosedLoopConcurrency(unittest.TestCase):
    """c workers overlap; c=1 never does."""

    def test_concurrency_one_never_overlaps(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "c1", 1, 6, 2)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        self.assertEqual(srv.peak, 1, "c=1 put more than one request in flight")

    def test_concurrency_four_really_overlaps(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "c4", 4, 12, 4)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        self.assertEqual(srv.peak, 4,
                         f"c=4 peaked at {srv.peak} concurrent requests")


class WarmupIsExcluded(unittest.TestCase):
    def test_measured_window_holds_no_warmup_request(self):
        srv, url = start_mock()
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "w", 2, 6, 3)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        phases = [r["phase"] for r in leg["records"]]
        self.assertEqual(phases.count("warmup"), 3)
        self.assertEqual(phases.count("measured"), 6)
        measured = report.ok_records(leg, "measured")
        self.assertTrue(all(r["phase"] == "measured" for r in measured))
        # And the with-warmup view still sees every request.
        self.assertEqual(len(report.ok_records(leg, None)), 9)

    def test_measured_wall_clock_excludes_the_warmup_phase(self):
        srv, url = start_mock(per_token_s=0.02)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "w2", 1, 4, 4)
        _stop(srv)
        s = leg["summary"]
        self.assertGreater(s["warm_wall_s"], 0.0)
        # 4 warm + 4 measured at c=1; the measured wall must be about half the
        # total, not all of it.
        self.assertLess(s["measured_wall_s"],
                        0.75 * (s["warm_wall_s"] + s["measured_wall_s"]))


class ChunkingIsRecovered(unittest.TestCase):
    """Two engines, one token per chunk and four, must not be compared raw."""

    def _leg(self, tmp, per_chunk, label, arm):
        srv, url = start_mock(tokens=16, tokens_per_chunk=per_chunk,
                              per_token_s=0.01, prefill_s=0.05)
        rc, leg = run_leg(url, tmp, label, 1, 6, 2, arm=arm)
        _stop(srv)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        return leg

    def test_tokens_per_chunk_comes_from_usage_not_from_the_chunk_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            fine = self._leg(tmp, 1, "fine", "THEIRS")
            coarse = self._leg(tmp, 4, "coarse", "OURS")
        a_fine = report.axes(report.ok_records(fine, "measured"))
        a_coarse = report.axes(report.ok_records(coarse, "measured"))
        self.assertAlmostEqual(sum(a_fine["toks_per_chunk"])
                               / len(a_fine["toks_per_chunk"]), 1.0, places=6)
        self.assertAlmostEqual(sum(a_coarse["toks_per_chunk"])
                               / len(a_coarse["toks_per_chunk"]), 4.0, places=6)
        # Both engines produced the same 16 tokens, so a chunk count would have
        # said one produced a quarter of the other.
        self.assertEqual(sum(a_fine["out_tokens"]), sum(a_coarse["out_tokens"]))

    def test_the_coarse_engine_pays_for_its_chunking_in_raw_ttft(self):
        with tempfile.TemporaryDirectory() as tmp:
            fine = self._leg(tmp, 1, "fine", "THEIRS")
            coarse = self._leg(tmp, 4, "coarse", "OURS")
        a_fine = report.axes(report.ok_records(fine, "measured"))
        a_coarse = report.axes(report.ok_records(coarse, "measured"))
        raw_f = report.percentile(a_fine["ttft"], 50)
        raw_c = report.percentile(a_coarse["ttft"], 50)
        cor_c = report.percentile(a_coarse["ttft_corrected"], 50)
        self.assertGreater(raw_c, raw_f,
                           "the coarse engine should look slower to first CHUNK")
        self.assertLess(cor_c, raw_c,
                        "the correction must remove chunking, not add to it")
        # It must not over-correct past the fine engine's own prefill.
        self.assertGreater(cor_c, 0.0)

    def test_first_chunk_tokens_tracks_the_servers_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            coarse = self._leg(tmp, 4, "coarse", "OURS")
        a = report.axes(report.ok_records(coarse, "measured"))
        est = sum(a["first_chunk_tokens"]) / len(a["first_chunk_tokens"])
        self.assertGreater(est, 2.0)
        self.assertLess(est, 6.0)


class UsageIsMandatory(unittest.TestCase):
    def test_a_server_without_usage_is_marked_chunks_not_counted_silently(self):
        srv, url = start_mock(usage=False)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "nousage", 1, 4, 2)
        _stop(srv)
        a = report.axes(report.ok_records(leg, "measured"))
        self.assertEqual(a["counted_by"], "chunks")
        self.assertEqual(a["n"], 0, "a leg with no usage must yield no axis")


class AcceptanceFromMetrics(unittest.TestCase):
    def test_the_metrics_delta_becomes_a_rate(self):
        srv, url = start_mock(drafted=10, accepted=6)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "acc", 1, 5, 2)
        _stop(srv)
        acc = report.acceptance(leg)
        self.assertEqual(acc["source"], "/metrics")
        # Five measured requests between the two scrapes.
        self.assertAlmostEqual(acc["draft"], 50.0)
        self.assertAlmostEqual(acc["accepted"], 30.0)
        self.assertAlmostEqual(acc["rate"], 0.6)

    def test_an_engine_without_metrics_falls_back_to_usage(self):
        srv, url = start_mock(metrics=False, accepted=7)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "theirs", 1, 5, 2, arm="THEIRS")
        _stop(srv)
        acc = report.acceptance(leg)
        self.assertEqual(acc["source"], "usage")
        self.assertEqual(acc["accepted"], 35)

    def test_an_engine_exposing_nothing_reports_absent_not_zero(self):
        srv, url = start_mock(metrics=False, accepted=0)
        with tempfile.TemporaryDirectory() as tmp:
            rc, leg = run_leg(url, tmp, "silent", 1, 4, 2, arm="THEIRS")
        _stop(srv)
        acc = report.acceptance(leg)
        self.assertEqual(acc["source"], "absent")
        self.assertNotIn("rate", acc)


class CorpusIsDeterministic(unittest.TestCase):
    """A published length histogram has to be reproducible from the seed."""

    def _sources(self, tmp):
        gsm = Path(tmp) / "gsm.jsonl"
        gsm.write_text("".join(
            json.dumps({"question": "q " * (5 + i % 20)}) + "\n"
            for i in range(200)))
        he = Path(tmp) / "he.jsonl"
        he.write_text("".join(
            json.dumps({"prompt": "def f%d():\n    pass\n" % i + "x " * (10 + i)})
            + "\n" for i in range(164)))
        son = Path(tmp) / "sonnet.txt"
        son.write_text("".join(f"line {i} of the verse here\n" for i in range(517)))
        return gsm, he, son

    def _build(self, tmp, out, man, seed=0, count=40):
        gsm, he, son = self._sources(tmp)
        rc = subprocess.run(
            [sys.executable, str(HARNESS / "build_corpus.py"),
             "--gsm8k", str(gsm), "--humaneval", str(he), "--sonnet", str(son),
             "--count", str(count), "--seed", str(seed),
             "--out", str(out), "--manifest", str(man)],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        return json.loads(Path(man).read_text())

    def test_same_seed_gives_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man")
            b = self._build(tmp, Path(tmp) / "b.json", Path(tmp) / "b.man")
        self.assertEqual(a["corpus_sha256"], b["corpus_sha256"])

    def test_a_different_seed_gives_different_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man")
            c = self._build(tmp, Path(tmp) / "c.json", Path(tmp) / "c.man", seed=1)
        self.assertNotEqual(a["corpus_sha256"], c["corpus_sha256"])

    def test_band_counts_sum_to_the_requested_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=137)
        self.assertEqual(sum(man["band_counts"].values()), 137)
        self.assertEqual(sum(v["n"] for v in man["realised_chars"].values()), 137)

    def test_every_source_is_pinned_by_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man")
        for name, src in man["sources"].items():
            self.assertEqual(len(src["sha256"]), 64, name)

    def test_the_bands_are_ordered_by_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            man = self._build(tmp, Path(tmp) / "a.json", Path(tmp) / "a.man",
                              count=200)
        med = {b: v["median"] for b, v in man["realised_chars"].items()}
        self.assertLess(med["S"], med["L"])
        self.assertLess(med["L"], med["XL"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
