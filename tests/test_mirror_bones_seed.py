"""Transport failure probes; these do not establish a live Catalog traversal."""
from contextlib import contextmanager
import threading
import time
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

spec = importlib.util.spec_from_file_location("mirror_bones", Path(__file__).parents[1] / "tools/mirror_bones_seed.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def plan_for(data, algorithm="sha256", name="archive.bin"):
    return {"repo_id": m.REPO, "revision": m.REVISION, "license": {"private": True}, "objects": [{
        "path": name, "size_bytes": len(data), "content_sha256": hashlib.sha256(data).hexdigest() if algorithm == "sha256" else None,
        "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest(), "integrity_algorithm": algorithm}]}


class Source:
    def __init__(self, data):
        self.data, self.calls = data, []

    @contextmanager
    def worker(self):
        yield self

    def download(self, item, offset, size, target):
        self.calls.append(offset)
        target.write_bytes(self.data[offset:offset+size])


class Store:
    def __init__(self):
        self.objects, self.puts, self.fetches, self.published = {}, [], [], None
        self.fail_at = None

    @contextmanager
    def worker(self):
        yield self

    def put(self, path, sha, size, state, save):
        if self.fail_at == len(self.puts):
            raise ConnectionError("interrupted")
        m.check_bytes(path, size, sha)
        self.objects[sha] = path.read_bytes()
        self.puts.append(sha)
        state["registered"] = True
        save()

    def fetch(self, sha, size, target):
        self.fetches.append(sha)
        target.write_bytes(self.objects[sha])
        m.check_bytes(target, size, sha)

    def publish(self, manifest, chunks):
        self.published = json.loads(manifest)
        return hashlib.sha256(manifest).hexdigest()

    def read_member(self, sha, target):
        target.write_bytes(self.objects[sha])
        return target


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        quiet = patch.object(m, "progress"); quiet.start(); self.addCleanup(quiet.stop)
        self.data = b"abcdefghijklmnopqrstuvwxyz"
        self.pin(plan_for(self.data))

    def pin(self, plan):
        fixture = patch.object(m, "EXPECTED_INVENTORY", m.inventory_identity(plan["objects"]))
        fixture.start(); self.addCleanup(fixture.stop)

    def run_mirror(self, store=None, source=None, plan=None):
        store = store or Store()
        source = source or Source(self.data)
        plan = plan or plan_for(self.data)
        self.pin(plan)
        result = m.mirror(plan, store, source, self.root/"stage", "project", "pack", chunk_size=8)
        return store, source, result

    def test_byte_exact_restore_without_hf_and_git_blob_verification(self):
        store, _, receipt = self.run_mirror(plan=plan_for(self.data, "git-blob-sha1", "nested/README.md"))
        self.assertEqual(m.restore(store, receipt["manifest_sha256"], self.root/"restored"), 1)
        self.assertEqual((self.root/"restored/nested/README.md").read_bytes(), self.data)
        self.assertFalse(store.published["training_admitted"])
        self.assertLessEqual(max(c["size_bytes"] for c in store.published["objects"][0]["chunks"]), 8)

    def test_resume_does_not_upload_completed_chunks_and_rehashes_remote_bytes(self):
        store, source = Store(), Source(self.data)
        store.fail_at = 1
        with self.assertRaises(ConnectionError):
            self.run_mirror(store, source)
        first = store.puts[0]
        store.fail_at = None
        self.run_mirror(store, source)
        self.assertEqual(store.puts.count(first), 1)
        self.assertIn(first, store.fetches)
        self.assertEqual(source.calls.count(0), 1)

    def test_resume_detects_remote_corruption_before_publish(self):
        store = Store(); store.fail_at = 1
        with self.assertRaises(ConnectionError):
            self.run_mirror(store)
        store.objects[store.puts[0]] = b"corrupt!"
        store.fail_at = None
        with self.assertRaises(m.MirrorError):
            self.run_mirror(store)
        self.assertIsNone(store.published)

    def test_bad_whole_source_never_publishes_final_pack(self):
        store = Store()
        for algorithm in ("sha256", "git-blob-sha1"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as directory:
                self.pin(plan_for(self.data, algorithm))
                with self.assertRaises(m.MirrorError):
                    m.mirror(plan_for(self.data, algorithm), store, Source(b"!"*len(self.data)),
                             Path(directory).resolve(), "project", "pack", 8)
                self.assertIsNone(store.published)

    def test_retained_corrupt_upload_chunk_is_not_reuploaded(self):
        store = Store(); store.fail_at = 0
        with self.assertRaises(ConnectionError):
            self.run_mirror(store)
        p = next((self.root/"stage").glob("*.chunk")); p.write_bytes(b"!"*8)
        store.fail_at = None
        # HTTP source resumes a full retained chunk without overwriting it.
        class Existing(Source):
            def download(self, item, offset, size, target):
                pass
        with self.assertRaises(m.MirrorError):
            self.run_mirror(store, Existing(self.data))
        self.assertEqual(store.puts, [])

    def test_manifest_refuses_paths_offsets_unknown_schema_and_algorithm(self):
        store, _, _ = self.run_mirror()
        mutations = [lambda x: x.update(schema="unknown"),
                     lambda x: x["objects"][0].update(path="../escape"),
                     lambda x: x["objects"][0].update(path=".mirror-restore/manifest.json"),
                     lambda x: x["objects"][0].update(integrity_algorithm="md5"),
                     lambda x: x["objects"][0]["chunks"][1].update(offset=7),
                     lambda x: x["objects"][0]["chunks"][0].update(sha256="0"),
                     lambda x: x["objects"].append(copy.deepcopy(x["objects"][0]))]
        for mutate in mutations:
            value = copy.deepcopy(store.published); mutate(value)
            with self.assertRaises(m.MirrorError):
                m.parse_manifest(value)

    def test_manifest_rejects_empty_or_same_size_substituted_inventory(self):
        store, _, _ = self.run_mirror()
        for mutate in (lambda x: x.update(objects=[]),
                       lambda x: x["objects"][0].update(path="different.bin"),
                       lambda x: x["objects"][0].update(git_blob_sha1="0"*40),
                       lambda x: x["objects"][0].update(content_sha256="0"*64)):
            value = copy.deepcopy(store.published); mutate(value)
            with self.assertRaises(m.MirrorError): m.parse_manifest(value)

    def interrupted_restore(self):
        store, _, receipt = self.run_mirror()
        original = store.read_member
        calls = []
        second = store.published["objects"][0]["chunks"][1]["sha256"]
        def read(sha, target):
            calls.append(sha)
            if sha == second: raise ConnectionError("interrupted restore")
            return original(sha, target)
        store.read_member = read
        output = self.root/"restore"
        with self.assertRaises(ConnectionError): m.restore(store, receipt["manifest_sha256"], output)
        store.read_member = lambda sha, target: (calls.append(sha), original(sha, target))[1]
        return store, receipt, output, calls

    def test_restore_resume_verifies_prefix_and_skips_completed_chunks(self):
        store, receipt, output, calls = self.interrupted_restore()
        first = store.published["objects"][0]["chunks"][0]["sha256"]
        partial = next((output/".mirror-restore").glob("source-*.partial"))
        self.assertEqual(partial.read_bytes(), self.data[:8])
        # An interrupted incomplete append is discarded after verifying the prefix.
        with partial.open("ab") as stream: stream.write(b"partial")
        m.restore(store, receipt["manifest_sha256"], output)
        self.assertEqual(calls.count(first), 1)
        self.assertEqual((output/"archive.bin").read_bytes(), self.data)

    def test_restore_refuses_corrupt_committed_prefix(self):
        store, receipt, output, calls = self.interrupted_restore()
        partial = next((output/".mirror-restore").glob("source-*.partial"))
        partial.write_bytes(b"!"*8)
        with self.assertRaises(m.MirrorError): m.restore(store, receipt["manifest_sha256"], output)
        self.assertFalse((output/"archive.bin").exists())

    def test_restore_journal_is_bound_to_manifest_identity(self):
        store, receipt, output, calls = self.interrupted_restore()
        value = copy.deepcopy(store.published); value["license"] = {"changed": True}
        data = m.encode(value); sha = hashlib.sha256(data).hexdigest(); store.objects[sha] = data
        with self.assertRaises(m.MirrorError): m.restore(store, sha, output)

    def test_symlink_escape_and_raw_scratch_refused(self):
        elsewhere = self.root/"elsewhere"; elsewhere.mkdir()
        (self.root/"link").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(m.MirrorError):
            m.under(self.root, "link/escape")
        with self.assertRaises(m.MirrorError):
            with m.workspace(self.root/"raw/stage"):
                pass

    def test_changed_project_checkpoint_refused(self):
        self.run_mirror()
        with self.assertRaises(m.MirrorError):
            m.mirror(plan_for(self.data), Store(), Source(self.data), self.root/"stage", "other", "pack", 8)

    def test_restore_does_not_replace_existing_mismatched_file(self):
        store, _, receipt = self.run_mirror()
        output = self.root/"restore"; output.mkdir()
        target = output/"archive.bin"; target.write_bytes(b"mine")
        with self.assertRaises(m.MirrorError):
            m.restore(store, receipt["manifest_sha256"], output)
        self.assertEqual(target.read_bytes(), b"mine")


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = {f"file-{i}.bin": bytes([i])*8+bytes([i+100])*8 for i in range(8)}
        self.plan = {"repo_id": m.REPO, "revision": m.REVISION, "license": {"private": True},
                     "objects": [plan_for(data, name=name)["objects"][0] for name, data in self.data.items()]}
        for name, value in (("EXPECTED_INVENTORY", m.inventory_identity(self.plan["objects"])), ("progress", lambda *a, **k: None)):
            guard = patch.object(m, name, value); guard.start(); self.addCleanup(guard.stop)

    def source(self, *, barrier=True, fail=False, corrupt=False):
        data = self.data
        class Factory:
            def __init__(self):
                self.lock = threading.Lock(); self.calls = []; self.active = self.maximum = self.clients = 0
                self.barrier = threading.Barrier(4) if barrier else None
                self.fail = fail
            @contextmanager
            def worker(self):
                owner = threading.get_ident()
                with self.lock: self.clients += 1
                factory = self
                class Client:
                    def download(self, item, offset, size, target):
                        if owner != threading.get_ident(): raise AssertionError("shared client")
                        with factory.lock:
                            factory.calls.append((item["path"], offset))
                            factory.active += 1
                            factory.maximum = max(factory.maximum, factory.active)
                        try:
                            if factory.barrier is not None and offset == 0 and int(item["path"][5]) < 4:
                                factory.barrier.wait(timeout=5)
                            if factory.fail and item["path"] == "file-0.bin" and offset == 8:
                                raise ConnectionError("one interrupted file")
                            if factory.fail and item["path"] != "file-0.bin":
                                time.sleep(0.03)  # keep the other three chunks in flight at failure
                            # Different completion order must not affect manifest identity.
                            time.sleep((3-int(item["path"][5])%4)*0.001)
                            content = data[item["path"]][offset:offset+size]
                            if corrupt and item["path"] == "file-0.bin": content = b"!"*size
                            target.write_bytes(content)
                        finally:
                            with factory.lock: factory.active -= 1
                yield Client()
        return Factory()

    def test_four_isolated_workers_merge_journal_and_keep_manifest_order(self):
        source = self.source(); store = Store()
        receipt = m.mirror(self.plan, store, source, self.root/"parallel", "project", "pack", 8)
        self.assertEqual(source.maximum, 4); self.assertEqual(source.clients, 4)
        self.assertEqual([x["path"] for x in store.published["objects"]], list(self.data))
        journal = m.document(self.root/"parallel/journal.json")
        self.assertEqual(len(journal["chunks"]), 17)
        self.assertTrue(all(x["registered"] for x in journal["chunks"].values()))
        with patch.object(m, "WORKERS", 1):
            serial = m.mirror(self.plan, Store(), self.source(barrier=False), self.root/"serial", "project", "pack", 8)
        self.assertEqual(receipt["manifest_sha256"], serial["manifest_sha256"])

    def test_failed_worker_retains_progress_resumes_without_reupload_and_never_publishes_partial(self):
        store = Store(); source = self.source(fail=True)
        with self.assertRaises(ConnectionError):
            m.mirror(self.plan, store, source, self.root/"stage", "project", "pack", 8)
        self.assertIsNone(store.published)
        self.assertNotIn("file-7.bin", {name for name, _ in source.calls})
        before = list(store.puts)
        self.assertTrue(before)
        m.mirror(self.plan, store, self.source(barrier=False), self.root/"stage", "project", "pack", 8)
        self.assertTrue(all(store.puts.count(sha) == 1 for sha in before))
        self.assertEqual(len(store.published["objects"]), 8)

    def test_largest_files_start_first_but_manifest_keeps_inventory_order(self):
        for index, name in enumerate(self.data):
            self.data[name] = bytes([index]) * (index+1)*8
        self.plan["objects"] = [plan_for(data, name=name)["objects"][0] for name, data in self.data.items()]
        source, store = self.source(barrier=False), Store()
        with patch.object(m, "EXPECTED_INVENTORY", m.inventory_identity(self.plan["objects"])), patch.object(m, "WORKERS", 1):
            m.mirror(self.plan, store, source, self.root/"largest", "project", "pack", 8)
        starts = [name for name, offset in source.calls if offset == 0]
        self.assertEqual(starts, list(reversed(self.data)))
        self.assertEqual([x["path"] for x in store.published["objects"]], list(self.data))

    def test_failure_event_precedes_settling_other_workers_and_omits_exception_text(self):
        source, store, events = self.source(fail=True), Store(), []
        def progress(event, **facts):
            if event == "worker_failed": events.append((facts, source.active))
        with patch.object(m, "progress", progress), self.assertRaises(ConnectionError):
            m.mirror(self.plan, store, source, self.root/"failure", "project", "pack", 8)
        self.assertEqual(events[0][0], {"error_type": "ConnectionError", "path": "file-0.bin", "phase": "source_download"})
        self.assertGreater(events[0][1], 0)
        self.assertNotIn("one interrupted file", json.dumps(events))
        self.assertIsNone(store.published)

    def test_concurrent_corrupt_source_refuses_final_publication(self):
        store = Store()
        with self.assertRaisesRegex(m.MirrorError, "Whole source SHA-256 mismatch"):
            m.mirror(self.plan, store, self.source(corrupt=True), self.root/"stage", "project", "pack", 8)
        self.assertIsNone(store.published)
        self.assertFalse((self.root/"stage/receipt.json").exists())


class PinnedInventoryTests(unittest.TestCase):
    def test_checked_plan_matches_trusted_inventory_and_substitution_is_refused(self):
        path = Path(__file__).parents[1]/"data/bones_seed/mirror-plan.json"
        if not path.exists(): self.skipTest("Private pinned inventory is not checked into git")
        value = m.read_plan(path)
        self.assertEqual(m.inventory_identity(value["objects"]), m.EXPECTED_INVENTORY)
        changed = copy.deepcopy(value["objects"])
        changed[0]["path"] = "same-size-substitute"
        with self.assertRaises(m.MirrorError): m.require_inventory(changed)


class FailureLogTests(unittest.TestCase):
    def test_status_only_comes_from_structured_cause_and_events_have_epoch(self):
        cause = Exception("https://secret.example/signed?token=secret")
        cause.response = SimpleNamespace(status_code=429)
        error = RuntimeError("private error"); error.__cause__ = cause
        self.assertEqual(m.failure_facts(error), {"error_type": "RuntimeError", "http_status": 429})
        cause.response.status_code = "503"
        self.assertEqual(m.failure_facts(error), {"error_type": "RuntimeError"})
        import io
        output = io.StringIO()
        with patch("sys.stdout", output), patch.object(m.time, "time", return_value=1700000000):
            m.progress("worker_failed", phase="catalog_result_read", **m.failure_facts(error))
        entry = json.loads(output.getvalue())
        self.assertEqual(entry["utc_epoch_seconds"], 1700000000)
        self.assertNotIn("secret", output.getvalue())


class HTTPTests(unittest.TestCase):
    def test_partial_range_redirect_strips_auth(self):
        seen = []
        def handle(req):
            seen.append(req)
            if req.url.host == "huggingface.co":
                return httpx.Response(302, headers={"location": "https://storage.example/object"})
            return httpx.Response(206, headers={"content-range": "bytes 5-7/10"}, stream=httpx.ByteStream(b"fgh"))
        client = httpx.Client(transport=httpx.MockTransport(handle))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()/"chunk"; target.write_bytes(b"de")
            m.HuggingFaceSource("secret", client).download({"path": "a", "size_bytes": 10}, 3, 5, target)
            self.assertEqual(target.read_bytes(), b"defgh")
        self.assertEqual(seen[0].headers["range"], "bytes=5-7")
        self.assertEqual(seen[0].headers["authorization"], "Bearer secret")
        self.assertNotIn("authorization", seen[1].headers)

    def test_ignored_or_wrong_range_does_not_append(self):
        for status, content_range in [(200, "bytes 1-2/3"), (206, "bytes 0-2/3")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                target = Path(directory).resolve()/"chunk"; target.write_bytes(b"a")
                client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(status,
                    headers={"content-range": content_range}, content=b"bc")))
                with self.assertRaises(m.MirrorError):
                    m.HuggingFaceSource("secret", client).download({"path": "a", "size_bytes": 3}, 0, 3, target)
                self.assertEqual(target.read_bytes(), b"a")

    def test_fresh_complete_file_accepts_200_only_with_exact_length(self):
        for length in ("3", "4", None):
            with self.subTest(length=length), tempfile.TemporaryDirectory() as directory:
                headers = {} if length is None else {"content-length": length}
                client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200,
                    headers=headers, stream=httpx.ByteStream(b"abc"))))
                path = Path(directory).resolve()/"chunk"
                if length == "3":
                    m.HuggingFaceSource("secret", client).download({"path": "a", "size_bytes": 3}, 0, 3, path)
                    self.assertEqual(path.read_bytes(), b"abc")
                else:
                    with self.assertRaises(m.MirrorError):
                        m.HuggingFaceSource("secret", client).download({"path": "a", "size_bytes": 3}, 0, 3, path)

    def test_trickle_total_deadline(self):
        class Slow(httpx.SyncByteStream):
            def __iter__(self):
                yield b"a"
        client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(206,
            headers={"content-range": "bytes 0-0/1"}, stream=Slow())))
        with tempfile.TemporaryDirectory() as directory, patch.object(m.time, "monotonic", side_effect=[0, 1801]):
            with self.assertRaises(m.MirrorError):
                m.HuggingFaceSource("secret", client).download({"path": "a", "size_bytes": 1}, 0, 1, Path(directory).resolve()/"chunk")


class SDKAdapterTests(unittest.TestCase):
    def test_catalog_workers_own_and_close_independent_clients(self):
        opened, closed = [], []
        @contextmanager
        def factory():
            catalog = object(); opened.append(catalog)
            try: yield catalog
            finally: closed.append(catalog)
        parent = m.CatalogStore(object(), "pack", catalog_factory=factory)
        with parent.worker() as first, parent.worker() as second:
            self.assertIsNot(first.catalog, second.catalog)
            self.assertIsNot(first.transport, second.transport)
            self.assertIsNot(first.catalog, parent.catalog)
            self.assertIs(first.catalog_factory, factory)
        self.assertEqual(closed, list(reversed(opened)))

    def test_saved_operation_reads_reattach_boundedly_with_fresh_clients(self):
        try:
            from sx_catalog_client.client import CatalogUnavailableError
        except (ImportError, AttributeError): self.skipTest("Run with installed sentientx Python")
        operation, seen, created = object(), [], []
        class Client:
            def result(self, ref, **kwargs):
                seen.append((self, ref, kwargs["timeout_s"]))
                if len(seen) < 3: raise CatalogUnavailableError("secret signed-url must not appear")
                return "verified"
        @contextmanager
        def factory():
            client = Client(); created.append(client); yield client
        store = m.CatalogStore(Client(), "pack", catalog_factory=factory)
        with patch.object(m.time, "sleep") as sleep, patch.object(m.time, "monotonic", side_effect=[0, 0, 2, 7]), patch.object(m, "progress") as log:
            self.assertEqual(store.operation_result(operation), "verified")
        self.assertEqual(len(created), 2)
        self.assertEqual([entry[1] for entry in seen], [operation]*3)
        self.assertEqual([entry[2] for entry in seen], [1800, 1798, 1793])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 5])
        self.assertNotIn("secret", str(log.call_args_list))

    def test_operation_read_reattachment_exhausts_and_never_retries_terminal_errors(self):
        try:
            from sx_catalog_client.client import CatalogUnavailableError, CatalogPermissionError, CatalogRegistrationFailed
        except (ImportError, AttributeError): self.skipTest("Run with installed sentientx Python")
        for error_type, expected_calls in ((CatalogUnavailableError, 3), (CatalogPermissionError, 1), (CatalogRegistrationFailed, 1)):
            calls = []
            class Client:
                def result(self, operation, **kwargs):
                    calls.append(operation); raise error_type("stop")
            @contextmanager
            def factory(): yield Client()
            with self.subTest(error=error_type), patch.object(m.time, "sleep"), patch.object(m, "progress"):
                with self.assertRaises(error_type):
                    m.CatalogStore(Client(), "pack", catalog_factory=factory).operation_result(object())
            self.assertEqual(len(calls), expected_calls)

    def test_transient_part_response_refreshes_grant_and_replays_identical_range(self):
        import io
        from uuid import UUID
        grants, payloads = [], []
        def grant(uid, number):
            grants.append((uid, number))
            return SimpleNamespace(url=f"https://private.example/part?secret={len(grants)}")
        def handle(request):
            payloads.append(request.content)
            return httpx.Response(503 if len(payloads) == 1 else 200,
                                  headers={"etag": "part-etag"}, content=b"private server details")
        store = m.CatalogStore(SimpleNamespace(staging_upload_part=grant), "pack",
                               httpx.Client(transport=httpx.MockTransport(handle)))
        uid = UUID(int=1)
        with patch.object(m.time, "sleep") as sleep, patch.object(m, "progress") as log:
            etag = store.upload_part(io.BytesIO(b"prePAYLOADsuffix"), uid, 15, 3, 7)
        self.assertEqual(etag, "part-etag")
        self.assertEqual(payloads, [b"PAYLOAD", b"PAYLOAD"])
        self.assertEqual(grants, [(uid, 15), (uid, 15)])
        sleep.assert_called_once_with(2)
        facts = log.call_args.kwargs
        self.assertEqual((facts["http_status"], facts["part_number"]), (503, 15))
        self.assertNotIn("secret", str(log.call_args_list))
        self.assertNotIn("private server", str(log.call_args_list))

    def test_nontransient_part_response_stops_without_retry(self):
        import io
        calls = []
        def handle(request):
            calls.append(request); return httpx.Response(403, content=b"private response")
        store = m.CatalogStore(SimpleNamespace(staging_upload_part=lambda *args:
            SimpleNamespace(url="https://private.example/part")), "pack",
            httpx.Client(transport=httpx.MockTransport(handle)))
        with patch.object(m.time, "sleep") as sleep:
            with self.assertRaises(m.MultipartTransferError) as raised:
                store.upload_part(io.BytesIO(b"data"), "upload", 15, 0, 4)
        self.assertEqual(len(calls), 1); sleep.assert_not_called()
        self.assertEqual(m.failure_facts(raised.exception), {"error_type": "MultipartTransferError",
            "reason": "multipart_http_failure", "http_status": 403, "part_number": 15})
        self.assertNotIn("private", str(raised.exception))

    def test_part_retry_preserves_saved_etags_and_never_restarts_upload_or_registration(self):
        try:
            from sx_catalog_client.client import InitiatedStagingUpload
            from sx_contracts.operations import OperationRef
        except (ImportError, AttributeError): self.skipTest("Run with installed sentientx Python")
        from uuid import UUID
        uid = UUID(int=1)
        sha = hashlib.sha256(b"abcdefgh").hexdigest()
        for statuses in ((503, 200), (503, 503, 503)):
            begins, grants, payloads, registrations, completions = [], [], [], [], []
            state = {"parts": {"1": "retained-etag"}}
            class Catalog:
                def begin_staging_upload(self, *args):
                    begins.append(args); return InitiatedStagingUpload(uid, "s3://private/staging", 4, 60)
                def staging_upload_part(self, upload, number):
                    grants.append((upload, number)); return SimpleNamespace(url="https://private.example/part")
                def complete_staging_upload(self, upload, **kwargs):
                    completions.append(kwargs["parts"]); return SimpleNamespace(source_uri="s3://private/staging")
                def register(self, request):
                    registrations.append(request)
                    return SimpleNamespace(ref=OperationRef.parse("experience/asset-registration/id"), resource_ref="asset:"+sha)
                def result(self, *args, **kwargs): return sha
                def register_asset_pack(self, *args): pass
            def handle(request):
                payloads.append(request.content)
                return httpx.Response(statuses[len(payloads)-1], headers={"etag": "new-etag"})
            store = m.CatalogStore(Catalog(), "pack", httpx.Client(transport=httpx.MockTransport(handle)))
            with self.subTest(statuses=statuses), tempfile.TemporaryDirectory() as directory, patch.object(m.time, "sleep") as sleep, patch.object(m, "progress"):
                path = Path(directory).resolve()/"chunk"; path.write_bytes(b"abcdefgh")
                if statuses[-1] == 200:
                    store.put(path, sha, 8, state, lambda: None)
                    self.assertEqual(state["parts"], {"1": "retained-etag", "2": "new-etag"})
                    self.assertEqual([(p.part_number, p.etag) for p in completions[0]], [(1, "retained-etag"), (2, "new-etag")])
                    self.assertEqual(len(registrations), 1)
                else:
                    with self.assertRaises(m.MultipartTransferError): store.put(path, sha, 8, state, lambda: None)
                    self.assertEqual(state["parts"], {"1": "retained-etag"})
                    self.assertNotIn("operation", state)
                    self.assertEqual(registrations, []); self.assertEqual(completions, [])
                    self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 5])
                self.assertEqual(len(begins), 1)
                self.assertEqual(grants, [(uid, 2)]*len(statuses))
                self.assertEqual(payloads, [b"efgh"]*len(statuses))

    def test_non_https_upload_grant_is_refused_before_bytes_leave(self):
        try:
            from sx_catalog_client.client import InitiatedStagingUpload
        except (ImportError, AttributeError):
            self.skipTest("Run with installed sentientx Python")
        from uuid import UUID
        catalog = SimpleNamespace(
            begin_staging_upload=lambda *args: InitiatedStagingUpload(UUID(int=1), "s3://private/staging", 4, 60),
            staging_upload_part=lambda *args: SimpleNamespace(url="http://storage.example/part"))
        transmitted = []
        client = httpx.Client(transport=httpx.MockTransport(lambda req: transmitted.append(req)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/"chunk"; path.write_bytes(b"data")
            with self.assertRaises(m.MirrorError):
                m.CatalogStore(catalog, "pack", client).put(path, hashlib.sha256(b"data").hexdigest(), 4, {}, lambda: None)
        self.assertEqual(transmitted, [])

    def test_catalog_read_verifies_hash_before_atomic_target_replace(self):
        try:
            from sx_contracts.catalog import ObjectRead, ObjectReads
            from sx_contracts.content import Sha256Digest
        except ImportError:
            self.skipTest("Run with installed sentientx Python")
        sha = hashlib.sha256(b"good").hexdigest()
        catalog = SimpleNamespace(access=lambda *args: ObjectReads((ObjectRead(Sha256Digest(sha), "https://private.example/member"),), 60))
        client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200,
            headers={"content-length": "4"}, stream=httpx.ByteStream(b"evil"))))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()/"chunk"
            with self.assertRaisesRegex(m.MirrorError, "SHA-256 mismatch"):
                m.CatalogStore(catalog, "pack", client).fetch(sha, 4, target)
            self.assertFalse(target.exists())

    def test_restore_bundle_grant_cache_is_bounded_and_expires(self):
        try:
            from sx_contracts.catalog import ObjectRead, ObjectReads
            from sx_contracts.content import Sha256Digest
        except ImportError:
            self.skipTest("Run with installed sentientx Python")
        sha = hashlib.sha256(b"good").hexdigest()
        calls = []
        def access(*args):
            calls.append(args)
            return ObjectReads((ObjectRead(Sha256Digest(sha), "https://private.example/member"),), 60)
        client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200,
            headers={"content-length": "4"}, stream=httpx.ByteStream(b"good"))))
        store = m.CatalogStore(SimpleNamespace(access=access), "pack", client)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()/"chunk"
            store.read_member(sha, target); store.read_member(sha, target)
            self.assertEqual(len(calls), 1)
            entry, _, grant = store._read_grant
            store._read_grant = (entry, 0, grant)
            store.read_member(sha, target)
            self.assertEqual(len(calls), 2)

    def test_resume_keeps_part_etags_and_operation_before_wait(self):
        try:
            from sx_catalog_client.client import InitiatedStagingUpload
            from sx_contracts.operations import OperationRef
        except (ImportError, AttributeError):
            self.skipTest("Run with installed sentientx Python for SDK adapter proof")
        from uuid import UUID
        upload_id = UUID("00000000-0000-0000-0000-000000000001")
        state, snapshots, put_parts = {}, [], []
        sha = hashlib.sha256(b"abcdefgh").hexdigest()
        class Catalog:
            def begin_staging_upload(self, *args):
                return InitiatedStagingUpload(upload_id, "s3://private/staging", 4, 60)
            def staging_upload_part(self, uid, number):
                return SimpleNamespace(url=f"https://storage.example/{number}")
            def complete_staging_upload(self, uid, **kwargs):
                self.parts = kwargs["parts"]
                return SimpleNamespace(source_uri="s3://private/staging")
            def register(self, request):
                return SimpleNamespace(ref=OperationRef.parse("experience/asset-registration/id"), resource_ref="asset:"+sha)
            def result(self, operation, **kwargs):
                self.assert_saved = snapshots[-1]["operation"]["ref"] == str(operation.ref)
                raise TimeoutError("stop after operation persistence")
        def handle(req):
            number = int(req.url.path[1:]); put_parts.append(number)
            if number == 2 and len(put_parts) == 2:
                raise httpx.ConnectError("interrupted")
            return httpx.Response(200, headers={"etag": f"part-{number}"})
        catalog = Catalog()
        store = m.CatalogStore(catalog, "pack", httpx.Client(transport=httpx.MockTransport(handle)))
        def save(): snapshots.append(copy.deepcopy(state))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve()/"chunk"; path.write_bytes(b"abcdefgh")
            with self.assertRaises(httpx.ConnectError): store.put(path, sha, 8, state, save)
            self.assertEqual(state["parts"], {"1": "part-1"})
            with self.assertRaises(TimeoutError): store.put(path, sha, 8, state, save)
            self.assertEqual(put_parts, [1, 2, 2])
            self.assertTrue(catalog.assert_saved)
            with self.assertRaises(TimeoutError): store.put(path, sha, 8, state, save)
            self.assertEqual(put_parts, [1, 2, 2])
            self.assertNotIn("https://", json.dumps(state))


if __name__ == "__main__":
    unittest.main()
