"""Crucible web application.

The flow, end to end: upload a CSV with a target column and a stated
prediction point; an audit job runs in the background and streams its
progress over Server-Sent Events; the finished audit lands on a triage board
of four evidence buckets; a person confirms or rejects each flagged column;
only then does the impact stage measure what the confirmed leaks were costing.

The pause for human review is deliberate. This tool triages and explains; it
never deletes a column on its own authority.
"""

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from crucible import __version__ as crucible_version
from crucible import models as model_catalogue
from crucible import impact, intake, stats
from crucible.audit import AuditRequest, run_audit
from crucible.providers import (KEY_VARIABLES, GeminiProvider, KeyPool, ProviderError,
                                detect_provider,
                                QuotaExhausted, resolve)

RECALL_MODEL = os.environ.get("CRUCIBLE_RECALL_MODEL", model_catalogue.DEFAULT_MODEL)
# Unset means "let the model decide", which is what it should mean: order
# sensitivity is a property of the model, so a single global count is either
# wasteful for a steady model or unsafe for a swinging one. Setting the variable
# overrides every model, which is a blunt instrument and is why it is not set.
SHUFFLE_OVERRIDE = os.environ.get("CRUCIBLE_SHUFFLES")
SHUFFLE_COUNT = int(SHUFFLE_OVERRIDE) if SHUFFLE_OVERRIDE else None


def shuffles_for(model_id: str) -> int:
    return SHUFFLE_COUNT or model_catalogue.shuffles_for(model_id)

# How often to send a comment line on an idle event stream. Comfortably
# under the 60-second idle timeout that hosted proxies typically enforce.
KEEPALIVE_SECONDS = 15

# A shared pool used only to let visitors try the tool without an account. It
# is built once, at import, so every request draws on the same daily budget and
# a reload cannot mint fresh quota.
_DEMO_POOL = GeminiProvider() if os.environ.get("CRUCIBLE_GEMINI_KEYS") else None


# The generated API reference sits at /api/docs rather than /docs, because the
# root of this deployment is a tool rather than a service and a visitor who
# lands on a JSON schema browser has been sent somewhere they did not ask for.
app = FastAPI(title="Crucible", docs_url="/api/docs", redoc_url=None)

# Jobs are held in memory, and each one carries a schema, per-column verdicts,
# statistics and an impact result. On a long-lived instance that is an
# unbounded leak of memory, and the uploaded table behind it an unbounded leak
# of disk. Both are bounded here: the oldest job is evicted once the cap is
# reached, and evicting a job deletes the directory its upload was written to.
#
# The cap is a count rather than a size because a job's footprint is dominated
# by the number of columns, which is small and bounded, while its lifetime is
# not bounded by anything at all.
JOBS: dict[str, dict] = {}
MAX_RETAINED_JOBS = int(os.environ.get("CRUCIBLE_MAX_JOBS", "40"))


def _evict_old_jobs() -> None:
    """Drop the oldest jobs, and their uploads, once the cap is exceeded.

    A job still streaming events is skipped: closing the file underneath a
    running audit would turn a slow job into a confusing failure. Those are
    caught on a later pass, once they have finished.
    """
    while len(JOBS) > MAX_RETAINED_JOBS:
        for job_id, job in JOBS.items():
            if job["status"] == "running":
                continue
            _discard_job(JOBS.pop(job_id))
            break
        else:
            return          # everything left is still running; try again later


def _discard_job(job: dict) -> None:
    directory = os.path.dirname(job.get("path") or "")
    if directory and os.path.basename(directory).startswith("crucible_"):
        shutil.rmtree(directory, ignore_errors=True)


def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(
            404,
            f"This audit is no longer held. The server keeps the most recent "
            f"{MAX_RETAINED_JOBS} and discards older ones along with their uploads. "
            f"Run the audit again to get a fresh result.")
    return job


async def _emit(job: dict, event: str, data: dict):
    job["events"].append({"event": event, "data": data})
    for listener_queue in job["listeners"]:
        await listener_queue.put({"event": event, "data": data})


@app.get("/api/health")
async def health():
    """Liveness for the host. Deliberately free of credentials and of anything
    that could fail for a reason unrelated to the process being up."""
    return {"status": "ok", "version": crucible_version, "jobs": len(JOBS)}


@app.get("/api/models")
async def models():
    """The catalogue, plus what credentials are available. No key, and no part
    of a key, appears in this response."""
    payload = {
        "models": model_catalogue.public_catalogue(),
        "default": RECALL_MODEL,
        "shuffles": {entry["id"]: shuffles_for(entry["id"])
                     for entry in model_catalogue.CATALOGUE},
        "shuffle_rationale": {entry["id"]: model_catalogue.shuffle_rationale(entry["id"])
                              for entry in model_catalogue.CATALOGUE},
        # Which providers this server can already reach without the visitor
        # supplying anything. Reports presence only; no key or fragment of one.
        "server_keys": sorted(
            name for name, variable in KEY_VARIABLES.items()
            if name != "gemini" and os.environ.get(variable)),
        "server_key_present": bool(os.environ.get("FEATHERLESS_API_KEY")),
        "demo_pool": _DEMO_POOL.status() if _DEMO_POOL else None,
    }
    return payload


STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX = os.path.join(STATIC, "index.html")
VERSIONED_ASSETS = ("app.js", "style.css")
_asset_stamp: dict = {}


def _asset_version() -> str:
    """A short hash of the interface's own files.

    The version used to be a literal in the HTML that a person had to remember
    to bump. That is a bad way to invalidate a cache and it failed exactly as
    you would expect: the interface changed, the URL did not, and browsers kept
    serving the copy they already had. Deriving it from the bytes means the URL
    changes when and only when the file does, and nobody has to remember
    anything.
    """
    signature = []
    for name in VERSIONED_ASSETS:
        path = os.path.join(STATIC, name)
        try:
            signature.append(str(os.path.getmtime(path)))
        except OSError:
            signature.append("0")
    key = "|".join(signature)
    if _asset_stamp.get("key") != key:
        digest = hashlib.blake2b(digest_size=6)
        for name in VERSIONED_ASSETS:
            try:
                with open(os.path.join(STATIC, name), "rb") as handle:
                    digest.update(handle.read())
            except OSError:
                continue
        _asset_stamp.update(key=key, hash=digest.hexdigest())
    return _asset_stamp["hash"]


@app.get("/", response_class=HTMLResponse)
async def index():
    """The interface, with its asset URLs stamped by content.

    Served through here rather than by the static mount so the stamp can be
    applied. `no-store` on the document itself is deliberate and cheap: the
    document is small, and it is the thing that has to be fresh for every other
    URL on the page to be correct.
    """
    with open(INDEX, encoding="utf-8") as handle:
        html = handle.read()
    stamp = _asset_version()
    for name in VERSIONED_ASSETS:
        # Matches the asset with or without an existing stamp, so removing the
        # placeholder from the source cannot quietly disable cache busting.
        html = re.sub(rf"{re.escape(name)}(\?v=[^\"']*)?", f"{name}?v={stamp}", html)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.post("/api/key/check")
async def key_check(api_key: str = Form("")):
    """Say which vendor issued a key and which catalogued models it unlocks.

    Recognition is by the key's own prefix, so nothing is sent to any provider
    and no call is spent. The key is not stored, not logged, and not echoed
    back: the response names a provider and some model identifiers.
    """
    issuer = detect_provider(api_key)
    # Every provider that serves at least one catalogued model, so an
    # unrecognized key can be routed by hand instead of being turned away.
    choices = sorted({entry["provider"] for entry in model_catalogue.CATALOGUE})
    if not issuer:
        return {
            "recognized": False,
            "unlocks": [],
            "providers": choices,
            "message": "This key does not match a format Crucible knows, which "
                       "usually just means the provider changed it. Nothing has "
                       "been sent anywhere. Say who issued it and the key will "
                       "be used as it is.",
        }
    unlocks = [entry["id"] for entry in model_catalogue.CATALOGUE
               if entry["provider"] == issuer]
    if unlocks:
        message = f"{issuer} key recognized. It unlocks " + ", ".join(unlocks) + "."
    else:
        message = f"{issuer} key recognized, but no catalogued model uses it."
    return {"recognized": True, "provider": issuer, "unlocks": unlocks,
            "providers": choices, "message": message}


@app.post("/api/audit")
async def start_audit(
    file: UploadFile = File(...),
    target: str = Form(...),
    prediction_point: str = Form(...),
    model: str = Form(RECALL_MODEL),
    api_key: str = Form(""),
    dictionary: UploadFile | None = File(None),
):
    if not prediction_point.strip():
        raise HTTPException(
            422,
            "The prediction point is required. It is a fact about how the model "
            "will be used, not about the data, so the tool cannot infer it.",
        )
    work_directory = tempfile.mkdtemp(prefix="crucible_")
    path = os.path.join(work_directory, file.filename or "data.csv")
    with open(path, "wb") as output_file:
        output_file.write(await file.read())

    try:
        table = intake.load_table(path)
        intake.check_target(table, target)
    except intake.IntakeError as error:
        raise HTTPException(422, str(error))
    except Exception as error:
        raise HTTPException(422, f"could not read table: {error}")

    feature_columns = [column for column in table.columns if column != target]
    if intake.names_are_anonymized(feature_columns):
        raise HTTPException(
            422,
            "Most column names look like placeholders (A1, var_3). The audit "
            "works by reading names, so this table cannot be judged. Halting "
            "is more honest than guessing.",
        )

    dictionary_report = None
    if dictionary is not None and dictionary.filename:
        dictionary_path = os.path.join(work_directory, dictionary.filename)
        with open(dictionary_path, "wb") as output_file:
            output_file.write(await dictionary.read())
        try:
            dictionary_report = intake.load_dictionary(dictionary_path, feature_columns, target)
        except intake.IntakeError as error:
            raise HTTPException(422, f"data dictionary: {error}")
        except Exception as error:
            raise HTTPException(422, f"could not read the data dictionary: {error}")

    chosen_model = model.strip() or RECALL_MODEL
    known_models = set(model_catalogue.BY_ID)
    if chosen_model not in known_models:
        raise HTTPException(422, f"unknown model {chosen_model!r}")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "path": path, "target": target,
        "prediction_point": prediction_point,
        "model": chosen_model,
        # Held only for the lifetime of this job, never written to disk and
        # never returned by any endpoint.
        "api_key": api_key.strip() or None,
        "dictionary": dictionary_report,
        "status": "running", "events": [], "listeners": [],
        "schema": intake.describe_table(table, target),
        "semantic": None, "statistical": None, "buckets": None, "shuffle_detail": None,
        "contested": None, "review": {}, "impact": None, "error": None,
    }
    _evict_old_jobs()
    asyncio.create_task(_run_audit(JOBS[job_id]))
    return {
        "job_id": job_id,
        "n_columns": len(feature_columns),
        "dictionary": _dictionary_summary(dictionary_report),
    }


def _dictionary_summary(report: dict | None) -> dict | None:
    if not report:
        return None
    return {
        "matched": len(report["matched"]),
        "missing": report["missing"],
        "unmatched_rows": report["unmatched_rows"],
        "name_column": report["name_column"],
        "description_column": report["description_column"],
    }


async def _run_audit(job: dict):
    """Run the packaged audit and stream its stages.

    The service holds no pipeline logic of its own. Whatever the command line
    does, this does, because they call the same function.
    """
    table = intake.load_table(job["path"])
    descriptions = (job["dictionary"] or {}).get("descriptions")

    async def on_stage(stage, detail):
        await _emit(job, "stage", {"stage": stage, "name": detail,
                                   "model": job["model"]})

    provider = None
    try:
        provider = _provider_for(job)
        report = await run_audit(
            AuditRequest(
                table=table, target=job["target"],
                prediction_point=job["prediction_point"],
                model=job["model"], descriptions=descriptions,
                shuffles=shuffles_for(job["model"]),
            ),
            provider=provider, on_stage=on_stage,
        )
        job.update({
            "semantic": report["semantic"],
            "statistical": report["statistical"],
            "buckets": report["buckets"],
            "contested": report["contested"],
            "shuffle_detail": report["per_shuffle"],
            "criterion": report["criterion"],
            "status": "awaiting_review",
        })
        await _emit(job, "done", {"status": "awaiting_review",
                                  "buckets": _bucket_counts(report["buckets"])})
    except QuotaExhausted as error:
        job["status"] = "error"
        job["error"] = (f"{error} Use your own key on the model step to continue "
                        f"straight away.")
        await _emit(job, "error", {"message": job["error"], "quota": True})
    except Exception as error:
        job["status"] = "error"
        job["error"] = str(error)
        await _emit(job, "error", {"message": str(error)})
    finally:
        # The shared demonstration pool outlives this job and serves every
        # other visitor, so it is never closed here. Only a provider built for
        # this one job is.
        if provider is not None and provider is not _DEMO_POOL:
            await provider.aclose()


def _provider_for(job: dict):
    """Pick the provider for this job.

    A caller-supplied key is used directly. Otherwise the model's own provider
    is built from the server's environment: for Gemini that is the shared
    demonstration pool, which is created once at import so its daily budget is
    genuinely shared rather than reset per request.
    """
    provider_name = model_catalogue.provider_for(job["model"])

    # A key the caller supplied always wins. Falling through to the shared pool
    # while someone is holding out their own key spends community quota that
    # was not offered and quietly ignores the credential they chose to give.
    if job.get("api_key"):
        issuer = detect_provider(job["api_key"])
        if issuer and issuer != provider_name:
            raise ProviderError(
                f"that key was issued by {issuer}, and {job['model']} is served by "
                f"{provider_name}. Pick a model this key can run, or paste the "
                f"matching key. The key was not sent anywhere.")
        if provider_name == "gemini":
            return GeminiProvider(pool=KeyPool([job["api_key"]],
                                               daily_limit=1_000_000, name="caller"))
        return resolve(provider_name, api_key=job["api_key"])

    if provider_name == "gemini":
        if _DEMO_POOL is None:
            raise ProviderError(
                "no Gemini keys are configured on this server; choose another "
                "model or supply your own key")
        return _DEMO_POOL
    return resolve(provider_name)


def _bucket_counts(buckets: dict) -> dict:
    counts = {bucket: 0 for bucket in "ABCD"}
    for bucket in buckets.values():
        counts[bucket] += 1
    return counts


@app.get("/api/audit/{job_id}/events")
async def events(job_id: str):
    job = _job(job_id)

    async def stream():
        # Replay events already emitted, then stay attached for live ones.
        for past_event in job["events"]:
            yield f"event: {past_event['event']}\ndata: {json.dumps(past_event['data'])}\n\n"
        if job["status"] in ("awaiting_review", "error", "complete"):
            return
        listener_queue: asyncio.Queue = asyncio.Queue()
        job["listeners"].append(listener_queue)
        try:
            while True:
                try:
                    live_event = await asyncio.wait_for(
                        listener_queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    # A single stage can take minutes, and every hosted proxy
                    # closes a connection that has gone quiet for a minute or
                    # two. A comment line is ignored by EventSource and keeps
                    # the socket alive. Without it the audit still completes on
                    # the server while the browser waits on a dead connection,
                    # which looks exactly like a hang and cannot be diagnosed
                    # from the front end.
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {live_event['event']}\ndata: {json.dumps(live_event['data'])}\n\n"
                if live_event["event"] in ("done", "error"):
                    return
        finally:
            job["listeners"].remove(listener_queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # A proxy that buffers the response defeats streaming entirely. nginx
        # sits in front of most hosts and needs telling explicitly.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@app.get("/api/audit/{job_id}")
async def state(job_id: str):
    """Everything the interface needs. Listed key by key on purpose: the job
    also holds the caller's API key, and an endpoint that returned the whole
    job would hand it back out."""
    job = _job(job_id)
    payload = {key: job[key] for key in (
        "id", "status", "target", "prediction_point", "model", "schema", "shuffle_detail",
        "semantic", "statistical", "buckets", "contested", "review", "impact", "error",
    )}
    payload["dictionary"] = _dictionary_summary(job["dictionary"])
    return payload


@app.get("/api/audit/{job_id}/preview")
async def preview(job_id: str, rows: int = 8):
    """The first rows of the table, plus the current drop list, so the
    interface can show the dataset before and after cleaning."""
    job = _job(job_id)
    table = intake.load_table(job["path"])
    drop_list = [column for column, decision in job["review"].items() if decision == "drop"]
    return {
        "columns": list(table.columns),
        "drop_list": drop_list,
        "target": job["target"],
        "n_rows": int(len(table)),
        "rows": table.head(rows).astype(str).fillna("").to_dict(orient="records"),
    }


@app.post("/api/audit/{job_id}/review")
async def review(job_id: str, verdicts: dict[str, str]):
    """Record the human reviewer's decisions: {column: "drop" | "keep"}."""
    job = _job(job_id)
    valid_columns = set(job["schema"]["feature_columns"])
    for column, decision in verdicts.items():
        if column not in valid_columns or decision not in ("drop", "keep"):
            raise HTTPException(422, f"bad review entry: {column}={decision}")
    job["review"].update(verdicts)
    return {"review": job["review"]}


@app.post("/api/audit/{job_id}/impact")
async def run_impact(job_id: str, run_id: str = ""):
    job = _job(job_id)
    drop_list = [column for column, decision in job["review"].items() if decision == "drop"]
    if not drop_list:
        raise HTTPException(422, "confirm at least one column as 'drop' before measuring impact")
    table = intake.load_table(job["path"])
    # N-81. The third arm is what the correlation screen would have removed on
    # this table, so the comparison answers "did the semantic screen beat the
    # cheap alternative here?" rather than only "are these columns worth
    # anything?".
    baseline_drops = sorted(
        column for column, entry in (job["statistical"] or {}).items()
        if entry.get("flagged"))
    reporter = _fit_reporter(run_id)
    try:
        result = await asyncio.to_thread(
            impact.quantify, table, job["target"], drop_list,
            baseline_drop_list=baseline_drops or None, on_event=reporter)
    except Exception as error:      # noqa: BLE001 - reported, never a bare 500
        # A learner refusing the data is a 422 about the data, not a 500 about
        # the server. Left unhandled it answers with a plain-text stack page,
        # and a browser parsing that as JSON reports a message about strings
        # that has nothing to do with the cause.
        if reporter:
            reporter("failed", {"message": str(error)})
        if isinstance(error, impact.ImpactError):
            raise HTTPException(422, str(error))
        raise HTTPException(422, f"the comparison could not be computed: {error}")
    job["impact"] = {"drop_list": drop_list, **result}
    job["status"] = "complete"
    return job["impact"]


# ── watching the fit happen ──────────────────────────────────────────────
#
# The measurement takes tens of seconds and used to show a spinner. It now
# reports itself: which arm is fitting, which fold, and one real tree off each
# fitted model together with a census of what the first forty trees split on
# first. That census is the argument of this whole tool in the model's own
# structure, because an arm still holding a leaked column reaches for it and the
# same forest without it does not.
#
# A caller opens the stream for a run id it chose, then posts the request
# carrying that id. Nothing about the measurement changes: the callback is
# observational and the endpoint still answers with the same JSON.

FIT_STREAMS: dict[str, asyncio.Queue] = {}
MAX_FIT_STREAMS = 24


def _fit_reporter(run_id: str | None):
    """An `on_event` callback that publishes to a run's stream, or nothing.

    `quantify` runs in a worker thread, so every publication is bounced back on
    to the event loop rather than touching the queue directly.
    """
    if not run_id or run_id not in FIT_STREAMS:
        return None
    loop = asyncio.get_running_loop()
    queue = FIT_STREAMS[run_id]

    def publish(name: str, payload: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"event": name, "data": payload})

    return publish


@app.get("/api/fit/{run_id}/events")
async def fit_events(run_id: str):
    """Server-sent events for one measurement run."""
    while len(FIT_STREAMS) >= MAX_FIT_STREAMS:
        FIT_STREAMS.pop(next(iter(FIT_STREAMS)), None)
    queue: asyncio.Queue = asyncio.Queue()
    FIT_STREAMS[run_id] = queue

    async def stream():
        yield "retry: 3000\n\n"
        try:
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"       # hosted proxies drop idle streams
                    continue
                yield f"event: {message['event']}\ndata: {json.dumps(message['data'], default=str)}\n\n"
                if message["event"] in ("done", "failed"):
                    break
        finally:
            FIT_STREAMS.pop(run_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


DEMO_TABLE = os.path.join(os.path.dirname(__file__), "static", "titanic.csv")


@app.post("/api/demo/impact")
async def demo_impact(request: dict):
    """Measure the demo dataset against an arbitrary drop list.

    The demo runs without an API key, so its detection stage is canned. The
    measurement is not: this fits the real learners on the real table, which is
    what makes the column editor meaningful in the demo rather than a control
    that redraws the same numbers.
    """
    drop_list = [str(column) for column in request.get("drop_list", [])]
    if not drop_list:
        raise HTTPException(422, "select at least one column to drop")
    table = intake.load_table(os.path.abspath(DEMO_TABLE))
    screened = stats.statistical_screen(table, "survived")
    baseline_drops = sorted(c for c, e in screened.items() if e.get("flagged"))
    reporter = _fit_reporter(str(request.get("run_id") or ""))
    try:
        result = await asyncio.to_thread(
            impact.quantify, table, "survived", drop_list,
            baseline_drop_list=baseline_drops or None, on_event=reporter)
    except impact.ImpactError as error:
        if reporter:
            reporter("failed", {"message": str(error)})
        raise HTTPException(422, str(error))
    return {"drop_list": drop_list, **result}


@app.get("/api/audit/{job_id}/report")
async def report(job_id: str):
    """The full audit as downloadable JSON: one row per column, plus a
    manifest recording exactly how the audit was run."""
    job = _job(job_id)
    if not job["buckets"]:
        raise HTTPException(409, "audit still running")
    descriptions = (job["dictionary"] or {}).get("descriptions", {})
    rows = []
    for column in job["schema"]["feature_columns"]:
        semantic = job["semantic"].get(column, {})
        statistical = job["statistical"].get(column) or {}
        rows.append({
            "column": column,
            "bucket": job["buckets"][column],
            "verdict": semantic.get("verdict"),
            "mechanism": semantic.get("mechanism"),
            "shuffle_votes": f"{semantic.get('leak_votes', 0)}/{semantic.get('shuffles_counted', 0)}",
            "model_reasons": semantic.get("reasons", []),
            # When a data dictionary was supplied, the documented description
            # travels with the verdict. That pairing is what makes a dropped
            # column defensible in writing: the reason and its source sit
            # together instead of the reason standing on its own.
            "documented_description": descriptions.get(column),
            "absolute_correlation": statistical.get("correlation"),
            "human_verdict": job["review"].get(column),
        })
    manifest = {
        "tool": "Crucible", "job": job["id"], "target": job["target"],
        "prediction_point": job["prediction_point"],
        "model": job["model"], "provider": model_catalogue.provider_for(job["model"]),
        "shuffle_count": shuffles_for(job["model"]),
        "derivation_criterion": model_catalogue.criterion_for(job["model"]),
        "grounded_in_data_dictionary": bool(descriptions),
        "columns_with_documentation": len(descriptions),
        "note": "sample rows were collected for this report and never sent to a model",
    }
    return JSONResponse(
        {"manifest": manifest, "columns": rows, "impact": job["impact"]},
        headers={"Content-Disposition": f"attachment; filename=crucible_{job_id}.json"},
    )


@app.get("/api/audit/{job_id}/cleaned.csv")
async def cleaned_csv(job_id: str):
    job = _job(job_id)
    drop_list = [column for column, decision in job["review"].items() if decision == "drop"]
    table = intake.load_table(job["path"]).drop(columns=drop_list)
    output_path = os.path.join(os.path.dirname(job["path"]), "cleaned.csv")
    table.to_csv(output_path, index=False)
    return FileResponse(output_path, filename="cleaned.csv")


app.mount(
    "/",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True),
    name="static",
)
