# Deploying to Render

Ten minutes, no card. The repository already has `render.yaml`, so Render reads
the settings from it and you only fill in the key.

## 1 · Connect the repository

Go to [dashboard.render.com](https://dashboard.render.com) and sign in with
GitHub. Pick **New** then **Blueprint**, and select `Crucible-Leakage`.

Render finds `render.yaml`, shows one service called `crucible`, and asks you to
confirm. The build command, the start command and the health check all come from
that file.

## 2 · Set the key

The blueprint declares `CRUCIBLE_GEMINI_KEYS` with `sync: false`, which means
Render prompts you for it and never stores it in the repository. Paste one or
more Gemini API keys, comma separated:

```
key1,key2,key3
```

Get them from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
Free-tier keys allow a small number of requests a day each, which is why the
field takes several: the server uses the least-used one on every call and sets
aside any key that reports its quota is spent.

Leave the other three key fields empty. Those unlock the OpenAI, Anthropic and
Featherless models for every visitor, and for a public deployment you want those
to stay locked behind a visitor's own key.

## 3 · Deploy

Press **Apply**. The first build takes about four minutes, most of it installing
scikit-learn and pandas. When the log reaches

```
Uvicorn running on http://0.0.0.0:10000
```

the service is live at `https://crucible-<something>.onrender.com`.

## 4 · Check it

```bash
curl https://your-service.onrender.com/api/health
```

You want `{"status":"ok","version":"0.1.0","jobs":0}`.

Then open the URL. The landing page loads, the Titanic replay runs, and the
downstream comparison fits real models, all without a key. Only the detection
stage needs one.

## What the free plan does

It sleeps after fifteen minutes with no traffic, and the next request waits about
forty seconds while it wakes. That is fine for a demonstration and bad for a
judge who clicks once and leaves. Two ways to handle it:

Hit the URL yourself a minute before anyone else does.

Or point any uptime pinger at `/api/health` every ten minutes, which keeps it
warm for free.

Memory on the free plan is 512 MB. An audit holds the uploaded table in memory,
so a very wide file can run it out. The service caps the number of concurrent
jobs and evicts the oldest, so it recovers rather than falling over.

## One worker, on purpose

`--workers 1` is in the start command because audit jobs live in the process's
own memory. A second worker would receive progress requests for jobs it has never
heard of and answer with a 404.

## Fitting takes minutes

The downstream comparison fits ninety models. On the free plan's single shared
CPU a wide table can take five minutes or more, and the interface says so before
you start it. The **Stop and just download the CSV** button exists for exactly
this: the cleaned file does not need the fits.

## Why the fit used to fail with a 502

Render's proxy will not hold an HTTP request open for minutes, and this stage
fits ninety models. The request was being cut off and the browser received the
proxy's own HTML error page, which it then tried to read as JSON.

The measurement now runs past the request that starts it. Posting to
`/api/audit/{job}/impact` returns straight away, the browser watches the event
stream for progress, and it polls `GET /api/audit/{job}/impact` for the answer,
which replies 202 until there is one. Nothing about the fitting changed.

`CRUCIBLE_MAX_ROWS` is set to 2000 in the blueprint for the same reason: on a
shared CPU the default of 5000 makes a wide table take twenty minutes. Raise it
if you move to a paid instance. Whatever it is set to, the report and the
interface both say how many rows were actually used.
