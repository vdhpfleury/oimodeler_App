---
name: oimodeler-app
description: Project-local guidance for oimodeler_App (Streamlit interferometric modeling tool for OIFITS/VLTI data). Use for any change to this repository — architecture, Streamlit pages, core/ logic, data validation, async fitting (MCMC), or deployment/security hardening for public hosting. Reflects the actual current code structure (core/ components/ pages/ services/ config/) and the active hardening effort tracked in docs/security_audit_2026-09.md. Complements the account-level oimodeler-webapp skill with repo-specific reality and open workstreams.
---

# oimodeler_App — project state and working rules

## What this app is

A Streamlit frontend letting scientists/students with no coding experience
model OIFITS interferometric data (VLTI/GRAVITY, MATISSE) with `oimodeler`:
upload data, build/select a model, fit it (random search today, MCMC/emcee
target), visualize, export. **Target deployment: public internet access,
no login, ~30 concurrent users.** That last constraint is now binding —
see "Security" below before writing any code that touches user input,
files, or long computations.

## Actual repository structure (not the aspirational one)

```
oimodeler_App/
├── 3_test.py         # real entry point today (streamlit run 3_test.py) — despite
│                      # its internal header comment reading "# app.py". Renaming
│                      # this to app.py and removing dev leftovers is tracked (V20).
├── core/              # pure logic: model_builder, fitting, code_generator,
│                      # csv_import, registry, results, component. No `import streamlit`
│                      # here — keep it that way, it's what makes core/ testable.
├── components/        # reusable Streamlit widgets (param_editor, plots)
├── pages/             # per-tab Streamlit render() functions, wired into 3_test.py's
│                      # st.tabs(). WARNING: Streamlit auto-discovers pages/ as a
│                      # multipage app, so these are also reachable directly by URL,
│                      # bypassing init_session_state(). Renaming to views/ is on the
│                      # security backlog (V11) — until then, never assume a page's
│                      # render() only runs after 3_test.py's setup.
├── services/          # data_service (OIFITS loading/caching), session (session_state init)
├── config/            # constants
├── docs/              # security_audit_2026-09.md — read before touching auth/files/perf
└── tutorial/Data/      # example datasets (GRAVITY, MATISSE) for users without their own data
```

There is no `jobs/`, no `tests/`, no `.streamlit/config.toml`, no `Dockerfile`
yet. These are open workstreams, not oversights to silently "fix" — see below.

## Non-negotiable rules from here on (public-deployment reality)

These come from a full security audit (`docs/security_audit_2026-09.md`,
2026-09-04) and apply to **all new code**, not just the backlog items:

1. **Every value coming from a Streamlit widget is hostile network input.**
   `min_value`/`max_value` on `number_input`/`slider`, the option list of a
   `selectbox`/`multiselect`, `max_chars` on `text_input` — none of these are
   enforced server-side by Streamlit 1.48. A forged WebSocket message can send
   anything. **Always re-validate in Python after reading the widget**, using
   (once merged) `core/validation.py`'s helpers (`num`, `choice`, `choices`,
   `text`). Never trust a widget value used for: array/image sizes, loop
   counts, MCMC steps/walkers, file paths, or anything passed to `eval`.
2. **Never build a filesystem path by concatenating a client-controlled
   string** (an uploaded filename, a `multiselect`/`text_input` value). Use
   `st.session_state.loaded_files` (or the future `services/storage.py`) as
   an allowlist: look up by key, reject unknown keys — never reconstruct
   `/tmp/{name}`-style paths.
3. **`@st.cache_resource` is a global, cross-session cache** — it is *not*
   session-isolated the way `st.session_state` is. Never cache on a key
   derived from user-controlled input alone (a raw filename), and never
   mutate a cached object in place (build a fresh one, or fold the mutating
   parameters into the cache key).
4. **Long computations (MCMC, random search, big images) must never run
   unbounded on the main thread.** At minimum: a concurrency semaphore and a
   per-session cooldown; the real target is an out-of-process worker/queue.
   See "Active workstreams" below.
5. **Never show a raw exception to the user** (`st.error(f"... {exc}")` leaks
   paths and internals — see V12). Show a generic message with a short
   correlation id, log the real exception server-side.
6. All user-facing text, code comments, and identifiers stay in **English**,
   regardless of the language used to discuss the project.

## Active workstreams (current priorities)

Four workstreams are active in parallel. Each has a dedicated subagent in
`.claude/agents/` — prefer delegating to it for that domain rather than
improvising:

| Workstream | Subagent | Tracks |
|---|---|---|
| Harden for public deployment (30 concurrent users) | `security-hardener` | `docs/security_audit_2026-09.md` §7 action plan (V1–V20) |
| Verify & validate existing features (no `tests/` yet) | `qa-verification` | dead code, duplicated logic (e.g. 3 copies of `_get_active_data_with_filter()`), functional regressions |
| Real MCMC + async execution | `async-fitting-architect` | replacing/extending `core/fitting.py`'s blocking random search; job queue design |
| Robust OIFITS upload validation | `oifits-validator` | GRAVITY vs MATISSE structure checks, file-format sanity before handing to `oimodeler` |

When a task clearly belongs to one of these domains, invoke that subagent
instead of doing the work inline — it carries the narrower, more detailed
brief for that area.

## Checklist before proposing code on this project

1. Does it keep `core/` free of `streamlit` imports?
2. Is every value read from a widget re-validated server-side before use
   (size, type, membership in an allowed set)?
3. Is any file path built only from a trusted allowlist, never from raw
   client input?
4. If the computation can take more than ~1s, is it bounded (semaphore,
   cooldown, or queued) rather than run straight on the main thread?
5. Are error messages shown to the user generic, with details only in logs?
6. Is the change consistent with `docs/security_audit_2026-09.md` rather
   than reintroducing something it flags?
