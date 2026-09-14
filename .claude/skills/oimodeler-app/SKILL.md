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

## Main workflow

This is the scientific workflow the app must support end to end, and the
order in which a user walks through it. Any feature work should be
understood in terms of where it sits in this sequence — e.g. "filtering"
means step 5, not step 3's target selection, and a change to step 7
(model definition) has downstream effects on 8-12 that need checking:

1. Load OIFITS file
2. Inspect metadata
3. Select target
4. Select observables
5. Filter the data
6. Define model
7. Define free/fixed parameters
8. Configure optimizer
9. Run fit
10. Inspect diagnostics
11. Inspect residuals
12. Export results
13. Save project

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

## Do not

- Rewrite `oimodeler` — it's a dependency, not code owned by this repo. If
  its behavior is wrong (e.g. the `eval()` in `oifitsFlagWithExpression`,
  V5), work around it at the boundary and/or file an upstream issue; don't
  fork its logic into `core/`.
- Duplicate mathematical/scientific implementations that `oimodeler`
  already provides — call into it rather than reimplementing chi²,
  model-image generation, or fitting math locally.
- Use an undocumented `oimodeler` API without first checking it against
  the actually-installed version (see "Development procedure" below) —
  `oimodeler` is pulled from `HEAD` with no pin (V14), so its surface can
  shift under you.
- Silently modify scientific units (e.g. converting mas/deg/rad, µm/m) —
  any unit conversion must be explicit and visible, never an implicit
  scale factor buried in a computation.
- Silently discard invalid data (a bad row, an out-of-range measurement) —
  reject with a clear message or flag it visibly; never drop it quietly
  and keep going.
- Overwrite a user's file without confirmation — this includes upload
  storage, saved projects, and exported results.
- Use `pickle` for persistent project files — it's neither safe against
  untrusted input nor a stable format; use JSON/HDF5 (per the existing run
  manager convention) or another explicit, inspectable format.
- Store secrets in source code — config/environment for anything sensitive
  (none exist yet; keep it that way).
- Store user data globally — every upload/session artifact is scoped per
  session (see V3/V9 in the security rules above; this is the same
  constraint restated for the data layer generally, not just uploads).
- Introduce a new dependency without justification — check
  `requirements.txt` first; a new package needs a concrete reason, not
  convenience.
- Create unnecessary abstractions (classes/factories/config layers) for a
  single use site — match the project's existing concise, flat style.

## Development procedure

Before modifying the code:

1. Inspect the existing architecture (this skill's "Actual repository
   structure" above, and the module you're about to touch).
2. Identify the relevant module — don't add logic to `pages/` that
   belongs in `core/` or `services/`, or vice versa.
3. Inspect existing implementations of the same kind of thing (e.g. how
   the other pages already load/filter/cache data) before writing a new
   one — several bugs in this codebase come from copy-pasted logic
   drifting apart (see `qa-verification`'s brief on
   `_get_active_data_with_filter()`).
4. Check the installed `oimodeler` API directly (it isn't pinned — don't
   assume a signature from memory or from `oimodeler`'s docs matches what's
   actually installed).
5. Make the smallest appropriate change. Never rewrite an entire file when
   a local, targeted modification is sufficient.
6. Run relevant tests.
7. Check for regressions — including in the workflow steps downstream of
   whatever you changed (see "Main workflow" above).
8. Only then refactor, if the refactor is actually warranted — not as a
   default step.

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

## Definition of done

A feature is complete only when all of the following hold — not just "it
runs once in a demo":

- It works in the Streamlit UI, not only as isolated `core/` logic.
- The underlying scientific computation is correct — verified against a
  known case where possible (a reference dataset in `tutorial/Data/`, or a
  hand-checked value), not just "it didn't crash."
- User sessions remain isolated (no shared/global state leak — see the
  security rules above).
- Invalid inputs are handled: rejected with a clear message, never a raw
  traceback, and never silently coerced or dropped.
- The code is documented where necessary (a non-obvious constraint or
  workaround gets a short comment; the rest is self-explanatory naming —
  see project code style, no docstring padding).
- Tests pass, including any new ones the change warrants.
- No existing functionality is broken — check the workflow steps
  downstream of the change (see "Main workflow" above), not just the step
  directly touched.
- The change doesn't introduce an unnecessary performance regression
  (e.g. a new O(n²) pass over spectral data, an avoidable full re-read of
  a cached file).

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
7. Does it respect "Do not" above (no `oimodeler` reimplementation, no
   pickle, no silent unit/data changes, no unjustified new dependency)?
8. Was the "Development procedure" followed — smallest appropriate change,
   existing implementations checked first, installed `oimodeler` API
   verified rather than assumed?
