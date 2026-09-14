# Proposal: user-supplied model components

Status: **proposal only — not implemented.** Written in response to a
feature request for "an option to input user models." Per the project's
security posture (public, no-login deployment, ~30 concurrent users), this
needs a design decision before any code is written.

## The constraint

Item 5 of the original request ("let a user provide their own model") most
naturally reads as "let the user supply arbitrary Python code that computes
a model." That is not acceptable for this app: it is headed for public,
no-login hosting, and accepting user-authored code for `exec`/`eval` is a
remote-code-execution hole, not a feature — one user's "model" would run
with the same privileges as everyone else's session, the app's file access,
and (per the existing security audit) the same process already flagged for
tightening around `oimFlagWithExpressionFilter`'s own internal `eval()`
(V5). This proposal exists to find what's left once that's off the table.

## What oimodeler supports natively

Inspected the installed package directly rather than assuming:

- **`oimComponentFitsImage`** — loads a 2D image (or 3D chromatic image
  cube) from a FITS file via `astropy.io.fits.open()` and Fourier-transforms
  it through the same engine every other component uses
  (`getComplexCoherentFlux` / `getImage`). No code execution: it is a
  binary/scientific data format parse, not a script. The app already
  accepts uploaded FITS files for OIFITS data, so this doesn't introduce a
  new *class* of upload risk, only a second use of an existing one.
- **`oimParamUserFunc`** — takes an actual Python `Callable` and calls it
  directly (`self.userfunc(*argvals)`). This is oimodeler's real "custom
  model" hook, but it is designed for script/notebook use: the caller
  already has a live Python function object. There is no way to expose
  this over a web form without either (a) accepting literal Python source
  and executing it, or (b) building an entirely separate, app-owned safe
  expression layer that *produces* a callable — at which point oimodeler's
  own hook isn't doing the safety work, our own parser is.
- **Everything else** (the existing registry) is a fixed set of analytic
  Fourier/radial-profile component classes defined in oimodeler's own
  source. Adding a genuinely new *shape* there means subclassing in
  oimodeler itself (see item 2's registry work) — it isn't a per-user,
  runtime-configurable thing.

So oimodeler gives exactly one safe, no-code path (image-based components)
and one real "custom model" hook that is unsafe to expose as-is
(`oimParamUserFunc`).

## Options considered

### Option A — FITS/image upload (`oimComponentFitsImage`)

The user uploads a 2D FITS image (or is offered a small in-app tool to
build one from a simple radius/intensity table — see "possible follow-up"
below); the app instantiates `oim.oimComponentFitsImage(fitsImage=path)`
and adds it to the model like any registry component, with `pa`, `scale`,
`x`, `y`, `f` exposed as usual.

- **Security**: no code execution surface at all. The only new risk is
  parsing an untrusted FITS file (same class of risk the Data tab already
  accepts, so it inherits — and should reuse — the OIFITS upload hardening
  work (`oifits-validator` workstream): size cap, `NAXIS`/`CDELT`
  sanity checks (`oimComponentFitsImage.loadImage` already rejects
  non-square images and mismatched pixel scales), and treating the file
  the same as any other per-session upload (allowlisted path, never
  concatenated from the raw filename, evicted like other cached data).
  The FT of a large image is a real compute cost, so it needs the same
  bounding the skill already calls for on long computations (size cap on
  `NAXIS1`/`NAXIS2`, plus the existing concurrency/cooldown work).
- **Effort**: low. It's a new registry-style entry point (`file_uploader`
  →  `oimComponentFitsImage`), not a new subsystem.
- **Limitation**: the user needs a FITS image in hand (or a tool to make
  one from something simpler — see below). It doesn't let someone type
  "a disk with a Gaussian bump" without preparing that image first.

### Option B — restricted parametric formula

Let the user type a small mathematical expression (e.g.
`exp(-((r-r0)/w)**2)` as a radial intensity profile, or `f(x,y)` for a 2D
one), parsed with Python's `ast` module against a hard allowlist of nodes
(numeric literals, `+ - * / **`, a fixed set of numpy ufuncs like `exp`,
`sin`, `sqrt`) and evaluated only on numeric arrays the app controls —
never `eval()`/`exec()` on the raw string, and never resolving arbitrary
names. The result is applied on a grid built by the app and fed either
into a small FITS image (reusing Option A's pipeline) or into
`oimParamUserFunc` via a callable *we* construct from the parsed and
validated AST.

- **Security**: safe *if and only if* the allowlist is genuinely closed
  (no attribute access, no subscripting into arbitrary objects, no
  `__`-dunder lookups, a recursion/length cap on the expression, a time
  budget on evaluation). This is a real, non-trivial piece of security
  engineering in its own right — small in code size, but the kind of
  thing that needs a dedicated review, not something to fold into a
  general feature pass. Getting an AST allowlist subtly wrong is a classic
  sandbox-escape source.
- **Effort**: medium. No oimodeler support to lean on; this is fully
  app-owned code, which also means it isn't "reimplementing oimodeler
  math" (nothing here duplicates a scientific model oimodeler already
  ships) but it is new attack surface the project owns and must maintain.
- **Benefit over Option A**: no external tool needed — a user can type a
  shape directly.

### Option C — arbitrary Python code (`exec`/`eval`)

Included only for contrast. Rejected outright: this is precisely what the
skill's security rules and the active hardening workstream forbid for a
public, no-login, multi-tenant deployment. Not a real option here.

## Recommendation

Ship **Option A** first: it has no code-execution surface, reuses upload
infrastructure and validation patterns the app already has (and that the
`oifits-validator` workstream is already hardening for the same threat
model), and is a small, ordinary feature addition — a new
`st.file_uploader` plus a registry-style entry for
`oim.oimComponentFitsImage`, with size/shape validation on the FITS file
mirroring the checks `loadImage()` already does internally.

**Possible follow-up, not v1**: a small in-app converter from a simple
uploaded table (radius, intensity — a CSV, i.e. plain data, not code) into
a FITS image server-side, so users who don't want to prepare a FITS file
by hand still don't need Option B's expression sandbox. This keeps the
"no code execution" property while covering the common case Option B was
partly meant to solve.

**Option B is a plausible v2**, but only as its own reviewed piece of work
with an explicit threat model for the expression sandbox (ideally reviewed
alongside the `security-hardener` workstream, given the project already
has one cautionary tale in this exact area —
`oimFlagWithExpressionFilter`'s internal `eval()`, V5 in the security
audit). It should not be built as a quick add-on to Option A.

Option C should not be built.
