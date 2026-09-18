# core/normalization.py
"""
Support for oimodeler's `oimParamNorm`: normalize a component's parameter
to `norm - sum(other params)`, evaluated dynamically — the standard way to
tie one component's flux to "everything else" in a multi-component SED
model. See the oimodeler docs/example this was integrated from:

    pt2 = oim.oimPt()
    g2  = oim.oimGauss(f=oim.oimInterp("wl", wl=[3e-6, 4e-6], values=[1, 2]))
    m2  = oim.oimModel(g2, pt2)
    pt2.params["f"] = oim.oimParamNorm(g2.params["f"])

`oim.oimParamNorm(params, norm=1.0)` takes a LIST OF LIVE oimParam objects
(not a macro + kwargs like oimInterp) and must be assigned to a component's
`.params[name]` AFTER every component in the model has already been built
— it references other components' *already-instantiated* parameter
objects directly. That's why, unlike interpolators (core/interp_registry.py,
applied per-component inside core/component.py's create_instance()), this
is applied at the MODEL level, once every component instance exists — see
core/model_builder.py's build_oim_model().

Storage (component_dict["normalizations"][param_name]):
    {
        "enabled": bool,
        "norm": float,                       # oimParamNorm's `norm` kwarg
        "refs": [{"component": <other component's name>,
                  "param": <that component's param name>}, ...],
    }

oimParamNorm.free is always False (see its source) — a normalized
parameter is never independently fittable, so it never gets its own
free/bounds UI, and must be excluded wherever the app treats a component's
plain scalar params as candidates for fitting (core/component.py's
generate_random_params()) or expects every model.getParameters() entry to
have `.value`/`.min`/`.max` (core/results.py — oimParamNorm has none of
those, only `.free` and the ability to be called as `p(wl, t)`).

Pure declarative + validation logic, no Streamlit/oimodeler import — safe
to import from core/ and unit test in isolation.
"""
from __future__ import annotations

from core.validation import InvalidInput


def validate_normalization_refs(
    target_comp_name: str, refs: list[dict], comp_list: list[dict], registry: dict,
) -> None:
    """Raises InvalidInput if a normalization's reference list is invalid:
    empty, self-referencing, pointing at an unknown component, or at a
    parameter that component doesn't actually have."""
    if not refs:
        raise InvalidInput(
            "Select at least one other component/parameter to normalize against."
        )

    comps_by_name = {c["name"]: c for c in comp_list}
    seen = set()
    for ref in refs:
        comp_name, param_name = ref.get("component"), ref.get("param")
        if comp_name == target_comp_name:
            raise InvalidInput(
                f"A component cannot be normalized against itself ({comp_name!r})."
            )
        ref_comp = comps_by_name.get(comp_name)
        if ref_comp is None:
            raise InvalidInput(f"Unknown component in normalization reference: {comp_name!r}.")
        ref_params = registry.get(ref_comp["type"], {}).get(
            "params", ref_comp.get("params", [])
        )
        if param_name not in ref_params:
            raise InvalidInput(
                f"{comp_name!r} has no parameter {param_name!r} to normalize against."
            )
        key = (comp_name, param_name)
        if key in seen:
            raise InvalidInput(f"Duplicate reference: {comp_name}.{param_name}.")
        seen.add(key)


def is_normalized(comp: dict, param_name: str) -> bool:
    """True if `param_name` has an enabled normalization on this component dict."""
    return comp.get("normalizations", {}).get(param_name, {}).get("enabled", False)
