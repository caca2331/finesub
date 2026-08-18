"""How one call decides which model answers it.

The layers run in one direction and only one:

    model_catalog   facts -- what each (provider, model) can do and costs
    model_routes    composition -- targets, model groups, task groups, presets
    execution_policy the execution identity a call carries into its checkpoint
    model_router    the per-call plan: candidates, filters, the answer
    client          (outside this package) the call itself

`config` is read by every layer and depends on none of them; `profiles` and
`capabilities` turn the user's switches into the questions the router asks;
`api_keys` reads the provider switches and named pools out of `config.toml`.
At import time nothing here reaches further into `llm` than `finesub.llm.agent`
(a driver config is part of the execution identity, and a spent subscription
pre-filters candidates); `test_import_boundaries` enforces both that and the
layer order above. The two places the direction really inverts -- `config`
reading a resolved route back, `model_router` classifying a `finesub.llm.client`
failure -- import inside the function and say so.

See docs/manual/model-routing.md for what a user sees, and
docs/llm_design_notes.md for why the layers are cut here.
"""
