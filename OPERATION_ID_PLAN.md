# Operation ID Plan: bdd_playwright_service

Canonical registry of `operation_id` values for every route in
`bdd_playwright_service`. These IDs feed the org-wide n8n custom-node
generator and the `ultrareview` cross-service catalogue, so they must be:

- **stable** -- never rename without coordinating downstream,
- **`verb_noun` snake_case** -- matches the org standard (nlp_service,
  worker_management, kag, analysis_service, video_compiler_service,
  video_analyzer_service),
- **globally unique** -- disambiguated against the rest of the fleet's verbs.

This service hosts four BDD generation streams (QA suite, accessibility,
documentation, demo video), a Gherkin upload endpoint, a job-status surface,
a package download, plus the health surface. Verbs are prefixed with `bdd_`
where the generic verb (`upload_*`, `download_*`, `get_*_status`) would
collide across the fleet.

## Files to edit

- `src/asgi.py` (root endpoint)
- `src/api/health.py`
- `src/api/routes/qa_suite.py`
- `src/api/routes/accessibility.py`
- `src/api/routes/documentation.py`
- `src/api/routes/demo_video.py`
- `src/api/routes/gherkin.py`
- `src/api/routes/jobs.py`
- `src/api/routes/packages.py`

## Mapping Tables

### `src/asgi.py` (tag: `health`)

| Method | Function | operation_id   | Rationale                                  |
|--------|----------|----------------|--------------------------------------------|
| GET    | `root`   | `service_root` | Service banner; matches the fleet standard. |

### `src/api/health.py` (tag: `health`)

| Method | Function       | operation_id   | Rationale                                                  |
|--------|----------------|----------------|------------------------------------------------------------|
| GET    | `health_check` | `health_check` | Primary health probe; matches the fleet standard verb.     |
| GET    | `health_live`  | `health_live`  | Liveness sub-probe; one-to-one with path.                  |
| GET    | `health_ready` | `health_ready` | Readiness sub-probe; one-to-one with path.                 |

### `src/api/routes/qa_suite.py` (tag: `bdd`)

| Method | Function          | operation_id        | Rationale                                                                  |
|--------|-------------------|---------------------|----------------------------------------------------------------------------|
| POST   | `start_qa_suite`  | `start_bdd_qa_suite`| `qa-suite` is BDD-stream specific; `bdd_` prefix avoids `start_*` collisions. |

### `src/api/routes/accessibility.py` (tag: `bdd`)

| Method | Function              | operation_id                       | Rationale                                                              |
|--------|-----------------------|------------------------------------|------------------------------------------------------------------------|
| POST   | `start_accessibility` | `start_bdd_accessibility_audit`    | Disambiguates from a hypothetical generic accessibility verb.          |

### `src/api/routes/documentation.py` (tag: `bdd`)

| Method | Function               | operation_id                  | Rationale                                                                          |
|--------|------------------------|-------------------------------|------------------------------------------------------------------------------------|
| POST   | `start_documentation`  | `start_bdd_documentation`     | Generic `documentation` namespace is crowded; `bdd_` keeps us in our verb namespace. |

### `src/api/routes/demo_video.py` (tag: `bdd`)

| Method | Function           | operation_id              | Rationale                                                                        |
|--------|--------------------|---------------------------|----------------------------------------------------------------------------------|
| POST   | `start_demo_video` | `start_bdd_demo_video`    | `demo_video` collides with the video_* services; `bdd_` keeps it scoped to BDD.  |

### `src/api/routes/gherkin.py` (tag: `bdd`)

| Method | Function         | operation_id                    | Rationale                                                                       |
|--------|------------------|---------------------------------|---------------------------------------------------------------------------------|
| POST   | `upload_gherkin` | `upload_bdd_gherkin_features`   | Generic `upload_*` collides at fleet scope; `bdd_gherkin_features` is specific. |

### `src/api/routes/jobs.py` (tag: `bdd`)

| Method | Function          | operation_id          | Rationale                                                                          |
|--------|-------------------|-----------------------|------------------------------------------------------------------------------------|
| GET    | `get_job_status`  | `get_bdd_job_status`  | Generic `get_job_status` collides across the fleet; domain-prefix.                 |
| GET    | `get_job_result`  | `get_bdd_job_result`  | Same -- domain-prefix.                                                              |

### `src/api/routes/packages.py` (tag: `bdd`)

| Method | Function           | operation_id           | Rationale                                                                |
|--------|--------------------|------------------------|--------------------------------------------------------------------------|
| GET    | `download_package` | `download_bdd_package` | Generic `download_*` collides; prefix with the BDD domain.               |

## Verification

```bash
cd bdd_playwright_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the union of every right-hand column above.
Drift (rename, missed decorator, fallback) fails the test loudly.

An additional AST-level decorator scan verifies that every `@router.<verb>`
and `@app.<verb>` decorator AND every `router.add_api_route(...)` call has
an explicit `operation_id=` kwarg (the `generate_unique_id_function`
fallback is defence-in-depth only, never the primary identifier).

```bash
python -c "
import ast, pathlib
root = pathlib.Path('src')
files = list(root.rglob('*.py'))
total = explicit = 0
for f in files:
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            # decorator: @router.<verb>(...) or @app.<verb>(...)
            if isinstance(fn, ast.Attribute) and fn.attr in {
                'get', 'post', 'put', 'patch', 'delete', 'head', 'options'
            }:
                obj = fn.value
                if isinstance(obj, ast.Name) and obj.id in {'router', 'app'}:
                    total += 1
                    if any(k.arg == 'operation_id' for k in node.keywords):
                        explicit += 1
            # router.add_api_route(...)
            elif isinstance(fn, ast.Attribute) and fn.attr == 'add_api_route':
                total += 1
                if any(k.arg == 'operation_id' for k in node.keywords):
                    explicit += 1
print(f'{explicit}/{total} routes have explicit operation_id')
assert explicit == total, 'every decorator must set operation_id explicitly'
"
```

## Disambiguation rationale

Concrete renames where the function name or path was generic enough to
collide across the fleet:

- `start_qa_suite` / `start_accessibility` / `start_documentation` /
  `start_demo_video` -> all `start_bdd_*` to keep the BDD streams out of
  the generic `start_*` namespace shared with other services.
- `upload_gherkin` -> `upload_bdd_gherkin_features` -- `upload_*` is too
  generic at fleet scope; gherkin features are the actual payload.
- `get_job_status` / `get_job_result` -> `get_bdd_job_*` -- generic
  job-status verbs are shared across services.
- `download_package` -> `download_bdd_package` -- generic `download_*`
  collides across the fleet.

## Future routes

When adding a route in this service, **always** set `operation_id="..."`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
