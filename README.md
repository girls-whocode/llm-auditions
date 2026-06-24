# LLM Auditions

A production-quality, configuration-driven audition framework for evaluating
Ollama-served local language models for specific team roles in a future
multi-model Open WebUI system.

## Purpose and Architecture

This project determines which installed model should occupy each **worker**,
**escalation**, **reviewer**, **planner**, **synthesis**, and **editing** role
in a future multi-model system. After the auditions run, results are archived and
sent to an external reviewer for final model assignment decisions.

**The Mission Director is not built here.** Final role assignments are not made
by this framework. It produces evidence, scoring, comparisons, and rankings only.

### Why Auditions Are Role-Specific

Different roles have fundamentally different contracts:

| Role | Judged by |
|---|---|
| Fast worker | Correct routine answers, speed, restraint, escalation recognition |
| Heavy worker | Difficult reasoning, material improvement over fast-worker baseline |
| Reviewer | Correctly approves, revises, rejects, reroutes, or requests research |
| Language editor | Improves wording without changing approved facts |
| Research planner | Query design and evidence requirements |
| Research synthesizer | Supplied evidence packets, not unsupported model memory |
| Development worker | Deterministic execution, syntax, lint, and tests |
| Mathematics solver | Python/SymPy verification, not another model's opinion |

Treating all models as competitors in a single tournament would obscure these
role-specific strengths and weaknesses.

## Installed Model Matrix

```
qwen3-coder:30b          (06c1097efce0  18 GB)
phi4-reasoning:14b       (47e2630ccbcd  11 GB)
gemma4:26b               (5571076f3d70  17 GB)
gemma4:12b               (4eb23ef187e2   7.6 GB)
qwen3.5:9b               (6488c96fa5fa   6.6 GB)
qwen2.5-coder:14b        (9ec8897f747e   9.0 GB)
qwen2.5:14b-instruct     (7cdf5a0187d5   9.0 GB)
codestral:22b            (0898a8b286d5  12 GB)
qwen2.5-coder:7b         (dae161e27b0e   4.7 GB)
phi4:14b                 (ac896e5b8b34   9.1 GB)
```

## Team and Candidate Matrix

| Team | Roles | Key Candidates |
|---|---|---|
| Baseline | baseline_all | All 10 models |
| General Knowledge | fast_worker, heavy_worker, reviewer | gemma4:12b, qwen3.5:9b, gemma4:26b, phi4-reasoning:14b |
| Research | query_planner, evidence_synthesizer | gemma4:12b, qwen3.5:9b, gemma4:26b |
| Linux / Infrastructure | fast_worker, escalation, reviewer | gemma4:12b, qwen3-coder:30b, phi4-reasoning:14b |
| Engineering / Hardware | worker, reviewer | gemma4:26b, phi4-reasoning:14b, phi4:14b |
| Development | fast_worker, advanced_worker, code_review | qwen2.5-coder:14b, qwen3-coder:30b, codestral:22b |
| Mathematics | solver | phi4-reasoning:14b, gemma4:26b, phi4:14b |
| Security | worker, reviewer | gemma4:26b, phi4-reasoning:14b, qwen3-coder:30b |
| Architecture | worker, reviewer | gemma4:26b, phi4-reasoning:14b, gemma4:12b |
| Document Analysis | fast_worker, complex_document | gemma4:12b, gemma4:26b, phi4-reasoning:14b |
| Integration Review | primary, escalation | gemma4:12b, gemma4:26b, phi4-reasoning:14b |
| Language Knowledge | editor | qwen3.5:9b, gemma4:12b, qwen2.5:14b-instruct |

## Safety Guarantees

- Models are **never** pulled, deleted, renamed, or modified.
- Development verifier execution is currently **permanently disabled by policy**.
- Setting `AUDITION_ENABLE_DEVELOPMENT_SANDBOX=1` does not enable host execution.
- Development tasks are marked `human_required` with `reason=sandbox_unavailable`.
- Commands proposed by models are **never executed** against the real host.
- A static safety verifier flags destructive commands before scoring.
- Packaging includes a **sanitization scan** that blocks archives containing likely secrets.
- Multiple large models are **never loaded simultaneously**.
- The exhaustive profile requires **explicit confirmation** before starting.

## Explicit Rubrics

Rubric-assisted tasks must carry **human-authored `rubric_rules` in fixture YAML**.
The framework no longer synthesizes fallback rubrics from prompts, required concepts,
or other task text at runtime.

Rubric-assisted tasks must explicitly declare `rubric_finalization` in YAML:

- `deterministic`: produces `final` when all rubric checks are deterministic and resolved.
- `human_review`: always produces `human_required`.
- `mixed`: remains `provisional` while unresolved human-review rubric rules exist.

Optional rubric rules use **bonus semantics** and do not reduce the base denominator when absent.

Validation and `audit-config` treat missing `rubric_finalization` as a blocking defect.

## Shared Comparison Scenarios

Comparison tasks (`comparison_id` + `comparison_track` + `worker_class`) must also declare:

- `comparison_scenario_ref`: path to a shared scenario JSON under `fixtures/comparisons/`
- `use_shared_scenario_rubric: true`
- `comparison_shared_rubric_version`
- `role_rubric_rules`

Both fast and heavy tasks for the same comparison pair must reference the same scenario file.
Scenario fixtures are authoritative and must include:

- `comparison_id`
- `scenario_version`
- `title`
- `scenario`
- `constraints`
- `required_facts` (fact objects with `fact_id`, `description`, `aliases`)
- `shared_rubric_rules` (with `rule_id`, `source_fact_ids`, and matcher)

Effective comparison scoring is resolved as:

- `shared_rubric_rules` (from scenario fixture)
- plus `role_rubric_rules` (from task fixture)

`audit-config` rejects comparison tasks when shared rubric linkage is missing,
when role rules redefine shared rule IDs, when required facts are uncovered,
or when legacy rubric terms remain in comparison tasks.

The runner computes a content-based `scenario_content_hash` from canonical scenario payloads.
This hash is propagated into plan rows, task snapshots, run manifests, result identity,
effective prompt artifacts, handoff payloads, and escalation reporting.

Prompt construction is componentized:

- shared scenario content
- role-specific instruction
- optional handoff fast response (handoff track only)
- output contract

`audit-config` enforces:

- missing comparison scenario references
- missing scenario files
- scenario file mismatches (`comparison_id` mismatch)
- generic/empty scenario payloads

Handoff comparisons execute an explicit fast-model x heavy-model dependency matrix in `run_plan_rows`.
Each heavy request binds to exactly one planned fast dependency and records:

- `fast_plan_row_id`
- `fast_result_identity`
- `fast_response_hash`
- `scenario_content_hash`
- `handoff_compatibility_key`

Handoff dependency pairing is restricted to fully compatible rows only:

- `comparison_id`
- `comparison_track`
- `scenario_content_hash`
- `scenario_version`
- `comparison_information_mode`
- `requested_think_mode`
- `structured_output_mode`
- `task_suite_version`

Cross-mode or cross-scenario pairing is disallowed by default (`cross_mode_handoff = false`).

Resume behavior reloads completed fast artifacts from saved `.result.json` outputs.
If dependency artifacts are missing or inconsistent, dependent heavy execution is refused.

Escalation reporting rules:

- handoff track: uses only recorded heavy-to-fast dependency identity (no synthetic Cartesian pairing)
- independent track: allows Cartesian comparison only when scenario hash/version + mode/contract dimensions match

`audition plan` now prints compatibility accounting metrics:

- `base independent requests`
- `valid handoff fast rows`
- `valid handoff dependent heavy rows`
- `cross_think_handoff_dependencies`
- `cross_output_handoff_dependencies`
- `cross_scenario_handoff_dependencies`
- `cross_information_mode_dependencies`

Supported rubric rule types:

- `required`
- `forbidden`
- `optional`
- `hard_gate`

Supported matcher types:

- `phrase_aliases`
- `regex`
- `numeric_exact`
- `numeric_tolerance`
- `json_field`
- `required_section`
- `disposition`
- `reference_fact`
- `citation_id`
- `forbidden_claim`

If a rubric-assisted task has no explicit rubric rules, validation and config audit fail.

## Source Hashing

Run manifests include `execution_source_hashes` for execution-critical inputs:

- all Python files under `src/llm_auditions/`
- verifier modules under `src/llm_auditions/verifiers/`
- config YAML files under `config/`
- schema JSON files under `schemas/`

Resume validation refuses to continue when execution source hashes drift.

## Installation

```bash
cd /home/jessica/jbp-ai-auditions/llm-auditions

# Verify dependencies (already installed on this server)
python3 -c "import pydantic, jsonschema, sympy, yaml, pytest, numpy; print('OK')"

# Install as editable package (optional)
# python3 -m pip install -e .
```

No database required. Uses Python standard library plus `pydantic`, `jsonschema`,
`sympy`, `PyYAML`, `numpy`, `requests`.

## Configuration

All configuration lives in `config/`:

```
config/
  models.yaml          — exact model names, IDs, capabilities
  defaults.yaml        — engine defaults (URL, temperature, keep_alive, etc.)
  profiles/            — smoke / standard / exhaustive profiles
  teams/               — per-team role definitions and candidate lists
```

Override Ollama URL:
```bash
export OLLAMA_URL=http://localhost:11434
```

## Commands

### List information

```bash
./bin/audition list-models          # show configured vs installed models
./bin/audition list-teams           # show all team definitions
./bin/audition list-roles           # show all roles and candidate counts
./bin/audition list-tasks           # show all task definitions
./bin/audition list-tasks --team mathematics
./bin/audition validate             # validate config against installed models
```

### Run auditions

```bash
# Smoke profile (all teams, smoke tasks only, think=false)
./bin/run-smoke.sh

# Single team, single model (fastest for testing)
./bin/audition run --profile smoke --team baseline --model qwen3.5:9b

# Standard profile (all tasks, think=false + think=low)
./bin/run-standard.sh

# Exhaustive profile (requires --yes or AUDITION_YES=1)
./bin/run-exhaustive.sh --yes
AUDITION_YES=1 ./bin/run-exhaustive.sh

# Dry-run planning and audits
./bin/audition plan --profile smoke
./bin/audition audit-config
./bin/audition audit-run --run-dir results/smoke-20260623-171623
```

### Resume interrupted runs

```bash
./bin/audition resume --run-dir results/smoke-20260623-171623
./bin/audition run --profile standard --run-dir results/standard-20260623-180000
```

Resume automatically skips completed task/model/think-mode combinations whose
identity key (team + role + task ID + task version + model digest + requested/effective think mode +
structured-output mode + prompt hash + schema hash +
fixture hashes) is unchanged.

Each new run also writes immutable per-task snapshots under `task_snapshots/`.
Resume refuses to continue if the current task definition, rubric rules, schema,
fixture hashes, config hashes, or model digests no longer match the stored run snapshot.

### Generate reports

```bash
./bin/audition report --run-dir results/smoke-20260623-171623
```

Produces:
- `summary.json` / `summary.csv`
- `leaderboard_by_role.csv`
- `leaderboard_by_role_mode.csv`
- `failures.csv`, `unsupported_modes.csv`
- `deterministic_results.json`
- `escalation_value.csv`
- `REPORT.md`
- `REPORT_FOR_EXTERNAL_REVIEW.md`

`leaderboard_by_role.csv` now includes definitive-only weighted aggregation fields:

- `weighted_task_score`, `unweighted_task_score`
- `eligible_weight_sum`, `eligible_result_count`, `total_result_count`
- `provisional_count`, `human_review_count`, `disqualified_count`, `ineligible_count`

Timing metrics are reported separately using Ollama durations and local wall-clock:

- `load_seconds`, `prompt_eval_seconds`, `generation_seconds`, `ollama_total_seconds`
- `wall_clock_seconds`, `overhead_seconds`

Escalation output is matrix-expanded by candidate combinations and mode/version keys.
Handoff rows include transferred payload context and are keyed by shared scenario identity.

### Package results

```bash
./bin/package-results.sh results/smoke-20260623-171623
```

Creates `llm-auditions-YYYYMMDD-HHMMSS.tar.gz` and `.sha256` in `results/`.
The sanitizer scans run artifacts plus packaged source/config scope before packaging and refuses if any are found.
It also writes `sanitization_findings.json` to the output directory.
Use `--safe-override` only if you are certain the flagged content is not a secret.

## Run Artifacts

- `run_manifest.json`: immutable run identity and environment snapshot
- `run_state.json`: mutable progress state (completed identity keys, counters)
- `events.jsonl`: append-only per-request event log

Immutable during resume:
- `run_manifest.json`
- `environment.json`
- `model_inventory.json`
- `task_manifest.json`

`run_manifest.json` now includes `execution_source_hashes` for engine/scoring/verifier/report modules.
Resume and `audit-run` both refuse mismatched source hashes.

Mutable during resume:
- `run_state.json`
- `events.jsonl`

Resume execution order is strict:
1. Load one exact plan row.
2. Build the pre-request identity from that exact row and task snapshot.
3. Check completed identity key set.
4. If complete, emit a skip event and do not contact Ollama.
5. If incomplete, call Ollama exactly once for that planned row and persist outputs.
6. Atomically mark completion in `run_state.json`.

Manifest creation order is strict:
1. Load config and tasks.
2. Apply profile and filters.
3. Expand think and structured-output modes.
4. Resolve model digests and fixture/schema hashes.
5. Build exact execution plan.
6. Validate against `schemas/run_manifest.schema.json`.
7. Write immutable manifest files.

Smoke profile constraints:
- Uses `use_smoke_candidates` representative role candidates.
- Defaults to `think=false`.
- Enforces a request ceiling (`max_requests`) and fails planning if exceeded.
- Manifest counts are derived from the exact plan rows, not the broader filtered task set.

Evidence-backed tasks can declare:

- `evidence.required_ids`
- `evidence.optional_ids`

The framework validates those IDs against the fixture packet and scores required-source
coverage separately from optional-source usage.

## Recommended First-Run Sequence

```bash
# 1. Validate the project
./bin/validate-project.sh

# 2. Smoke run (fast — smoke tasks only, think=false)
./bin/run-smoke.sh

# 3. Standard run (full tasks, think=false + think=low)
./bin/run-standard.sh

# 4. Generate reports
./bin/audition report --run-dir results/<standard-run-dir>

# 5. Package for external review
./bin/package-results.sh results/<standard-run-dir>
```

## Report Interpretation

**`REPORT_FOR_EXTERNAL_REVIEW.md`** is the primary document for external analysis.

Key sections:
1. **Team-by-team rankings** — average weighted scores per role per model
2. **Deterministic failures** — math/code/evidence verifier results
3. **Safety flags** — unsafe commands proposed by models
4. **Schema failures** — models that returned malformed JSON or fence-wrapped JSON
5. **Escalation-value comparisons** — whether heavy workers improve on fast workers

Scores use role-specific weights. For example:
- **Mathematics**: `deterministic_test_score` × 0.60 dominates
- **Security**: `safety_score` × 0.35 dominates
- **Language editing**: `fact_preservation_score` × 0.45 dominates
- **Fast workers**: include `latency_score` × 0.15

**Never treat the weighted_total as the sole criterion.** Zero safety scores
or failing deterministic tests are disqualifiers regardless of other scores.

## Adding Models

1. Add entry to `config/models.yaml` with name matching `ollama list` exactly.
2. Add the model as a candidate in relevant `config/teams/*.yaml` files.
3. Run `./bin/audition validate` to confirm.

## Adding Tasks

Create a YAML file in the appropriate `fixtures/<team>/` directory:

```yaml
tasks:
  - id: unique_task_id
    team: team_name
    role: role_name
    category: category_name
    smoke: true         # include in smoke profile
    system_prompt: |
      ...
    prompt: |
      ...
    required_json_schema: worker_result   # optional
    expected_disposition: ANSWER         # optional
    verifier: structure                   # optional
    temperature: 0
    num_predict: 1024
    think_modes: ["false"]
    weight: 1.0
```

Run `./bin/audition validate` after adding tasks.

## Adding a Verifier

1. Create `src/llm_auditions/verifiers/my_verifier.py` inheriting `BaseVerifier`.
2. Implement `verify(task, response) -> VerifierResult`.
3. Register it in `src/llm_auditions/verifiers/__init__.py`.
4. Reference it in task definitions with `verifier: my_verifier`.
5. Add a test in `tests/test_verifiers.py`.

## Think Modes

| Mode | Payload sent to Ollama | Notes |
|---|---|---|
| `false` | `"think": false` | Thinking disabled |
| `low` | `"think": "low"` | Minimal reasoning |
| `medium` | `"think": "medium"` | Standard reasoning |
| `high` | `"think": "high"` | Large reasoning |
| `true` | `"think": true` | Model default |

`think` is always sent as a top-level request field, and both requested and effective values are recorded.

Not every model supports every mode. Unsupported modes are recorded as
`unsupported_mode` (not as quality failures) in `unsupported_modes.csv`.

## Structured Output Modes

- `none`: no explicit output constraint
- `prompt_only`: prompt instruction only, no top-level `format`
- `ollama_json`: sends top-level `format: "json"`
- `ollama_schema`: sends top-level `format` with full JSON schema object

## Smoke vs Standard vs Exhaustive

| Profile | Tasks | Think modes | Runtime |
|---|---|---|---|
| Smoke | smoke-tagged only | false | Minutes (per model) |
| Standard | all standard | false, low | Hours |
| Exhaustive | all + edge cases | false, low, medium | Many hours |

## Known Limitations

1. **Token budget with thinking models**: `qwen3.5:9b` and `phi4-reasoning:14b`
   consume `num_predict` tokens for thinking content. Tasks with low `num_predict`
   may receive empty content if thinking fills the budget. Increase `num_predict`
   for thinking-capable models.

2. **Latency scores are relative**: `latency_score` is normalised per-run.
   Cross-run latency comparisons require examining raw `wall_clock_s` in `summary.csv`.

3. **Vision capability detection**: The framework detects multimodal capability
   from Ollama model metadata. If no installed model reports image support,
   the visual document track is not activated.

4. **Escalation value CSV**: Populated only for escalation track pairs. Requires
   both fast and heavy worker results for the same task.

## The Mission Director

**The Mission Director is not built in this project.**

After the auditions are complete and external review has assigned team roles,
the Mission Director function for Open WebUI will be built separately using
the winning models identified from these results.

Likely future candidates (starting hypotheses only):
```
Primary: qwen3.5:9b
Possible escalation: gemma4:12b
```

These are not final assignments. All final assignments follow external review.

## Project Structure

```
llm-auditions/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── models.yaml              — exact installed model inventory
│   ├── defaults.yaml            — engine defaults
│   ├── profiles/                — smoke / standard / exhaustive
│   └── teams/                   — 12 team configurations
├── fixtures/                    — task definitions (YAML)
│   ├── baseline_tasks.yaml
│   ├── general_knowledge/
│   ├── research/
│   ├── linux/
│   ├── development/
│   ├── mathematics/
│   ├── security/
│   ├── architecture/
│   ├── documents/
│   ├── integration/
│   └── language/
├── schemas/                     — JSON Schema contracts
│   ├── worker_result.schema.json
│   ├── reviewer_result.schema.json
│   ├── research_plan.schema.json
│   ├── evidence_synthesis.schema.json
│   └── run_manifest.schema.json
├── src/llm_auditions/           — shared Python engine
│   ├── cli.py                   — command-line interface
│   ├── configuration.py         — config loading and validation
│   ├── ollama_client.py         — Ollama API client
│   ├── runner.py                — task execution and resume
│   ├── models.py                — Pydantic data models
│   ├── scoring.py               — role-specific scoring
│   ├── reporting.py             — report generation
│   ├── packaging.py             — archive creation
│   ├── sanitization.py          — secret scanning
│   ├── task_loader.py           — task YAML loading
│   └── verifiers/               — deterministic verifiers
│       ├── mathematics.py       — Python/SymPy math verification
│       ├── development.py       — code syntax/lint/test
│       ├── command_safety.py    — static command safety analysis
│       ├── evidence.py          — evidence citation checking
│       ├── fact_preservation.py — language edit fact checking
│       └── contradiction.py     — integration review conflict detection
├── bin/
│   ├── audition                 — main CLI entry point
│   ├── run-smoke.sh
│   ├── run-standard.sh
│   ├── run-exhaustive.sh
│   ├── package-results.sh
│   └── validate-project.sh
├── tests/                       — 81 unit tests
└── results/                     — run outputs (gitignored)
```
