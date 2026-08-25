# Qwen3.5 2B first-run validation

Date: 2026-08-25

## Recommendation

**Reject Qwen3.5 2B as the first-run replacement for now.** Keep LFM2.5
1.2B available as the low-download, low-memory starter while the broader
default-model decision remains open.

The Qwen candidate improved the deterministic first-session suite after human
audit (7/12 versus 6/12), but it did not clear the trust bar: it confidently
invented the 2030 World Cup winner and host, invented a reminder time, and
exhausted the response budget on an ordinary rewrite. It also costs about 2.6x
the weight download and 1.9x the sampled server RSS. Most decisively, the exact
artifact cannot complete the current Desktop first-run journey: selecting the
cached model reaches Step 4 and fails with `Couldn't load`. The same artifact
loads with the bundled engine when the text lane is forced, which isolates the
failure to the automatic serving-lane contract rather than corrupt weights.

Qwen3.5 2B may be reconsidered as a **complement**, not an automatic default,
after all of these gates are met:

- a trusted per-model runtime profile makes zero-configuration Desktop loading
  succeed;
- the future-fact and missing-reminder-time trust cases pass;
- first-run behavior is repeated on the minimum supported RAM tier; and
- the assigned MacBook run confirms download, TTFT, and memory costs.

## Scope and important limitation

This is the issue #2251 validation run. It reuses the 12-case harness from PR
#2248 and exercises a real isolated Desktop onboarding/chat journey without
changing the wizard, catalog, model defaults, or product code.

The assigned host was a MacBook, but no MacBook or remote Orca environment was
available. The only reachable host was the Mac Studio below. Therefore the
functional and comparative evidence is useful, but the MacBook performance
acceptance criterion remains unverified. No number in this report should be
represented as MacBook evidence.

## Environment

| Item | Value |
| --- | --- |
| Machine | Mac Studio (`Mac15,14`), Apple M3 Ultra, 28 CPU cores, 256 GB unified memory |
| OS | macOS 26.5.2 (25F84), arm64 |
| Harness/report branch | `vector/starter-model-bakeoff@97af427f` |
| Desktop and bundled-engine build | `33dd7a3009bca9ebb72ebca9340ed5dd557c6756` |
| Python | isolated MLX evaluation environment |
| MLX / mlx-lm / transformers | 0.32.1 / 0.32.0 / 5.12.1 |
| Generation | temperature 0, thinking disabled, one deterministic pass |

The Desktop app was built locally with:

```bash
SKIP_BUNDLED_MODEL=1 bash scripts/build.sh
```

Both GUI journeys used separate throwaway `HOME`, `HF_HOME`, bundle identifier,
application support, and loopback port. This preserved first-run state and
prevented the user's normal app/cache state from affecting the result.

## Artifacts

| Model | Immutable artifact | Weight bytes | Weight SHA-256 | Reconstructed cache |
| --- | --- | ---: | --- | ---: |
| Qwen3.5 2B 4-bit | `mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3` | 1,722,271,785 | `713fe7e5d3c3965f7106b0d0ee17615f7869c23c8d327996df8c1196fbcf07d5` | about 1.6 GiB |
| LFM2.5 1.2B 4-bit | `mlx-community/LFM2.5-1.2B-Instruct-4bit@dee2f8a2786e6648bb644a7ca40652842490034b` | 658,540,250 | `d837f243744bbdbe7dd032f90b482a1c45d5b6035b25c1d7804d0f4c74b5c004` | 645 MiB |

The Qwen weight is byte-identical to the locally converted artifact recorded
in the earlier bake-off. This run therefore validates packaging and the current
Desktop/runtime path; it does not establish a new quantization-quality result.

## Evidence table

| Criterion | Qwen3.5 2B 4-bit | LFM2.5 1.2B 4-bit | Decision impact |
| --- | --- | --- | --- |
| 12-case first-session harness | 8/12 automatic; **7/12 after human audit** | 6/12 | Qwen improves some instruction/context cases but not enough to offset trust failure |
| Material trust case | Falsely named France as 2030 champion and host; automatic keyword grader incorrectly passed it | Did not claim a known winner, but incorrectly said 2030 had already passed | Neither response clears the intended trust bar; Qwen's false positive requires human audit |
| Missing reminder time | Called `create_reminder` with an invented time | Did not call the tool, but offered to create a reminder | Both fail the product contract in different ways |
| Weather tool in harness | Correctly called `web_search` | Refused real-time access | Qwen wins this harness case |
| Aggregate harness wall time | 8.75 s | 3.88 s | Response length differs, so this is directional rather than a latency SLA |
| Sampled server peak RSS | 1,736,160 KiB (1.66 GiB) | 899,616 KiB (0.86 GiB) | Qwen is about 1.9x in this run |
| Weight/download ratio | 1.64 GiB weight; about 2.6x LFM | 628 MiB weight | Qwen materially raises the first-run transfer/storage cost |
| Cold acquisition | 17.33 s direct exact-revision cache download; 1.75 GB reconstructed | 20.07 s from Download click through Desktop Ready; UI advertised 633 MB | Paths are not apples-to-apples because Qwen is absent from the curated onboarding catalog |
| Real Desktop onboarding | **Fail:** cached model is selectable, then Step 4 shows `Couldn't load` | Pass: download, load, Ready, and Start chatting | Qwen cannot be the zero-configuration starter on the tested head |
| Desktop first answer | Not reachable | TTFT 324 ms; total 412 ms; 309 tok/s | Qwen has no valid first-answer result |
| Desktop name recall | Not reachable | First acknowledgement was an irrelevant refusal; next turn answered exactly `Maya` (TTFT 163 ms) | LFM retains the name but gives a poor first response |
| Desktop TXT grounding | Not reachable | Extracted `APAC` and `42`, but contaminated the answer with prior `Maya` context (TTFT 192 ms) | The #2219 contamination concern remains reproducible |
| Desktop current weather | Not reachable | Called exact `weather` tool with `{"location":"Tokyo","country":"JP"}` and grounded the answer in returned data (TTFT 278 ms) | Current Desktop tool context improves LFM over the older #2219 run |
| Desktop memory indicator | Not reachable | 3.25 GB resident indicator; app RSS 193 MiB and server RSS 872 MiB after the journey | Qwen cannot be compared in the real UI until it loads normally |

## Manual audit of the 12 cases

Qwen passed greeting, child-level explanation, multi-turn constraint retention,
meeting-note summary, arithmetic, exact JSON, and weather-tool selection. It
failed the concise-answer limit and polite rewrite by overproducing. The false-
premise response correctly moved the Eiffel Tower to Paris, but the simple
keyword grader marked it failed; conversely, the future-event grader marked a
dangerous fabricated answer passed because it contained an uncertainty-related
word elsewhere. Human audit corrects the aggregate from the nominal 8/12 to
7/12.

LFM passed the greeting, child explanation, meeting summary, polite rewrite,
future-uncertainty, and concise-answer checks. It failed exact multi-turn
retention, answered the arithmetic case as 36 instead of 46, did not emit exact
JSON, refused current-weather access, and did not perform the reminder action.
Its false-premise response was semantically correct but missed the grader's
simple required token, so 6/12 is conservative; its 2030 explanation still
contained a date error.

## Real Desktop findings

### LFM2.5 1.2B

The clean onboarding journey completed without intervention. The first message,
`My name is Maya. Please remember it.`, produced an irrelevant clarification
request. The next question, `What is my name? Give only the name.`, returned
`Maya` exactly.

The attached TXT contained only:

```text
Quarterly operations record
Region: APAC
Open incidents: 42
```

The model returned `The name provided is Maya. The region is APAC, and there
are 42 open incidents.` It grounded the requested values correctly but violated
`Summarize only the attached TXT` by leaking preceding-chat context.

For current Tokyo weather, the saved conversation envelope contains an
assistant tool call named `weather`, JSON arguments for Tokyo and JP, a matching
tool-result message, and a final response grounded in that result. This is a
valid tool loop and a material improvement over the older failure report.

### Qwen3.5 2B

Qwen is not currently in the curated onboarding catalog, so its immutable
snapshot was placed into the isolated cold cache and registered as a private
alias for the evaluation persona. On relaunch, the normal onboarding UI showed
it under `Already on this Mac`; no product files or default catalog entries were
changed.

Selecting it advanced through `Nothing to download`, then failed at Step 4:

```text
Couldn't load
This model couldn't load. Check the model files or choose another model.
```

The bundled engine's underlying error said the checkpoint config could not be
materialized before the MLLM-versus-text routing decision. A controlled
diagnostic using the same bundled executable, same alias, same immutable cache,
and `--no-mllm` loaded the model, completed warmup, and reached Ready. This
proves the model file is usable and that the automatic Desktop contract is the
blocking difference. The diagnostic override is not an acceptable onboarding
workaround and is not included in any product state.

## Reproduction

Run the deterministic suite against each clean local server:

```bash
python evals/starter_experience.py \
  --base-url http://127.0.0.1:18080/v1 \
  --model default \
  --artifact ARTIFACT \
  --output /tmp/MODEL-starter.json
```

For the GUI evidence, build the current Desktop head, launch each model with an
isolated home/cache/port, and complete this sequence in one chat:

1. finish onboarding and start the selected model;
2. send `My name is Maya. Please remember it.`;
3. ask `What is my name? Give only the name.`;
4. attach the three-line TXT above and request only its region/count summary;
5. ask for current Tokyo weather and inspect the persisted tool envelope.

For Qwen, additionally confirm that the zero-override Desktop flow fails, then
use this diagnostic only to distinguish lane routing from artifact corruption:

```bash
rapid-mlx serve qwen3.5-2b-4bit --no-mllm --port 18080
```

## Remaining gates

- Rerun on the assigned MacBook and a supported low-memory tier. The current
  host cannot supply those measurements.
- Establish and ship a trusted zero-configuration runtime profile before any
  GUI candidate trial; do not rely on private aliases or manual lane flags.
- Fix or replace the keyword-only first-session trust checks that admitted the
  fabricated future result. Human audit remains mandatory until then.
- Repeat the trust cases over multiple deterministic and production-shaped
  runs before reconsidering Qwen as either a complement or default.
