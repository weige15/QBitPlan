# Controller prototype decision and claims ledger

Status: **engineering prototype; simulated and non-evidentiary**

## Decision log

| Status | Decision or question | Rationale | Evidence required to change it |
|---|---|---|---|
| Accepted for prototype | Implement the contract-fixed OLS selectors before a neural router | Minimizes architecture assumptions and gives static, direct, independent, and causal seams now | A superseding execution contract or evidence that OLS cannot exercise the accepted estimand |
| Accepted for prototype | Keep the prototype separate from `ExperimentPlan -> ArtifactBundle` | Synthetic results must not enter immutable Stage-1 evidence lineage | Valid `P_exec`, quality artifacts, target sets, and an accepted integration issue |
| Accepted for prototype | Use all 256 synthetic profiles | Exercises canonical IDs and executable-constrained selection | A real measured `P_exec` from a new immutable inventory |
| Rejected | Train an MLP, GRU, attention controller, or MoE-style router now | No real executable target labels exist; added capacity would not repair missing evidence | Successful prerequisite artifacts and a predeclared comparison protocol |
| Rejected | Treat average bit width as a cost objective | It does not establish resident memory, transfer bytes, kernel latency, or end-to-end latency | Accepted dimension-specific measured or estimated cost evidence |
| Rejected | Use final correctness, future hidden states, or selected outcomes as features | Violates the causal decision boundary | None; this remains prohibited |
| Unresolved | Which profiles are executable after the CPU-first OOM fix? | Current directly executed inventory has empty `P_exec` | New UUID-pinned RTX 3090 smoke and complete immutable inventory |
| Unresolved | Do query-only features predict real target sets? | Synthetic separability is constructed | Functional-quality outcomes, epsilon-feasible profiles, training target sets, held-out validation results |
| Unresolved | Does upstream context add value beyond independent scoring? | Synthetic recurrence is constructed | Paired held-out interaction test with actual prefix contexts and accepted bootstrap |
| Unresolved | Is controller overhead amortized? | No controller, probe, feedback, transfer, or kernel measurements exist | Directly measured overhead with separate resident bytes, transfer bytes, latency, stalls, and switches |

## Claims ledger

| Claim ID | Classification | Claim | Supporting artifact | Permitted interpretation |
|---|---|---|---|---|
| CP-1 | Evidence: simulated | The direct query-only implementation exceeds the static implementation on the deterministic fixture | `controller-prototype-metrics.json`, Figure 1, focused test | Code-path and fixture behavior only |
| CP-2 | Evidence: simulated | The causal interaction implementation exactly recovers the constructed prefix recurrence | `controller-prototype-metrics.json`, Figure 1, causal-prefix focused test | Representation and causal-plumbing check only |
| CP-3 | Evidence: simulated | The additive independent implementation fails on most later-group recurrence decisions | Figure 1 per-group curve | Expected negative control for this fixture only |
| CP-4 | Proposed Design | The same interfaces can consume future real target and context artifacts | typed interfaces and module contracts | Integration hypothesis, not demonstrated Stage-1 evidence |

**Evidence.** No novelty claim is made. No comparison with QAQ, DP-LLM,
MoBiQuant, MixQuant, IMPQ, Any-Precision LLM, MatGPTQ, or other literature is
made by this prototype.
