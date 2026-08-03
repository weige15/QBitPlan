# ADR-0014: Stage-1 stop/continue rules

- Status: ACCEPTED
- Date: 2026-08-04
- Decision owners: project maintainers

## Context

The originating decision ticket is [Set Stage-1 stop/continue rules](https://github.com/weige15/QBitPlan/issues/19).
The Stage-1 signal, interaction, and oracle-headroom tests are defined by
[ADR-0011](0011-stage-1-signal-test-and-success-threshold.md),
[ADR-0012](0012-stage-1-interaction-test-and-success-threshold.md), and
[ADR-0013](0013-stage-1-oracle-headroom-and-stop-criterion.md). The map's
predeclared selection rule requires the scientific ordering signal →
interaction → oracle headroom and requires stopping before sophisticated
controller work when the oracle offers little benefit over a static profile.

This ADR defines what evidence opens the next gate, what valid results stop
the sequence, and which failures require protocol revision rather than
scientific interpretation. It does not select a controller architecture or
authorize implementation.

## Decision

### Ordered gates

Stage 1 uses three ordered falsification gates.

1. **Signal gate.** A causal precision-profile predictor succeeds only when
   its feasible-profile hit rate improves over both the frozen static baseline
   and the additive independent baseline by at least 0.05, with the paired
   95% bootstrap lower bound above zero for both comparisons.
2. **Interaction gate.** An interaction-aware planner succeeds only when its
   feasible-profile hit rate improves over the additive independent baseline
   by at least 0.05, with the paired 95% bootstrap lower bound above zero.
3. **Oracle-headroom gate.** Useful headroom exists only when the
   Pareto-dominance opportunity rate over the frozen static profile is at least
   0.05 on the fixed cost frame, with the paired 95% bootstrap lower bound
   above zero.

The signal and interaction gates use the primary feasible-profile hit-rate
endpoints and their accepted paired query contracts. The oracle gate uses the
accepted offline Pareto opportunity-rate endpoint and does not introduce a
scalar cost objective.

### Transitions and stopping

- Only a successful signal gate permits the interaction gate.
- Only a successful interaction gate permits the oracle-headroom gate.
- Only useful oracle headroom clears the stop criterion for considering later
  controller decisions.
- A valid practical null, inconclusive result, or negative result stops the
  ordered sequence at that gate.
- Continuing means proceeding to later research or protocol decisions. It
  does not authorize implementation, controller architecture selection,
  amortization claims, or serving claims.

For valid results, a positive improvement below the five-point margin is a
practical null; an uncertainty interval including zero is inconclusive; and a
worse predictor or planner result is negative. The oracle opportunity-rate
gate uses useful headroom, practical null, and inconclusive classifications
because its primary opportunity rate is non-negative. Existing gate ADRs
remain authoritative for endpoint-specific reporting.

### Protocol-revision boundary

Protocol revision is reserved for validity failures, including:

- causal-boundary or query-split violations;
- use of forbidden final IDs or post-decision information;
- broken paired evaluation;
- missing required artifacts or uncertainty calculations;
- failure of the frozen static quality gate; or
- no interpretable common cost dimension.

A valid scientific result must not be reinterpreted by changing the protocol.
Revisiting a stopped gate requires a superseding ADR. A secondary signed
cost delta does not override the oracle gate's primary classification.

## Source facts, inference, and project preference

**Source facts.** ADR-0011 defines the signal endpoint, static and
independent baselines, five-point margin, paired bootstrap rule, and outcome
taxonomy. ADR-0012 defines the interaction endpoint, causal upstream
boundary, identical cost contract, five-point margin, and paired bootstrap
rule. ADR-0013 defines the offline oracle-headroom endpoint, frozen static
comparator, fixed 256-query cost frame, Pareto policy, five-point margin, and
paired bootstrap rule. `SCIENTIFIC_STANDARDS.md` requires causal information
boundaries, paired evaluation, explicit uncertainty, honest null reporting,
and separation of evidence classes. The map selection rule supplies the
signal → interaction → oracle-headroom ordering and the stop-before-controller
condition.

**Inference.** The strict sequence treats each gate as a prerequisite
falsification checkpoint: predictive signal must precede interaction testing,
interaction value must precede oracle-headroom interpretation, and useful
gross oracle headroom must precede consideration of later controller work.
This preserves the distinction between a valid scientific null and an
invalid protocol execution.

**Project preference.** The project prefers conservative five-percentage-point
practical margins, paired uncertainty, explicit practical-null/inconclusive/
negative reporting, and no controller authorization from gross oracle
headroom alone.

## Consequences and retained uncertainty

- The result is bounded to the pinned model, backend, execution configuration,
  manifests, non-final MATH evaluation boundaries, and included cost
  dimensions established by the related ADRs.
- Exact bootstrap seeds and replicate counts, prompt and answer-normalization
  details, raw-artifact schema, and detailed measurement procedures remain
  consolidated-protocol fields.
- If the signal predictor passes one baseline comparison but not the other,
  the conjunctive signal gate is not successful. The consolidated protocol
  must report comparator-specific outcomes rather than inventing a single
  precedence label; this retained reporting detail does not permit
  continuation.
- A useful oracle result is necessary but not sufficient for any later
  controller implementation or serving claim.

## Related records

- [ADR-0003: Stage-1 cost vector and profile-comparison policy](0003-stage-1-cost-contract.md)
- [ADR-0009: Stage-1 degradation and epsilon selection](0009-stage-1-degradation-and-epsilon.md)
- [ADR-0011: Stage-1 signal test and success threshold](0011-stage-1-signal-test-and-success-threshold.md)
- [ADR-0012: Stage-1 interaction test and success threshold](0012-stage-1-interaction-test-and-success-threshold.md)
- [ADR-0013: Stage-1 oracle-headroom test and stop criterion](0013-stage-1-oracle-headroom-and-stop-criterion.md)
- [Set Stage-1 stop/continue rules](https://github.com/weige15/QBitPlan/issues/19)
- [Accept the consolidated Stage-1 protocol and ADR set](https://github.com/weige15/QBitPlan/issues/20)
