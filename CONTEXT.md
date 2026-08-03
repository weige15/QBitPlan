# QBitPlan domain context

This document is the canonical glossary for terms used in issues, ADRs,
experiments, plots, and paper drafts. It is not an implementation
specification.

## Decision language

- **ACCEPTED** — adopted for the current project phase.
- **PROPOSED** — a bounded direction recorded for review, not authorization to
  implement.
- **OPEN** — intentionally unresolved; do not fill it in by convention.

## Core terms

### Query

The complete model input whose precision requirements are being planned. It
may include the instruction, context, and task-specific input for one
evaluation unit.

### Ground truth

The externally defined task target, such as a correct answer or reference
continuation. It is distinct from a model's reference execution.

### High-precision reference

The declared reference execution against which quantized behavior and
degradation are compared.

### Layer group

A contiguous set of Transformer layers assigned one precision decision.

### Bit profile

An ordered vector of bit-width decisions, one per layer group.

### Precision budget

A constraint on execution cost. It is not synonymous with an average-bit
target.

### Cost vector

The ordered set of execution-cost dimensions reported for a query/profile
execution. Its components remain separate and may be measured, estimated, or
explicitly omitted; it is not an average-bit target or an arbitrary scalar
coefficient.

### Precision shape

The distribution of extra precision across layer groups. Two queries may have
similar total cost but different precision shapes.

### Degradation

A loss of task or model behavior relative to the declared high-precision
reference.

### Quantization difficulty

The minimum cost required for a query to remain within an accepted degradation
threshold. It is not a human difficulty label.

### Interaction

A condition in which the value or sensitivity of one layer group depends on
precision choices made for other groups or on upstream execution context.

### Upstream context

The precision decisions and resulting execution state that exist before a
later layer group runs.

### Independent planner

A planner that scores group choices additively.

### Interaction-aware planner

A planner whose score or decision can depend on other precision choices or
upstream execution context.

### Causal signal

Information available before the decision it influences.

### Promotion

A causal change from lower to higher precision for a future layer group.

### Oracle profile

An offline bit profile selected using outcomes unavailable to the online
controller. It estimates achievable headroom and is not an online controller.

### Hardware-executable profile

A discrete bit profile that the declared execution backend can execute.

### Simulated cost

A cost calculated from a proxy, analytical model, or lookup table rather than
directly observed execution.

### Measured cost

A cost obtained directly from the declared hardware and execution backend.

### Primary quality metric

The task-appropriate metric declared before evaluation as the main quality
outcome.

### Diagnostic metric

A supporting metric used to explain or analyze behavior; it does not silently
replace the primary quality metric.

### Under-precision

A selected bit profile that violates the accepted quality threshold.

### Over-precision

A cheaper bit profile that would also have satisfied the accepted quality
threshold.

### Profile medoid

An executable representative bit profile selected from a cluster of profiles.
