# QBitPlan domain context

## Purpose

This document defines the shared vocabulary used in issues, ADRs, code,
experiments, plots, and paper drafts.

## Core terms

### Query

The complete model input whose precision requirements are being planned.

### Layer group

A contiguous set of Transformer layers assigned one precision decision in
the v0.1 formulation.

### Bit profile

An ordered vector of bit-width decisions, one per layer group.

Example:

`[4, 4, 8, 4, 8, 8, 4, 4]`

### Precision budget

A constraint on execution cost. A precision budget is not necessarily an
average-bit target.

### Precision shape

The distribution of extra precision across layer groups. Two queries may
need similar total cost but different precision shapes.

### Degradation

A loss of behavior relative to the selected high-precision reference.
Exact task correctness is final; KL divergence and hidden-state distance are
diagnostic measures.

### Quantization difficulty

The minimum measured hardware cost required for a query to stay within an
accepted degradation threshold.

### Interaction

A condition in which the value or sensitivity of one group depends on the
precision choices made for other groups.

### Upstream context

The precision decisions and resulting execution state before a group runs.

### Independent planner

A planner that treats group contributions as additive. Independent MCKP is
the mandatory baseline, not an assumption that the system is truly
independent.

### Interaction-aware planner

A planner whose score for a group can depend on other precision choices or
upstream execution context.

### Causal signal

Information available before the decision it influences. A hidden state
after group g may influence groups g+1 onward, but not group g retroactively.

### Promotion

A causal change from lower to higher precision for a future group.

### Oracle profile

A profile selected using measured outcomes unavailable to the online
controller. It estimates achievable headroom and supplies labels or bounds.

### Under-precision

A selected profile violates the accepted quality threshold.

### Over-precision

A cheaper profile would also have satisfied the quality threshold.

### Profile medoid

An executable representative profile selected from a cluster of profiles.

### Simulated cost

A cost calculated from a proxy, analytical model, or lookup table.

### Measured cost

A cost obtained directly from the declared hardware and execution backend.
