# QBitPlan domain context

This document defines the shared vocabulary used in issues, ADRs, code,
experiments, plots, and paper drafts. It is a glossary, not an implementation
specification.

## Decision language

- **ACCEPTED** means the project has adopted the term, invariant, or decision
  for the current phase.
- **PROPOSED** means a bounded direction documented for review, not an
  authorization to implement it.
- **OPEN** means that the project has intentionally not selected an option.
  An open choice must not be filled in by convention or implication.

## Document roles and source-of-truth hierarchy

When project documents disagree, report the conflict explicitly and resolve it
according to this order:

1. Accepted ADRs — accepted technical or research decisions.
2. CONTEXT.md — project terminology.
3. SCIENTIFIC_STANDARDS.md — scientific and reporting invariants.
4. The originating GitHub issue — task scope and acceptance criteria.
5. Documents under docs/research/ — research background and inputs, not
   accepted specifications.
6. README.md — public orientation.

## Core terms

### Query

The complete model input whose precision requirements are being planned. A
query may include the instruction, context, and any task-specific input that
the model receives for one evaluation unit.

### Ground truth

The externally defined task target, such as a correct answer or reference
continuation. Ground truth is distinct from the output of the high-precision
reference execution.

### High-precision reference

The declared reference execution against which quantized behavior and
degradation are compared. Its numeric format is OPEN; it is not itself the
task ground truth.

### Layer group

A contiguous set of Transformer layers assigned one precision decision in the
proposed v0.1 formulation.

### Bit profile

The canonical term for an ordered vector of bit-width decisions, one per layer
group. In proposed v0.1 examples, each decision is 4 or 8.

Example: [4, 4, 8, 4, 8, 8, 4, 4].

### Precision budget

A constraint on execution cost. It is not synonymous with an average-bit
target.

### Precision shape

The distribution of extra precision across layer groups. Two queries may have
similar total cost but different precision shapes.

### Degradation

A task- or distribution-level loss relative to the declared high-precision
reference. The exact primary metric and accepted threshold are OPEN for the
experiment protocol. For tasks with answer correctness, correctness remains
the final quality metric; KL divergence and hidden-state distance are
diagnostic unless a later protocol says otherwise.

### Quantization difficulty

The minimum measured hardware cost required for a query to remain within the
experiment's accepted degradation threshold. It is not a human difficulty
label.

### Interaction

A condition in which the value or sensitivity of one layer group depends on
precision choices made for other groups or on the upstream execution context.

### Upstream context

The precision decisions and resulting execution state that exist before a
later layer group runs.

### Independent planner

A planner that uses additive group contributions, including the mandatory
independent MCKP baseline. This is a comparison baseline, not a claim that
layer groups are truly independent.

### Interaction-aware planner

A planner whose score or decision can depend on other precision choices or
upstream execution context. The exact algorithm is OPEN.

### Causal signal

Information available before the decision it influences. A signal observed
after group g may influence groups g+1 onward, but may not retroactively
influence group g.

### Promotion

A causal change from lower to higher precision for a future layer group.

### Oracle profile

An offline bit profile selected using measured outcomes unavailable to the
online controller. It estimates achievable headroom and may supply labels or
bounds; it is not an online controller.

### Hardware-executable profile

A discrete bit profile that the eventually declared execution backend can
execute. The specific hardware, backend, and measured cost remain OPEN.

### Simulated cost

A cost calculated from a proxy, analytical model, or lookup table rather than
directly observed execution on the declared hardware and backend.

### Measured cost

A cost obtained directly from the declared hardware and execution backend.

### Primary quality metric

The task-appropriate metric declared before evaluation as the main quality
outcome. For tasks with externally judged answers, this is task correctness.
For a language-modeling task without task labels, perplexity may be primary.

### Diagnostic metric

A supporting metric used to explain or analyze behavior, such as KL divergence,
answer flips, or hidden-state distance. It must not silently replace the
primary quality metric.

### Under-precision

A selected bit profile violates the accepted quality threshold.

### Over-precision

A cheaper bit profile would also have satisfied the accepted quality
threshold.

### Profile medoid

An executable representative bit profile selected from a cluster of profiles.
