# TMLR Submission Roadmap

Revision plan based on pre-review feedback, prioritized by dependency order.

---

## Phase 1 — Additional Experiments *(run before editing the manuscript)*

These should be completed first, as the results will directly inform several text changes.

- [ ] Re-run **IFEval** and **TruthfulQA-MC2** with **3 seeds** for the following configurations, on both models:
  - Baseline (4.0× for 1B / 2.67× for 3B)
  - 2.4× (equilibrium point)
  - Most aggressive ratio (1.6× for 1B / 1.07× for 3B)
- [ ] Report **mean ± standard deviation** for these configurations in the manuscript

---

## Phase 2 — Editorial / Administrative Changes

- [ ] Update author affiliation to **Independent Researcher**
- [ ] Rename **MAW → PPM (Peak-to-Peak Magnitude)** throughout the entire manuscript (no reference to the previous name needed)

---

## Phase 3 — Claim Adjustments *(after Phase 1 data is available)*

- [ ] Soften language around the Llama-1B correlation (p=0.096 is not statistically significant — remove "robust" or equivalent wording)
- [ ] Revise or remove the combined two-model correlation, which may violate statistical independence assumptions
- [ ] Update IFEval claims with variance data from Phase 1
- [ ] Be more explicit about the modest **absolute** magnitude of IFEval gains (alongside the relative gains already reported)

---

## Phase 4 — Technical Clarifications

- [ ] Add detail on the PPM scoring: tensor shapes, reduction axes, and normalization (if any)
- [ ] Explicitly document the **decoding configuration** used across all benchmarks (temperature, top-p, max tokens)
- [ ] Clarify seed usage — the efficiency tables mention seeds 42, 123, 456, but it is not clear whether multiple seeds were used for the main benchmark evaluations

---

## Phase 5 — Related Work and Limitations

- [ ] Expand **Related Work** to discuss STAT and NIRVANA, framing the scope difference rather than providing direct comparisons
- [ ] Strengthen the **Limitations** section to explicitly acknowledge:
  - Absence of comparisons with stronger pruning baselines (SliceGPT, STAT, NIRVANA)
  - No post-pruning recovery experiments (e.g., LoRA fine-tuning)
  - Hardware-agnostic pruning (no dimension alignment to multiples of 8/16/64)
  - Single inference environment (Colab L4)

---

## Out of Scope

The following were considered and deliberately excluded to preserve the paper's focus:

- **Knowledge Distillation** — would shift the scope from "what width pruning does" to "compression technique comparison"
- **Direct benchmark comparisons with STAT/NIRVANA** — different experimental setups make fair comparison unfeasible within the current scope
