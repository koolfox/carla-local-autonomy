# Research manuscript: evidence checklist and reviewer matrix

This is the paper's **proposed evaluation protocol**, not a promise of acceptance
or a universal journal checklist. Requirements concern the chosen claim. The
method paper should focus on measured ADAS perception benefits of error-source-aware
data acquisition; a dataset descriptor instead needs a sufficiently useful released
dataset and technical validation. Software completion is not an empirical result.

## Claim scope and exclusions

Claim CARLA as an upstream simulator, not this project's invention. Distinguish
annotation supervision from a behavioral teacher, RGB-only from RGB-plus-speed,
visible boxes from amodal geometry, shadow inference from driver-in-the-loop
control, and saved-image replay from simulator rerun. Do not claim online weight
updates or sim-to-real success without corresponding experiments. Contributing a
system is legitimate; feature aggregation alone does not prove new methodology.

## Gate-to-evidence matrix

| Reviewer question | Required retained evidence | Executable support | Not established by software alone |
|---|---|---|---|
| What exactly is new? | nearest-work comparison, formal decision rule and falsifiable H1/H2 | study protocol, methods/claim registry | priority, importance or novelty |
| Did the claimed implementation run? | selected source commit, exact source bytes, lock, CLI and logs | source snapshot with Git object proof; JUnit; CI | runtime on unseen hardware |
| Are images and labels aligned? | frame/timing/calibration diagnostics and failed attempts | common dataset/teacher verification; corruption/golden tests | rendered semantics and real sensor timing |
| Are partitions independent? | source split inventories, lineage and observed corridor audit | voxel/imitation split checks; 3D transitive graph | all unobserved roads or route gaps |
| Are hard valid cases retained? | separate clean-hard, capture-fault, label-fault and mixed regimes | diagnosis confusion/action analysis, unknown decisions | actual fault prevalence |
| Does supervision deserve trust? | blinded paired reviews, adjudication, sample/group sampling rule | audit references, agreement and conservative quality bounds | correctness from agreement alone |
| Is the teacher available only when permitted? | observation/action and runtime input contract | timing and RGB-plus-speed checks | absence of hidden deployment information in arbitrary plugins |
| Is a baseline fairly implemented? | citations, exact algorithm version, fit data, settings, cost | internal heuristics; published greedy k-center component | noisy-oracle competing papers not yet reproduced |
| Are budgets matched? | all render/probe/feature/review/repair/train attempts and fixed conversions | validated ledgers; measured process resources; bound result rows | missing undisclosed work, GPU/energy not measured |
| Are generations reproducible? | seeds, failed-run registry, scenario recipes, actual map/actor settings | frozen plans, cycle receipts, source identity | bitwise cross-GPU simulator determinism |
| Can the data be large? | storage/peak-memory/throughput profile on representative runs | canonical shards, recovery inventory, training adapters | measured long-run scale/performance |
| Are results aggregated correctly? | predictions, exact image order, ontology, threshold artifacts | pooled COCO, support-aware ratios, paired/group intervals | generalization from toy fixtures |
| Is model selection independent of test? | locked checkpoint/protocol and test-open event | separate test CLI, immutable source binding | proving historical first access without an external record |
| Are uncertainty and power defensible? | pilot-only plan, effect size, seeds/groups, raw paired runs | paired/crossed bootstrap, Holm helper | independence assumptions and adequate actual sample size |
| Is superiority demonstrated? | same-budget task-matched comparisons across learners and holdouts | run lattice and source-bound CSV generation | effect direction/magnitude until data exist |
| Does the proposed mechanism matter? | remove attribution/quality/exploration/coverage separately | ablation planner and output references | causal attribution from uncontrolled comparisons |
| Does it generalize? | withheld map/weather/routes; external real data when claimed | scoped reviewer/claim gate | sim-to-real transfer or operational safety |
| Can another person reproduce it? | relocated package, exact replay commands, independent execution log | closure verification and saved-prediction COCO replay | independent retraining or independence of a named person |
| Can the artifacts be shared? | owner-reviewed per-asset licenses, provenance, author approvals | rights inventory checks, no automatic uploads | legal permission, choice of project license |
| Does the prose match results? | claim/table/figure inventory, limitations and author approval | fixed role checker and actual result predicates | peer review of prose or statistics |

## Experiments before a methods-paper submission

Study A tests diagnosis and action selection with distinct known fault regimes,
including clean but difficult night/occluded/small-object cases. Keep injected truth
separate from allowed signals. Report false rejection of valid hard samples,
confusion among error sources, abstention, and group-level uncertainty. Add a
blinded independently audited subset; record disagreements and corrections.

Study B compares equal-cost selection on a shared candidate pool: random, coverage,
uncertainty/diversity, error-only, noise-aware, capability-matched review/recapture,
and the full method. Include a faithful nearest noisy-oracle detection baseline
and disclose any adaptation. Greedy k-center is a useful published component, not
a substitute for the nearest competing method. The same independent review budget
and supported actions must be available to the strongest control.

Study C runs the real adaptive generation/retraining cycle with the same all-attempt
budget. Lock initial weights, data augmentation, optimization steps, stopping rule,
threshold selection and search effort. Compare at least two justified learner
families; retain all generation/training seed cells. Report AP and per-condition
support, coverage, invalid/unknown sample rates, actual wall/GPU/audit/storage costs,
and failures. Do not average shard AP. A result confined to injected corruption
must be described as such, not as universal performance improvement.

Use independent acquisition, model-selection validation and locked test roles.
Register a smallest practically important effect before opening test. Determine
seed/group counts with an independent pilot and disclosed compute limits, not a
supposed universal Q1 minimum. Estimate seed and held-out-group uncertainty at their
correct levels; retain failed runs and explain missing cells rather than dropping
unfavorable trials. Report negative/null results and parameter sensitivity.

## Submission assembly

Use Introduction (problem and contribution), Related Work (nearest competing
methods), Method (formal decision/action rule), Implementation (scope), Experiments
(preregistered design), Results (only measured outputs), Discussion, Limitations,
Ethics/rights and Code/Data Availability. Every result table should link to a
registered producer and a recomputation command. Add dataset/model cards, environment
and hardware, map/sensor/actor readback, seed inventories, sample selection rules,
blinded QA protocol, costs, expected runtime, licenses and a reviewer troubleshooting
page. Obtain human author approval and an independent rerun. Never invent its author.

After choosing the actual venue/article type, recheck its current scope, review
anonymity, length, templates and availability policies. Quartiles depend on ranking
system/year/category and are not certified here. IEEE T-ITS explicitly expects a
transportation-system application/benefit [R1]; Scientific Data's descriptor format
emphasizes data records, technical validation and reuse [R2]. They are different
publication tracks, not interchangeable ways to bypass missing empirical evidence.

## Primary sources used for this protocol

[R1] IEEE ITS Society, *IEEE Transactions on Intelligent Transportation Systems*,
Scope and Information for Authors. https://ieee-itss.org/pub/t-its/ (checked 2026-10-16).

[R2] Nature, *Scientific Data: Submission guidelines / Data Descriptors*.
https://www.nature.com/sdata/submission-guidelines . Article-type policy should be
rechecked at submission; no quartile or acceptance probability is inferred.

[R3] O. Sener and S. Savarese, *Active Learning for Convolutional Neural Networks:
A Core-Set Approach*, ICLR 2018, Algorithm 1. https://arxiv.org/abs/1708.00489 .
Implementation in this repository covers greedy k-center, not the robust solver.

[R4] *Deep Active Learning with Noisy Oracle in Object Detection*, 2023 preprint.
https://arxiv.org/abs/2310.00372 . Directly relevant comparison; not yet reproduced
by the internal heuristics or the greedy k-center component.

[R5] *Active Data Acquisition in Autonomous Driving Simulation*, 2023 preprint.
https://arxiv.org/abs/2306.13923 . Error-driven acquisition alone is not a claim of
first use in simulation. Compare exact inputs, objectives, actions and costs.
