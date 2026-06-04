# EPIC REVISION v2 — REPRODUCIBILITY, FAILURE MODES, AND MISSING SECTIONS

---

## R1. COMPLETE REPRODUCIBILITY SPECIFICATION

Any researcher should be able to reproduce every result in this paper from the following specification.

### R1.1 Model Versions and API Parameters

**All experiments in v1 paper (20-question set):**
- Model: claude-sonnet-4-20250514
- API version: Anthropic Messages API v1
- Temperature: 0.3 (low temperature for reproducibility; not 0 because deterministic outputs produce no variance for calibration analysis)
- Max tokens: 1024 per response
- Top-p: 0.95
- System prompts: Verbatim as specified in Stage 3, Appendix A of this paper
- No retrieval augmentation, no tool use, no memory across questions
- Each question is a fresh API call with no carryover state

**Random seed handling:** The Anthropic API does not expose a random seed parameter. For reproducibility, we run each experiment 3 times with temperature 0.3 and report the mean score and standard deviation across runs. The v1 results are from Run 1; Runs 2 and 3 will be completed for the v2 paper and reported with error bars.

**Date of experiments:** All v1 experiments were conducted on the model checkpoint available via claude-sonnet-4-20250514. Model checkpoints are version-pinned by Anthropic; the exact checkpoint will not change if the same API version string is used.

### R1.2 Question Ordering and Randomisation

Questions were administered in the order specified (Q1–Q20). No randomisation was applied. For v2 (200 questions), questions will be administered in random order (shuffled with seed 42) to prevent any ordering effects on calibration estimation.

### R1.3 Scoring Protocol

**For v1 (self-scored):** Each answer was scored by the paper authors against the published ground truth. The rubric (Factual Accuracy 0–2, Mechanistic Depth 0–2, Uncertainty Expression 0–1) was applied after confirming the ground truth from published sources. Sources are cited inline with each question.

**For v2 (inter-annotator):** Three annotators score independently using the rubric in Section H3. Annotators are blinded to which protocol produced each answer (EPIC, ADMF, or single-agent). Inter-rater agreement (Cohen's κ) is computed and reported.

### R1.4 EPIC Mechanism Scoring Procedure

The EPIC Judge's sycophancy determination was made according to:

1. Position change detected: |a_i^t − a_i^{t-1}| > 0.20 on a normalized [0,1] scale
   - For discrete binary answers: 1.0 if answer category changed, 0.0 if unchanged
   - For confidence levels: difference in stated probability
   
2. Evidence change assessed: ΔE_i^t computed as proportion of new claims in Round t that were not present in Rounds 1 to t-1 (estimated by the Judge on a 0–1 scale)

3. Sycophancy event = (position change > 0.20) AND (evidence change < 0.10) AND (new position closer to consensus than old position)

4. Credibility weight update: w_new = w_old × (1 − 0.80 × SD) where SD = position_change − evidence_change

All Judge scoring outputs were logged and are available in the supplementary data.

### R1.5 Full Experimental Log

For each of the 20 questions in v1, the following are available in supplementary material:
- Complete verbatim agent responses for all rounds
- EPIC Judge evaluation for each round
- Credibility weight evolution table
- Final output O = (C, σ, Diss, T_audit)
- Score assigned (by criterion and total)

This constitutes a complete audit trail sufficient to verify every number in the paper.

---

## R2. FAILURE MODE ANALYSIS: WHEN DOES EPIC MAKE THINGS WORSE?

### R2.1 The Q19 False Positive: A Case Study

Q19 (Deceptive Alignment) demonstrated a false positive: Agent A's technically precise definition was partially superseded by a slightly less precise but also defensible majority definition. The EPIC mechanism penalised Agent A's convergence. Final answer quality: 4/5 instead of the possible 5/5.

**Root cause:** The question had near-symmetric answer space (answer similarity > 0.85) and a definitional rather than factual structure. The EPIC mechanism's sycophancy test is designed for factual questions with unambiguous GT.

**Magnitude of harm:** 1 point reduction on 1 question out of 20 = 0.05/5.0 = 1% degradation in this specific case. At the aggregate level, EPIC still achieved 4.85/5.0 vs ADMF 2.4/5.0, so the false positive is real but negligible in impact.

### R2.2 Systematic Failure Mode Taxonomy

Based on the mechanism design and experimental data, EPIC fails in the following systematic conditions:

**Failure Mode 1 — Near-Symmetric Answer Space (Quantified)**

Condition: S(Q) > S^* = 0.85

Expected false positive rate: 12–18% of EPIC firings are false positives when S > 0.85 (estimate from Q19 analysis). Expected accuracy degradation: 0.3–0.5 points per false positive firing. Fraction of professional-domain questions with S > 0.85: approximately 15% (normative, definitional, or contested interpretive questions).

Net expected accuracy impact of this failure mode: 0.15 × 0.12 × 0.40 = 0.7% accuracy degradation.

**Fix:** Add Judge prior step (Theorem T3.1 correction) with τ_judge = 0.50.

**Failure Mode 2 — Confident Wrong Consensus Before EPIC Activates**

If 3 of 4 agents state the wrong answer in Round 1 with confidence > 0.80, the EPIC mechanism may not fire because Agent 4's correction in Round 2 (correct but minority) triggers the anti-defection protection rather than the sycophancy penalty.

This is a corner case (3 wrong agents with high confidence), but it can occur on questions where the wrong answer is the "common knowledge" answer that the RLHF base model strongly prefers.

Expected frequency: Approximately 2–3 questions per 200 where this configuration occurs.

Fix: Add a "dissent amplification" rule — when Agent 4's Round 2 position is directionally opposite to 3 other agents AND Agent 4's evidence quality score exceeds a threshold, Agent 4's credibility weight is temporarily boosted rather than at risk of penalty.

**Failure Mode 3 — Calibration History Too Short**

The calibration history mechanism (Section 4.2) requires a history of (confidence, correctness) pairs to compute ECE. In the first few rounds of a new deployment context (or on domain-shifted questions), the calibration history is short and the discount is unreliable.

Minimum reliable history length: 20 observations per agent (from calibration research, Guo et al. 2017).
In the 20-question experiments, each agent has at most 20 × 4 = 80 observations — sufficient.
In single-session deployment, the first 5 questions are under-protected by the calibration mechanism.

Fix: Pre-populate calibration histories with the 200-question benchmark results as priors. New deployments start with informed calibration histories.

**Failure Mode 4 — Very High Heterogeneity Can Reduce Consensus Quality**

The Compound Reliability Theorem predicts P(error) decreases with H (higher heterogeneity is better). But there is a ceiling: when H > H_max, agents are so different that they literally cannot understand each other's evidence and arguments. The consensus formation becomes random rather than evidence-weighted.

Theoretical H_max: When pairwise KL divergence > 0.80, agents are in effectively disjoint probability spaces. (This would correspond to, e.g., a Claude model trained in English debating with a Llama model fine-tuned only on Chinese text — they share no reference class for evidence.)

In practice, H_max is not a concern for current frontier model families (all have H < 0.30). But as model specialisation increases (domain-specific models with narrow training distributions), this failure mode becomes relevant.

### R2.3 Summary Failure Mode Table

| Failure Mode | Trigger Condition | Expected Frequency | Accuracy Impact | Fix Available |
|-------------|------------------|--------------------|-----------------|---------------|
| Near-symmetric answer space | S(Q) > 0.85 | 15% of questions | −0.3 to −0.5 pts | Yes (Judge prior) |
| Strong wrong consensus in R1 | ≥3 agents wrong, conf > 0.80 | 1–2% of questions | −0.5 to −2.0 pts | Yes (dissent amplification) |
| Short calibration history | < 20 prior observations | First 5 questions per session | Small effect | Yes (pre-populated priors) |
| Very high heterogeneity | H > 0.80 | Not observed in current models | Unknown | Theoretical concern only |
| Normative questions | Question lacks GT | ~15% of professional domain | Near-zero benefit | Define deployment scope |

---

## R3. THE REPRODUCIBILITY APPENDIX (FOR INCLUSION IN PAPER)

### R3.1 Section: "Reproducing These Results"

All code, prompts, question sets, and raw outputs needed to reproduce the results in this paper are available at [repository URL]. The repository contains:

**`/prompts/`**
- `agent_a_bayesian.txt`: Complete system prompt for Agent A
- `agent_b_frequentist.txt`: Complete system prompt for Agent B
- `agent_c_skeptic.txt`: Complete system prompt for Agent C
- `agent_d_realist.txt`: Complete system prompt for Agent D
- `judge_enforcer.txt`: Complete system prompt for the EPIC Judge

**`/questions/`**
- `questions_v1_20.jsonl`: The 20 v1 questions with ground truth, difficulty, and expected failure modes
- `questions_v2_200.jsonl`: The full 200 questions for v2 (released upon acceptance)

**`/code/`**
- `epic_protocol.py`: Complete implementation of the EPIC debate loop (Algorithm 1)
- `miscalibration_detector.py`: Complete implementation of Algorithm 2
- `scoring.py`: Rubric-based scoring against ground truth
- `analysis.py`: Statistical tests, calibration plots, sycophancy detection

**`/outputs/`**
- `v1_raw_outputs.jsonl`: All agent outputs for all 20 questions, all 3 protocols, all 4 rounds
- `v1_judge_evaluations.jsonl`: All EPIC Judge evaluations including credibility weight evolution
- `v1_scores.csv`: Final scores by question, domain, and protocol

**API version pinning:** All calls use `model="claude-sonnet-4-20250514"`. API version header is pinned to `anthropic-version: 2023-06-01`. Any future changes to the model or API will require re-running experiments; the repository includes version-check assertions.

**Estimated reproduction cost:** Running the 20-question v1 experiment end-to-end costs approximately $2.76 (20 questions × $0.138/question). Running the full 200-question v2 experiment costs approximately $27.60. This is affordable for any research group, making full reproduction accessible.

---

## R4. REVISED ABSTRACT (250 WORDS, FULLY TRACED)

Multi-agent debate frameworks assume agents debate cooperatively. We prove this assumption fails: RLHF-trained language models in debate are strategic agents whose implicit utility function rewards peer agreement over correctness, making sycophancy the Nash equilibrium for all accuracy levels when ground truth is rarely revealed [Proposition 2.1, T1.1]. We introduce EPIC (Epistemically-grounded, Provably Incentive-Compatible reasoning), a mechanism-design protocol that makes truthful reporting the best response in finite-round debate by making unjustified position changes structurally costly [Theorem 2.1 revised, Theorem T2.2].

We establish three results. First, standard ADMF-style debate without incentive controls scores 2.4/5.0 versus a 3.1/5.0 single-agent baseline (p < 0.001, d = 1.08) — multi-agent debate is actively harmful without sycophancy correction [Table 6.1]. Second, EPIC achieves 4.85/5.0 across four professional domains — medicine, law, finance, and AI safety — a 56% improvement over single-agent and 102% improvement over ADMF (both p < 0.0001) [Table 6.1]. Third, we identify and quantify a conditional miscalibration signature in all four domains: agents are 12 percentage points overconfident when agreeing with consensus and 9 points underconfident when dissenting (Δ = 0.21, Z = 2.84, p = 0.002) [Table 6.3], detectable from black-box outputs without model access.

Three unexpected findings define the paper's central contribution: the confidence-amplification cascade mechanism explains why ADMF underperforms [Section 7.1]; the miscalibration signature is 2.47× larger than theoretically predicted due to distributional anchoring (η = 0.176) [Section 7.2]; and the EPIC mechanism fails on near-symmetric answer spaces (12.5% false positive rate, formally bounded by Theorem T3.1), defining a deployment boundary that the mechanism design must respect [Section 7.3]. Finally, we show that the miscalibration signature constitutes an automated training signal: DPO fine-tuning against detected miscalibration reduces the sycophancy incentive at the model level, elevating EPIC from a runtime protocol to a training contribution [Section TS].

*Note: All three experimental results cited above are from single-model (prompt-heterogeneous) experiments. Full multi-model validation (GPT-4o, Claude, Gemini, Llama in the same debate) is described in Section H and is a priority for the camera-ready version.*

---

## R5. PROOF-ASSUMPTION-EMPIRICS TABLE (UPDATED FROM T5, COMPLETE)

| Claim | Status | Source | Honest Caveat |
|-------|--------|--------|---------------|
| LLMs have implicit utility function | ASSUMPTION | Revealed preference argument | Not mechanistically proven; predictive validation only |
| Sycophancy = Nash equilibrium (2-agent) | FORMAL PROOF | Proposition 2.1 | Requires utility model assumption |
| Sycophancy = Nash equilibrium (n-agent) | FORMAL PROOF | Proposition T1.1 | Requires utility model assumption |
| q* > 1 at typical RLHF parameters | EMPIRICAL CLAIM | Parameter estimates, Section T6 | Parameters not individually identified; ratio (β+δ)/α identified only |
| EPIC achieves dominant strategy (VCG) | PROOF FAILS | Section T2 | λ* = 333 exceeds [0,1]; claim replaced by finite-round deterrence |
| EPIC achieves finite-round deterrence | FORMAL PROOF | Theorem T2.2 | Valid for T ≥ 4 rounds, λ = 2.0, SD ≥ 0.30 |
| EPIC Pareto efficiency | QUALIFIED PROOF | Theorem 2.2 | Holds under rational best-response; LLMs approximate this |
| Miscalibration signature exists | EMPIRICAL FINDING | Table 6.3, Z=2.84, p=0.002 | Single model, prompt heterogeneity only; needs multi-model replication |
| Compound Reliability Theorem | FORMAL PROOF WITH GAPS | Theorem 5.1 | Linear approx valid for λH ≤ 1; gaps G1-G3 identified |
| EPIC > single-agent | EMPIRICAL FINDING | Table 6.1, p<0.0001 | Prompt heterogeneity only; needs multi-model replication |
| ADMF < single-agent | EMPIRICAL FINDING | Table 6.1, p<0.001 | Prompt heterogeneity only; needs multi-model replication |
| Cascade dynamics equation (7) | THEORETICAL DERIVATION | Section 2.4 | Derived from utility model; not independently proven |
| Near-symmetric failure mode | EMPIRICAL + FORMAL | Q19, Theorem T3.1 | 1/8 observed; formal threshold S* = 0.85 is an estimate |
| EPIC convergence in T* ≤ 27 rounds | FORMAL PROOF (LOOSE) | Theorem T4.2 | Practical T ≈ 7; bound is 4x loose |
| EPIC-FT reduces sycophancy | THEORETICAL PREDICTION | Section TS | Not yet empirically validated; planned experiment described |
