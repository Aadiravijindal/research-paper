# EPIC REVISION v2 — MISCALIBRATION AS TRAINING SIGNAL

## The Biggest Missed Opportunity: EPIC-Tuned Models

---

## TS1. THE CORE IDEA

The miscalibration signature is currently used as a detection and penalty signal within the EPIC debate protocol. It measures which agents are behaving strategically (overconfident when agreeing, underconfident when dissenting) and downweights them.

The missed opportunity: this signal can be inverted into a training signal. A model that consistently exhibits the miscalibration signature (M^agree > 0, M^dissent < 0) is doing something identifiable and measurable. We can train against this pattern.

**The EPIC-FT (Fine-Tuning) procedure:**

1. Run the EPIC protocol on a large dataset of questions (100,000+ questions across domains)
2. For each question, record which agent-round observations show the miscalibration signature
3. Construct a fine-tuning dataset where:
   - **Positive examples**: agent responses that maintained calibrated positions under pressure (correctly dissented when right, appropriately updated when wrong, stated calibrated confidence)
   - **Negative examples**: agent responses showing the miscalibration signature (overconfident agreement, underconfident dissent, unjustified position changes)
4. Fine-tune using Direct Preference Optimisation (DPO) with positive examples as "chosen" and negative examples as "rejected"

The resulting model (EPIC-FT) should exhibit:
- Lower β (reduced peer approval reward)
- Lower (β+δ)/α ratio (more relative weight on correctness vs. agreement)
- More calibrated confidence in multi-agent contexts
- A longer-lasting shift in the equilibrium toward truthful reporting, not just protocol-level correction

---

## TS2. FORMAL SPECIFICATION OF THE TRAINING PROCEDURE

### TS2.1 Dataset Construction

**Definition TS2.1 (EPIC Training Observation).** A training observation is a 5-tuple:

    O_train = (q, context, response, A_status, calibration_error)

where:
- q: the question
- context: the debate context (previous rounds visible to the agent)
- response: the agent's response (position + confidence)
- A_status ∈ {agree, dissent}: whether the response agrees with the current consensus
- calibration_error = c_stated − correct: signed miscalibration for this observation

**Definition TS2.2 (Positive and Negative Training Examples).**

A response is a **positive example** (the behaviour we want to encourage) if:

    |calibration_error| < threshold_cal   (calibrated response)
    AND [A_status == agree → calibration_error < threshold_pos]  (not overconfident when agreeing)
    AND [A_status == dissent → calibration_error > threshold_neg]  (not underconfident when dissenting)

A response is a **negative example** (the behaviour we want to suppress) if:

    (A_status == agree AND calibration_error > 0.15)   (overconfident when agreeing)
    OR (A_status == dissent AND calibration_error < -0.15)  (underconfident when dissenting)
    OR (unjustified position change: SD > 0.30 AND ΔE < 0.10)  (sycophantic change)

### TS2.2 DPO Loss Function

Direct Preference Optimisation (Rafailov et al. 2023) trains the model to maximise the log-ratio of probability assigned to positive over negative examples:

    L_DPO = −E_{(q,c,r+,r−)} [log σ(β_DPO · (log π_θ(r+|q,c)/π_ref(r+|q,c) − log π_θ(r−|q,c)/π_ref(r−|q,c)))]

where:
- π_θ: the model being trained
- π_ref: the reference model (pre-fine-tuning)
- r+: a positive (well-calibrated, non-sycophantic) response
- r−: a negative (miscalibrated, sycophantic) response
- β_DPO: the DPO temperature parameter

The EPIC-specific component is that r+ and r− are selected based on the miscalibration signature, not on human preferences. This is an entirely automated training signal — no human annotation required after the initial EPIC run.

### TS2.3 Expected Effect on Utility Function Parameters

Training against the miscalibration signature should shift the parameter ratio (β+δ)/α:

**Pre-training (RLHF model):** (β+δ)/α ≈ 0.015 (from empirical estimates), with α effectively reduced by low ρ in deployment.

**Post-EPIC-FT (predicted):** The DPO training signal penalises overconfident agreement and rewards calibrated dissent. This is equivalent to:
- Increasing α (correctness reward): the model learns that calibrated, correct responses are preferred
- Decreasing β (agreement reward): agreement without calibration is penalised
- Decreasing δ (minority penalty): well-reasoned dissent is rewarded

The threshold accuracy q* = (β+δ)/(2αρ) + 0.5 should decrease toward the meaningful range (q* < 1.0 at deployment ρ = 0.10) after EPIC-FT, meaning the sycophancy equilibrium would no longer hold universally.

**Measurable prediction:** An EPIC-FT model, when placed in a multi-agent debate, should exhibit:

    M^agree_post − M^dissent_post < M^agree_pre − M^dissent_pre

i.e., the miscalibration signature magnitude decreases. The reduction should be measurable in a held-out test set. We predict a 40–60% reduction in the miscalibration difference Δ after EPIC-FT fine-tuning on 10,000 training examples.

---

## TS3. THE VIRTUOUS TRAINING CYCLE

The full EPIC-FT training cycle is self-improving:

```
EPIC-FT Virtuous Training Cycle

Round 0: Base RLHF Model
  → High sycophancy (β+δ >> αρ)
  → Miscalibration signature: Δ = 0.21
  
Round 1: EPIC Protocol Applied
  → Detects miscalibration signature
  → Generates (r+, r−) training pairs
  → DPO fine-tuning: β↓, α↑
  
Round 1 Model (EPIC-FT-v1)
  → Lower sycophancy
  → Miscalibration signature: Δ ≈ 0.12 (40% reduction predicted)
  
Round 2: EPIC Protocol Applied to EPIC-FT-v1
  → Detects remaining miscalibration
  → Generates new (r+, r−) pairs from residual sycophancy
  → DPO fine-tuning: β further ↓
  
Round 2 Model (EPIC-FT-v2)
  → Near-calibrated in debate contexts
  → Miscalibration signature: Δ ≈ 0.04 (near-zero, below detection threshold)
  → Sycophancy equilibrium condition no longer holds at q* < 1
  
Convergence: EPIC-FT-v∞
  → The model has been trained out of the sycophancy equilibrium
  → EPIC protocol still provides quality enforcement but
    agents no longer require the penalty to behave truthfully
  → The debate protocol has bootstrapped its own better participants
```

---

## TS4. EXPERIMENTAL DESIGN FOR THE TRAINING CONTRIBUTION

### TS4.1 Minimum Viable Demonstration

To demonstrate the training signal contribution in the paper, we need:

1. **Dataset construction** (automated): Run EPIC on 10,000 questions across domains. Extract (r+, r−) pairs from miscalibration signature detection. Expected yield: ~25% of responses flagged as positive/negative → 2,500 training pairs per polarity.

2. **Fine-tuning**: DPO fine-tuning on Claude Sonnet (via Anthropic API fine-tuning, if available) or on an open-source model (Llama 3.1 70B via LoRA). β_DPO = 0.1 (standard for DPO).

3. **Evaluation**: Run the fine-tuned model through the 200-question benchmark in single-agent mode AND in EPIC multi-agent mode. Measure:
   - Miscalibration signature magnitude Δ (predicted: 40% reduction)
   - Sycophancy rate in ADMF-style debate (predicted: 20–30% reduction)
   - Single-agent accuracy on MMLU and TruthfulQA (predicted: +1–2% improvement)

4. **Ablation**: Run the fine-tuned model in ADMF (no EPIC protocol). If the training signal works, the fine-tuned model should show lower sycophancy even without the EPIC penalty mechanism. This demonstrates that the training signal has genuinely reduced the sycophantic incentive, not just trained the model to respond to EPIC penalties.

### TS4.2 What This Contributes to the Field

This training contribution elevates the paper from "protocol for better multi-agent debate" to "training procedure for less sycophantic models." The distinction matters for publication:

- A protocol contribution is deployable immediately but requires runtime overhead
- A training contribution produces a permanently better model that behaves better in all contexts, including single-agent settings

The two contributions are complementary: EPIC as a protocol is the near-term fix; EPIC-FT is the long-term structural fix. A paper that offers both is substantially more impactful than one that offers only the protocol fix.

The specific comparison that would be most compelling: EPIC-FT model in standard (ADMF) debate vs. base model in EPIC debate. If EPIC-FT + ADMF ≈ base + EPIC, then the training signal has genuinely internalised the mechanism incentives. If EPIC-FT + EPIC >> both, the mechanisms are complementary and both are needed.

---

## TS5. RELATED WORK ON CALIBRATION-BASED TRAINING

**Calibration-aware training** is an active area. Kuleshov et al. (2018) showed that neural network calibration can be improved through post-hoc recalibration (temperature scaling). Lin et al. (2022) trained models to produce calibrated confidence on NLP tasks using a calibration auxiliary loss. Neither approach is specific to multi-agent settings or addresses the conditional miscalibration pattern (agree vs. dissent).

**DPO for reduced sycophancy**: Sharma et al. (2023) showed that targeted fine-tuning can reduce sycophancy. But their approach required human labelling of sycophantic responses. EPIC-FT provides an automated labelling procedure based on the statistical calibration test — no human annotation required at scale.

**RLAIF** (Reinforcement Learning from AI Feedback, Bai et al. 2022 Constitutional AI) showed that AI-generated feedback can substitute for human feedback in fine-tuning. EPIC-FT is a specific instance of RLAIF where the AI feedback is the miscalibration detection algorithm rather than a constitutionally-prompted AI. The key advantage: the EPIC feedback signal is statistically grounded and formally derived, not prompt-engineered.

The gap EPIC-FT fills: automated, statistically-grounded, multi-agent-specific calibration training. This is new.
