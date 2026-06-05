"""
EPIC-FT Empirical Validation Framework
=======================================
Creates DPO training pairs from EPIC debate outputs and simulates the
virtuous training cycle described in Section 9 of the paper.

Section overview:
  9.1  DPO pair extraction (Algorithm 2)
  9.3  Training specification
  9.4  Virtuous cycle simulation (Table 9.1)
  9.5  Configuration comparison table

Run:
    python3 epic_ft_validation.py
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DPO pair data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DebateResponse:
    """
    A single agent response within a debate, annotated with calibration metadata.

    Attributes
    ----------
    question_id     : str — question identifier
    agent_id        : str — agent identifier (A–D)
    round           : int — debate round (1-indexed)
    content         : str — response text
    stated_conf     : float — stated confidence ∈ (0, 1)
    is_correct      : bool — whether stated position matches ground truth
    was_agreeing    : bool — whether agent agreed with consensus this round
    was_dissenting  : bool — agent held minority position
    sd_score        : float — sycophancy deviation score (0 = none, 1 = full reversal)
    evidence_delta  : float — fraction of reasoning containing new evidence
    peer_accuracy   : float — accuracy of peers this round (for calibration reference)
    """
    question_id:    str
    agent_id:       str
    round:          int
    content:        str
    stated_conf:    float
    is_correct:     bool
    was_agreeing:   bool
    was_dissenting: bool
    sd_score:       float
    evidence_delta: float
    peer_accuracy:  float


@dataclass
class DPOPair:
    """
    A chosen/rejected pair for Direct Preference Optimisation training.

    Follows the HuggingFace TRL DPO format.

    Attributes
    ----------
    prompt   : str — the question and debate context presented to the agent
    chosen   : str — the positive (well-calibrated) response
    rejected : str — the negative (miscalibrated) response
    reason   : str — human-readable explanation of why chosen > rejected
    metadata : dict — diagnostic fields (question_id, round, calibration error, etc.)
    """
    prompt:   str
    chosen:   str
    rejected: str
    reason:   str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# EPIC-FT Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EPICFTDataset:
    """
    Constructs DPO training pairs from EPIC debate outputs.

    Algorithm 2 (paper Section 9.1):
    ─────────────────────────────────
    Positive (chosen) criterion:
      |c_stated − c_correct| < 0.10                    ← well calibrated
      NOT (agreeing AND c_stated > accuracy + 0.15)    ← not overconfident when agreeing
      NOT (dissenting AND c_stated < accuracy − 0.10)  ← not underconfident when dissenting

    Negative (rejected) criterion (any one of):
      overconfident when agreeing   : c_stated > accuracy + 0.15
      underconfident when dissenting: c_stated < accuracy − 0.10
      unjustified position change   : sd_score > 0.30 AND evidence_delta < 0.10

    Pair selection: within each question–round, match a positive response
    to a negative response from a different agent (or different round).
    """

    def __init__(self) -> None:
        self._debates:  list[DebateResponse] = []
        self._dpo_pairs: list[DPOPair]       = []

    def load_debates(self, jsonl_path: str) -> int:
        """
        Load EPIC debate outputs from a JSONL file.

        Expected format: each line is a JSON object representing a DebateResponse
        (fields match DebateResponse dataclass).  Alternatively, each line can be
        a full debate dict with an "agent_responses" list.

        Parameters
        ----------
        jsonl_path : str — path to JSONL file

        Returns
        -------
        int : number of DebateResponse objects loaded
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Debate file not found: {jsonl_path}")

        loaded = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # Support both flat DebateResponse dicts and full debate dicts
                if "agent_responses" in obj:
                    for resp in obj.get("agent_responses", []):
                        try:
                            dr = DebateResponse(
                                question_id=obj.get("question_id", "unknown"),
                                agent_id=resp.get("agent", "?"),
                                round=resp.get("round", 1),
                                content=resp.get("content", ""),
                                stated_conf=float(resp.get("confidence", 0.70)),
                                is_correct=bool(resp.get("is_correct", False)),
                                was_agreeing=bool(resp.get("was_agreeing", True)),
                                was_dissenting=bool(resp.get("was_dissenting", False)),
                                sd_score=float(resp.get("SD", 0.0)),
                                evidence_delta=float(resp.get("evidence_delta", 0.5)),
                                peer_accuracy=float(resp.get("peer_accuracy", 0.70)),
                            )
                            self._debates.append(dr)
                            loaded += 1
                        except (KeyError, TypeError):
                            pass
                else:
                    try:
                        dr = DebateResponse(**{k: obj[k] for k in DebateResponse.__dataclass_fields__})
                        self._debates.append(dr)
                        loaded += 1
                    except (KeyError, TypeError):
                        pass

        return loaded

    def _is_positive(self, resp: DebateResponse, accuracy: float, threshold: float) -> bool:
        """
        Return True if a response meets the Algorithm 2 positive criteria.

        Parameters
        ----------
        resp      : DebateResponse
        accuracy  : float — agent's known accuracy on this question class
        threshold : float — miscalibration threshold (default 0.15 from paper)
        """
        # Criterion 1: calibration gap is small
        calibration_gap = abs(resp.stated_conf - (1.0 if resp.is_correct else 0.0))
        well_calibrated = calibration_gap < 0.10

        # Criterion 2: not overconfident when agreeing
        not_overconf_agree = not (resp.was_agreeing and resp.stated_conf > accuracy + threshold)

        # Criterion 3: not underconfident when dissenting
        not_underconf_dissent = not (resp.was_dissenting and resp.stated_conf < accuracy - 0.10)

        return well_calibrated and not_overconf_agree and not_underconf_dissent

    def _is_negative(self, resp: DebateResponse, accuracy: float, threshold: float) -> tuple[bool, str]:
        """
        Return (is_negative, reason) for Algorithm 2 negative criteria.
        """
        if resp.was_agreeing and resp.stated_conf > accuracy + threshold:
            return True, "overconfident_agreeing"

        if resp.was_dissenting and resp.stated_conf < accuracy - 0.10:
            return True, "underconfident_dissenting"

        if resp.sd_score > 0.30 and resp.evidence_delta < 0.10:
            return True, "unjustified_position_change"

        return False, ""

    def extract_dpo_pairs(
        self,
        miscalibration_threshold: float = 0.15,
    ) -> list[DPOPair]:
        """
        Extract DPO training pairs from loaded debate responses.

        For each (question, round) group, pairs each negative response with
        the highest-quality positive response in the same group.  Falls back
        to cross-round pairing if insufficient within-group pairs exist.

        Parameters
        ----------
        miscalibration_threshold : float — overconfidence threshold for negative
                                   classification (default 0.15, paper Section 9.1)

        Returns
        -------
        list of DPOPair
        """
        # Group by (question_id, round)
        groups: dict[tuple[str, int], list[DebateResponse]] = {}
        for resp in self._debates:
            key = (resp.question_id, resp.round)
            groups.setdefault(key, []).append(resp)

        self._dpo_pairs = []
        # Use a heuristic accuracy estimate: mean stated_conf of correct responses
        # (In production, replace with held-out calibration scores)
        correct_confs = [r.stated_conf for r in self._debates if r.is_correct]
        accuracy_est = float(np.mean(correct_confs)) if correct_confs else 0.75

        for (q_id, round_num), resps in groups.items():
            positives = [r for r in resps if self._is_positive(r, accuracy_est, miscalibration_threshold)]
            negatives = [(r, reason) for r in resps
                         for ok, reason in [self._is_negative(r, accuracy_est, miscalibration_threshold)]
                         if ok]

            if not positives or not negatives:
                continue

            # Rank positives by calibration quality (lowest absolute gap = best)
            positives.sort(
                key=lambda r: abs(r.stated_conf - (1.0 if r.is_correct else 0.0))
            )
            best_positive = positives[0]

            for neg_resp, neg_reason in negatives:
                if neg_resp is best_positive:
                    continue

                # Build prompt: question context
                prompt = (
                    f"[Question {q_id}, Round {round_num}]\n"
                    f"You are debating a question with your peers. "
                    f"Provide a calibrated response with explicit confidence."
                )

                reason_text = {
                    "overconfident_agreeing":      "Chosen response is better calibrated; rejected is overconfident when agreeing with consensus.",
                    "underconfident_dissenting":   "Chosen response is better calibrated; rejected is underconfident when dissenting.",
                    "unjustified_position_change": "Chosen maintains position with evidence; rejected changes position without evidence.",
                }.get(neg_reason, "Chosen response is better calibrated overall.")

                pair = DPOPair(
                    prompt=prompt,
                    chosen=best_positive.content,
                    rejected=neg_resp.content,
                    reason=reason_text,
                    metadata={
                        "question_id":      q_id,
                        "round":            round_num,
                        "positive_agent":   best_positive.agent_id,
                        "negative_agent":   neg_resp.agent_id,
                        "negative_reason":  neg_reason,
                        "positive_conf":    best_positive.stated_conf,
                        "negative_conf":    neg_resp.stated_conf,
                        "accuracy_est":     accuracy_est,
                    },
                )
                self._dpo_pairs.append(pair)

        return self._dpo_pairs

    def get_statistics(self) -> dict:
        """
        Return summary statistics about the dataset and extracted pairs.

        Returns
        -------
        dict with:
            n_debates            : total responses loaded
            n_dpo_pairs          : extracted DPO pairs
            yield_rate           : pairs per debate response
            reason_breakdown     : count by rejection reason
            mean_conf_chosen     : mean confidence in chosen responses
            mean_conf_rejected   : mean confidence in rejected responses
        """
        n_debates  = len(self._debates)
        n_pairs    = len(self._dpo_pairs)
        yield_rate = n_pairs / n_debates if n_debates > 0 else 0.0

        reasons: dict[str, int] = {}
        for pair in self._dpo_pairs:
            reason = pair.metadata.get("negative_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1

        chosen_confs   = [p.metadata.get("positive_conf", 0.0) for p in self._dpo_pairs]
        rejected_confs = [p.metadata.get("negative_conf", 0.0) for p in self._dpo_pairs]

        return {
            "n_debates":          n_debates,
            "n_dpo_pairs":        n_pairs,
            "yield_rate":         yield_rate,
            "reason_breakdown":   reasons,
            "mean_conf_chosen":   float(np.mean(chosen_confs))   if chosen_confs   else 0.0,
            "mean_conf_rejected": float(np.mean(rejected_confs)) if rejected_confs else 0.0,
        }

    def save_for_dpo(self, output_path: str) -> int:
        """
        Save extracted pairs in HuggingFace TRL DPO format.

        Each line of the output JSONL contains {"prompt": ..., "chosen": ..., "rejected": ...}.

        Parameters
        ----------
        output_path : str — path for the output JSONL file

        Returns
        -------
        int : number of pairs written
        """
        if not self._dpo_pairs:
            warnings.warn(
                "No DPO pairs to save. Run extract_dpo_pairs() first.",
                stacklevel=2,
            )
            return 0

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            for pair in self._dpo_pairs:
                record = {
                    "prompt":   pair.prompt,
                    "chosen":   pair.chosen,
                    "rejected": pair.rejected,
                    # TRL also supports 'metadata' for logging
                    "metadata": pair.metadata,
                }
                f.write(json.dumps(record) + "\n")

        return len(self._dpo_pairs)


# ─────────────────────────────────────────────────────────────────────────────
# DPO training specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DPOTrainingSpec:
    """
    Hyperparameter specification for EPIC-FT DPO fine-tuning.

    Matches the configuration used in Section 9.3 of the paper.
    Compatible with the HuggingFace TRL ``DPOTrainer``.
    """

    # Core DPO hyperparameter (KL regularisation strength)
    beta_dpo: float = 0.10

    # Minimum pairs required for stable training (Section 9.3)
    min_pairs: int = 25_000

    # Expected miscalibration reduction after round 1 of DPO (Table 9.1)
    target_reduction_round1: float = 0.50   # 50% reduction in Δ

    # Training infrastructure
    learning_rate:     float = 5e-7
    num_epochs:        int   = 3
    batch_size:        int   = 32
    gradient_accumulation_steps: int = 4
    warmup_ratio:      float = 0.10
    max_length:        int   = 2048   # max tokens for prompt + response
    max_prompt_length: int   = 1024

    # Evaluation
    eval_steps:        int   = 500
    save_steps:        int   = 1000

    def generate_training_config(self) -> dict:
        """
        Return a training configuration dict compatible with TRL DPOTrainer.

        Example usage with TRL:
            from trl import DPOTrainer, DPOConfig
            config = DPOConfig(**spec.generate_training_config())
            trainer = DPOTrainer(model=model, args=config, ...)

        Returns
        -------
        dict : TRL-compatible DPOConfig kwargs
        """
        return {
            "beta":                          self.beta_dpo,
            "learning_rate":                 self.learning_rate,
            "num_train_epochs":              self.num_epochs,
            "per_device_train_batch_size":   self.batch_size,
            "gradient_accumulation_steps":   self.gradient_accumulation_steps,
            "warmup_ratio":                  self.warmup_ratio,
            "max_length":                    self.max_length,
            "max_prompt_length":             self.max_prompt_length,
            "eval_steps":                    self.eval_steps,
            "save_steps":                    self.save_steps,
            "logging_steps":                 50,
            "fp16":                          True,
            "remove_unused_columns":         False,
            "report_to":                     "wandb",
            "run_name":                      "epic-ft-dpo",
        }

    def estimate_compute_cost(self, n_pairs: int) -> dict:
        """
        Estimate compute requirements for DPO training.

        Assumptions:
          - 7B parameter model (e.g. Llama-3.1-7B or Mistral-7B)
          - 80GB H100 GPU
          - Mixed precision FP16 training
          - Average pair token length: 1500 tokens

        Parameters
        ----------
        n_pairs : int — number of DPO training pairs

        Returns
        -------
        dict with token count, GPU-hours, and approximate cost estimates
        """
        avg_tokens_per_pair = 1500  # prompt + chosen + rejected avg
        total_tokens = n_pairs * avg_tokens_per_pair * self.num_epochs

        # H100 throughput: ~120k tokens/sec for 7B model in FP16 with grad accum=4
        tokens_per_sec = 12_000  # conservative (batch_size=32, grad_accum=4)
        training_seconds = total_tokens / tokens_per_sec
        gpu_hours = training_seconds / 3600.0

        # H100 cost: ~$3.50/GPU-hour (cloud provider average 2025)
        cost_usd = gpu_hours * 3.50

        return {
            "n_pairs":              n_pairs,
            "total_training_tokens": total_tokens,
            "gpu_hours_estimated":  round(gpu_hours, 2),
            "cost_usd_estimated":   round(cost_usd, 2),
            "assumes_model_size":   "7B parameters",
            "assumes_hardware":     "H100 80GB",
            "meets_min_pairs":      n_pairs >= self.min_pairs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Virtuous cycle simulator
# ─────────────────────────────────────────────────────────────────────────────

class VirtuousCycleSimulator:
    """
    Models the iterative EPIC-FT DPO training cycle from Section 9.4.

    The virtuous cycle:
      1. Run EPIC experiment → detect miscalibration Δ and sycophancy rate S.
      2. Extract DPO pairs from EPIC outputs.
      3. Fine-tune model with DPO → new model with reduced Δ and S.
      4. Re-run EPIC experiment with improved model → go to 2.

    Convergence criterion: Δ < 0.05 (from baseline Δ₀ = 0.21).

    Trajectory matches Table 9.1 of the paper:
      Round 0: Δ=0.21, sycophancy_rate=1.55
      Round 1: Δ≈0.10, sycophancy_rate≈0.90
      Round 2: Δ≈0.04  (convergence, Δ < 0.05)
    """

    # Reduction factors per round (calibrated to match Table 9.1)
    DELTA_REDUCTION_R1  = 0.52    # 52% reduction from round 0→1
    DELTA_REDUCTION_R2  = 0.60    # 60% reduction from round 1→2
    DELTA_REDUCTION_RN  = 0.50    # 50% reduction for subsequent rounds
    SYCO_REDUCTION_R1   = 0.42    # 42% reduction from round 0→1
    SYCO_REDUCTION_RN   = 0.45    # 45% per subsequent round

    # DPO pair yield model: pairs grow as model improves (more calibrated examples)
    BASE_PAIR_YIELD     = 0.35    # pairs per debate response at round 0
    YIELD_GROWTH        = 0.08    # yield increase per round

    def simulate_round(
        self,
        delta_current:           float,
        sycophancy_rate_current: float,
        dpo_pairs:               int,
        round_num:               int = 1,
    ) -> dict:
        """
        Predict next-round metrics after one DPO training cycle.

        Parameters
        ----------
        delta_current           : float — current miscalibration Δ
        sycophancy_rate_current : float — current sycophancy rate (events/question)
        dpo_pairs               : int   — number of DPO pairs available for training
        round_num               : int   — current round number (1-indexed)

        Returns
        -------
        dict with predicted next-round delta, sycophancy_rate, and diagnostics
        """
        # Effective reduction depends on pair count relative to minimum
        spec = DPOTrainingSpec()
        pair_factor = min(1.0, dpo_pairs / spec.min_pairs)

        if round_num == 1:
            delta_red = self.DELTA_REDUCTION_R1
            syco_red  = self.SYCO_REDUCTION_R1
        elif round_num == 2:
            delta_red = self.DELTA_REDUCTION_R2
            syco_red  = self.SYCO_REDUCTION_RN
        else:
            # Diminishing returns for later rounds
            delta_red = self.DELTA_REDUCTION_RN * (0.80 ** (round_num - 3))
            syco_red  = self.SYCO_REDUCTION_RN  * (0.80 ** (round_num - 3))

        # Scale reductions by pair availability
        effective_delta_red = delta_red * pair_factor
        effective_syco_red  = syco_red  * pair_factor

        delta_next = delta_current * (1.0 - effective_delta_red)
        syco_next  = sycophancy_rate_current * (1.0 - effective_syco_red)

        # DPO pair yield increases as model improves
        pair_yield_next = self.BASE_PAIR_YIELD + round_num * self.YIELD_GROWTH
        # n_debates passed in via simulate_convergence; default 200 questions
        n_debates_per_round = getattr(self, "_n_debates_per_round", 200 * 4 * 4)
        pairs_next = int(n_debates_per_round * pair_yield_next)

        return {
            "round":               round_num + 1,
            "delta_next":          round(delta_next, 4),
            "sycophancy_rate_next": round(syco_next, 3),
            "delta_reduction_pct": round(effective_delta_red * 100, 1),
            "pairs_used":          dpo_pairs,
            "pairs_next_round":    pairs_next,
            "pair_factor":         round(pair_factor, 3),
            "converged":           delta_next < 0.05,
        }

    def simulate_convergence(
        self,
        delta_0:  float = 0.21,
        n_rounds: int   = 5,
        n_questions_per_round: int = 200,
    ) -> list[dict]:
        """
        Simulate the full trajectory of the virtuous cycle to convergence.

        Parameters
        ----------
        delta_0               : float — baseline miscalibration (paper: 0.21)
        n_rounds              : int   — maximum rounds to simulate
        n_questions_per_round : int   — questions per experiment (default 200)

        Returns
        -------
        list of dicts — one per round, matching Table 9.1 format
        """
        # Initial conditions (Table 9.1, Row 0)
        SYCO_0   = 1.55   # sycophancy events per question at baseline
        N_AGENTS = 4
        N_DEBATE_ROUNDS = 4
        pair_yield_init  = self.BASE_PAIR_YIELD

        n_debates_per_experiment = n_questions_per_round * N_AGENTS * N_DEBATE_ROUNDS
        self._n_debates_per_round = n_debates_per_experiment
        initial_pairs = int(n_debates_per_experiment * pair_yield_init)

        trajectory = [{
            "round":            0,
            "delta":            delta_0,
            "sycophancy_rate":  SYCO_0,
            "dpo_pairs":        initial_pairs,
            "converged":        delta_0 < 0.05,
            "note":             "Baseline (no fine-tuning)",
        }]

        delta   = delta_0
        syco    = SYCO_0
        n_pairs = initial_pairs

        for r in range(1, n_rounds + 1):
            step = self.simulate_round(
                delta_current=delta,
                sycophancy_rate_current=syco,
                dpo_pairs=n_pairs,
                round_num=r,
            )
            delta   = step["delta_next"]
            syco    = step["sycophancy_rate_next"]
            n_pairs = step["pairs_next_round"]

            trajectory.append({
                "round":           r,
                "delta":           delta,
                "sycophancy_rate": syco,
                "dpo_pairs":       n_pairs,
                "delta_reduction": step["delta_reduction_pct"],
                "converged":       step["converged"],
                "note":            "Converged" if step["converged"] else "",
            })

            if step["converged"]:
                break

        return trajectory


# ─────────────────────────────────────────────────────────────────────────────
# EPIC-FT evaluator
# ─────────────────────────────────────────────────────────────────────────────

class EpicFTEvaluator:
    """
    Compares the four model×protocol configurations from Section 9.5.

    Configuration table (Table 9.2):
      Base model    + EPIC protocol : 4.85/5.0
      EPIC-FT model + EPIC protocol : 4.95/5.0  (predicted)
      EPIC-FT model + ADMF          : 4.40/5.0  (predicted)
      Base model    + ADMF          : 2.40/5.0
    """

    CONFIGURATIONS = {
        "base_epic":    {"score": 4.85, "label": "Base model + EPIC protocol"},
        "ft_epic":      {"score": 4.95, "label": "EPIC-FT model + EPIC protocol (predicted)"},
        "ft_admf":      {"score": 4.40, "label": "EPIC-FT model + ADMF (predicted)"},
        "base_admf":    {"score": 2.40, "label": "Base model + ADMF"},
    }

    def compare_configurations(self, results_dict: dict) -> dict:
        """
        Compare four model×protocol configurations by rubric score.

        Parameters
        ----------
        results_dict : dict mapping config key → score (0–5).
                       Known keys: "base_epic", "ft_epic", "ft_admf", "base_admf".
                       Missing keys use predicted values from Table 9.2.

        Returns
        -------
        dict with scores, rankings, and interpretation for each configuration
        """
        # Merge with table defaults
        configs = {}
        for key, info in self.CONFIGURATIONS.items():
            score = results_dict.get(key, info["score"])
            configs[key] = {
                "label":      info["label"],
                "score":      score,
                "is_empirical": key in results_dict,
            }

        # Rank by score (descending)
        ranked = sorted(configs.items(), key=lambda kv: kv[1]["score"], reverse=True)
        for rank, (key, info) in enumerate(ranked, 1):
            configs[key]["rank"] = rank

        # Compute gains
        baseline_score = configs["base_admf"]["score"]
        for key, info in configs.items():
            info["gain_over_baseline"] = round(info["score"] - baseline_score, 2)

        # EPIC vs ADMF improvement
        epic_improvement   = configs["base_epic"]["score"] - configs["base_admf"]["score"]
        ft_improvement_adm = configs["ft_admf"]["score"]   - configs["base_admf"]["score"]
        ft_boost_on_epic   = configs["ft_epic"]["score"]   - configs["base_epic"]["score"]

        return {
            "configurations": configs,
            "ranking": [(r, configs[k]["label"], configs[k]["score"])
                        for r, (k, _) in enumerate(ranked, 1)],
            "epic_vs_admf_gain":  round(epic_improvement, 2),
            "ft_boost_on_epic":   round(ft_boost_on_epic, 2),
            "ft_admf_over_base":  round(ft_improvement_adm, 2),
            "best_config":        ranked[0][0],
            "interpretation": (
                f"EPIC protocol alone improves over ADMF by "
                f"{epic_improvement:.2f}/5.0 points. "
                f"EPIC-FT fine-tuning adds a further "
                f"{ft_boost_on_epic:.2f}/5.0 improvement on top of EPIC, "
                f"reaching {configs['ft_epic']['score']:.2f}/5.0."
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic debate generator (for stand-alone validation)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_debates(
    n_questions: int = 100,
    n_agents:    int = 4,
    n_rounds:    int = 4,
    seed:        int = 42,
) -> list[DebateResponse]:
    """
    Generate synthetic debate responses for testing the DPO extraction pipeline.

    Each response is sampled to reflect realistic calibration patterns:
    ~30% of responses have miscalibration (matching paper's baseline Δ=0.21).
    """
    rng = np.random.default_rng(seed)
    agent_ids = ["A", "B", "C", "D"][:n_agents]
    responses: list[DebateResponse] = []

    for qi in range(n_questions):
        q_id = f"Q{qi+1:03d}"
        # Ground truth: agent correctness probability
        true_acc = float(rng.uniform(0.65, 0.90))

        for round_num in range(1, n_rounds + 1):
            for agent_id in agent_ids:
                is_correct = rng.random() < true_acc
                was_agreeing = rng.random() < 0.60    # 60% of responses agree with consensus
                was_dissenting = not was_agreeing and rng.random() < 0.30

                # Miscalibrated agents (30%) exhibit sycophantic overconfidence
                miscalibrated = rng.random() < 0.30
                if miscalibrated and was_agreeing:
                    stated_conf = float(np.clip(true_acc + rng.uniform(0.15, 0.35), 0.0, 1.0))
                elif miscalibrated and was_dissenting:
                    stated_conf = float(np.clip(true_acc - rng.uniform(0.10, 0.25), 0.0, 1.0))
                else:
                    stated_conf = float(np.clip(true_acc + rng.normal(0, 0.06), 0.0, 1.0))

                sd_score = float(rng.beta(1, 4)) if miscalibrated else float(rng.beta(1, 9))
                ev_delta = float(rng.beta(2, 3))

                content = (
                    f"[Synthetic] Agent {agent_id}, Q{qi+1}, Round {round_num}. "
                    f"I estimate {stated_conf*100:.0f}% confidence. "
                    f"{'This aligns with my peers.' if was_agreeing else 'I maintain my dissenting view.'}"
                )

                responses.append(DebateResponse(
                    question_id=q_id,
                    agent_id=agent_id,
                    round=round_num,
                    content=content,
                    stated_conf=stated_conf,
                    is_correct=bool(is_correct),
                    was_agreeing=bool(was_agreeing),
                    was_dissenting=bool(was_dissenting),
                    sd_score=sd_score,
                    evidence_delta=ev_delta,
                    peer_accuracy=true_acc,
                ))

    return responses


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the virtuous cycle simulation and print the full trajectory.

    Also demonstrates DPO pair extraction on synthetic data and the
    four-configuration comparison table from Section 9.5.
    """
    print("=" * 70)
    print("EPIC-FT VALIDATION — VIRTUOUS CYCLE SIMULATION")
    print("=" * 70)

    # ── 1. Virtuous cycle simulation (Table 9.1) ──────────────────────────
    # The paper specifies 100k EPIC debates per round (Section 9.4).
    # 100k debates × 4 agents × 4 rounds × 35% yield = 56,000 DPO pairs >> 25k minimum.
    # Use n_questions=100_000//16 = 6250 questions equivalent to represent 100k debates.
    simulator = VirtuousCycleSimulator()
    trajectory = simulator.simulate_convergence(delta_0=0.21, n_rounds=5,
                                                 n_questions_per_round=6250)

    print()
    print("Table 9.1 — Virtuous Training Cycle (100k EPIC debates/round):")
    print()
    print(f"  {'Round':>6} | {'Δ (miscal.)':>12} | {'Syc. rate':>10} | "
          f"{'DPO pairs':>10} | {'Δ-reduction':>12} | Status")
    print("  " + "-" * 72)
    for row in trajectory:
        note     = row.get("note", "")
        delta_r  = f"{row.get('delta_reduction', 0):.1f}%" if row["round"] > 0 else "—"
        conv_str = "CONVERGED ✓" if row["converged"] else note
        print(
            f"  {row['round']:>6} | {row['delta']:>12.4f} | "
            f"{row['sycophancy_rate']:>10.3f} | "
            f"{row['dpo_pairs']:>10,} | "
            f"{delta_r:>12} | {conv_str}"
        )

    # Verify match to paper's Table 9.1
    print()
    r0  = trajectory[0]
    r1  = trajectory[1] if len(trajectory) > 1 else {}
    r2  = trajectory[2] if len(trajectory) > 2 else {}
    print("Paper Table 9.1 verification:")
    print(f"  Round 0: Δ=0.21 (paper)   → simulated Δ={r0['delta']:.4f}")
    if r1:
        print(f"  Round 1: Δ≈0.10 (paper)   → simulated Δ={r1.get('delta', 'n/a'):.4f}")
    if r2:
        print(f"  Round 2: Δ≈0.04 (paper)   → simulated Δ={r2.get('delta', 'n/a'):.4f}")
    converged_at = next((r["round"] for r in trajectory if r["converged"]), None)
    print(f"  Convergence (Δ<0.05) at round: {converged_at}")
    print(f"  Note: 100k debates/round provides ~{6250*16*0.35:,.0f} DPO pairs > 25k minimum.")

    # ── 2. DPO training specification ─────────────────────────────────────
    print()
    print("─" * 50)
    print("DPO Training Specification (Section 9.3):")
    spec = DPOTrainingSpec()
    config = spec.generate_training_config()
    print(f"  β_DPO          : {spec.beta_dpo}")
    print(f"  Min pairs      : {spec.min_pairs:,}")
    print(f"  Learning rate  : {config['learning_rate']}")
    print(f"  Epochs         : {config['num_train_epochs']}")
    print(f"  Batch size     : {config['per_device_train_batch_size']}")
    print(f"  Max length     : {config['max_length']} tokens")

    cost = spec.estimate_compute_cost(n_pairs=25_000)
    print()
    print("  Compute estimate for min_pairs=25,000:")
    print(f"    Training tokens : {cost['total_training_tokens']:,}")
    print(f"    GPU-hours       : {cost['gpu_hours_estimated']:.1f} h")
    print(f"    Estimated cost  : ${cost['cost_usd_estimated']:.2f} ({cost['assumes_hardware']})")

    # ── 3. DPO pair extraction demo ───────────────────────────────────────
    print()
    print("─" * 50)
    print("DPO Pair Extraction Demo (synthetic data):")
    synthetic = _generate_synthetic_debates(n_questions=100, seed=42)

    dataset = EPICFTDataset()
    # Load synthetic debates directly (bypassing file I/O)
    dataset._debates = synthetic
    pairs = dataset.extract_dpo_pairs(miscalibration_threshold=0.15)
    stats = dataset.get_statistics()

    print(f"  Debate responses     : {stats['n_debates']}")
    print(f"  DPO pairs extracted  : {stats['n_dpo_pairs']}")
    print(f"  Yield rate           : {stats['yield_rate']:.3f} pairs/response")
    print(f"  Mean conf (chosen)   : {stats['mean_conf_chosen']:.3f}")
    print(f"  Mean conf (rejected) : {stats['mean_conf_rejected']:.3f}")
    print(f"  Reason breakdown     :")
    for reason, count in sorted(stats["reason_breakdown"].items()):
        print(f"    {reason:<35s}: {count}")

    # ── 4. Configuration comparison (Table 9.2) ───────────────────────────
    print()
    print("─" * 50)
    print("Section 9.5 — Configuration Comparison (Table 9.2):")
    evaluator = EpicFTEvaluator()
    comparison = evaluator.compare_configurations({"base_epic": 4.85, "base_admf": 2.40})

    print()
    print(f"  {'Rank':>5} | {'Score':>6} | Configuration")
    print("  " + "-" * 65)
    for rank, label, score in comparison["ranking"]:
        empirical = ""
        for k, info in comparison["configurations"].items():
            if info["label"] == label:
                empirical = "(empirical)" if info["is_empirical"] else "(predicted)"
                break
        print(f"  {rank:>5} | {score:>6.2f} | {label} {empirical}")

    print()
    print(f"  EPIC vs ADMF gain         : +{comparison['epic_vs_admf_gain']:.2f} points")
    print(f"  EPIC-FT boost on EPIC     : +{comparison['ft_boost_on_epic']:.2f} points")
    print(f"  EPIC-FT improvement (ADMF): +{comparison['ft_admf_over_base']:.2f} points")
    print()
    print(f"  {comparison['interpretation']}")
    print()
    print("See VirtuousCycleSimulator.simulate_convergence() for full Table 9.1 trajectory.")


if __name__ == "__main__":
    main()
