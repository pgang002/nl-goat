# Project Plan: NLP-Grounded Autonomous Exploitation Agent (NL-GOAT)

**Working title:** *"From Text to Exploit: Grounding Autonomous Penetration-Testing Agents in Natural-Language Vulnerability Knowledge"*

## 1. One-line pitch
An LLM agent that builds its own structured vulnerability knowledge base by reading unstructured security text (CVE descriptions, vendor advisories, exploit-db writeups), then uses that knowledge to autonomously plan and execute exploitation in an isolated lab network — with the planning policy improved via RL based on execution outcomes.

## 1a. Locked-in v1 scope decisions
- **Vuln classes:** SQL Injection (CWE-89) + Command Injection (CWE-78). Chosen for large/well-annotated CVE corpora, formulaic advisory language (good NER/classification targets), unambiguous automatable success signals (data returned / command output captured — directly usable as RL reward), trivial isolated-lab deployment (DVWA, Vulhub, custom vulnerable apps), and enough difference between the two (data-layer vs OS-layer) to support a generalization claim. Phase-2 expansion candidates once v1 works: Path Traversal, SSRF.
- **Agent framework:** LangGraph (explicit state-machine control, cleaner RL integration than CrewAI/AutoGen).
- **Cloud:** AWS is the target environment. Azure subscription is confirmed available as fallback; AWS credit availability pending supervisor confirmation. Architecture below is written AWS-first — flag if we need an Azure-equivalent mapping (VNet instead of VPC, AKS/Container Instances instead of ECS, Azure ML instead of SageMaker, etc.) in case AWS credits don't come through.

## 2. Why this is novel (your paper's contribution)
Most "LLM agent does pentesting" papers (PentestGPT, hackingBuddyGPT, etc.) lean on the LLM's *parametric* knowledge of CVEs, which is unreliable, stale, and unverifiable. Your contribution is the **NLP layer in between**: a dedicated information-extraction + classification pipeline that turns raw security text into a structured, queryable knowledge graph (product → version → vulnerability → ATT&CK technique → known exploitation pattern), which the agent then *grounds* its decisions in via retrieval. You can measure and report:
- Extraction accuracy (NER/classification F1) as a standalone NLP contribution
- End-to-end agent success rate **with vs without** the grounding layer (this is your key ablation and the paper's core empirical claim)
- How RL fine-tuning on execution feedback changes planning quality over time

This framing gives you two publishable angles in one project: an NLP/IE paper and a systems/agent paper, which is useful if you want to split output into more than one publication later.

## 3. System architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: NLP Knowledge Extraction Pipeline                  │
│  Sources: NVD/CVE feed, vendor advisories, exploit-db,       │
│  security blogs/writeups (scraped, license-checked)          │
│                                                                │
│  → NER model: extract {product, version, vuln_type, vector}  │
│  → Multi-label classifier: CVE text → MITRE ATT&CK technique │
│  → Embedding index (RAG store) of raw + structured records   │
│  → Output: Vulnerability Knowledge Graph (Neo4j or simple    │
│    JSON graph + vector DB, e.g. pgvector/FAISS)              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Agentic Planning & Execution Layer                 │
│  Framework: LangGraph (recommend over CrewAI/AutoGen for     │
│  explicit state-machine control, easier RL integration)      │
│                                                                │
│  Agent loop: Recon → Service Fingerprint → Knowledge-Graph    │
│  Query (RAG) → Candidate Exploit Ranking → Action Selection   │
│  → Execute (sandboxed tool call) → Observe Result → Replan    │
│                                                                │
│  Tools exposed to agent: nmap wrapper, service banner parser, │
│  metasploit-rpc wrapper, custom exploit scripts, shell-in-    │
│  container executor (all sandboxed, logged, rate-limited)     │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: RL Optimization Layer                              │
│  Environment: Gym-style wrapper around the lab network        │
│  State: current recon findings + KG query results + history   │
│  Action: which candidate exploit/technique to attempt next    │
│  Reward: +success (host compromised), -noisy/failed actions,  │
│  -time/step penalty, +partial credit for correct ATT&CK stage │
│  Algorithm: PPO (via TRL or stable-baselines3) fine-tuning    │
│  the planning policy / LLM action-selection head              │
└─────────────────────────────────────────────────────────────┘
```

## 4. Isolated lab environment (AWS)
This is the part you must get right for both ethics and reproducibility:
- Dedicated AWS account or strictly isolated VPC, **no internet egress** from the target subnet (private subnet, no IGW/NAT route)
- Target hosts: Dockerized intentionally-vulnerable services scoped to v1's two vuln classes — DVWA and a custom Flask/PHP app for SQL Injection cases, Vulhub images / custom scripts with shell-out patterns for Command Injection cases, each mapped to real CVEs pulled from the extraction pipeline — deployed on EC2 or ECS within the isolated subnet
- Agent runs in a separate "attacker" subnet with explicit, logged, one-way connectivity to the target subnet (simulate via security groups / NACLs)
- All traffic captured (VPC Flow Logs + pcap on attacker host) for your evaluation metrics and for the paper's reproducibility section
- IAM: least-privilege roles per component; tag everything `project=nl-goat` for cost tracking
- Document this setup explicitly in the paper's methodology — reviewers will ask, and a clean isolated-range description heads off ethics-desk-reject risk

## 5. Fine-tuning plan
- Base model: Llama-3.1-8B or Mistral-7B (small enough for single-GPU LoRA, large enough to plan competently)
- Two fine-tuning targets (can be same model, different adapters):
  1. **Extraction model**: fine-tune on a labeled subset of CVE text (you can bootstrap labels using GPT-4/Claude-assisted annotation + manual correction of a few hundred examples — disclose this in your data section)
  2. **Planning model**: fine-tune on agent trajectory data — collect (state, action, outcome) traces from early agent runs (even rule-based/heuristic baseline runs), then LoRA/QLoRA fine-tune so the model's action proposals better match successful trajectories before RL even starts (this is standard "SFT warm-start before RL")
- Tooling: HuggingFace `transformers` + `peft` for LoRA, `bitsandbytes` for QLoRA quantization, train on a single AWS `g5.xlarge`/`g5.2xlarge` spot instance or SageMaker training job

## 6. RL design details
- Use `trl`'s `PPOTrainer` if you want to RL-tune the LLM directly on text-generation-as-action, OR wrap the agent in a custom `gymnasium.Env` and use `stable-baselines3` PPO over a discrete/parameterized action space (simpler to get working first — recommend starting here, moving to direct LLM-RL only if time allows)
- Reward shaping is your most important design decision — document it carefully, it's a common reviewer question:
  - Large positive: confirmed exploitation (shell/flag obtained)
  - Small positive: correct intermediate ATT&CK stage reached (e.g., correct service fingerprinted)
  - Small negative: failed/noisy action (e.g., wrong exploit attempted, triggers IDS-like detection rule you simulate)
  - Step penalty: discourage excessive exploration
- Baselines to compare against: (a) no-RL fine-tuned agent, (b) no-NLP-grounding agent (pure LLM parametric knowledge), (c) random/heuristic baseline, (d) published prior work if reproducible (PentestGPT-style)

## 7. MLOps / tooling stack mapping
| Requirement | Tool | Notes |
|---|---|---|
| Experiment logging | Weights & Biases | Log NLP training curves, RL reward curves, agent trajectory replays |
| Versioning | Git + DVC | DVC for datasets/model checkpoints (too large for git), Git for code |
| Model registry | W&B Artifacts or MLflow Model Registry | Version your extraction model, SFT model, RL-tuned model separately |
| CI/CD | GitHub Actions | Lint/test on PR; on merge to main, trigger build → push container to ECR |
| Deployment | AWS ECR + ECS (or SageMaker endpoints) | Containerize agent + NLP pipeline separately |
| Infra-as-code | Terraform (recommended) | Define the isolated VPC/lab as code — huge reproducibility win for the paper |
| Compute | EC2 spot / SageMaker training jobs | g5 instances for fine-tuning, smaller instances for agent runtime |

## 8. Evaluation plan
- **NLP extraction**: precision/recall/F1 against a held-out manually-verified CVE annotation set
- **Agent end-to-end**: success rate, mean steps-to-compromise, mean noisy-actions-per-success, across a held-out set of target configurations not seen during RL training (critical — avoid overfitting to your own lab)
- **Ablations**: with/without NLP grounding, with/without RL, with/without SFT warm-start
- **Generalization test**: hold out an entire CVE/service family from training, test if the agent can still reason about it via the NLP-grounded knowledge graph (this is your strongest novelty claim if it works)

## 9. Suggested timeline (14–16 weeks, adjust to your availability)
1. Weeks 1–2: Confirm AWS credits (Azure fallback mapping ready if not); data collection (SQLi + Command Injection CVE/advisory corpus only), AWS lab infra (Terraform), isolated VPC + DVWA/Vulhub target containers up and reachable
2. Weeks 3–4: NLP extraction pipeline (NER + ATT&CK classifier), build knowledge graph/RAG store
3. Weeks 5–6: Baseline agent (LangGraph, no RL, no fine-tuning) — get *something* exploiting a target end-to-end
4. Weeks 7–8: SFT fine-tuning on collected trajectories; logging/versioning/CI-CD wired up properly
5. Weeks 9–11: RL training loop, reward shaping iteration, ablation runs
6. Weeks 12–13: Full evaluation sweep, generalization holdout tests
7. Weeks 14–16: Write-up, figures, arXiv preprint, target-journal formatting

## 10. Target venues
- ACM AISec Workshop (co-located CCS) — strong fit, realistic acceptance bar for a first paper
- Elsevier *Computers & Security* journal — good fit for the full NLP+agent+eval framing
- IEEE TIFS — stronger venue, aim here if results are clean and ablations are convincing
- arXiv preprint in parallel regardless of journal outcome — gets it on your resume/citable immediately

## 11. Ethics/legal note for your methods section
Explicitly state: all testing conducted against self-deployed, intentionally vulnerable software in an isolated, internet-disconnected AWS environment with no third-party systems involved; no real-world exploitation performed; code/data release (if any) will exclude anything that could be repurposed as a turnkey attack tool against unpatched real systems. This single paragraph heads off most reviewer/ethics-desk concerns.

## 12. Main risks & mitigations
- **Risk:** RL training is sample-inefficient / agent never improves → **Mitigation:** start with SFT warm-start, keep action space small/discrete initially, only widen once basic loop works
- **Risk:** NLP extraction quality too low to be useful → **Mitigation:** scope to a narrow CVE category first (e.g., web app CVEs only) rather than all of NVD
- **Risk:** Scope creep (this is a lot of moving parts) → **Mitigation:** get the *full pipeline* working end-to-end at toy scale (1 NLP type, 1 vulnerable service, no RL) by week 6 before adding sophistication
