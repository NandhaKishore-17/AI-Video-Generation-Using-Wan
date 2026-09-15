# Kaalapadhivugal (காலப் பதிவுகள்) — Autonomous AI Documentary Channel System

An end-to-end autonomous AI-driven documentary channel system dedicated to researching, scripting, planning, generating, and publishing high-fidelity historical documentary episodes focusing on South Indian, Tamil, and World ancient history.

---

## 🏛️ Canonical Channel Identity

| Attribute | Tamil | English |
| :--- | :--- | :--- |
| **Channel Name** | காலப் பதிவுகள் | **Kaalapadhivugal** |
| **English Meaning** | — | Records of Time |
| **Documentary Host** | யாழினி | **Yaazhini** |
| **Channel Tagline** | *கடந்த காலத்தை, சான்றுகளுடன் மீண்டும் பார்ப்போம்.* | *Let’s revisit the past, through the evidence.* |
| **Handle** | `@kaalapadhivugal` | `@kaalapadhivugal` |
| **Host Reference Image** | `assets/yaazhini_presenter.jpg` | `assets/yaazhini_presenter.jpg` |

### Host Persona & Visual Consistency
- **Name**: Yaazhini (யாழினி)
- **Visual Identity**: South Indian female documentary presenter with warm skin tone, expressive natural face, dark wavy hair, traditional jhumka earrings, wearing an earthy olive-green kurti with patterned dupatta, presenting from an archaeological study with ancient artifacts and historical maps in the background.
- **Tone & Delivery**: Grounded, respectful, curious, evidence-focused narration in classical and conversational Tamil with broadcast bilingual subtitles.

---

## ⚙️ Architecture & Completed Phases

1. **Phase 1 — Foundation & State Architecture**:
   - Central configuration (`autonomous/config.py`), directory management, environment loading.
2. **Phase 2 — State Management, Recovery & Single-Instance Mutex**:
   - Strict SQLite state tracking (`universe_platform.db`), process locking (`lock.pid`), automated crash recovery (`autonomous/recovery_manager.py`).
3. **Phase 3 — Autonomous Topic Discovery & Measurable Scoring**:
   - 8-metric weighted evaluation, recency penalties, duplicate suppression (`autonomous/topic_discovery.py`, `autonomous/topic_selector.py`).
4. **Phase 4 — Authoritative Research Engine & Vector Indexing**:
   - Academic and historical retrieval, CPU-native Qdrant vector storage, evidence dossier generation (`autonomous/research_engine.py`).
5. **Phase 5 — Claim Verification & Script Generation**:
   - Deterministic claim extraction, cross-referencing, multi-scene bilingual script generation (`autonomous/script_generator.py`).
6. **Phase 5.1 — Content Quality Gate**:
   - Deterministic verification preventing shallow scripts: minimum unique claims (≥3), core claims (≥2), distinct topic aspects (≥2), scene repetition prevention.
7. **Phase 5.2 — Autonomous Research Expansion**:
   - Self-healing recovery from `CONTENT_INSUFFICIENT` without hallucination: targeted expansion query generation, vector assimilation, re-verification loop.
8. **Phase 6 — Autonomous Visual Planning**:
   - Deterministic visual scene planning transforming scripts into evidence-grounded visual plans (`autonomous/visual_planner.py`).
   - Epistemic grounding taxonomy (`HISTORICAL_CLAIM_GROUNDED`, `RECONSTRUCTION`, `ILLUSTRATIVE`, `ATMOSPHERIC`, `ABSTRACT`).
   - Strict 17-field Phase 7 contract schema, conservative prompt sanitization, depth/motion strategies.
9. **Phase 6.1 — Global Channel Identity Standardization**:
   - Single canonical channel identity configuration (`autonomous_settings.channel`).
   - Fully configuration-driven presenter resolution with zero hardcoding.
   - Clean retirement of all legacy prototypes across code, tests, and configurations.

---

## 🚀 Running the System

### Running Autonomous Channel Master CLI
```bash
python run_channel.py
```

### Running Test Suite
```bash
python -m pytest tests/ -q
```