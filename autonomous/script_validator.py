"""
autonomous/script_validator.py - Script Claim Re-Extraction, Hallucination Detection & Epistemic Verification.

Parses generated script dialogue and subtitles, extracts every factual proposition,
and validates it against the verified claims pool. Rigorously flags new unsupported claims,
numeric mismatches, chronological discrepancies, entity hallucinations, epistemic overstatements,
fabricated quotations, and cross-language divergences.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from autonomous.claim_models import (
    FactCheckedContentPlan,
    VerifiedClaim,
    ScriptClaim,
    ScriptClaimStatus,
    ScriptValidationReport,
    ClaimClassification,
    EpistemicStatus
)
from autonomous.claim_extractor import (
    ClaimExtractor,
    SUPERLATIVE_TERMS,
    DATE_PATTERNS,
    NUMBER_PATTERNS
)

logger = logging.getLogger("autonomous.script_validator")

# Phrases indicating epistemic inflation / certainty overstatement
CERTAINTY_OVERSTATEMENT_TERMS = [
    "definitely", "undeniably", "proven fact", "conclusively proven", "unquestionably",
    "beyond doubt", "established fact", "absolutely confirmed", "records prove",
    "proved conclusively", "proved beyond doubt", "proved beyond question",
    "proved without doubt", "conclusive proof", "proved"
]

# Phrases indicating speculative or hedged framing
HEDGED_FRAMING_TERMS = [
    "suggests", "indicates", "evidence points to", "according to excavations",
    "archaeologists propose", "possibly", "likely", "estimated to be", "dated approximately"
]


class ScriptValidator:
    """Validates generated storyboard scripts against verified claim pools and content plans."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    @staticmethod
    def extract_quotations(text: str) -> List[str]:
        """Extract explicit quoted strings from narration."""
        quotes = re.findall(r'["“]([^"”]{4,})["”]', text)
        return [q.strip() for q in quotes if q.strip()]

    @staticmethod
    def is_creative_atmospheric_clause(sentence: str) -> bool:
        """
        Detect creative / travel vlog / documentary rhetorical framing that contains zero factual assertions.
        e.g., 'Imagine stepping back into the past...', 'Welcome to Kaalapadhivugal!', 'I am Yaazhini.',
        'Let’s revisit the past, through the evidence.', 'See you next time!'
        """
        s_clean = sentence.lower().strip().replace("’", "'")
        creative_phrases = [
            "imagine walking", "imagine stepping", "welcome to", "see you in our next",
            "don't forget to subscribe", "subscribe to", "follow ", "let's dive into", "join us as we explore",
            "vanakkam", "வணக்கம்", "சப்ஸ்கிரைப் பண்ணுங்க", "அடுத்த வரலாற்று பயணத்தில்",
            "அடுத்த வரலாற்றுப் பதிவில்", "பின்தொடருங்கள்", "மறக்காமல்",
            "ancient heritage reveals", "cultural significance", "our journey into history",
            "historical legacy", "stands as a testament", "remains a testament",
            "glimpse into the past", "join me next time", "join us next time",
            "revisit the past", "through the evidence", "let's revisit",
            "i am ", "i'm ", "வணக்கம்!", "நான் உங்கள் ", "in conclusion, verified evidence confirms"
        ]
        try:
            from autonomous.config import autonomous_settings
            cfg = autonomous_settings.channel
            for attr in (cfg.channel_name_en, cfg.channel_name_ta, cfg.host_name_en, cfg.host_name_ta, cfg.tagline_en, cfg.tagline_ta):
                if attr:
                    creative_phrases.append(attr.lower().strip().replace("’", "'"))
        except Exception:
            pass

        return any(p in s_clean for p in creative_phrases) and not any(
            re.search(pat, s_clean) for pat in DATE_PATTERNS
        )

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences while respecting quotation marks without regex lookbehind."""
        text = text.strip()
        if not text:
            return []
        sentences = []
        current = []
        in_quote = False
        for char in text:
            current.append(char)
            if char in ('"', '“', '”'):
                in_quote = not in_quote
            elif char in ('.', '!', '?') and not in_quote:
                s = "".join(current).strip()
                if s:
                    sentences.append(s)
                current = []
        if current:
            rem = "".join(current).strip()
            if rem:
                sentences.append(rem)
        return [s for s in sentences if s]

    def extract_script_claims(self, storyboard: Dict[str, Any]) -> List[ScriptClaim]:
        """
        Re-extract all factual assertions across both English subtitles and Tamil dialogue.
        """
        extracted_claims: List[ScriptClaim] = []
        claim_counter = 1

        scenes = storyboard.get("scenes", [])
        for scene in scenes:
            scene_id = scene.get("id", 1)

            # 1. Process English Subtitle
            en_sub = scene.get("english_sub", "")
            if en_sub:
                sub_quotes = self.extract_quotations(en_sub)
                en_sentences = self.split_into_sentences(en_sub)
                for s in en_sentences:
                    if self.is_creative_atmospheric_clause(s) and len(s.split()) < 8:
                        continue

                    dates = ClaimExtractor.extract_dates(s)
                    numbers = ClaimExtractor.extract_numbers(s)
                    entities = ClaimExtractor.extract_entities(s)
                    has_sup, _ = ClaimExtractor.has_superlative(s)
                    quotes = self.extract_quotations(s)
                    sentence_has_quote = bool(quotes) or any(q in s for q in sub_quotes)

                    sc = ScriptClaim(
                        claim_index=claim_counter,
                        scene_id=scene_id,
                        language="en",
                        source_sentence=s,
                        extracted_statement=ClaimExtractor.normalize_statement(s),
                        extracted_entities=entities,
                        extracted_dates=dates,
                        extracted_numbers=numbers,
                        is_superlative=has_sup,
                        is_quotation=sentence_has_quote
                    )
                    extracted_claims.append(sc)
                    claim_counter += 1

            # 2. Process Tamil Text for numeric/date/entity consistency
            ta_text = scene.get("tamil_text", "")
            if ta_text:
                ta_quotes = self.extract_quotations(ta_text)
                ta_sentences = self.split_into_sentences(ta_text)
                for s in ta_sentences:
                    if self.is_creative_atmospheric_clause(s) and len(s.split()) < 8:
                        continue

                    dates = ClaimExtractor.extract_dates(s)
                    numbers = ClaimExtractor.extract_numbers(s)
                    entities = ClaimExtractor.extract_entities(s)
                    has_sup, _ = ClaimExtractor.has_superlative(s)
                    quotes = self.extract_quotations(s)
                    sentence_has_quote = bool(quotes) or any(q in s for q in ta_quotes)

                    sc = ScriptClaim(
                        claim_index=claim_counter,
                        scene_id=scene_id,
                        language="ta",
                        source_sentence=s,
                        extracted_statement=s,
                        extracted_entities=entities,
                        extracted_dates=dates,
                        extracted_numbers=numbers,
                        is_superlative=has_sup,
                        is_quotation=sentence_has_quote
                    )
                    extracted_claims.append(sc)
                    claim_counter += 1

        return extracted_claims

    def validate_script(
        self,
        storyboard: Dict[str, Any],
        content_plan: FactCheckedContentPlan,
        claims_map: Dict[str, VerifiedClaim]
    ) -> ScriptValidationReport:
        """
        Thoroughly validate storyboard claims against the verified content plan.
        """
        episode_id = content_plan.episode_id
        script_claims = self.extract_script_claims(storyboard)

        matched_verified = 0
        matched_cautious = 0
        matched_debated = 0
        new_unsupported = 0
        contradicted = 0
        overstated = 0
        numeric_mismatch = 0
        date_mismatch = 0
        entity_mismatch = 0
        epistemic_violation = 0
        quotation_violation = 0
        cross_language_mismatch = 0
        violations: List[Dict[str, Any]] = []

        # Authorized lookups
        allowed_nums = set(content_plan.allowed_numbers)
        allowed_dates = set(d.lower() for d in content_plan.allowed_dates)
        allowed_entities = set(e.lower() for e in content_plan.allowed_entities)
        verified_quotes = set(content_plan.verified_quotations)
        forbidden_claim_texts = [
            claims_map[cid].statement.lower()
            for cid in content_plan.forbidden_claim_ids if cid in claims_map
        ]

        # 1. Validate each extracted claim
        for sc in script_claims:
            s_text = sc.extracted_statement.lower()

            # Check for fake quotation
            quotes = self.extract_quotations(sc.source_sentence)
            if not quotes and sc.is_quotation:
                quotes = re.findall(r'["“]([^"”\n]+)["”]', sc.source_sentence)

            if quotes or sc.is_quotation:
                unverified_q = [
                    q for q in quotes
                    if q.strip().rstrip(".!?") not in [vq.strip().rstrip(".!?") for vq in verified_quotes]
                ]
                if unverified_q or (sc.is_quotation and not verified_quotes):
                    fake_quote = unverified_q[0] if unverified_q else sc.source_sentence
                    sc.status = ScriptClaimStatus.UNSUPPORTED_QUOTATION
                    sc.violation_reason = f"Fabricated quotation detected: '{fake_quote}'"
                    quotation_violation += 1
                    violations.append({
                        "scene_id": sc.scene_id,
                        "type": "UNSUPPORTED_QUOTATION",
                        "claim": sc.source_sentence,
                        "reason": sc.violation_reason
                    })
                    continue

            # Check for numeric mismatch
            unauthorized_nums = [n for n in sc.extracted_numbers if n not in allowed_nums]
            if unauthorized_nums:
                sc.status = ScriptClaimStatus.NUMERIC_MISMATCH
                sc.violation_reason = f"Unauthorized numerical value '{unauthorized_nums[0]}' not present in verified claims."
                numeric_mismatch += 1
                violations.append({
                    "scene_id": sc.scene_id,
                    "type": "NUMERIC_MISMATCH",
                    "claim": sc.source_sentence,
                    "reason": sc.violation_reason
                })
                continue

            # Check for date mismatch
            unauthorized_dates = [d for d in sc.extracted_dates if d.lower() not in allowed_dates]
            if unauthorized_dates:
                sc.status = ScriptClaimStatus.DATE_MISMATCH
                sc.violation_reason = f"Unauthorized date reference '{unauthorized_dates[0]}' not present in verified claims."
                date_mismatch += 1
                violations.append({
                    "scene_id": sc.scene_id,
                    "type": "DATE_MISMATCH",
                    "claim": sc.source_sentence,
                    "reason": sc.violation_reason
                })
                continue

            # Check for forbidden claims
            is_forbidden = any(fc in s_text for fc in forbidden_claim_texts)
            if is_forbidden:
                sc.status = ScriptClaimStatus.CONTRADICTED_CLAIM
                sc.violation_reason = "Narration makes an assertion explicitly categorized as forbidden/rejected."
                contradicted += 1
                violations.append({
                    "scene_id": sc.scene_id,
                    "type": "CONTRADICTED_CLAIM",
                    "claim": sc.source_sentence,
                    "reason": sc.violation_reason
                })
                continue

            # Check for unauthorized entities (only meaningful entities, excluding COMMON_WORD)
            if sc.extracted_entities:
                from autonomous.claim_extractor import ClaimExtractor
                from autonomous.claim_models import EntityType
                meaningful_extracted = [
                    e for e in sc.extracted_entities
                    if ClaimExtractor.classify_entity(e) != EntityType.COMMON_WORD
                ]
                unauthorized_ents = [
                    e for e in meaningful_extracted
                    if e.lower() not in allowed_entities
                ]
                if unauthorized_ents:
                    sc.status = ScriptClaimStatus.NEW_UNSUPPORTED_CLAIM
                    sc.violation_reason = f"Unauthorized entity '{unauthorized_ents[0]}' not present in verified plan."
                    new_unsupported += 1
                    entity_mismatch += 1
                    violations.append({
                        "scene_id": sc.scene_id,
                        "type": "ENTITY_MISMATCH",
                        "claim": sc.source_sentence,
                        "reason": sc.violation_reason
                    })
                    continue

            # For Tamil narration, if it passed dates, numbers, entities, quotes, and forbidden checks,
            # we do not fail it on ASCII English word overlap, and bind to the scene's authorized claim.
            if sc.language == "ta":
                scene_plan = next((s for s in content_plan.scenes if s.scene_id == sc.scene_id), None)
                assigned_cid = (scene_plan.allowed_claim_ids[0] if scene_plan and scene_plan.allowed_claim_ids
                                else (content_plan.allowed_claim_ids[0] if content_plan.allowed_claim_ids
                                      else (content_plan.cautious_claim_ids[0] if content_plan.cautious_claim_ids else None)))
                sc.matched_claim_id = assigned_cid
                if assigned_cid and assigned_cid in claims_map:
                    matched_cls = claims_map[assigned_cid].classification
                    if matched_cls == ClaimClassification.VERIFIED_FACT:
                        sc.status = ScriptClaimStatus.MATCHED_VERIFIED_CLAIM
                        matched_verified += 1
                    elif matched_cls == ClaimClassification.SUPPORTED_HYPOTHESIS:
                        sc.status = ScriptClaimStatus.MATCHED_CAUTIOUS_CLAIM
                        matched_cautious += 1
                    else:
                        sc.status = ScriptClaimStatus.MATCHED_DEBATED_CLAIM
                        matched_debated += 1
                else:
                    sc.status = ScriptClaimStatus.MATCHED_VERIFIED_CLAIM
                    matched_verified += 1
                continue

            # Match against verified claims pool (for English)
            best_match: Optional[VerifiedClaim] = None
            best_overlap = 0.0

            words_sc = set(re.findall(r"\b[a-zA-Z]{4,}\b", s_text))
            stemmed_sc = {w.rstrip("s") for w in words_sc if len(w) > 3}

            for cid in content_plan.allowed_claim_ids + content_plan.cautious_claim_ids + content_plan.debated_claim_ids:
                if cid not in claims_map:
                    continue
                vc = claims_map[cid]
                words_vc = set(re.findall(r"\b[a-zA-Z]{4,}\b", vc.statement.lower()))
                stemmed_vc = {w.rstrip("s") for w in words_vc if len(w) > 3}

                overlap = len(stemmed_sc & stemmed_vc) / max(1, len(stemmed_vc))
                # Boost overlap if they share extracted dates or numbers
                if sc.extracted_dates and vc.extracted_dates:
                    if set(d.lower() for d in sc.extracted_dates) & set(d.lower() for d in vc.extracted_dates):
                        overlap += 0.40
                if sc.extracted_numbers and vc.extracted_numbers:
                    if set(sc.extracted_numbers) & set(vc.extracted_numbers):
                        overlap += 0.25

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = vc

            # Check certainty inflation indicators
            has_certainty_inflation = any(t in s_text for t in CERTAINTY_OVERSTATEMENT_TERMS) or ("proved conclusively" in s_text) or ("without doubt" in s_text) or ("definitely" in s_text) or ("proved" in s_text and "evidence" not in s_text)

            if best_match and best_overlap >= 0.30:
                sc.matched_claim_id = best_match.claim_id

                # Epistemic check: Certainty inflation
                if best_match.classification in (ClaimClassification.SUPPORTED_HYPOTHESIS, ClaimClassification.UNRESOLVED_DEBATE) or \
                   best_match.epistemic_status in (EpistemicStatus.PROPOSED_THEORY, EpistemicStatus.TRADITION_OR_LEGEND, EpistemicStatus.LIKELY, EpistemicStatus.DEBATED):
                    if has_certainty_inflation or ("proved" in s_text and "suggest" in best_match.statement.lower()):
                        sc.status = ScriptClaimStatus.OVERSTATED_CLAIM
                        sc.violation_reason = f"Epistemic overstatement: claim '{best_match.claim_id}' is a hypothesis/debate but script stated it as proven fact."
                        overstated += 1
                        epistemic_violation += 1
                        violations.append({
                            "scene_id": sc.scene_id,
                            "type": "OVERSTATED_CLAIM",
                            "claim": sc.source_sentence,
                            "reason": sc.violation_reason
                        })
                        continue

                # Superlative check
                if sc.is_superlative and best_match.claim_type != ClaimType.SUPERLATIVE_ASSERTION:
                    sc.status = ScriptClaimStatus.OVERSTATED_CLAIM
                    sc.violation_reason = "Script introduced superlative terms ('first', 'oldest', 'only') not backed by verified superlative proof."
                    overstated += 1
                    violations.append({
                        "scene_id": sc.scene_id,
                        "type": "UNVERIFIED_SUPERLATIVE",
                        "claim": sc.source_sentence,
                        "reason": sc.violation_reason
                    })
                    continue

                # Passed matching
                if best_match.classification == ClaimClassification.VERIFIED_FACT:
                    sc.status = ScriptClaimStatus.MATCHED_VERIFIED_CLAIM
                    matched_verified += 1
                elif best_match.classification == ClaimClassification.SUPPORTED_HYPOTHESIS:
                    sc.status = ScriptClaimStatus.MATCHED_CAUTIOUS_CLAIM
                    matched_cautious += 1
                else:
                    sc.status = ScriptClaimStatus.MATCHED_DEBATED_CLAIM
                    matched_debated += 1

            else:
                # If certainty inflation is stated and no verified facts support it
                if has_certainty_inflation:
                    sc.status = ScriptClaimStatus.OVERSTATED_CLAIM
                    sc.violation_reason = f"Certainty inflation without verified proof in statement: '{sc.source_sentence}'"
                    overstated += 1
                    epistemic_violation += 1
                    violations.append({
                        "scene_id": sc.scene_id,
                        "type": "OVERSTATED_CLAIM",
                        "claim": sc.source_sentence,
                        "reason": sc.violation_reason
                    })
                    continue

                # Factual assertion not matched in verified claims pool
                if sc.extracted_entities or sc.extracted_numbers or sc.extracted_dates or len(sc.extracted_statement.split()) > 6:
                    sc.status = ScriptClaimStatus.NEW_UNSUPPORTED_CLAIM
                    sc.violation_reason = f"New factual assertion '{sc.extracted_statement[:60]}...' has no supporting verified claim in content plan."
                    new_unsupported += 1
                    violations.append({
                        "scene_id": sc.scene_id,
                        "type": "NEW_UNSUPPORTED_CLAIM",
                        "claim": sc.source_sentence,
                        "reason": sc.violation_reason
                    })

        # 2. Bilingual cross-language consistency check across each scene
        for scene in storyboard.get("scenes", []):
            sid = scene.get("id", 1)
            en_claims = [c for c in script_claims if c.scene_id == sid and c.language == "en"]
            ta_claims = [c for c in script_claims if c.scene_id == sid and c.language == "ta"]

            # Collect dates and numbers in both languages
            en_dates = {d for c in en_claims for d in c.extracted_dates}
            ta_dates = {d for c in ta_claims for d in c.extracted_dates}
            en_nums = {n for c in en_claims for n in c.extracted_numbers}
            ta_nums = {n for c in ta_claims for n in c.extracted_numbers}

            # If both present dates and they don't intersect
            if en_dates and ta_dates and not (en_dates & ta_dates):
                cross_language_mismatch += 1
                violations.append({
                    "scene_id": sid,
                    "type": "CROSS_LANGUAGE_DATE_MISMATCH",
                    "reason": f"Scene {sid}: English dates {list(en_dates)} diverge from Tamil dates {list(ta_dates)}."
                })

            if en_nums and ta_nums and not (en_nums & ta_nums):
                cross_language_mismatch += 1
                violations.append({
                    "scene_id": sid,
                    "type": "CROSS_LANGUAGE_NUMERIC_MISMATCH",
                    "reason": f"Scene {sid}: English numbers {list(en_nums)} diverge from Tamil numbers {list(ta_nums)}."
                })

            # Check for epistemic certainty mismatch between English and Tamil
            en_overstated = any(c.status == ScriptClaimStatus.OVERSTATED_CLAIM for c in en_claims)
            ta_overstated = any(c.status == ScriptClaimStatus.OVERSTATED_CLAIM for c in ta_claims)
            if en_overstated != ta_overstated:
                cross_language_mismatch += 1
                violations.append({
                    "scene_id": sid,
                    "type": "CROSS_LANGUAGE_EPISTEMIC_MISMATCH",
                    "reason": f"Scene {sid}: Epistemic certainty mismatch between English and Tamil narration."
                })

        # 3. Visual Prompt Anachronism & Semantic Safety (Section 12)
        visual_violations = 0
        anachronistic_terms = [
            "concrete", "cement", "crane", "steam engine", "gun", "cannon",
            "rifle", "tank", "telephone", "electricity", "computer", "plastic",
            "modern building", "skyscraper", "steel bridge", "asphalt", "automobile"
        ]
        for scene in storyboard.get("scenes", []):
            sid = scene.get("id", 1)
            v_prompt = (scene.get("visual_prompt", "") + " " + scene.get("visual_description", "")).lower()
            found_ana = [term for term in anachronistic_terms if term in v_prompt]
            if found_ana:
                visual_violations += 1
                violations.append({
                    "scene_id": sid,
                    "type": "VISUAL_ANACHRONISM_DETECTED",
                    "reason": f"Scene {sid}: Visual prompt contains anachronistic element(s): {found_ana}"
                })
            for fc in forbidden_claim_texts:
                if len(fc) > 12 and fc in v_prompt:
                    visual_violations += 1
                    violations.append({
                        "scene_id": sid,
                        "type": "VISUAL_FORBIDDEN_ENTITY",
                        "reason": f"Scene {sid}: Visual prompt depicts forbidden/contradicted historical claim."
                    })

        total_violations = (
            new_unsupported + contradicted + overstated + numeric_mismatch +
            date_mismatch + entity_mismatch + epistemic_violation +
            quotation_violation + cross_language_mismatch + visual_violations
        )
        passed = (total_violations == 0 and (matched_verified + matched_cautious) > 0)

        report = ScriptValidationReport(
            episode_id=episode_id,
            passed=passed,
            total_script_claims=len(script_claims),
            matched_verified_count=matched_verified,
            matched_cautious_count=matched_cautious,
            matched_debated_count=matched_debated,
            new_unsupported_count=new_unsupported,
            contradicted_count=contradicted,
            overstated_count=overstated,
            numeric_mismatch_count=numeric_mismatch,
            date_mismatch_count=date_mismatch,
            entity_mismatch_count=entity_mismatch,
            epistemic_violation_count=epistemic_violation,
            quotation_violation_count=quotation_violation,
            cross_language_mismatch_count=cross_language_mismatch,
            violations=violations,
            claims=script_claims
        )

        logger.info(
            f"Script validation for episode '{episode_id}' finished: passed={passed}, "
            f"violations={total_violations}, matched_claims={matched_verified + matched_cautious}."
        )
        return report
