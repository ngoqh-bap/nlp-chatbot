import csv
import json
import os
from typing import Any, Dict, List, Set, Tuple, Optional

from .preprocess import normalize_text

uts_ner = None  # optional underthesea NER removed; use model/deterministic extraction only


def _span_in_original(text: str, surface: str) -> Optional[Tuple[int, int]]:
    """Return [start, end) Unicode indices of `surface` in `text`, or None."""
    if not surface:
        return None
    t = text.casefold()
    s = surface.casefold()
    start = t.find(s)
    if start < 0:
        return None
    return start, start + len(surface)


def _load_entity_patterns(path: str) -> List[Tuple[str, str]]:
    patterns: List[Tuple[str, str]] = []
    if not os.path.isfile(path):
        return patterns

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            label = (item.get("label") or "").strip()
            pat = item.get("pattern")
            if isinstance(pat, str):
                pat = normalize_text(pat)
            else:
                continue
            if label and pat:
                patterns.append((label, pat))
    return patterns


def _extract_by_ner(text: str) -> List[Dict[str, Any]]:
    if uts_ner is None:
        return []

    try:
        ner_result = uts_ner(text)
    except (ValueError, RuntimeError):
        return []

    found: List[Dict[str, Any]] = []
    buffer_tokens: List[str] = []
    current_tag: str = ""

    def flush():
        nonlocal buffer_tokens, current_tag, found
        if buffer_tokens and current_tag:
            span = " ".join(buffer_tokens)
            found.append({"label": current_tag, "text": span, "source": "ner"})
        buffer_tokens = []
        current_tag = ""

    for item in ner_result:
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                token = item[0]
                tag = item[-1]
            else:
                continue
        else:
            continue

        if isinstance(tag, str) and tag.startswith("B-"):
            flush()
            current_tag = tag[2:]
            buffer_tokens = [token]
        elif (
                isinstance(tag, str) and tag.startswith("I-") and current_tag == tag[2:]
        ):
            buffer_tokens.append(token)
        else:
            flush()

    flush()
    return found


class EntityExtractor:
    def __init__(self, data_dir: str, patterns_path: str, synonym_map: Optional[Dict[str, str]] = None) -> None:
        self.data_dir = data_dir
        self.synonym_map = synonym_map or {}

        self.entity_patterns: List[Tuple[str, str]] = _load_entity_patterns(
            patterns_path
        )

        self.dict_phrases: List[Tuple[str, str]] = self._load_dictionary_phrases()

        self.entity_label_alias: Dict[str, str] = {
            "NAM_TUYEN_SINH": "NAM_HOC",
            "NAM": "NAM_HOC",
            "PHUONG_THUC_TUYEN_SINH": "PHUONG_THUC_XET_TUYEN",
            "CHUNG_CHI": "CHUNG_CHI_UU_TIEN",
            "TO_HOP": "TO_HOP_MON",
            "KHOI_THI": "TO_HOP_MON",
        }

    def _load_dictionary_phrases(self) -> List[Tuple[str, str]]:
        phrases: List[Tuple[str, str]] = []

        def add_file(path: str, handler):  # type: ignore
            if os.path.isfile(path):
                with open(path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        handler(r, phrases)

        base = self.data_dir

        add_file(
            os.path.join(base, "majors.csv"),
            lambda r, p: (
                (
                    p.append(("MA_NGANH", normalize_text(r.get("major_code") or "")))
                    if r.get("major_code")
                    else None
                ),
                (
                    p.append(("TEN_NGANH", normalize_text(r.get("major_name") or "")))
                    if r.get("major_name")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "admission_methods.csv"),
            lambda r, p: (
                (
                    p.append(
                        (
                            "PHUONG_THUC_XET_TUYEN",
                            normalize_text(r.get("admission_method") or ""),
                        )
                    )
                    if r.get("admission_method")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "tuition.csv"),
            lambda r, p: (
                (
                    p.append(
                        (
                            "HOC_PHI_CATEGORY",
                            normalize_text(r.get("program_type") or ""),
                        )
                    )
                    if r.get("program_type")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "scholarships.csv"),
            lambda r, p: (
                (
                    p.append(("HOC_BONG_TEN", normalize_text(r.get("scholarship_name") or "")))
                    if r.get("scholarship_name")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "admission_scores.csv"),
            lambda r, p: (
                (
                    p.append(
                        (
                            "TEN_NGANH",
                            normalize_text(r.get("program_name") or ""),
                        )
                    )
                    if r.get("program_name")
                    else None
                ),
                [
                    p.append(("TO_HOP_MON", normalize_text(th.strip())))
                    for th in (r.get("subject_combination") or "").split(",")
                    if th.strip()
                ],
                [
                    p.append(("NAM_HOC", normalize_text(k)))
                    for k in r.keys()
                    if k and k.strip().isdigit() and len(k.strip()) == 4
                ],
            ),
        )

        add_file(
            os.path.join(base, "admission_targets.csv"),
            lambda r, p: (
                (
                    p.append(("MA_NGANH", normalize_text(r.get("major_code") or "")))
                    if r.get("major_code")
                    else None
                ),
                (
                    p.append(("MA_XET_TUYEN", normalize_text(r.get("admission_code") or "")))
                    if r.get("admission_code")
                    else None
                ),
                (
                    p.append(("TEN_NGANH", normalize_text(r.get("major_name") or "")))
                    if r.get("major_name")
                    else None
                ),
                (
                    p.append(("CHUYEN_NGANH", normalize_text(r.get("program_name") or "")))
                    if r.get("program_name")
                    else None
                ),
                [
                    p.append(("TO_HOP_MON", normalize_text(th)))
                    for th in (normalize_text(r.get("subject_combination") or "")).split(",")
                    if th
                ],
            ),
        )

        add_file(
            os.path.join(base, "admissions_schedule.csv"),
            lambda r, p: (
                (
                    p.append(
                        (
                            "PHUONG_THUC_XET_TUYEN",
                            normalize_text(r.get("admission_method") or ""),
                        )
                    )
                    if r.get("admission_method")
                    else None
                ),
                (
                    p.append(("THOI_GIAN_BUOC", normalize_text(r.get("timeline") or "")))
                    if r.get("timeline")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "contact_info.csv"),
            lambda r, p: (
                (
                    p.append(("DON_VI_LIEN_HE", normalize_text(r.get("university_name") or "")))
                    if r.get("university_name")
                    else None
                ),
                (
                    p.append(("DIA_CHI", normalize_text(r.get("address") or "")))
                    if r.get("address")
                    else None
                ),
                (
                    p.append(("EMAIL", normalize_text(r.get("email") or "")))
                    if r.get("email")
                    else None
                ),
                (
                    p.append(("DIEN_THOAI", normalize_text(r.get("phone") or "")))
                    if r.get("phone")
                    else None
                ),
                (
                    p.append(("HOTLINE", normalize_text(r.get("hotline") or "")))
                    if r.get("hotline")
                    else None
                ),
                (
                    p.append(("WEBSITE", normalize_text(r.get("website") or "")))
                    if r.get("website")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "cefr_conversion.csv"),
            lambda r, p: (
                (
                    p.append(("CHUNG_CHI_UU_TIEN", normalize_text("IELTS")))
                    if r.get("IELTS")
                    else None
                ),
                (
                    p.append(("CHUNG_CHI_UU_TIEN", normalize_text("TOEFL iBT")))
                    if r.get("TOEFL iBT")
                    else None
                ),
                (
                    p.append(("CHUNG_CHI_UU_TIEN", normalize_text("TOEIC")))
                    if r.get("TOEIC")
                    else None
                ),
            ),
        )

        add_file(
            os.path.join(base, "subject_combinations.csv"),
            lambda r, p: (
                (
                    p.append(("TO_HOP_MON", normalize_text(r.get("combination_code") or "")))
                    if r.get("combination_code")
                    else None
                ),
                (
                    p.append(("TO_HOP_MON_TEN", normalize_text(r.get("subject_names") or "")))
                    if r.get("subject_names")
                    else None
                ),
                [
                    p.append(("MON_HOC", normalize_text(subject.strip())))
                    for subject in (r.get("subject_names") or "").split(",")
                    if subject.strip()
                ],
                [
                    p.append(("KY_THI", normalize_text(exam_type.strip())))
                    for exam_type in (r.get("exam_type") or "").split(",")
                    if exam_type.strip()
                ],
            ),
        )

        add_file(
            os.path.join(base, "admission_conditions.csv"),
            lambda r, p: (
                (
                    p.append(("NAM_HOC", normalize_text(r.get("nam") or "")))
                    if r.get("nam")
                    else None
                ),
                (
                    p.append(
                        (
                            "PHUONG_THUC_XET_TUYEN",
                            normalize_text(r.get("admission_method") or ""),
                        )
                    )
                    if r.get("admission_method")
                    else None
                ),
                (
                    p.append(
                        (
                            "DIEU_KIEN_XET_TUYEN",
                            normalize_text(r.get("requirements") or ""),
                        )
                    )
                    if r.get("requirements")
                    else None
                ),
            ),
        )

        cleaned: List[Tuple[str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for lbl, phr in phrases:
            if not lbl or not phr:
                continue
            key = (lbl, phr)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(key)
        return cleaned

    def _extract_by_patterns(self, norm_text: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        for label, pat in self.entity_patterns:
            if pat and pat in norm_text:
                norm_pat = normalize_text(pat)
                fixed_label = label

                if "điểm sàn" in norm_pat:
                    fixed_label = "DIEM_SAN"
                elif "điểm chuẩn" in norm_pat:
                    fixed_label = "DIEM_CHUAN"

                found.append({"label": fixed_label, "text": pat, "source": "pattern"})
        return found

    def _extract_by_dictionaries(self, norm_text: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []

        for label, phrase in self.dict_phrases:
            if phrase and phrase in norm_text:
                found.append({"label": label, "text": phrase, "source": "dictionary"})

        tokens = norm_text.split()
        expanded_tokens = []

        for token in tokens:
            canonical = self.synonym_map.get(token, token)
            expanded_tokens.append(canonical)

        expanded_text = " ".join(expanded_tokens)

        if expanded_text != norm_text:
            for label, phrase in self.dict_phrases:
                if phrase and phrase in expanded_text:
                    already_found = any(
                        e["label"] == label and e["text"] == phrase
                        for e in found
                    )
                    if not already_found:
                        found.append({"label": label, "text": phrase, "source": "dictionary"})

        return found

    def extract(self, text: str) -> List[Dict[str, Any]]:
        norm = normalize_text(text)

        results: List[Dict[str, Any]] = []
        results.extend(self._extract_by_patterns(norm))
        results.extend(self._extract_by_dictionaries(norm))
        results.extend(_extract_by_ner(text))

        seen: Set[Tuple[str, str]] = set()
        dedup: List[Dict[str, Any]] = []

        for ent in results:
            raw_label = (ent.get("label") or "").strip()
            raw_text = ent.get("text") or ""
            norm_t = normalize_text(raw_text)

            canon_label = self.entity_label_alias.get(raw_label, raw_label)

            if "điểm chuẩn" in norm_t:
                canon_label = "DIEM_CHUAN"

            span = _span_in_original(text, raw_text)
            if span is None:
                continue
            start, end = span

            key = (canon_label, norm_t)
            if key not in seen:
                seen.add(key)
                dedup.append(
                    {
                        "label": canon_label,
                        "text": raw_text,
                        "start": start,
                        "end": end,
                        "source": ent.get("source"),
                    }
                )

        return dedup
