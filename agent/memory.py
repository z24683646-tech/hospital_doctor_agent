"""Small Markdown memory for evaluation reflections.

The baseline stores one reflection field per patient. Future prompts read the
latest patient reflections as simple reference notes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_PATH = BASE_DIR / "data" / "memory_data" / "memory.md"
DEFAULT_DATA_DIR = BASE_DIR / "data" / "memory_data"


class MarkdownMemory:
    def __init__(
        self,
        path: Union[str, Path],
        *,
        max_notes: int = 3,
        max_note_chars: int = 1200,
    ):
        self.path = Path(path)
        self.max_notes = int(max_notes)
        self.max_note_chars = int(max_note_chars)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Baseline Doctor Memory\n\n", encoding="utf-8")

    def load_notes(self, limit: Optional[int] = None) -> List[str]:
        """读取最近几条评估反思，新的在前。"""
        text = self.path.read_text(encoding="utf-8")
        notes = []
        for block in text.split("\n## "):
            block = block.strip()
            if block and not block.startswith("# Baseline"):
                notes.append(self._truncate("## " + block))
        notes.reverse()
        return notes[: int(limit or self.max_notes)]

    def append_case_reflection(
        self,
        *,
        patient_id: str,
        evaluation_reflection: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """训练评估后，每个患者只保存一个 reflection 字段。"""
        case_block = self._case_block(patient_id, evaluation_reflection or {})
        self._upsert_case_block(patient_id, case_block)

    def _case_block(self, patient_id: str, reflection: Dict[str, Any]) -> str:
        reflection_value = reflection.get("reflection") if "reflection" in reflection else reflection
        if not isinstance(reflection_value, dict):
            reflection_value = {"summary": reflection_value}

        lines = [
            "## Case %s Reflection" % patient_id,
            "",
            "### Reflection",
            "",
        ]
        for label, key in [
            ("Profile", "profile"),
            ("Diagnosis", "diagnosis_reflection"),
            ("Examination", "examination_reflection"),
            ("Treatment", "treatment_reflection"),
            ("Future Strategy", "future_strategy"),
        ]:
            value = self._truncate(reflection_value.get(key))
            if value:
                lines.append("- **%s:** %s" % (label, value))

        extra_items = [
            (key, value)
            for key, value in reflection_value.items()
            if key
            not in {
                "profile",
                "diagnosis_reflection",
                "examination_reflection",
                "treatment_reflection",
                "future_strategy",
            }
        ]
        for key, value in extra_items:
            lines.append("- **%s:** %s" % (str(key), self._truncate(value)))

        return "\n".join(lines).rstrip() + "\n"

    def _upsert_case_block(self, patient_id: str, new_block: str) -> None:
        text = self.path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        header = parts[0].rstrip() if parts else "# Baseline Doctor Memory"
        case_blocks = []
        for raw_block in parts[1:]:
            block = "## " + raw_block.strip()
            if not self._is_patient_block(block, patient_id):
                case_blocks.append(block)
        body = "\n\n".join([*case_blocks, new_block]).strip()
        self.path.write_text("%s\n\n%s\n" % (header, body), encoding="utf-8")

    def _is_patient_block(self, block: str, patient_id: str) -> bool:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        return first_line in {
            "## Case %s" % patient_id,
            "## Case %s Reflection" % patient_id,
        }

    def _truncate(self, value: Any, max_chars: Optional[int] = None) -> str:
        text = str(value or "").strip()
        limit = int(max_chars or self.max_note_chars)
        if len(text) <= limit:
            return text
        return text[: limit - 18].rstrip() + "\n...[truncated]"


class DoctorMemory:
    """Structured memory for doctor agent with JSON-based storage and keyword retrieval.

    All thresholds/switches/paths are read from config.yaml[memory] section;
    defaults below match hardcoded legacy behavior exactly.
    """

    def __init__(self, config: dict):
        memory_config = config if isinstance(config, dict) else {}
        configured_path = memory_config.get("md_path")
        self.max_notes = int(memory_config.get("max_notes", 100))
        self.max_note_chars = int(memory_config.get("max_note_chars", 500))

        md_path = Path(configured_path) if configured_path else DEFAULT_MEMORY_PATH
        if not md_path.is_absolute():
            md_path = BASE_DIR / md_path
        self.md_path = md_path
        self.data_dir = self.md_path.parent
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 读取 config.yaml[memory] 新增覆盖配置（__init__ 全部所需配置） ----------
        def _cfg_int(key: str, default: int) -> int:
            try:
                return int(memory_config.get(key, default))
            except (TypeError, ValueError):
                return int(default)
        def _cfg_bool(key: str, default: bool) -> bool:
            value = memory_config.get(key, default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
        def _resolve_path(key: str, default_filename: str) -> Path:
            raw = memory_config.get(key, "")
            raw_str = str(raw or "").strip()
            if not raw_str:
                return self.data_dir / default_filename
            p = Path(raw_str)
            return p if p.is_absolute() else BASE_DIR / p

        # 字符长度阈值（原类常量 MAX_*）
        self.max_entry_chars: int = _cfg_int("max_entry_chars", 50)
        self.max_prompt_chars: int = _cfg_int("max_prompt_chars", 200)

        # 3 个 JSON 存储路径
        self.disease_map_path: Path = _resolve_path("disease_map_path", "disease_map.json")
        self.error_lessons_path: Path = _resolve_path("error_lessons_path", "error_lessons.json")
        self.confusion_pairs_path: Path = _resolve_path("confusion_pairs_path", "confusion_pairs.json")

        # 功能开关（诊断报告和最终治疗方案不保存，减少 token 消耗）
        self.enable_auto_migrate: bool = _cfg_bool("enable_auto_migrate", True)
        self.save_diagnosis_report: bool = False
        self.save_final_treatment: bool = False
        self.enable_deduplicate_perfect_success: bool = False

        # 列表长度阈值（大幅减少，降低 token 消耗）
        self.max_symptoms_per_case: int = _cfg_int("max_symptoms_per_case", 5)
        self.max_exams_per_case: int = _cfg_int("max_exams_per_case", 5)
        self.max_treatments_per_case: int = _cfg_int("max_treatments_per_case", 5)
        self.max_diagnosis_reports_per_case: int = 0
        self.max_final_treatments_per_case: int = 0
        self.max_lessons_per_deduction_type: int = _cfg_int("max_lessons_per_deduction_type", 3)
        # 夹逼上限：get_error_patterns 最多 10 条，retrieve_relevant 最多 3 条（与需求严格对齐，config过大也不会超）
        self.max_error_patterns_limit: int = min(_cfg_int("max_error_patterns_limit", 5), 10)
        self.max_retrieve_relevant_limit: int = min(_cfg_int("max_retrieve_relevant_limit", 3), 3)
        # --------------------------------------------------------------------------

        self.disease_map: Dict[str, Dict[str, Any]] = self._load_json(self.disease_map_path, {})
        self.error_lessons: List[Dict[str, str]] = self._load_json(self.error_lessons_path, [])
        self.confusion_pairs: List[Dict[str, str]] = self._load_json(self.confusion_pairs_path, [])

        migrated = False
        if self.enable_auto_migrate:
            for disease, entry in self.disease_map.items():
                if not isinstance(entry, dict):
                    continue
                need_migrate = False
                if "department" not in entry:
                    entry["department"] = "未知科室"
                    need_migrate = True
                if "diagnosis_reports" not in entry:
                    entry["diagnosis_reports"] = []
                    need_migrate = True
                if "final_treatments" not in entry:
                    entry["final_treatments"] = []
                    need_migrate = True
                if need_migrate:
                    migrated = True

        if migrated:
            try:
                self._save_json(self.disease_map_path, self.disease_map)
            except OSError:
                pass

        if not self.md_path.exists():
            self.md_path.write_text("# Doctor Memory\n\n", encoding="utf-8")

    def _load_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    def _save_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _as_scalar_text(self, value: Any) -> str:
        """Extract first scalar from list/tuple/set and truncate; else just truncate as-is.

        Fixes the bug where list-valued scalar fields (e.g. diagnosis=["心绞痛"])
        became "['心绞痛']" (stringified list brackets) when used as keys/scalars.
        """
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            first = next((x for x in list(value) if x is not None and str(x).strip()), None)
            return self._truncate_entry(first if first is not None else "")
        return self._truncate_entry(value)

    def _truncate_entry(self, value: Any) -> str:
        text = str(value or "").strip()
        if len(text) <= self.max_entry_chars:
            return text
        return text[: self.max_entry_chars - 3] + "..."

    def load_notes(self, limit: Optional[int] = None) -> List[str]:
        """Load recent case reflections from Markdown (backward compatible).

        Pre-pends two segments auto-generated by the two canonical methods:
        - retrieve_relevant() -> 经验·标准路径参考 (正确答案/标准检查/标准治疗)
        - get_error_patterns() -> 扣分教训·避免重犯
        so that baseline agent.py (no modifications) always injects them.
        约束：两个结构化前缀的**总字符** ≤ self.max_prompt_chars (默认 200)，避免浪费 token。
        """
        exp_fragment = ""
        err_fragment = ""
        try:
            exp_result = self.retrieve_relevant(query="")
            if isinstance(exp_result, list) and exp_result:
                single_parts: List[str] = []
                for exp in exp_result:
                    if not isinstance(exp, dict):
                        continue
                    d = self._truncate_entry(exp.get("disease", ""))
                    s = self._truncate_entry(exp.get("key_symptoms", ""))
                    e = self._truncate_entry(exp.get("exam_path", ""))
                    t = self._truncate_entry(exp.get("treatment_notes", ""))
                    err = self._truncate_entry(exp.get("error_note", ""))
                    if not d and not s and not e and not t:
                        continue
                    head = d
                    body = []
                    if s:
                        body.append("症" + s)
                    if e:
                        body.append("检" + e)
                    if t:
                        body.append("治" + t)
                    if err:
                        body.append("⚠" + err)
                    if head and body:
                        one = head + ":" + ";".join(body)
                    elif head:
                        one = head
                    elif body:
                        one = ";".join(body)
                    else:
                        one = ""
                    if one:
                        single_parts.append(one)
                if single_parts:
                    exp_fragment = "|".join(single_parts)
            elif isinstance(exp_result, str) and exp_result.strip():
                exp_fragment = exp_result.strip()
        except Exception:
            exp_fragment = ""

        try:
            err_raw = self.get_error_patterns()
            if isinstance(err_raw, list):
                joined = "; ".join(str(x) for x in err_raw if x).strip()
                if joined:
                    err_fragment = joined
            elif isinstance(err_raw, str) and err_raw.strip():
                err_fragment = err_raw.strip()
        except Exception:
            err_fragment = ""

        prefix: List[str] = []
        label_exp = "经验·标准路径参考:"
        label_err = "扣分教训·避免重犯:"
        total_budget = self.max_prompt_chars
        used = 0
        if exp_fragment:
            candidate_exp = label_exp + exp_fragment
            if len(candidate_exp) <= total_budget:
                prefix.append(candidate_exp)
                used += len(candidate_exp) + 2
            else:
                allowed = max(0, total_budget - len(label_exp) - 3)
                if len(exp_fragment) > allowed:
                    exp_fragment = exp_fragment[:allowed] + "..."
                if exp_fragment:
                    prefix.append(label_exp + exp_fragment)
                    used = len(prefix[-1])
        if err_fragment:
            remaining = max(0, total_budget - used)
            if remaining > len(label_err):
                allowed_err = max(0, remaining - len(label_err) - 3)
                if len(err_fragment) > allowed_err:
                    err_fragment = err_fragment[:allowed_err] + "..."
                if err_fragment:
                    prefix.append(label_err + err_fragment)

        text = self.md_path.read_text(encoding="utf-8")
        notes = []
        for block in text.split("\n## "):
            block = block.strip()
            if block and not block.startswith("# Doctor") and not block.startswith("# Baseline"):
                notes.append(self._truncate_md("## " + block))
        notes.reverse()
        md_limit = max(0, int(limit or self.max_notes))
        md_notes = notes[:md_limit]
        return prefix + md_notes

    def _truncate_md(self, value: Any) -> str:
        text = str(value or "").strip()
        if len(text) <= self.max_note_chars:
            return text
        return text[: self.max_note_chars - 18].rstrip() + "\n...[truncated]"

    def append_case_reflection(
        self,
        *,
        patient_id: str,
        evaluation_reflection: Optional[Dict[str, Any]] = None,
        eval_result: Optional[Dict[str, Any]] = None,
        case_summary: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """Append case reflection. If eval_result and case_summary provided, upgrade to structured save."""
        if eval_result is not None or case_summary is not None:
            self.save_case_experience(
                patient_id=patient_id,
                eval_result=eval_result or {},
                case_summary=case_summary or (evaluation_reflection or {}),
            )
        case_block = self._build_case_block(patient_id, evaluation_reflection or {})
        self._upsert_md_block(patient_id, case_block)

    def _build_case_block(self, patient_id: str, reflection: Dict[str, Any]) -> str:
        reflection_value = reflection.get("reflection") if isinstance(reflection, dict) and "reflection" in reflection else reflection
        profile_text = ""
        diagnosis_text = ""
        if isinstance(reflection_value, dict):
            profile_text = self._truncate_entry(str(reflection_value.get("profile") or ""))
            diagnosis_text = self._truncate_entry(str(reflection_value.get("diagnosis_reflection") or ""))
        lines = [
            "## Case %s Reflection" % patient_id,
            "",
            "- Profile: %s" % (profile_text or "-"),
            "- Diagnosis: %s" % (diagnosis_text or "-"),
            "",
        ]
        raw = "\n".join(lines).rstrip()
        if len(raw) > self.max_note_chars:
            raw = raw[: max(0, self.max_note_chars - 11)] + "[truncated]"
        return raw + "\n"

    def _upsert_md_block(self, patient_id: str, new_block: str) -> None:
        text = self.md_path.read_text(encoding="utf-8")
        parts = text.split("\n## ")
        header = parts[0].rstrip() if parts else "# Doctor Memory"
        case_blocks = []
        for raw_block in parts[1:]:
            block = "## " + raw_block.strip()
            if not self._is_patient_block(block, patient_id):
                case_blocks.append(block)
        body = "\n\n".join([*case_blocks, new_block]).strip()
        self.md_path.write_text("%s\n\n%s\n" % (header, body), encoding="utf-8")

    def _is_patient_block(self, block: str, patient_id: str) -> bool:
        first_line = block.splitlines()[0].strip() if block.strip() else ""
        return first_line in {
            "## Case %s" % patient_id,
            "## Case %s Reflection" % patient_id,
        }

    def _to_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if v is not None and v != ""]
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace("，", ",").split(",") if p.strip()]
            return parts if parts else ([value] if value.strip() else [])
        if isinstance(value, dict):
            return [k for k, v in value.items() if v]
        return []

    def _extract_missing(
        self,
        submitted: Any,
        expected: Any,
        *,
        label: str,
    ) -> List[str]:
        submitted_items = [self._truncate_entry(str(x)).lower() for x in self._to_list(submitted)]
        expected_items = [self._truncate_entry(str(x)) for x in self._to_list(expected)]
        missing: List[str] = []
        for exp in expected_items:
            if not exp:
                continue
            exp_low = exp.lower()
            if not any(exp_low in s_item for s_item in submitted_items):
                missing.append("%s缺%s" % (label, exp))
        return missing

    def _extract_extra(
        self,
        submitted: Any,
        expected: Any,
        *,
        label: str,
    ) -> List[str]:
        submitted_items = [self._truncate_entry(str(x)) for x in self._to_list(submitted)]
        expected_items = [self._truncate_entry(str(x)).lower() for x in self._to_list(expected)]
        extra: List[str] = []
        for sub in submitted_items:
            if not sub:
                continue
            sub_low = sub.lower()
            if not any(e_item in sub_low for e_item in expected_items):
                extra.append("%s多%s" % (label, sub))
        return extra

    def _extract_abnormal_exams(self, examination_results: Any) -> List[str]:
        normal_phrases = {
            "未见异常", "无异常", "无明显异常", "未见明显异常",
            "正常", "未见特殊", "无特殊", "大致正常",
            "normal", "within normal", "unremarkable", "negative",
        }
        abnormal_flags = {
            "↑", "↓", "异常", "偏高", "偏低", "阳性", "增高", "降低",
            "升高", "下降", "轻度升高", "明显升高", "危急",
            "high", "low", "abnormal", "elevated", "decreased",
            "positive", "h", "l",
        }
        results: List[str] = []
        if not isinstance(examination_results, dict):
            return results
        for exam_name, detail in examination_results.items():
            name_str = str(exam_name).strip()
            if not name_str:
                continue
            is_abnormal = False
            if isinstance(detail, dict):
                status = str(detail.get("status") or detail.get("result") or detail.get("conclusion") or "").lower()
                value = str(detail.get("value") or detail.get("level") or "").lower()
                text = " ".join(str(v) for v in detail.values() if v is not None).lower()
                joined = " ".join([status, value, text])
            else:
                joined = str(detail).lower()
            if any(np in joined for np in normal_phrases):
                continue
            if any(flag in joined for flag in abnormal_flags):
                is_abnormal = True
            if is_abnormal:
                results.append("%s(异常)" % self._truncate_entry(name_str))
        return results

    def _is_duplicate_case(
        self,
        entry: Dict[str, Any],
        *,
        standard_exams: List[str],
        standard_treatments: List[str],
    ) -> bool:
        if int(entry.get("count", 0)) < 1:
            return False
        existing_std_exams = set(str(x) for x in entry.get("standard_exams", []) if x)
        new_std_exams = set(str(x) for x in standard_exams if x)
        existing_std_treats = set(str(x) for x in entry.get("standard_treatments", []) if x)
        new_std_treats = set(str(x) for x in standard_treatments if x)
        return existing_std_exams == new_std_exams and existing_std_treats == new_std_treats

    def save_case_experience(
        self,
        *,
        patient_id: str,
        eval_result: Dict[str, Any],
        case_summary: Dict[str, Any],
    ) -> None:
        """Save structured experience in 3 phases:
        1) Save case basics first (department + disease + ordered exams +
           diagnosis report + final treatment plan), with Chinese fields
           ("科室"/"诊断书"/"最终治疗方案") preferred.
        2) Compare against ground truth to extract deduction reasons (lessons)
           and correct standard paths (standard_exams / standard_treatments).
        3) Archive by department to accelerate future retrieval.
        Deduction fields in eval_result: diagnosisAccuracy, examinationPrecision,
        treatmentOverallScore, treatmentSafety, diagnosisDetail{submitted,expected,
        matched,status}, examinationDetail{ordered,expected,matched},
        treatmentDetail{submitted,reference,reasoning}, ground_truth.
        """
        eval_dict = eval_result if isinstance(eval_result, dict) else {}
        case_dict = case_summary if isinstance(case_summary, dict) else {}
        ground_truth = eval_dict.get("ground_truth") or {}
        gt_dict = ground_truth if isinstance(ground_truth, dict) else {}

        # ========= Phase 1: Save CASE BASICS first =========
        # 1-1) Department (按科室分层)
        department_raw = (
            case_dict.get("科室")
            or case_dict.get("所属科室")
            or case_dict.get("科室分类")
            or case_dict.get("department")
            or gt_dict.get("科室")
            or gt_dict.get("department")
        )
        department = self._as_scalar_text(department_raw or "未知科室")

        # 1-2) Disease (主键) + ground_truth 完整性标志
        diagnosis_detail = eval_dict.get("diagnosisDetail") or {}
        gt_diagnosis = gt_dict.get("diagnosis") or gt_dict.get("disease")
        expected_diagnosis_raw = gt_diagnosis  # 只用 ground_truth.diagnosis 作为"正确答案主键"，不再 fallback diagnosisDetail.expected
        submitted_diagnosis_raw = (
            diagnosis_detail.get("submitted")
            or case_dict.get("诊断")
            or case_dict.get("疾病")
            or case_dict.get("diagnosis")
            or case_dict.get("disease")
        )
        # ground_truth 完整性标志：标准答案是否存在（非常重要，后面所有逻辑依赖这个）
        gt_has_diagnosis = bool(self._as_scalar_text(gt_diagnosis or ""))
        gt_has_exams = bool([x for x in self._to_list(gt_dict.get("examinations") or gt_dict.get("exams")) if x])
        gt_has_treatments = bool([x for x in self._to_list(gt_dict.get("treatments")) if x])
        diagnosis_correct = False
        submitted_disease = self._as_scalar_text(submitted_diagnosis_raw)
        gt_disease_scalar = self._as_scalar_text(gt_diagnosis or "")
        if gt_has_diagnosis and submitted_disease:
            diagnosis_correct = submitted_disease == gt_disease_scalar
        # 主键：有 gt_diagnosis 强制用它（即使 submitted 是别的病，也存到正确疾病名下）；没有 gt 才用 submitted
        disease = self._as_scalar_text((gt_diagnosis or submitted_diagnosis_raw) or "unknown")

        # 1-3) Symptoms + 只存异常检查结果 (约束 ②)
        symptoms = [
            self._truncate_entry(str(s))
            for s in self._to_list(
                case_dict.get("主诉")
                or case_dict.get("症状")
                or case_dict.get("现病史")
                or case_dict.get("symptoms")
            )[: self.max_symptoms_per_case]
            if s
        ]
        abnormal_exams_list = self._extract_abnormal_exams(
            case_dict.get("检查结果")
            or case_dict.get("examination_results")
            or {}
        )
        for ab in abnormal_exams_list:
            if ab and ab not in symptoms:
                symptoms.append(ab)

        # 1-4) 需要的检查 (ordered exams)
        examination_detail = eval_dict.get("examinationDetail") or {}
        ordered_exams_raw = (
            examination_detail.get("ordered")
            or case_dict.get("检查")
            or case_dict.get("检查项目")
            or case_dict.get("examinations")
            or case_dict.get("exams")
        )
        submitted_exams = [
            self._truncate_entry(str(e)) for e in self._to_list(ordered_exams_raw)[: self.max_exams_per_case] if e
        ]
        # 诊断错误 且 有 ground_truth：submitted_exams 不合并到正确疾病 entry.exams（避免错误开单污染正确疾病的开单集合）
        if gt_has_diagnosis and not diagnosis_correct:
            all_exams = []
        else:
            all_exams = list(dict.fromkeys(submitted_exams))

        # 1-5) 治疗诊断书 diagnosis report（功能开关：save_diagnosis_report）
        diagnosis_reports_new: List[str] = []
        if self.save_diagnosis_report:
            diagnosis_report_raw = (
                case_dict.get("诊断书")
                or case_dict.get("诊断证明")
                or case_dict.get("诊断报告")
                or case_dict.get("diagnosis_report")
                or case_dict.get("diagnosis_letter")
            )
            if isinstance(diagnosis_report_raw, list):
                diagnosis_reports_new = [
                    self._truncate_entry(str(x))
                    for x in diagnosis_report_raw[: self.max_diagnosis_reports_per_case]
                    if x
                ]
            elif diagnosis_report_raw:
                diagnosis_reports_new = [self._truncate_entry(str(diagnosis_report_raw))]

        # 1-6) 最终治疗方案 final treatment plan + submitted treatments（功能开关：save_final_treatment）
        treatment_detail = eval_dict.get("treatmentDetail") or {}
        submitted_treatments_raw = (
            treatment_detail.get("submitted")
            or case_dict.get("治疗")
            or case_dict.get("treatments")
        )
        final_plan_raw = (
            case_dict.get("最终治疗方案")
            or case_dict.get("最终方案")
            or case_dict.get("治疗方案")
            or case_dict.get("final_treatment_plan")
            or case_dict.get("final_treatment")
            or case_dict.get("treatment_plan")
            or submitted_treatments_raw
        )
        submitted_treatments = [
            self._truncate_entry(str(t))
            for t in self._to_list(submitted_treatments_raw)[: self.max_treatments_per_case]
            if t
        ]
        final_treatments_new: List[str] = []
        if self.save_final_treatment:
            if isinstance(final_plan_raw, list):
                final_treatments_new = [
                    self._truncate_entry(str(x))
                    for x in final_plan_raw[: self.max_final_treatments_per_case]
                    if x
                ]
            elif final_plan_raw:
                final_treatments_new = [self._truncate_entry(str(final_plan_raw))]
        # 诊断错误 且 有 ground_truth：submitted_treatments / final_treatments 不合并（避免错误方案污染正确疾病的方案集合）
        if gt_has_diagnosis and not diagnosis_correct:
            all_treatments = []
        else:
            all_treatments = list(
                dict.fromkeys(submitted_treatments + [t for t in final_treatments_new if t])
            )

        # 1-7) 写入 disease_map 基础条目 & count +1
        is_new_entry = disease not in self.disease_map
        if is_new_entry:
            self.disease_map[disease] = {
                "department": department,
                "symptoms": [],
                "exams": [],
                "treatments": [],
                "diagnosis_reports": [],
                "final_treatments": [],
                "standard_exams": [],
                "standard_treatments": [],
                "count": 0,
            }
        entry = self.disease_map[disease]
        if department and department != entry.get("department") and entry.get("department") in (None, "", "未知科室"):
            entry["department"] = department
        for s in symptoms:
            if s and s not in entry["symptoms"]:
                entry["symptoms"].append(s)
        for e in all_exams:
            if e and e not in entry["exams"]:
                entry["exams"].append(e)
        for t in all_treatments:
            if t and t not in entry["treatments"]:
                entry["treatments"].append(t)
        for dr in diagnosis_reports_new:
            if dr and dr not in entry.setdefault("diagnosis_reports", []):
                entry["diagnosis_reports"].append(dr)
        for ft in final_treatments_new:
            if ft and ft not in entry.setdefault("final_treatments", []):
                entry["final_treatments"].append(ft)
        entry["count"] = int(entry.get("count", 0)) + 1

        # ========= Phase 2: Compare with ground truth → 正确答案 + 扣分原因 =========
        # 2-1) 正确答案 (standard paths 标准答案) 只用 ground_truth 字段 → 绝对优先，不用参考分
        # standard_exams / standard_treatments：只从 gt_dict 取（disease_map 里存的"正确答案"）
        standard_exams = [
            self._truncate_entry(str(e))
            for e in self._to_list(gt_dict.get("examinations") or gt_dict.get("exams"))[: self.max_exams_per_case]
            if e
        ]
        standard_treatments = [
            self._truncate_entry(str(t))
            for t in self._to_list(gt_dict.get("treatments"))[: self.max_treatments_per_case]
            if t
        ]
        # *_for_compare：仅用于计算缺失/多余扣分项，gt 缺时 fallback 到裁判参考值（不存进 standard_* 正确答案）
        gt_exams_for_compare = (
            gt_dict.get("examinations")
            or gt_dict.get("exams")
            or examination_detail.get("expected")
        )
        gt_treatments_for_compare = (
            gt_dict.get("treatments")
            or treatment_detail.get("reference")
        )

        # 2-2) 扣分原因 lessons + ground_truth 缺字段告警（非常重要，没有标准答案就不能正确学习）
        lessons: List[Dict[str, str]] = []

        # === ground_truth 完整性告警（最高优先级，没有标准答案的话其他教训都白学）===
        if not (gt_has_diagnosis or gt_has_exams or gt_has_treatments):
            lessons.append({"reason": "⚠️完全无ground_truth标准答案（需补eval_result.ground_truth字段）", "amount": ""})
        else:
            if gt_has_diagnosis and not gt_has_exams:
                lessons.append({"reason": "⚠️gt缺标准答案:examinations（正确检查项缺失）", "amount": ""})
            if gt_has_diagnosis and not gt_has_treatments:
                lessons.append({"reason": "⚠️gt缺标准答案:treatments（正确治疗方案缺失）", "amount": ""})
            if not gt_has_diagnosis and (gt_has_exams or gt_has_treatments):
                lessons.append({"reason": "⚠️gt缺标准答案:diagnosis（正确疾病名称缺失）", "amount": ""})

        # === 扣分项：诊断（详细区分：明确错/缺失/gt缺失） ===
        diag_acc = eval_dict.get("diagnosisAccuracy")
        if diag_acc is not None and isinstance(diag_acc, (int, float)) and diag_acc < 1:
            acc_str = str(int(round((1 - float(diag_acc)) * 100)))
            submitted_disease = self._as_scalar_text(submitted_diagnosis_raw)
            if gt_has_diagnosis and submitted_diagnosis_raw and submitted_disease != disease and submitted_disease:
                lessons.append({
                    "reason": "诊断错:%s→%s" % (submitted_disease, disease),
                    "amount": acc_str,
                })
            elif gt_has_diagnosis and not submitted_disease:
                lessons.append({"reason": "⚠️诊断缺失:gt=%s,未解析到submitted诊断" % disease, "amount": acc_str})
            elif not gt_has_diagnosis:
                lessons.append({"reason": "⚠️诊断扣分但gt无正确疾病名,无法定位正确主键", "amount": acc_str})
            else:
                lessons.append({"reason": "诊断不准确", "amount": acc_str})

        exam_prec = eval_dict.get("examinationPrecision")
        if exam_prec is not None and isinstance(exam_prec, (int, float)):
            exam_score_str = str(int(round((1 - float(exam_prec)) * 100)))
            missing_exams = self._extract_missing(ordered_exams_raw, gt_exams_for_compare, label="检查")
            extra_exams = self._extract_extra(ordered_exams_raw, gt_exams_for_compare, label="检查")
            for reason in missing_exams[: self.max_lessons_per_deduction_type]:
                lessons.append({"reason": self._truncate_entry(reason), "amount": exam_score_str})
            for reason in extra_exams[: self.max_lessons_per_deduction_type]:
                lessons.append({"reason": self._truncate_entry(reason), "amount": exam_score_str})

        treat_overall = eval_dict.get("treatmentOverallScore")
        treat_safety = eval_dict.get("treatmentSafety")
        if treat_overall is not None and isinstance(treat_overall, (int, float)):
            treat_score_str = str(int(round((1 - float(treat_overall)) * 100)))
            missing_treats = self._extract_missing(submitted_treatments_raw, gt_treatments_for_compare, label="治疗")
            extra_treats = self._extract_extra(submitted_treatments_raw, gt_treatments_for_compare, label="治疗")
            for reason in missing_treats[: self.max_lessons_per_deduction_type]:
                lessons.append({"reason": self._truncate_entry(reason), "amount": treat_score_str})
            for reason in extra_treats[: self.max_lessons_per_deduction_type]:
                lessons.append({"reason": self._truncate_entry(reason), "amount": treat_score_str})
            if treat_safety is not None and isinstance(treat_safety, (int, float)) and float(treat_safety) < 1:
                lessons.append({
                    "reason": "治疗安全性不足",
                    "amount": str(int(round((1 - float(treat_safety)) * 100))),
                })

        diag_status = diagnosis_detail.get("status") if isinstance(diagnosis_detail, dict) else None
        if diag_status and str(diag_status).lower() not in {"ok", "correct", "true", "pass", "success"}:
            lesson_reason = "诊断状态:%s" % self._truncate_entry(str(diag_status))
            if not any(l.get("reason") == lesson_reason for l in lessons):
                lessons.append({"reason": lesson_reason, "amount": ""})

        old_deductions = eval_dict.get("deductions")
        if isinstance(old_deductions, list):
            for d in old_deductions:
                if isinstance(d, dict):
                    reason = self._truncate_entry(d.get("reason") or d.get("description") or "")
                    amount = self._truncate_entry(d.get("amount") or d.get("score") or "")
                    if reason:
                        lessons.append({"reason": reason, "amount": amount})

        # 2-3) 约束 ③：不存重复病例（基于疾病名+标准检查+标准治疗三重匹配，所有病例都去重）
        has_error = bool(lessons)
        should_skip = False
        if not is_new_entry:
            if self._is_duplicate_case(
                entry,
                standard_exams=standard_exams,
                standard_treatments=standard_treatments,
            ):
                should_skip = True
        if should_skip:
            entry["count"] = max(0, int(entry.get("count", 1)) - 1)
            return

        # 2-4) 约束③通过后，才将 standard paths 写入 entry（只保留 ground_truth 的标准路径，减少 token 消耗）
        for e in standard_exams:
            if e and e not in entry.setdefault("standard_exams", []):
                entry["standard_exams"].append(e)
        for t in standard_treatments:
            if t and t not in entry.setdefault("standard_treatments", []):
                entry["standard_treatments"].append(t)

        # ========= Phase 3: 扣分教训 + 混淆对 + 持久化 =========
        # 3-1) 扣分 lessons 去重写入 error_lessons.json
        for lesson in lessons:
            reason = lesson.get("reason", "")
            amount = lesson.get("amount", "")
            if not reason:
                continue
            if not any(
                l.get("reason") == reason and l.get("amount") == amount
                for l in self.error_lessons
            ):
                self.error_lessons.append({"reason": reason, "amount": amount})

        # 3-2) 混淆诊断对 confusion_pairs 去重写入（双条件：gt有正确疾病 + 提交了错误诊断）
        submitted = submitted_disease
        if gt_has_diagnosis and submitted and submitted != disease:
            pair_key = tuple(sorted([disease, submitted]))
            if not any(
                tuple(sorted([p.get("a", ""), p.get("b", "")])) == pair_key
                for p in self.confusion_pairs
            ):
                self.confusion_pairs.append({"a": disease, "b": submitted})

        old_confused = eval_dict.get("confused_with") or eval_dict.get("misdiagnosis")
        if old_confused:
            confused_disease = self._truncate_entry(
                old_confused if isinstance(old_confused, str) else old_confused.get("disease") or ""
            )
            correct_disease = disease
            if confused_disease and confused_disease != correct_disease:
                pair_key = tuple(sorted([correct_disease, confused_disease]))
                if not any(
                    tuple(sorted([p.get("a", ""), p.get("b", "")])) == pair_key
                    for p in self.confusion_pairs
                ):
                    self.confusion_pairs.append({"a": correct_disease, "b": confused_disease})

        # 3-3) JSON 原子落盘（3 个结构化文件：disease_map / error_lessons / confusion_pairs
        self._save_json(self.disease_map_path, self.disease_map)
        self._save_json(self.error_lessons_path, self.error_lessons)
        self._save_json(self.confusion_pairs_path, self.confusion_pairs)

    def _split_keywords(self, text: Any) -> List[str]:
        """Split any text/list into lowercase stripped keyword tokens (len>=2), dedup, cap 12."""
        raw: List[str] = []
        if isinstance(text, (list, tuple, set)):
            for item in text:
                s = str(item or "").strip()
                if s:
                    raw.extend(re.split(r"[\s，,。；;：:、（）()\[\]【】]+", s))
        else:
            s = str(text or "").strip()
            if s:
                raw.extend(re.split(r"[\s，,。；;：:、（）()\[\]【】]+", s))
        seen: set = set()
        out: List[str] = []
        for token in raw:
            t = token.strip().lower()
            if len(t) >= 2 and t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= 12:
                    break
        return out

    def _match_error_note_for_disease(self, disease: str) -> str:
        """Pick first lesson from error_lessons whose reason mentions this disease (or related tokens)."""
        d_tokens = self._split_keywords(disease)
        if not d_tokens and not disease:
            return ""
        d_lower = str(disease or "").lower()
        for lesson in self.error_lessons:
            if not isinstance(lesson, dict):
                continue
            reason = str(lesson.get("reason") or "").strip()
            if not reason:
                continue
            r_lower = reason.lower()
            if d_lower and d_lower in r_lower:
                return self._truncate_entry(reason)
            if any(t and t in r_lower for t in d_tokens):
                return self._truncate_entry(reason)
        return ""

    def _compact_experience_text(self, exp: Dict[str, Any]) -> Dict[str, Any]:
        """Trim an experience dict so the total serialized JSON text ≤50 chars."""
        disease = self._truncate_entry(exp.get("disease", ""))
        key_symptoms = self._truncate_entry(exp.get("key_symptoms", ""))
        exam_path = self._truncate_entry(exp.get("exam_path", ""))
        treatment_notes = self._truncate_entry(exp.get("treatment_notes", ""))
        error_note = self._truncate_entry(exp.get("error_note", ""))

        def _total_len() -> int:
            sample = {
                "disease": disease,
                "key_symptoms": key_symptoms,
                "exam_path": exam_path,
                "treatment_notes": treatment_notes,
                "error_note": error_note,
            }
            try:
                return len(json.dumps(sample, ensure_ascii=False))
            except Exception:
                return sum(len(str(v)) for v in sample.values()) + 20

        # 逐字段从非核心到核心截断：error_note → treatment_notes → exam_path → key_symptoms
        while _total_len() > self.max_entry_chars and error_note:
            if len(error_note) <= 3:
                error_note = ""
            else:
                error_note = error_note[: max(0, len(error_note) - 4)]
        while _total_len() > self.max_entry_chars and treatment_notes:
            if len(treatment_notes) <= 3:
                treatment_notes = ""
            else:
                treatment_notes = treatment_notes[: max(0, len(treatment_notes) - 4)]
        while _total_len() > self.max_entry_chars and exam_path:
            if len(exam_path) <= 3:
                exam_path = ""
            else:
                exam_path = exam_path[: max(0, len(exam_path) - 4)]
        while _total_len() > self.max_entry_chars and key_symptoms:
            if len(key_symptoms) <= 3:
                key_symptoms = ""
            else:
                key_symptoms = key_symptoms[: max(0, len(key_symptoms) - 4)]
        # 最后兜底：disease 也要保证不超
        while _total_len() > self.max_entry_chars and len(disease) > 3:
            disease = disease[:-1]

        result: Dict[str, Any] = {
            "disease": disease,
            "key_symptoms": key_symptoms,
            "exam_path": exam_path,
            "treatment_notes": treatment_notes,
        }
        if error_note:
            result["error_note"] = error_note
        return result

    def retrieve_relevant(
        self,
        *,
        symptoms: Optional[List[str]] = None,
        query: str = "",
        summary: Optional[str] = None,
        inquiry_summary: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """根据问诊摘要/症状关键词检索最相关的历史经验（纯 Python 关键词匹配，不调用 LLM）。

        返回：最多 3 条，每条 5 字段，单条总 JSON 文本 ≤50 字，
        格式：[{"disease","key_symptoms","exam_path","treatment_notes","error_note"(可选)}, ...]
        """
        effective_limit = self.max_retrieve_relevant_limit
        if top_k is not None:
            try:
                effective_limit = min(max(int(top_k), 0), self.max_retrieve_relevant_limit)
            except (TypeError, ValueError):
                pass
        keywords: List[str] = []
        if symptoms:
            keywords.extend(self._split_keywords(symptoms))
        if inquiry_summary:
            keywords.extend(self._split_keywords(inquiry_summary))
        if summary:
            keywords.extend(self._split_keywords(summary))
        if query:
            keywords.extend(self._split_keywords(query))
        seen_kw: set = set()
        dedup_kw: List[str] = []
        for k in keywords:
            if k and k not in seen_kw:
                seen_kw.add(k)
                dedup_kw.append(k)
            if len(dedup_kw) >= 12:
                break

        all_diseases = list(self.disease_map.keys())
        scored: List[tuple] = []

        if dedup_kw:
            for disease in all_diseases:
                info = self.disease_map.get(disease) or {}
                disease_lower = str(disease).lower()
                dep_lower = str(info.get("department") or "").strip().lower()
                syms_lower = [str(s).lower() for s in info.get("symptoms", [])]
                exams_lower = [str(e).lower() for e in (info.get("exams", []) + info.get("standard_exams", []))]
                treats_lower = [str(t).lower() for t in (info.get("treatments", []) + info.get("standard_treatments", []))]
                score = 0
                for kw in dedup_kw:
                    if kw in disease_lower:
                        score += 3
                    if any(kw in s for s in syms_lower):
                        score += 3
                    if any(kw in e for e in exams_lower):
                        score += 2
                    if any(kw in t for t in treats_lower):
                        score += 1
                    if len(kw) >= 2 and kw in dep_lower:
                        score += 1
                if score > 0:
                    count = int(info.get("count", 0) or 0)
                    scored.append((score, count, disease))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            picked_diseases: List[str] = [item[2] for item in scored[: effective_limit]]
        else:
            picked_diseases = sorted(
                all_diseases,
                key=lambda d: (self.disease_map.get(d) or {}).get("count", 0),
                reverse=True,
            )[: effective_limit]

        results: List[Dict[str, Any]] = []
        for disease in picked_diseases:
            info = self.disease_map.get(disease, {}) or {}
            symptoms_list = [self._truncate_entry(str(s)) for s in (info.get("symptoms") or []) if s][:3]
            merged_exams = [self._truncate_entry(str(e)) for e in (info.get("standard_exams") or []) if e][:3]
            merged_treats = [self._truncate_entry(str(t)) for t in (info.get("standard_treatments") or []) if t][:3]

            exp: Dict[str, Any] = {
                "disease": self._truncate_entry(str(disease)),
                "key_symptoms": "、".join(symptoms_list),
                "exam_path": "+".join(merged_exams),
                "treatment_notes": "+".join(merged_treats),
            }
            error_note = self._match_error_note_for_disease(disease)
            if error_note:
                exp["error_note"] = error_note
            results.append(self._compact_experience_text(exp))
            if len(results) >= effective_limit:
                break

        return results

    def get_error_patterns(self, limit: Optional[int] = None) -> List[str]:
        """Return top error lessons as concise string list, total text ≤max_prompt_chars, capped 10 items.

        Each element: "reason(-amount)" or just "reason"; empty element never returned.
        Count of elements ≤ min(limit|config, 10), and combined "; ".join() ≤ max_prompt_chars.
        """
        raw_limit = int(limit if limit is not None else self.max_error_patterns_limit)
        effective_limit = max(0, min(raw_limit, 10))
        lessons = self.error_lessons[:effective_limit]
        parts: List[str] = []
        total = 0
        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue
            reason = lesson.get("reason", "")
            amount = lesson.get("amount", "")
            fragment = "%s(-%s)" % (reason, amount) if amount else str(reason)
            fragment = self._truncate_entry(fragment)
            if not fragment:
                continue
            if total + len(fragment) + 2 > self.max_prompt_chars:
                break
            parts.append(fragment)
            total += len(fragment) + 2
        return parts


def build_memory(config: Dict[str, Any]) -> Union[MarkdownMemory, DoctorMemory]:
    memory_config = config.get("memory") if isinstance(config.get("memory"), dict) else {}
    use_doctor = bool(memory_config.get("use_doctor_memory", True))

    if use_doctor:
        return DoctorMemory(memory_config)

    configured_path = memory_config.get("md_path") if isinstance(memory_config, dict) else None
    max_notes = memory_config.get("max_notes", 3) if isinstance(memory_config, dict) else 3
    max_note_chars = memory_config.get("max_note_chars", 1200) if isinstance(memory_config, dict) else 1200
    memory_path = Path(configured_path) if configured_path else DEFAULT_MEMORY_PATH
    if not memory_path.is_absolute():
        memory_path = BASE_DIR / memory_path
    return MarkdownMemory(
        memory_path,
        max_notes=int(max_notes),
        max_note_chars=int(max_note_chars),
    )

