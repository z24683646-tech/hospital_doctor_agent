"""Small Markdown memory for evaluation reflections.

The baseline stores one reflection field per patient. Future prompts read the
latest patient reflections as simple reference notes.
"""

from __future__ import annotations

import json
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

        # 4 个 JSON 存储路径
        self.disease_map_path: Path = _resolve_path("disease_map_path", "disease_map.json")
        self.error_lessons_path: Path = _resolve_path("error_lessons_path", "error_lessons.json")
        self.confusion_pairs_path: Path = _resolve_path("confusion_pairs_path", "confusion_pairs.json")
        self.department_diseases_path: Path = _resolve_path("department_diseases_path", "department_diseases.json")

        # 功能开关
        self.enable_auto_migrate: bool = _cfg_bool("enable_auto_migrate", True)
        self.enable_department_index: bool = _cfg_bool("enable_department_index", True)
        self.save_diagnosis_report: bool = _cfg_bool("save_diagnosis_report", True)
        self.save_final_treatment: bool = _cfg_bool("save_final_treatment", True)
        self.enable_deduplicate_perfect_success: bool = _cfg_bool("enable_deduplicate_perfect_success", True)

        # 列表长度阈值
        self.max_symptoms_per_case: int = _cfg_int("max_symptoms_per_case", 10)
        self.max_exams_per_case: int = _cfg_int("max_exams_per_case", 10)
        self.max_treatments_per_case: int = _cfg_int("max_treatments_per_case", 10)
        self.max_diagnosis_reports_per_case: int = _cfg_int("max_diagnosis_reports_per_case", 3)
        self.max_final_treatments_per_case: int = _cfg_int("max_final_treatments_per_case", 3)
        self.max_lessons_per_deduction_type: int = _cfg_int("max_lessons_per_deduction_type", 3)
        self.max_error_patterns_limit: int = _cfg_int("max_error_patterns_limit", 5)
        # --------------------------------------------------------------------------

        self.disease_map: Dict[str, Dict[str, Any]] = self._load_json(self.disease_map_path, {})
        self.error_lessons: List[Dict[str, str]] = self._load_json(self.error_lessons_path, [])
        self.confusion_pairs: List[Dict[str, str]] = self._load_json(self.confusion_pairs_path, [])
        self.department_diseases: Dict[str, List[str]] = self._load_json(self.department_diseases_path, {})

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
                dep = str(entry.get("department") or "未知科室")
                if dep not in self.department_diseases:
                    self.department_diseases[dep] = []
                    migrated = True
                if disease not in self.department_diseases[dep]:
                    self.department_diseases[dep].append(disease)
                    migrated = True

        if migrated:
            try:
                self._save_json(self.disease_map_path, self.disease_map)
                self._save_json(self.department_diseases_path, self.department_diseases)
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

        Pre-pends two segments auto-generated by the three canonical methods:
        - retrieve_relevant() -> 经验·标准路径参考 (正确答案/标准检查/标准治疗)
        - get_error_patterns() -> 扣分教训·避免重犯
        so that baseline agent.py (no modifications) always injects them.
        """
        prefix: List[str] = []
        try:
            similar_str = self.retrieve_relevant(query="")
            if isinstance(similar_str, str) and similar_str.strip():
                prefix.append("经验·标准路径参考:" + similar_str)
        except Exception:
            pass
        try:
            error_str = self.get_error_patterns()
            if isinstance(error_str, str) and error_str.strip():
                prefix.append("扣分教训·避免重犯:" + error_str)
        except Exception:
            pass

        text = self.md_path.read_text(encoding="utf-8")
        notes = []
        for block in text.split("\n## "):
            block = block.strip()
            if block and not block.startswith("# Doctor") and not block.startswith("# Baseline"):
                notes.append(self._truncate_md("## " + block))
        notes.reverse()
        md_limit = int(limit or self.max_notes) - len(prefix)
        if md_limit < 0:
            md_limit = 0
        md_notes = notes[:md_limit]
        result = prefix + md_notes
        return result

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

    def _is_duplicate_perfect_success(
        self,
        entry: Dict[str, Any],
        *,
        diagnosis_accuracy,
        examination_precision,
        treatment_overall,
        treatment_safety,
        standard_exams: List[str],
        standard_treatments: List[str],
        is_error: bool,
    ) -> bool:
        if is_error:
            return False
        try:
            diag_ok = diagnosis_accuracy is None or float(diagnosis_accuracy) >= 1.0
            exam_ok = examination_precision is None or float(examination_precision) >= 1.0
            treat_ok = treatment_overall is None or float(treatment_overall) >= 1.0
            safe_ok = treatment_safety is None or float(treatment_safety) >= 1.0
        except (TypeError, ValueError):
            return False
        if not (diag_ok and exam_ok and treat_ok and safe_ok):
            return False
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

        # 1-2) Disease (主键)
        diagnosis_detail = eval_dict.get("diagnosisDetail") or {}
        gt_diagnosis = gt_dict.get("diagnosis") or gt_dict.get("disease")
        expected_diagnosis_raw = gt_diagnosis or diagnosis_detail.get("expected")
        submitted_diagnosis_raw = (
            diagnosis_detail.get("submitted")
            or case_dict.get("诊断")
            or case_dict.get("疾病")
            or case_dict.get("diagnosis")
            or case_dict.get("disease")
        )
        disease = self._as_scalar_text(expected_diagnosis_raw or submitted_diagnosis_raw or "unknown")

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
        # 2-1) 正确答案 (standard paths 标准答案) 先计算，不写入entry（避免干扰约束③去重对比基准）
        gt_exams_raw = (
            gt_dict.get("examinations")
            or gt_dict.get("exams")
            or examination_detail.get("expected")
        )
        standard_exams = [
            self._truncate_entry(str(e)) for e in self._to_list(gt_exams_raw)[: self.max_exams_per_case] if e
        ]
        gt_treatments_raw = gt_dict.get("treatments") or treatment_detail.get("reference")
        standard_treatments = [
            self._truncate_entry(str(t)) for t in self._to_list(gt_treatments_raw)[: self.max_treatments_per_case] if t
        ]

        # 2-2) 扣分原因 lessons
        lessons: List[Dict[str, str]] = []

        diag_acc = eval_dict.get("diagnosisAccuracy")
        if diag_acc is not None and isinstance(diag_acc, (int, float)) and diag_acc < 1:
            acc_str = str(int(round((1 - float(diag_acc)) * 100)))
            submitted_disease = self._as_scalar_text(submitted_diagnosis_raw)
            if expected_diagnosis_raw and submitted_diagnosis_raw and submitted_disease != disease and submitted_disease:
                lessons.append({
                    "reason": "诊断错:%s→%s" % (submitted_disease, disease),
                    "amount": acc_str,
                })
            else:
                lessons.append({"reason": "诊断不准确", "amount": acc_str})

        exam_prec = eval_dict.get("examinationPrecision")
        if exam_prec is not None and isinstance(exam_prec, (int, float)):
            exam_score_str = str(int(round((1 - float(exam_prec)) * 100)))
            missing_exams = self._extract_missing(ordered_exams_raw, gt_exams_raw, label="检查")
            extra_exams = self._extract_extra(ordered_exams_raw, gt_exams_raw, label="检查")
            for reason in missing_exams[: self.max_lessons_per_deduction_type]:
                lessons.append({"reason": self._truncate_entry(reason), "amount": exam_score_str})
            for reason in extra_exams[: self.max_lessons_per_deduction_type]:
                lessons.append({"reason": self._truncate_entry(reason), "amount": exam_score_str})

        treat_overall = eval_dict.get("treatmentOverallScore")
        treat_safety = eval_dict.get("treatmentSafety")
        if treat_overall is not None and isinstance(treat_overall, (int, float)):
            treat_score_str = str(int(round((1 - float(treat_overall)) * 100)))
            missing_treats = self._extract_missing(submitted_treatments_raw, gt_treatments_raw, label="治疗")
            extra_treats = self._extract_extra(submitted_treatments_raw, gt_treatments_raw, label="治疗")
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

        # 2-3) 约束 ③：不存重复的满分成功病例（功能开关：enable_deduplicate_perfect_success）
        has_error = bool(lessons)
        should_skip_perfect = False
        if self.enable_deduplicate_perfect_success and not is_new_entry:
            diag_acc_value = eval_dict.get("diagnosisAccuracy")
            exam_prec_value = eval_dict.get("examinationPrecision")
            treat_overall_value = eval_dict.get("treatmentOverallScore")
            treat_safety_value = eval_dict.get("treatmentSafety")
            if self._is_duplicate_perfect_success(
                entry,
                diagnosis_accuracy=diag_acc_value,
                examination_precision=exam_prec_value,
                treatment_overall=treat_overall_value,
                treatment_safety=treat_safety_value,
                standard_exams=standard_exams,
                standard_treatments=standard_treatments,
                is_error=has_error,
            ):
                should_skip_perfect = True
        if should_skip_perfect:
            entry["count"] = max(0, int(entry.get("count", 1)) - 1)
            return

        # 2-4) 约束③通过后，才将 standard paths / all merged paths 写入 entry（此时不影响去重的对比基准）
        for e in standard_exams:
            if e and e not in entry.setdefault("standard_exams", []):
                entry["standard_exams"].append(e)
        for t in standard_treatments:
            if t and t not in entry.setdefault("standard_treatments", []):
                entry["standard_treatments"].append(t)
        all_exams = list(dict.fromkeys(all_exams + standard_exams))
        all_treatments = list(dict.fromkeys(all_treatments + standard_treatments))
        for e in all_exams:
            if e and e not in entry["exams"]:
                entry["exams"].append(e)
        for t in all_treatments:
            if t and t not in entry["treatments"]:
                entry["treatments"].append(t)

        # ========= Phase 3: 按科室归档 + 持久化 =========
        # 3-1) 更新 department → diseases 分层索引 (O(1) 定位科室加速检索) 开关：enable_department_index
        if self.enable_department_index:
            dep_for_index = str(entry.get("department") or department or "未知科室")
            if dep_for_index not in self.department_diseases:
                self.department_diseases[dep_for_index] = []
            if disease not in self.department_diseases[dep_for_index]:
                self.department_diseases[dep_for_index].append(disease)

        # 3-2) 扣分 lessons 去重写入 error_lessons.json
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

        # 3-3) 混淆诊断对 confusion_pairs 去重写入
        if expected_diagnosis_raw and submitted_diagnosis_raw:
            correct_disease = disease
            submitted = self._as_scalar_text(submitted_diagnosis_raw)
            if submitted and submitted != correct_disease:
                pair_key = tuple(sorted([correct_disease, submitted]))
                if not any(
                    tuple(sorted([p.get("a", ""), p.get("b", "")])) == pair_key
                    for p in self.confusion_pairs
                ):
                    self.confusion_pairs.append({"a": correct_disease, "b": submitted})

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

        # 3-4) JSON 原子落盘（enable_department_index=False 时跳过分桶索引落盘，避免空数据覆盖）
        self._save_json(self.disease_map_path, self.disease_map)
        self._save_json(self.error_lessons_path, self.error_lessons)
        self._save_json(self.confusion_pairs_path, self.confusion_pairs)
        if self.enable_department_index:
            self._save_json(self.department_diseases_path, self.department_diseases)

    def retrieve_relevant(self, *, symptoms: Optional[List[str]] = None, query: str = "") -> str:
        """Retrieve relevant memory via keyword matching, accelerated by department index.

        Returns ≤200 chars. Department hits narrow candidates to a small per-department
        subset (O(1) lookup) instead of scanning all diseases.
        """
        keywords: List[str] = []
        if symptoms:
            keywords.extend([str(s).strip().lower() for s in symptoms if s])
        if query:
            query_str = str(query).strip()
            keywords.extend([q.strip().lower() for q in query_str.split() if q.strip()])
            keywords.append(query_str.lower())
        keywords = [k for k in keywords if k][:12]

        search_text = " ".join(keywords)

        matched_departments: List[str] = []
        if self.enable_department_index:
            for dep_name in self.department_diseases.keys():
                if not dep_name or dep_name == "未知科室":
                    continue
                dep_low = str(dep_name).strip().lower()
                if dep_low and (dep_low in search_text or any(dep_low in k for k in keywords)):
                    matched_departments.append(dep_name)

        candidate_diseases: List[str] = []
        if matched_departments:
            seen_dep: set = set()
            for dep in matched_departments:
                for d in self.department_diseases.get(dep, []) or []:
                    if d and d not in seen_dep:
                        seen_dep.add(d)
                        candidate_diseases.append(d)

        matched: List[str] = []
        candidate_pool = candidate_diseases if candidate_diseases else list(self.disease_map.keys())
        if keywords:
            for disease in candidate_pool:
                info = self.disease_map.get(disease) or {}
                disease_symptoms = [s.lower() for s in info.get("symptoms", [])]
                if any(any(k in ds for ds in disease_symptoms) for k in keywords) or \
                   any(k in str(disease).lower() for k in keywords):
                    matched.append(disease)

        if not matched:
            fallback_pool = candidate_pool or list(self.disease_map.keys())
            matched = sorted(
                fallback_pool,
                key=lambda d: (self.disease_map.get(d) or {}).get("count", 0),
                reverse=True,
            )[:3]

        parts: List[str] = []
        total = 0
        for disease in matched[:3]:
            info = self.disease_map.get(disease, {}) or {}
            dep_tag = str(info.get("department") or "")
            s_list = info.get("symptoms", [])[:3]
            s_text = ",".join([str(x) for x in s_list]) if s_list else ""
            exams_std = info.get("standard_exams", [])[:3]
            treats_std = info.get("standard_treatments", [])[:3]

            frag_parts: List[str] = []
            if dep_tag and dep_tag != "未知科室":
                frag_parts.append("[%s]" % self._truncate_entry(dep_tag))
            frag_parts.append(str(disease))
            if s_text:
                frag_parts.append("症:%s" % s_text)
            if exams_std:
                frag_parts.append("★检:%s" % ",".join([str(x) for x in exams_std]))
            if treats_std:
                frag_parts.append("★治:%s" % ",".join([str(x) for x in treats_std]))
            fragment = "|".join(frag_parts)
            fragment = self._truncate_entry(fragment)
            if total + len(fragment) + 2 > self.max_prompt_chars:
                break
            parts.append(fragment)
            total += len(fragment) + 2

        result = "; ".join(parts)
        if len(result) > self.max_prompt_chars:
            result = result[: self.max_prompt_chars - 3] + "..."
        return result

    def get_error_patterns(self, limit: Optional[int] = None) -> str:
        """Return top error lessons as concise string, ≤max_prompt_chars total."""
        effective_limit = int(limit if limit is not None else self.max_error_patterns_limit)
        if effective_limit < 0:
            effective_limit = 0
        lessons = self.error_lessons[:effective_limit]
        parts: List[str] = []
        total = 0
        for lesson in lessons:
            reason = lesson.get("reason", "")
            amount = lesson.get("amount", "")
            fragment = "%s(-%s)" % (reason, amount) if amount else reason
            fragment = self._truncate_entry(fragment)
            if total + len(fragment) + 2 > self.max_prompt_chars:
                break
            parts.append(fragment)
            total += len(fragment) + 2

        result = "; ".join(parts)
        if len(result) > self.max_prompt_chars:
            result = result[: self.max_prompt_chars - 3] + "..."
        return result


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

