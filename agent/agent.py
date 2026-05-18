"""A readable baseline doctor agent.

The SDK handles train/test orchestration and service calls. This file shows a
clear reference flow: decide action -> ask/order exam/diagnose -> train reflection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hospital_agent_sdk import AgentBuilder, BasicAgent, load_config
from .memory import build_memory
from .prompt import (
    DEPARTMENT_PROMPT,
    DISEASE_AND_TREATMENT_PROMPT,
    DOCTOR_SYSTEM_PROMPT,
    EVALUATION_REFLECTION_PROMPT,
    EXAM_CATEGORY_PROMPT,
    EXAM_ITEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    NEXT_ACTION_PROMPT,
    format_prompt,
)


BASE_DIR = Path(__file__).resolve().parents[1]
REF_DATA_DIR = BASE_DIR / "data" / "ref_data"


class MyDoctorAgent(BasicAgent):
    """Baseline doctor agent with explicit, easy-to-read steps."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, memory: Any = None):
        super().__init__(config=config, memory=memory)

        # 标准目录来自 data/ref_data。所有类别、科室、检查、疾病都尽量从这些标准名称中选择。
        self.examination_catalog = load_examination_catalog()
        self.disease_catalog = load_disease_catalog()
        self.exam_categories = list(self.examination_catalog.keys())
        self.departments = list(self.disease_catalog.keys())
        self.exam_category_map = build_name_map(self.exam_categories)
        self.department_map = build_name_map(self.departments)

    async def train(self, patient_id: str) -> Dict[str, Any]:
        return await self._run_agent(patient_id=patient_id, mode="train")

    async def test(self, patient_id: str) -> Dict[str, Any]:
        return await self._run_agent(patient_id=patient_id, mode="test")

    async def _run_agent(self, patient_id: str, mode: str) -> Dict[str, Any]:
        memory_notes = self.memory.load_notes() if self.memory and hasattr(self.memory, "load_notes") else []

        case_state: Dict[str, Any] = {
            "patient_id": patient_id,
            "mode": mode,
            "memory_notes": memory_notes,
            "chat_history": [],
            "ordered_examinations": [],
            "invalid_examinations": [],
            "examination_results": {},
            "decision_trace": [],
            "exam_decision_trace": [],
        }

        # 步骤 1：循环判断下一步动作。这里只决定“问诊 / 开检查 / 诊断”。
        while True:
            chat_history = case_state.get("chat_history", [])
            examinations = self._examination_context(case_state)
            action_prompt = format_prompt(
                NEXT_ACTION_PROMPT,
                {
                    "memory_notes": memory_notes,
                    "chat_history": chat_history,
                    "examinations": examinations,
                },
            )
            if not case_state["chat_history"]:
                fallback_action = {
                    "action": "ask_patient",
                    "question": "请您按时间顺序描述这次最主要的不适、何时开始、如何变化，以及伴随症状。",
                    "reason": "需要先了解主诉和现病史。",
                }
            else:
                fallback_action = {
                    "action": "final_diagnosis",
                    "question": "",
                    "reason": "已有信息足以进入最终诊疗，或继续收集信息的收益有限。",
                }

            decision = await self._call_llm(
                prompt=action_prompt,
                default=fallback_action,
                prompt_name="next_action",
                patient_id=patient_id,
            )
            action = str(decision.get("action", "")).strip().lower()
            if action not in {"ask_patient", "order_examination", "final_diagnosis"}:
                action = fallback_action["action"]

            question = clean_text(decision.get("question"))
            if action == "ask_patient" and not question:
                question = "请您描述这次最主要的不适、开始时间和伴随症状。"

            decision = {
                "action": action,
                "question": question if action == "ask_patient" else "",
                "reason": clean_text(decision.get("reason")),
            }
            case_state["decision_trace"].append(decision)

            # 步骤 2A：如果继续问诊，就把本轮问题发给患者，并保存问答历史。
            if action == "ask_patient":
                answer = await self.actions.ask_patient(
                    patient_id,
                    {
                        "question": decision["question"],
                        "chat_history": case_state["chat_history"],
                    },
                )
                case_state["chat_history"].extend(
                    [
                        {"from": "doctor", "text": decision["question"]},
                        {"from": "patient", "text": answer},
                    ]
                )
                continue

            # 步骤 2B：如果开检查，先选检查类别，再从该类别里选具体标准检查名称。
            if action == "order_examination":
                chat_history = case_state.get("chat_history", [])
                examinations = self._examination_context(case_state)

                # 检查步骤 1：先让模型从所有标准检查类别中选一个类别。
                category_prompt = format_prompt(
                    EXAM_CATEGORY_PROMPT,
                    {
                        "memory_notes": memory_notes,
                        "chat_history": chat_history,
                        "examinations": examinations,
                        "exam_categories": self.exam_categories,
                    },
                )
                category_decision = await self._call_llm(
                    prompt=category_prompt,
                    default={"category": self.exam_categories[0] if self.exam_categories else "", "reason": ""},
                    prompt_name="exam_category",
                    patient_id=patient_id,
                )
                category = match_standard_name(category_decision.get("category"), self.exam_category_map)
                if not category:
                    category = self.exam_categories[0] if self.exam_categories else ""
                    for candidate, names in self.examination_catalog.items():
                        if any(name not in case_state["ordered_examinations"] for name in names):
                            category = candidate
                            break

                # 检查步骤 2：只给该类别下的检查名称，让模型选择具体要开的检查。
                item_prompt = format_prompt(
                    EXAM_ITEM_PROMPT,
                    {
                        "memory_notes": memory_notes,
                        "chat_history": chat_history,
                        "examinations": examinations,
                        "category": category,
                        "exam_items": self.examination_catalog.get(category, []),
                    },
                )
                item_decision = await self._call_llm(
                    prompt=item_prompt,
                    default={"examinations": [], "reason": category_decision.get("reason", "")},
                    prompt_name="exam_item",
                    patient_id=patient_id,
                )
                allowed_exam_map = build_name_map(self.examination_catalog.get(category, []))
                already_ordered = set(as_text_list(case_state["ordered_examinations"]))
                examinations = []
                for item in as_text_list(item_decision.get("examinations")):
                    standard_name = match_standard_name(item, allowed_exam_map)
                    if not standard_name or standard_name in already_ordered or standard_name in examinations:
                        continue
                    examinations.append(standard_name)
                exam_plan = {
                    "category": category,
                    "examinations": examinations,
                    "reason": clean_text(item_decision.get("reason") or category_decision.get("reason")),
                }
                case_state["exam_decision_trace"].append(exam_plan)
                if not examinations:
                    break

                exam_response = await self.actions.order_examination(
                    patient_id,
                    examinations,
                    reason=exam_plan.get("reason", ""),
                )
                case_state["ordered_examinations"].extend(
                    as_text_list(exam_response.get("normalized_items"))
                )
                case_state["invalid_examinations"].extend(
                    as_text_list(exam_response.get("invalid_items"))
                )
                results = exam_response.get("results") or {}
                if isinstance(results, dict):
                    case_state["examination_results"].update(results)
                continue

            break

        # 步骤 3：进入诊断。先选科室，再从该科室疾病列表里选一个疾病并生成治疗方案。
        chat_history = case_state.get("chat_history", [])
        examinations = self._examination_context(case_state)

        # 诊断步骤 1：先从标准科室中选择一个最相关科室。
        department_prompt = format_prompt(
            DEPARTMENT_PROMPT,
            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
                "departments": self.departments,
            },
        )
        department_decision = await self._call_llm(
            prompt=department_prompt,
            default={"department": self.departments[0] if self.departments else "", "reason": ""},
            prompt_name="department",
            patient_id=patient_id,
        )
        department = match_standard_name(department_decision.get("department"), self.department_map)
        if not department:
            department = self.departments[0] if self.departments else ""

        # 诊断步骤 2：只给该科室下的标准疾病名称，让模型选一个疾病并给出治疗方案。
        disease_prompt = format_prompt(
            DISEASE_AND_TREATMENT_PROMPT,
            {
                "memory_notes": memory_notes,
                "chat_history": chat_history,
                "examinations": examinations,
                "department": department,
                "diseases": self.disease_catalog.get(department, []),
            },
        )
        diseases = self.disease_catalog.get(department, [])
        default_diagnosis = diseases[0] if diseases else "未明确诊断"
        final_plan = await self._call_llm(
            prompt=disease_prompt,
            default={
                "diagnosis": default_diagnosis,
                "treatment_plan": "当前信息不足以制定特异性治疗方案；建议补充病史、体格检查和必要辅助检查后再决策。",
                "reasoning": "LLM 未返回可解析的最终方案，使用保守兜底结果。",
            },
            prompt_name="disease_and_treatment",
            patient_id=patient_id,
        )
        diagnosis = match_standard_name(
            final_plan.get("diagnosis"),
            build_name_map(self.disease_catalog.get(department, [])),
        ) or default_diagnosis
        treatment_plan = clean_text(final_plan.get("treatment_plan"))
        if not treatment_plan:
            treatment_plan = "当前信息不足以制定特异性治疗方案；建议补充关键病史、体格检查和必要辅助检查后再决策。"
        reasoning = clean_text(final_plan.get("reasoning"))
        if not reasoning:
            reasoning = clean_text(department_decision.get("reason")) or "基于问诊和检查结果形成诊疗方案。"

        final_plan = {
            "department": department,
            "diagnosis": diagnosis,
            "treatment_plan": treatment_plan,
            "reasoning": reasoning,
        }
        case_state["final_plan"] = final_plan

        final_result = await self.actions.prescribe_treatment(
            patient_id=patient_id,
            diagnosis=[diagnosis],
            treatment_plan=treatment_plan,
            reasoning=reasoning,
        )

        # 步骤 4：训练模式下先评估，再基于评估做一次反思，最后把反思写入 memory。
        evaluation_report = None
        evaluation_reflection = None
        if mode == "train":
            evaluation_report = await self.actions.evaluation(
                patient_id=patient_id,
                final_result=final_result,
            )
            case_state["evaluation_report"] = evaluation_report

            reflection_prompt = format_prompt(
                EVALUATION_REFLECTION_PROMPT,
                {
                    "chat_history": case_state.get("chat_history", []),
                    "evaluation_details": self._evaluation_details(evaluation_report or {}),
                },
            )
            evaluation_reflection = await self._call_llm(
                prompt=reflection_prompt,
                default={"reflection": {"profile": "", "future_strategy": ""}},
                prompt_name="evaluation_reflection",
                patient_id=patient_id,
            )
            case_state["evaluation_reflection"] = evaluation_reflection

        if mode == "train" and self.memory:
            self.memory.append_case_reflection(
                patient_id=patient_id,
                evaluation_reflection=evaluation_reflection,
            )

        return final_result

    async def _call_llm(
        self,
        *,
        prompt: str,
        default: Dict[str, Any],
        prompt_name: str = "",
        patient_id: str = "",
    ) -> Dict[str, Any]:
        response = await self.llm.call(prompt, system_prompt=DOCTOR_SYSTEM_PROMPT, temperature=0.2)
        parsed = parse_json_object(response)
        if parsed is not None:
            self._write_prompt_log(
                prompt_name=prompt_name,
                patient_id=patient_id,
                system_prompt=DOCTOR_SYSTEM_PROMPT,
                user_prompt=prompt,
                response=response,
            )
            return parsed

        # LLM 偶尔会输出不完整 JSON；这里做一次轻量修复，仍失败就使用兜底结果。
        repair_prompt = "请修复以下内容为合法 JSON 对象：\n\n%s" % response
        repaired = await self.llm.call(repair_prompt, system_prompt=JSON_REPAIR_SYSTEM_PROMPT, temperature=0)
        parsed = parse_json_object(repaired)
        result = parsed if parsed is not None else dict(default)
        self._write_prompt_log(
            prompt_name=prompt_name,
            patient_id=patient_id,
            system_prompt=DOCTOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            response=response,
        )
        return result

    def _write_prompt_log(
        self,
        *,
        prompt_name: str,
        patient_id: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
    ) -> None:
        if self.logger is None:
            return
        output_dir = getattr(self.logger, "output_dir", None)
        if output_dir is None:
            return
        prompt_dir = Path(output_dir) / "llm_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        filename = "%s_%s.txt" % (prompt_name, patient_id)
        path = prompt_dir / filename
        content = "\n".join(
            [
                "timestamp: %s" % datetime.now(timezone.utc).astimezone().isoformat(),
                "prompt_name: %s" % prompt_name,
                "patient_id: %s" % patient_id,
                "",
                "system_prompt:",
                system_prompt,
                "",
                "user_prompt:",
                user_prompt,
                "",
                "response:",
                response,
                "",
                "=" * 80,
                "",
            ]
        )
        with path.open("a", encoding="utf-8") as file:
            file.write(content)

    def _examination_context(self, case_state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ordered_examinations": unique_preserve_order(
                case_state.get("ordered_examinations", [])
            ),
            "examination_results": case_state.get("examination_results", {}),
        }

    def _evaluation_details(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "diagnosisDetail": evaluation_report.get("diagnosisDetail", {}),
            "examinationDetail": evaluation_report.get("examinationDetail", {}),
            "treatmentDetail": evaluation_report.get("treatmentDetail", {}),
        }


def build_name_map(names: Iterable[str]) -> Dict[str, str]:
    result = {}
    for name in names:
        normalized = normalize_name(name)
        if normalized and normalized not in result:
            result[normalized] = name
    return result


def load_examination_catalog() -> Dict[str, List[str]]:
    data = json.loads((REF_DATA_DIR / "examinations_catalog.json").read_text(encoding="utf-8"))
    catalog: Dict[str, List[str]] = {}
    for category, items in data.get("examinations", {}).items():
        names = []
        for item in items if isinstance(items, list) else []:
            name = item.get("name") if isinstance(item, dict) else item
            if str(name or "").strip():
                names.append(str(name).strip())
        if names:
            catalog[str(category)] = names
    return catalog


def load_disease_catalog() -> Dict[str, List[str]]:
    data = json.loads((REF_DATA_DIR / "diseases_catalog.json").read_text(encoding="utf-8"))
    catalog: Dict[str, List[str]] = {}
    for department, names in data.get("diseases", {}).items():
        clean_names = [str(name).strip() for name in names if str(name).strip()]
        if clean_names:
            catalog[str(department)] = clean_names
    return catalog


def match_standard_name(value: Any, name_map: Dict[str, str]) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return name_map.get(normalize_name(text), "")


def as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        value = value.values()
    if not isinstance(value, Iterable):
        return [str(value).strip()] if str(value).strip() else []
    items = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in as_text_list(values):
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def parse_json_object(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = text.strip("`").strip()

    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_name(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[\s\-_/，,。.;；:：、（）()\[\]【】]+", "", text)


def clean_text(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    config = load_config("config.yaml")
    memory = build_memory(config)
    agent = MyDoctorAgent(config=config, memory=memory)
    AgentBuilder(agent).start()
