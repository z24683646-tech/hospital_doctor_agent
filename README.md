# Hospital Agent Baseline Example

这是一个虚拟诊疗比赛的医生 Agent 示例项目，展示如何基于 `hospital-agent-sdk` 开发、训练、测试和部署自己的医生 Agent。

## 目录说明

```text
baseline_example/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # 需要修改：医生 Agent 主逻辑，包含 train/test 和诊疗策略
│   ├── prompt.py         # 需要修改：Prompt 模板和输出格式约束
│   └── memory.py         # 需要修改：训练反思、病例经验、检索记忆等
├── data/
│   ├── memory_data/
│   │   └── memory.md     # 可自定义：baseline 默认的本地文件记忆
│   └── ref_data/
│       ├── departments.json
│       ├── diseases_catalog.json
│       └── examinations_catalog.json
├── config.yaml           # 可配置：训练患者数量、输出目录、memory 路径等
├── train.py              # 可选修改：本地训练入口，默认可直接使用
├── test.py               # 可选修改：本地测试入口或自定义测试脚本
├── requirements.txt      # 必须维护：新增第三方库需要写在这里
├── Dockerfile            # 一般不需要修改：部署入口，默认启动 python3 -m agent.agent
└── README.md
```

参赛时需要重点修改 `agent/agent.py`、`agent/prompt.py`、`agent/memory.py`：`agent.py` 负责诊疗流程和 action 调用策略，`prompt.py` 负责模型输入和输出格式约束，`memory.py` 负责训练反思、病例经验、检索记忆等记忆逻辑。`config.yaml` 可以按需配置训练患者数量、随机种子、输出目录、记忆路径等参数。

注意：

- `data/memory_data/` 可以根据你选用的记忆存储形式自定义。baseline 默认使用本地 `memory.md`，你也可以改成 JSON、SQLite、向量数据库或其他形式。
- 记忆的存储形式不限定；如果使用数据库或对象存储等外部服务，需要使用可直接访问的外部存储服务。
- 部署环境不能依赖 `docker-compose` 拉起额外服务，因此不要把数据库、向量库等作为同一个提交中的 compose 依赖。
- `data/ref_data/` 是标准科室、疾病和检查名称，不要修改；提交结果中的诊断和检查名称必须使用这里提供的标准名称。
- `Dockerfile` 一般不需要修改，以免影响平台部署和评测，默认已经包含部署入口和启动命令。
- `train.py` 和 `test.py` 分别是训练和测试入口，可以按需修改；一般建议先直接使用默认入口，只有在需要自定义训练或测试流程时再调整。

## 如何开发医生 Agent

主要修改 `agent/agent.py` 中的 `MyDoctorAgent`。参赛者需要实现或调整 `train` 和 `test` 流程，并通过 `self.actions` 调用比赛能力：

- `ask_patient`：询问患者。
- `order_examination`：申请检查。
- `prescribe_treatment`：提交诊断和治疗方案。
- `evaluation`：训练阶段获取单个病例的评测结果。

如果要调整模型输入，可以修改 `agent/prompt.py`；如果要保存训练经验、病例总结或向量检索结果，可以修改或替换 `agent/memory.py`。

## 医生可用 Actions

所有 action 都通过 `self.actions.<action_name>(...)` 调用，通常在 `async def train(...)` 或 `async def test(...)` 中使用，需要 `await`。

### `ask_patient`

向患者提问，返回患者回复文本。适合收集主诉、现病史、既往史、用药史、过敏史、家族史等信息。

输入：

```python
answer = await self.actions.ask_patient(
    patient_id=patient_id,
    input_data={
        "question": "请描述这次最主要的不适、开始时间和伴随症状。",
        "chat_history": [
            {"from": "doctor", "text": "之前的问题"},
            {"from": "patient", "text": "之前的回答"},
        ],
    },
)
```

参数说明：

- `patient_id: str`：当前病例 ID，由 SDK 在 `train(patient_id)` / `test(patient_id)` 中传入。
- `input_data: dict`：发给患者服务的 JSON 对象。baseline 使用 `question` 和 `chat_history`，也可以加入自己的上下文字段。

输出：

```python
"患者回答文本"
```

返回值是 `str`。SDK 会自动记录问诊轮数，最终 `prescribe_treatment` 的返回结果中会包含 `conversation_rounds`。

### `order_examination`

申请检查，返回检查结果。检查名称应尽量使用 `data/ref_data/examinations_catalog.json` 中的标准名称。

输入：

```python
exam_response = await self.actions.order_examination(
    patient_id=patient_id,
    items=["血常规", "尿常规"],
    reason="鉴别感染、贫血和泌尿系统异常。",
)
```

参数说明：

- `patient_id: str`：当前病例 ID。
- `items: Iterable[str]`：要申请的检查名称列表。
- `reason: str`：申请检查的原因，可为空字符串。

输出：

```json
{
  "requested_items": ["血常规", "尿常规"],
  "normalized_items": ["血常规", "尿常规"],
  "invalid_items": [],
  "invalid_message": "",
  "results": {
    "血常规": {
      "item_name": "血常规",
      "result": "检查结果文本",
      "status": "completed"
    }
  }
}
```

字段说明：

- `requested_items`：本次请求的检查名称。
- `normalized_items`：服务端认可并完成的检查名称。
- `invalid_items`：无效或非标准检查名称。
- `invalid_message`：存在无效检查时的提示，通常表示需要使用标准检查名称。
- `results`：检查结果字典，key 是检查名称；每个结果包含 `item_name`、`result`、`status`。`status` 可能是 `completed` 或 `normal`。

### `prescribe_treatment`

提交最终诊断和治疗方案，并结束当前病例。测试阶段每个病例必须最终调用这个 action，返回值会写入 `final_results.jsonl` 并作为提交结果。

输入：

```python
final_result = await self.actions.prescribe_treatment(
    patient_id=patient_id,
    diagnosis=["肺炎"],
    treatment_plan="建议抗感染治疗，结合病情补液、退热、止咳，并监测生命体征和复查相关指标。",
    reasoning="结合问诊和检查结果，诊断与治疗方案匹配。",
)
```

参数说明：

- `patient_id: str`：当前病例 ID。
- `diagnosis: Any`：最终诊断，建议使用标准疾病名称列表。
- `treatment_plan: str`：治疗方案文本。
- `reasoning: str`：诊疗依据，可为空字符串。

输出：

```json
{
  "patient_id": "patient_xxx",
  "team_id": "your-team-id",
  "doctor_id": "your-team-id",
  "diagnosis": ["肺炎"],
  "treatment_plan": "治疗方案文本",
  "reasoning": "诊疗依据文本",
  "ordered_examinations": ["血常规", "胸部CT"],
  "conversation_rounds": 3,
  "finished": true
}
```

字段说明：

- `diagnosis`、`treatment_plan`、`reasoning`：你提交的最终诊疗结果。
- `ordered_examinations`：当前病例中已成功申请的检查，SDK 会去重并保持顺序。
- `conversation_rounds`：当前病例调用 `ask_patient` 的次数。
- `finished`：是否完成病例；调用该 action 后固定为 `true`。

### `evaluation`

训练阶段用于评估单个已完成病例，通常在 `prescribe_treatment` 之后调用，用于反思和改进策略。测试阶段的临时 token 不能成功调用该 action，部署测试逻辑中不要依赖它。

输入：

```python
report = await self.actions.evaluation(
    patient_id=patient_id,
    final_result=final_result,
)
```

参数说明：

- `patient_id: str`：当前病例 ID。
- `final_result: dict | None`：`prescribe_treatment` 的返回结果。如果不传，SDK 会尝试读取当前病例最近一次最终结果。

输出：

```json
{
  "patientId": "patient_xxx",
  "status": "evaluated",
  "overallScore": 0.85,
  "diagnosisAccuracy": 1.0,
  "examinationPrecision": 0.8,
  "treatmentMatchingScore": 0.75,
  "treatmentSafety": 1.0,
  "treatmentEffectivenessAlignment": 0.8,
  "treatmentPersonalization": 0.7,
  "diagnosisDetail": {
    "submitted": ["肺炎"],
    "expected": ["肺炎"],
    "matched": ["肺炎"],
    "accuracy": 1.0,
    "status": "correct"
  },
  "examinationDetail": {
    "ordered": ["血常规", "胸部CT"],
    "expected": ["血常规", "胸部CT"],
    "matched": ["血常规", "胸部CT"],
    "precision": 1.0,
    "coverage": 1.0
  },
  "treatmentDetail": {
    "submitted": "你提交的治疗方案",
    "reference": "参考治疗方案",
    "overallScore": 0.75,
    "safety": 1.0,
    "effectivenessAlignment": 0.8,
    "personalization": 0.7,
    "reasoning": "评测模型给出的解释"
  },
  "ground_truth": {}
}
```

评测报告字段可能随比赛服务调整而变化，训练代码应优先读取自己需要的字段，并对缺失字段做好兜底。

## Quick Start

安装依赖：

```bash
cd baseline_example
pip install -r requirements.txt
```

配置本地训练环境变量：

```bash
export SERVICE_BASE_URL=https://baconroot-hospital-service.ms.show # 不能修改！！ 比赛后端服务地址，用于训练和测试中获取患者回复、检查结果、提交评估等
export SERVICE_TRAIN_TOKEN=<your-train-service-token> # 训练阶段访问令牌，即登录时的密钥，用于访问本队伍训练数据和评测接口。
export MODEL_API_KEY=<your-model-api-key> # 大语言模型调用密钥，用于训练阶段调用模型服务。
export TEAM_ID=<your-team-id> # 队伍标识，用于记录训练结果和提交记录归属。
```

运行本地训练：

```bash
python train.py
```

训练患者可以在 `config.yaml` 中配置：

- `training.patient_count`：随机抽取的训练患者数量。
- `training.random_seed`：随机种子。
- `training.patient_ids`：指定患者 ID；如果填写，则优先使用指定列表。

## 启动测试服务

本地启动 Agent 测试服务：

```bash
python -m agent.agent
```

服务默认监听 `0.0.0.0:7860`，测试接口为 `POST /test`。本地调用示例：

```bash
curl -X POST http://127.0.0.1:7860/test \
  -H 'Content-Type: application/json' \
  -d '{"contestServiceToken":"<temporary-service-token>"}'
```

测试阶段通常只需要配置：

```bash
export SERVICE_BASE_URL=<contest-service-url>
export TEAM_ID=<your-team-id>
```

测试请求中的临时 token 会由比赛平台传入；部署环境不需要提供 `MODEL_API_KEY`。

## 部署

部署到 ModelScope Studio 创空间时，提交最新代码后，在空间页面点击上线，等待空间构建状态变为“运行中”即可。

部署后需要保证服务监听 `7860` 端口，并暴露 `POST /test` 接口。默认通过 `Dockerfile` 启动 `python3 -m agent.agent`，会监听 `0.0.0.0:7860`；一般不需要修改部署入口。提交前建议先在本地运行 `python train.py` 和 `python -m agent.agent`，确认训练流程和测试服务都能正常工作。
