# Smart Home Agent Evaluation Benchmark

This repository contains a structured evaluation pipeline for testing a smart-home agentic assistant. The benchmark is stored in `./eval/eval_dataset_full.json` and is expected to contain about 900 test cases. Each test case defines the user prompt, setup conditions, deterministic checks, and soft LLM-as-a-judge criteria.

The evaluation pipeline has three main stages:

1. **Collect agent outputs** from the benchmark.
2. **Run deterministic evaluation** using fixed checks over tool calls, appliance commands, final states, notes, schedules, and final responses.
3. **Run LLM judge evaluation** for trajectory-level and response-level judgment.


## Benchmark Dataset Format

Each item in `eval_dataset_full.json` represents one test case. The dataset should keep the existing fields:

- `id`
- `task_type`
- `prompt`

Then each test case must contain exactly these three structured sections:

- `preconditions`
- `deterministic_eval`
- `llm_judge_eval`

### Important Note Rule

If a prompt contains a `[TODAY NOTES]` section, remove that part from `prompt` and move the note content into:

```json
"preconditions": {
  "notes_setup": [
    {
      "date": "today",
      "note_text": "note content here"
    }
  ],
  "state_setup": []
}
```

The user prompt should only contain the actual user request, not injected note setup text.

---

## Full Test Case Schema

```json
{
  "id": "tc_001",
  "task_type": "single",
  "prompt": "Turn on the bedroom light.",
  "preconditions": {
    "notes_setup": [],
    "state_setup": []
  },
  "deterministic_eval": {
    "must_check_note": false,
    "must_check_schedule": false,
    "expected_state": [
      {
        "espID": 1,
        "device_type": "actuator",
        "device_name": "led1",
        "value": true
      }
    ],
    "required_tool_calls": [],
    "abandon_tool_calls": [],
    "required_appliances": [
      {
        "espID": 1,
        "device_type": "actuator",
        "device_name": "led1",
        "action": "set"
      }
    ],
    "abandon_appliance": [],
    "ordering_constraints": []
  },
  "llm_judge_eval": {
    "judge_focus": [
      "intent_understanding",
      "reasonableness_of_flow",
      "grounded_response",
      "handling_of_ambiguity_or_infeasibility",
      "consistency_between_actions_and_reply"
    ],
    "required_behaviors": [
      "The agent should understand that the user wants the bedroom light turned on."
    ],
    "acceptable_behaviors": [
      "The agent may directly execute the appliance command and then confirm the result."
    ],
    "forbidden_behaviors": [
      "The agent must not claim the light was turned on if no successful action is shown."
    ],
    "response_requirements": {
      "must_confirm_result": true,
      "must_explain_if_not_executed": false,
      "must_be_consistent_with_environment_change": true,
      "must_not_overclaim": true
    },
    "trajectory_judgment": {
      "required_step_patterns": [
        {
          "actor": "Agent",
          "action": "respond",
          "tool": ""
        }
      ],
      "preferred_step_patterns": [],
      "forbidden_step_patterns": [],
      "reference_flow": [
        {
          "actor": "User",
          "action": "request",
          "tool": "",
          "args": {}
        },
        {
          "actor": "Controller",
          "action": "return_success",
          "tool": "appliance",
          "args": {
            "espID": 1,
            "device_type": "actuator",
            "device_name": "led1",
            "action": "set",
            "value": true
          }
        },
        {
          "actor": "Agent",
          "action": "respond",
          "tool": "",
          "args": {}
        }
      ]
    },
    "judge_note": "Checks whether the agent turns on the requested light and gives a grounded confirmation."
  }
}
```

---

## Section 1: `preconditions`

`preconditions` defines the environment before the agent runs.

```json
{
  "preconditions": {
    "notes_setup": [
      {
        "date": "yyyy-mm-dd/today",
        "note_text": "string"
      }
    ],
    "state_setup": [
      {
        "espID": 0,
        "device_type": "actuator",
        "device_name": "string",
        "action": "set",
        "value": "any"
      }
    ]
  }
}
```

Rules:

- Use `notes_setup` only when the test case depends on an existing note.
- Use `date: "today"` for notes that should be attached to the current test date.
- Use `state_setup` only for explicit appliance initialization needed before inference.
- `state_setup` should use actuator setup only.
- `action` should remain `"set"`.
- Use empty arrays when no setup is needed.
- Use only valid house devices and valid values.

Example:

```json
"preconditions": {
  "notes_setup": [
    {
      "date": "today",
      "note_text": "Remember to water the plants in the evening."
    }
  ],
  "state_setup": [
    {
      "espID": 3,
      "device_type": "actuator",
      "device_name": "pump",
      "action": "set",
      "value": false
    }
  ]
}
```

---

## Section 2: `deterministic_eval`

`deterministic_eval` contains hard checks that can be verified by code.

```json
{
  "deterministic_eval": {
    "must_check_note": true,
    "must_check_schedule": false,
    "expected_state": [
      {
        "espID": 0,
        "device_type": "actuator/sensor",
        "device_name": "string",
        "value": "any"
      }
    ],
    "required_tool_calls": ["tool_name"],
    "abandon_tool_calls": ["tool_name"],
    "required_appliances": [
      {
        "espID": 0,
        "device_type": "actuator/sensor",
        "device_name": "string",
        "action": "get/set"
      }
    ],
    "abandon_appliance": [
      {
        "espID": 0,
        "device_type": "actuator/sensor",
        "device_name": "string",
        "action": "get/set"
      }
    ],
    "ordering_constraints": [
      ["tool_name_1", "tool_name_2", "final_answer"]
    ]
  }
}
```

Rules:

- `must_check_note` should be `true` only when the agent is expected to verify added notes.
- `must_check_schedule` should be `true` only when the agent is expected to verify created scheduled behavior in `schedule_trigger.json`.
- `expected_state` should contain only final physically verifiable states.
- `required_tool_calls` and `abandon_tool_calls` must use real tool names only.
- `required_appliances` and `abandon_appliance` describe expected appliance-level `get` or `set` behavior.
- `ordering_constraints` must be a list of ordered sequences. The earlier item must appear before the later item, but they do not need to be consecutive.
- Do not put natural-language evaluation here.
- Use empty lists when a check is not needed.

### Deterministic Action Tags

The deterministic evaluator extracts these tags from the agent output:

```text
<tool_call>{"name":"tool_name", "arguments":{...}}</tool_call>
<appliance>[{"espID":1, "device_type":"actuator", "device_name":"led1", "action":"set", "value":true}]</appliance>
<final_answer>Final response to the user.</final_answer>
```

The evaluator parses tool calls, appliance executions, final answers, and ordering sequences from these tags.

---

## Section 3: `llm_judge_eval`

`llm_judge_eval` is the soft evaluation schema used by the LLM judge.

```json
{
  "llm_judge_eval": {
    "judge_focus": [
      "intent_understanding",
      "reasonableness_of_flow",
      "grounded_response",
      "handling_of_ambiguity_or_infeasibility",
      "consistency_between_actions_and_reply"
    ],
    "required_behaviors": ["string"],
    "acceptable_behaviors": ["string"],
    "forbidden_behaviors": ["string"],
    "response_requirements": {
      "must_confirm_result": true,
      "must_explain_if_not_executed": false,
      "must_be_consistent_with_environment_change": true,
      "must_not_overclaim": true
    },
    "trajectory_judgment": {
      "required_step_patterns": [
        {
          "actor": "Agent",
          "action": "call_tool/respond/refuse/ask_clarification",
          "tool": "tool_name"
        }
      ],
      "preferred_step_patterns": [
        {
          "actor": "Agent",
          "action": "call_tool/respond/refuse/ask_clarification",
          "tool": "tool_name"
        }
      ],
      "forbidden_step_patterns": [
        {
          "actor": "Agent",
          "action": "call_tool/respond",
          "tool": "tool_name"
        }
      ],
      "reference_flow": [
        {
          "actor": "User/Agent/Controller",
          "action": "request/call_tool/return_success/respond/refuse/ask_clarification",
          "tool": "tool_name",
          "args": {}
        }
      ]
    },
    "judge_note": "short objective note"
  }
}
```

Rules:

- `judge_focus` tells the judge what to care about.
- `required_behaviors` are mandatory soft behaviors.
- `acceptable_behaviors` should allow reasonable alternative valid flows.
- `forbidden_behaviors` are soft failures not fully captured by deterministic checks.
- `response_requirements` controls final response quality.
- `required_step_patterns` should include only key mandatory patterns.
- `preferred_step_patterns` are good but not mandatory.
- `forbidden_step_patterns` are trajectory patterns that should fail the case.
- `reference_flow` is a soft reference, not the only valid path.
- `judge_note` should be short, objective, and directly tied to pass/fail behavior.

---

## Valid Failure Types

Both deterministic and LLM judge reports should use the same failure type taxonomy:

| Fail Type | Meaning |
|---|---|
| `STATE_MISMATCH` | Final observed state, note state, schedule state, or reported result does not match the expected result or agent claim. |
| `WRONG_ACTION_TARGET` | Agent used the wrong tool, appliance, room, device attribute, or unrelated target. |
| `MISSING_REQUIRED_ACTION` | Agent failed to perform a required tool call, appliance interaction, note check, schedule check, or required step. |
| `EXECUTION_FAILURE_HANDLING` | Tool or appliance execution failed, had no effect, lacked evidence of success, or was handled incorrectly. |
| `TRAJECTORY_VIOLATION` | Agent violated soft trajectory rules, strict order, required step patterns, or forbidden step patterns. |
| `MISSING_FINAL_RESPONSE` | Agent did not return a valid final response. |

---

## Evaluation Pipeline

### 1. Collect Agent Outputs

Run:

```bash
python ./eval/eval_agent_output_collecting.py
```

This script:

- Loads `./eval/eval_dataset_full.json`.
- Resets appliances to default values before each case.
- Clears `schedule_trigger.json` and `note_storage.json` before each case.
- Applies `preconditions.notes_setup` using `add_note`.
- Applies `preconditions.state_setup` using `send_command`.
- Builds the agent from `system_prompt_doc/role.txt` and `system_prompt_doc/instruction.txt`.
- Runs the agent on `case["prompt"]`.
- Saves the trajectory, execution time, final state checks, notes after inference, and schedule triggers after inference.

Default output example:

```text
./eval/agent_output_gemini_3fewshot.json
```

Each output item has this general structure:

```json
{
  "id": "tc_001",
  "inference_ouput": "...agent trajectory...",
  "execution_time": 12.34,
  "state_check": [
    {
      "espID": 1,
      "device_type": "actuator",
      "device_name": "led1",
      "expected": true,
      "actual": true
    }
  ],
  "notes_after": {},
  "schedule_trigger_after": {}
}
```

> Note: the current code uses the field name `inference_ouput`. Keep this spelling consistent unless all related scripts are updated.

---

### 2. Run Deterministic Evaluation

Run:

```bash
python ./eval/eval_deterministic_eval.py
```

This script compares benchmark expectations against collected agent outputs. It checks:

- note persistence when `must_check_note` is enabled
- expected final states
- required tool calls
- forbidden tool calls
- required appliance executions
- forbidden appliance executions
- ordering constraints
- final answer existence

Default output example:

```text
./eval/deterministic_report_gemini_3fewshot.json
```

Report format:

```json
{
  "summary": {
    "total_cases_in_benchmark": 900,
    "total_cases_evaluated": 900,
    "pass": 800,
    "fail": 100,
    "missing_in_benchmark": [],
    "missing_in_output": []
  },
  "results": [
    {
      "id": "tc_001",
      "result": "PASS",
      "fail_types": [],
      "fail_reasons": [],
      "details": {}
    }
  ]
}
```

---

### 3. Run LLM Judge Evaluation

Run:

```bash
python ./eval/eval_llm_judge_eval.py
```

This script sends each case to the judge model with:

- input prompt
- `llm_judge_eval`
- tool list
- tool/appliance/final-answer tag format
- notes after inference
- schedule triggers after inference
- agent memory / trajectory

Default output example:

```text
./eval/llm_judge_report_gemini_0shot_3.json
```

Report format:

```json
{
  "summary": {
    "total_cases_in_benchmark": 900,
    "total_cases_evaluated": 900,
    "pass": 780,
    "fail": 120,
    "missing_in_benchmark": [],
    "missing_in_output": []
  },
  "results": [
    {
      "id": "tc_001",
      "verdict": "PASS",
      "fail_types": [],
      "reason": "The agent completed the requested action and gave a grounded confirmation.",
      "matched_required_behaviors": [],
      "violated_forbidden_behaviors": [],
      "trajectory_assessment": "The trajectory is valid.",
      "response_assessment": "The final response is consistent with the observed action."
    }
  ]
}
```

---

## LLM Judge System Prompt

The judge system prompt is stored in:

```text
./eval/llm_judge_sysprompt.txt
```

The judge is instructed to:

- judge only from provided evidence
- not assume hidden actions
- allow reasonable alternative flows
- prefer failure when required evidence is missing
- return exactly one JSON object
- use only the allowed fail types

This makes the judge stricter than a normal chat model while still allowing flexible valid trajectories.

---

## Typical Run Order

```bash
# 1. Collect trajectories and environment results
python ./eval/eval_agent_output_collecting.py

# 2. Run hard deterministic checks
python ./eval/eval_deterministic_eval.py

# 3. Run soft LLM-as-a-judge checks
python ./eval/eval_llm_judge_eval.py
```

---
