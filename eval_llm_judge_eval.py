import json
import os
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from local_llm import load_system_prompt, Copilot
import time


ALLOWED_FAIL_TYPES = {
    "STATE_MISMATCH",
    "WRONG_ACTION_TARGET",
    "MISSING_REQUIRED_ACTION",
    "EXECUTION_FAILURE_HANDLING",
    "TRAJECTORY_VIOLATION",
    "MISSING_FINAL_RESPONSE",
}

REQUIRED_JUDGE_KEYS = {
    "verdict",
    "fail_types",
    "reason",
    "matched_required_behaviors",
    "violated_forbidden_behaviors",
    "trajectory_assessment",
    "response_assessment",
}


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_optional_json(tool_list: Optional[Union[str, List[Any], Dict[str, Any]]]) -> Any:
    if tool_list is None:
        return []

    if isinstance(tool_list, (list, dict)):
        return tool_list

    if isinstance(tool_list, str):
        if os.path.isfile(tool_list):
            return load_json_file(tool_list)
        return tool_list

    return []


def build_user_prompt(
    input_prompt: str,
    llm_judge_eval: Dict[str, Any],
    tool_list: Any,
    notes_after: Any,
    schedule_after: Any,
    agent_memory: str,
) -> str:
    tool_list_text = tool_list if isinstance(tool_list, str) else json.dumps(tool_list, ensure_ascii=False, indent=2)
    llm_judge_eval_text = json.dumps(llm_judge_eval, ensure_ascii=False, indent=2)
    notes_after_text = json.dumps(notes_after, ensure_ascii=False, indent=2)
    schedule_after_text = json.dumps(schedule_after, ensure_ascii=False, indent=2)

    return f"""Evaluate the following smart home agent test case.

[INPUT PROMPT]
{input_prompt}

[LLM_JUDGE_EVAL]
{llm_judge_eval_text}

[TOOL LIST]
{tool_list_text}

[TOOL/APPLIANCE PATTERN]
Tool is calling by:
<tool_call>{{"name":"<tool_name>", "arguments":{{...}}}}</tool_call>
Appliance is executin by:
<appliance>...json_config...</appliance>
Final answer is returned by:
<final_answer>...your final answer for the user...</final_answer>

[NOTES AFTER]
{notes_after_text}

[SCHEDULE TRIGGERS AFTER]
{schedule_after_text}

[AGENT MEMORY / TRAJECTORY]
{agent_memory}

Now judge whether the agent should PASS or FAIL based only on the provided evidence and the llm_judge_eval rules.
Remember:
- reference_flow is soft unless strict_order_required is true
- alternative valid flows are allowed
- unsupported claims must fail
- if a required behavior is not evidenced, treat it as missing
"""


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Judge output is not a JSON object.")
        return parsed
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in judge output.")

        parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Extracted judge output is not a JSON object.")
        return parsed


def validate_judge_result(result: Dict[str, Any]) -> Dict[str, Any]:
    missing_keys = REQUIRED_JUDGE_KEYS - set(result.keys())
    if missing_keys:
        raise ValueError(f"Judge output missing keys: {sorted(missing_keys)}")

    verdict = result["verdict"]
    fail_types = result["fail_types"]
    reason = result["reason"]
    matched_required_behaviors = result["matched_required_behaviors"]
    violated_forbidden_behaviors = result["violated_forbidden_behaviors"]
    trajectory_assessment = result["trajectory_assessment"]
    response_assessment = result["response_assessment"]

    if verdict not in {"PASS", "FAIL"}:
        raise ValueError("Judge output 'verdict' must be either 'PASS' or 'FAIL'.")

    if not isinstance(fail_types, list) or not all(isinstance(x, str) for x in fail_types):
        raise ValueError("Judge output 'fail_types' must be a list of strings.")

    invalid_fail_types = [x for x in fail_types if x not in ALLOWED_FAIL_TYPES]
    if invalid_fail_types:
        raise ValueError(f"Judge output has invalid fail_types: {invalid_fail_types}")

    if verdict == "PASS" and fail_types:
        raise ValueError("Judge output for PASS must have empty fail_types.")

    if verdict == "FAIL" and not fail_types:
        raise ValueError("Judge output for FAIL must have at least one fail_type.")

    if not isinstance(reason, str):
        raise ValueError("Judge output 'reason' must be a string.")

    if not isinstance(matched_required_behaviors, list) or not all(isinstance(x, str) for x in matched_required_behaviors):
        raise ValueError("Judge output 'matched_required_behaviors' must be a list of strings.")

    if not isinstance(violated_forbidden_behaviors, list) or not all(isinstance(x, str) for x in violated_forbidden_behaviors):
        raise ValueError("Judge output 'violated_forbidden_behaviors' must be a list of strings.")

    if not isinstance(trajectory_assessment, str):
        raise ValueError("Judge output 'trajectory_assessment' must be a string.")

    if not isinstance(response_assessment, str):
        raise ValueError("Judge output 'response_assessment' must be a string.")

    return {
        "verdict": verdict,
        "fail_types": fail_types,
        "reason": reason.strip(),
        "matched_required_behaviors": matched_required_behaviors,
        "violated_forbidden_behaviors": violated_forbidden_behaviors,
        "trajectory_assessment": trajectory_assessment.strip(),
        "response_assessment": response_assessment.strip(),
    }


def infer_valid_judgment(
    llm: Copilot,
    system_prompt: str,
    user_prompt: str,
    case_id: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    last_raw_response = ""

    for _ in range(max_retries):
        raw_response = llm.infer(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        last_raw_response = raw_response

        try:
            parsed = extract_json_object(raw_response)
            return validate_judge_result(parsed)
        except Exception as exc:
            last_error = exc
            time.sleep(5)

    return {
        "verdict": "FAIL",
        "fail_types": ["EXECUTION_FAILURE_HANDLING"],
        "reason": f"Judge model returned invalid JSON after {max_retries} attempts for case {case_id}: {last_error}",
        "matched_required_behaviors": [],
        "violated_forbidden_behaviors": [],
        "trajectory_assessment": "Unable to assess trajectory reliably because judge output format was invalid.",
        "response_assessment": f"Raw judge output could not be validated as required JSON. Last raw output: {last_raw_response[:500]}",
    }


def run_llm_judge_eval(
    benchmark_path: str,
    agent_output_path: str,
    system_prompt_path: str = "./eval/llm_judge_sysprompt.txt",
    tool_list: Optional[Union[str, List[Any], Dict[str, Any]]] = None,
    model: str = "gemini-3-flash-preview:cloud",
    host: str = "http://localhost:11434",
) -> Dict[str, Any]:
    benchmark_data = load_json_file(benchmark_path)
    agent_output_data = load_json_file(agent_output_path)
    system_prompt = load_system_prompt(system_prompt_path)
    tool_list_data = load_optional_json(tool_list)

    benchmark_by_id = {item["id"]: item for item in benchmark_data}
    output_by_id = {item["id"]: item for item in agent_output_data}

    benchmark_ids = [item["id"] for item in benchmark_data]
    benchmark_id_set = set(benchmark_by_id.keys())
    output_id_set = set(output_by_id.keys())

    missing_in_output = sorted(benchmark_id_set - output_id_set)
    missing_in_benchmark = sorted(output_id_set - benchmark_id_set)

    llm = Copilot(host=host, model=model)
    results: List[Dict[str, Any]] = []

    for case_id in tqdm(benchmark_ids, desc="LLM Judge Evaluating", unit="case"):
        if case_id not in output_by_id:
            continue

        benchmark_case = benchmark_by_id[case_id]
        output_case = output_by_id[case_id]

        user_prompt = build_user_prompt(
            input_prompt=benchmark_case.get("prompt", ""),
            llm_judge_eval=benchmark_case.get("llm_judge_eval", {}),
            tool_list=tool_list_data,
            notes_after=output_case.get("notes_after", {}),
            schedule_after=output_case.get("schedule_trigger_after", {}),
            agent_memory=output_case.get("inference_ouput", ""),
        )

        judgment = infer_valid_judgment(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            case_id=case_id,
        )

        results.append({
            "id": case_id,
            **judgment,
        })

    pass_count = sum(1 for item in results if item["verdict"] == "PASS")
    fail_count = sum(1 for item in results if item["verdict"] == "FAIL")

    report = {
        "summary": {
            "total_cases_in_benchmark": len(benchmark_data),
            "total_cases_evaluated": len(results),
            "pass": pass_count,
            "fail": fail_count,
            "missing_in_benchmark": missing_in_benchmark,
            "missing_in_output": missing_in_output,
        },
        "results": results,
    }

    return report


def save_report(report: Dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    benchmark_path = "./eval/eval_dataset_full.json"
    agent_output_path = "./eval/agent_output_gemini_0shot.json"
    system_prompt_path = "./eval/llm_judge_sysprompt.txt"
    output_path = "./eval/llm_judge_report_gemini_0shot_3.json"
    tool_list = "./tool_list.json"

    report = run_llm_judge_eval(
        benchmark_path=benchmark_path,
        agent_output_path=agent_output_path,
        system_prompt_path=system_prompt_path,
        tool_list=tool_list,
    )
    save_report(report, output_path)