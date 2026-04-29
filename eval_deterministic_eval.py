import json
import re
from typing import Any, Dict, List, Optional, Tuple


# Extract to list of tool calls, appliance executions and final response returned
def extract_agent_actions(inference_output: str):
    pattern = re.compile(
        r'(<tool_call>.*?</tool_call>|<appliance>.*?</appliance>|<final_answer>.*?</final_answer>)',
        re.DOTALL
    )
    matches = pattern.findall(inference_output)
    results = [m.strip() for m in matches]  # remove whitespace
    return results


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _strip_tag(raw: str, tag: str) -> str:
    return re.sub(rf'^<{tag}>|</{tag}>$', '', raw.strip(), flags=re.DOTALL).strip()


def parse_tool_call(raw: str) -> Optional[Dict[str, Any]]:
    content = _strip_tag(raw, "tool_call")
    data = _safe_json_loads(content)
    if isinstance(data, dict):
        return data

    # Fallback for slightly malformed json-like text
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', content)
    if not name_match:
        return None

    args_match = re.search(r'"arguments"\s*:\s*(\{.*\})', content, flags=re.DOTALL)
    args = {}
    if args_match:
        parsed_args = _safe_json_loads(args_match.group(1))
        if isinstance(parsed_args, dict):
            args = parsed_args

    return {"name": name_match.group(1), "arguments": args}


def parse_appliance(raw: str) -> List[Dict[str, Any]]:
    content = _strip_tag(raw, "appliance")
    data = _safe_json_loads(content)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def parse_final_answer(raw: str) -> str:
    return _strip_tag(raw, "final_answer")


def values_equal(expected: Any, actual: Any) -> bool:
    # Prevent Python treating True == 1 and False == 0 as equal
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected is actual
    return expected == actual


def tool_name_matches(required_tool_name: str, actual_tool_name: str) -> bool:
    return required_tool_name == actual_tool_name


def appliance_matches(rule: Dict[str, Any], actual: Dict[str, Any]) -> bool:
    keys_to_check = ["espID", "device_type", "device_name", "action"]
    for key in keys_to_check:
        if key in rule and actual.get(key) != rule.get(key):
            return False
    if "value" in rule and not values_equal(rule.get("value"), actual.get("value")):
        return False
    return True


def flatten_notes(notes_after: Any) -> List[str]:
    results: List[str] = []
    if not isinstance(notes_after, dict):
        return results

    for _, note_bucket in notes_after.items():
        if isinstance(note_bucket, dict):
            for _, note_text in note_bucket.items():
                if isinstance(note_text, str):
                    results.append(note_text)
    return results


def build_action_views(inference_output: str) -> Dict[str, Any]:
    raw_actions = extract_agent_actions(inference_output)

    parsed_tool_calls: List[Dict[str, Any]] = []
    parsed_appliances: List[Dict[str, Any]] = []
    final_answers: List[str] = []

    flat_sequence: List[str] = []

    for raw in raw_actions:
        if raw.startswith("<tool_call>"):
            tool_data = parse_tool_call(raw)
            if tool_data:
                parsed_tool_calls.append(tool_data)
                tool_name = tool_data.get("name")
                if isinstance(tool_name, str):
                    flat_sequence.append(tool_name)

        elif raw.startswith("<appliance>"):
            appliance_list = parse_appliance(raw)
            if appliance_list:
                flat_sequence.append("appliance")
                for item in appliance_list:
                    parsed_appliances.append(item)
                    flat_sequence.append(
                        f"appliance:{item.get('espID')}:{item.get('device_type')}:{item.get('device_name')}:{item.get('action')}"
                    )

        elif raw.startswith("<final_answer>"):
            final_text = parse_final_answer(raw)
            if "..." not in final_text.strip():
                final_answers.append(final_text)
                flat_sequence.append("final_answer")

    return {
        "raw_actions": raw_actions,
        "tool_calls": parsed_tool_calls,
        "appliances": parsed_appliances,
        "final_answers": final_answers,
        "flat_sequence": flat_sequence,
        "tool_names": [tc["name"] for tc in parsed_tool_calls if isinstance(tc.get("name"), str)],
    }


def is_subsequence(pattern: List[str], sequence: List[str]) -> bool:
    if not pattern:
        return True

    i = 0
    for step in sequence:
        if step == pattern[i]:
            i += 1
            if i == len(pattern):
                return True
    return False


def check_ordering_constraints(ordering_constraints: List[List[str]], flat_sequence: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    For each pattern:
    - If 0 or 1 elements from the pattern exist in the actual sequence, skip it.
    - Otherwise, require the existing elements to appear in the same order.
    """
    violations: List[Dict[str, Any]] = []

    for pattern in ordering_constraints or []:
        if not pattern:
            continue

        existing_steps = [step for step in pattern if step in flat_sequence]

        # Pattern may contain tool that does not exist in actual list -> do not fail directly
        if len(existing_steps) <= 1:
            continue

        if not is_subsequence(existing_steps, flat_sequence):
            violations.append({
                "pattern": pattern,
                "relevant_subpattern": existing_steps,
                "actual_sequence": flat_sequence,
            })

    return len(violations) == 0, violations


def evaluate_single_case(
    benchmark_case: Dict[str, Any],
    output_case: Dict[str, Any]
) -> Dict[str, Any]:
    det = benchmark_case.get("deterministic_eval", {}) or {}
    inference_output = output_case.get("inference_ouput", "") or ""
    parsed = build_action_views(inference_output)

    fail_types: List[str] = []
    fail_reasons: List[Dict[str, Any]] = []

    # 1) must_check_note
    note_check_detail = {
        "must_check_note": bool(det.get("must_check_note", False)),
        "attempted_add_note_texts": [],
        "missing_note_texts_in_notes_after": [],
    }

    if det.get("must_check_note", False):
        attempted_note_texts: List[str] = []
        for tool_call in parsed["tool_calls"]:
            if tool_call.get("name") == "add_note":
                note_text = (tool_call.get("arguments") or {}).get("note_text")
                if isinstance(note_text, str) and note_text.strip():
                    attempted_note_texts.append(note_text.strip())

        note_check_detail["attempted_add_note_texts"] = attempted_note_texts
        notes_after_texts = flatten_notes(output_case.get("notes_after", {}))

        missing_note_texts = [t for t in attempted_note_texts if t not in notes_after_texts]
        note_check_detail["missing_note_texts_in_notes_after"] = missing_note_texts

        # no add_note tool call -> NOT FAIL
        if missing_note_texts:
            fail_types.extend(["STATE_MISMATCH", "EXECUTION_FAILURE_HANDLING"])
            fail_reasons.append({
                "check": "must_check_note",
                "detail": note_check_detail,
            })

    # 2) expected_state
    expected_state_detail = {
        "expected_state": det.get("expected_state", []),
        "checked": [],
        "missing_state_entries": [],
    }

    actual_state_checks = output_case.get("state_check", []) or []

    for expected in det.get("expected_state", []) or []:
        matched = None
        for actual_item in actual_state_checks:
            if (
                actual_item.get("espID") == expected.get("espID")
                and actual_item.get("device_type") == expected.get("device_type")
                and actual_item.get("device_name") == expected.get("device_name")
            ):
                matched = actual_item
                break

        if matched is None:
            expected_state_detail["missing_state_entries"].append(expected)
            continue

        actual_value = matched.get("actual")
        expected_value = expected.get("value")
        is_ok = values_equal(expected_value, actual_value)

        expected_state_detail["checked"].append({
            "target": {
                "espID": expected.get("espID"),
                "device_type": expected.get("device_type"),
                "device_name": expected.get("device_name"),
            },
            "expected_value": expected_value,
            "actual_value": actual_value,
            "pass": is_ok,
        })

    if expected_state_detail["missing_state_entries"] or any(not x["pass"] for x in expected_state_detail["checked"]):
        fail_types.append("STATE_MISMATCH")
        fail_reasons.append({
            "check": "expected_state",
            "detail": expected_state_detail,
        })

    # 3) required_tool_calls
    required_tools = det.get("required_tool_calls", []) or []
    missing_required_tools = [
        tool_name for tool_name in required_tools
        if not any(tool_name_matches(tool_name, actual) for actual in parsed["tool_names"])
    ]
    required_tool_detail = {
        "required_tool_calls": required_tools,
        "observed_tool_calls": parsed["tool_names"],
        "missing_required_tool_calls": missing_required_tools,
    }
    if missing_required_tools:
        fail_types.append("MISSING_REQUIRED_ACTION")
        fail_reasons.append({
            "check": "required_tool_calls",
            "detail": required_tool_detail,
        })

    # 4) abandon_tool_calls
    abandon_tools = det.get("abandon_tool_calls", []) or []
    found_abandon_tools = [
        tool_name for tool_name in abandon_tools
        if any(tool_name_matches(tool_name, actual) for actual in parsed["tool_names"])
    ]
    abandon_tool_detail = {
        "abandon_tool_calls": abandon_tools,
        "observed_tool_calls": parsed["tool_names"],
        "found_abandon_tool_calls": found_abandon_tools,
    }
    if found_abandon_tools:
        fail_types.append("WRONG_ACTION_TARGET")
        fail_reasons.append({
            "check": "abandon_tool_calls",
            "detail": abandon_tool_detail,
        })

    # 5) required_appliances
    required_appliances = det.get("required_appliances", []) or []
    missing_required_appliances = [
        appliance_rule for appliance_rule in required_appliances
        if not any(appliance_matches(appliance_rule, actual) for actual in parsed["appliances"])
    ]
    required_appliance_detail = {
        "required_appliances": required_appliances,
        "observed_appliances": parsed["appliances"],
        "missing_required_appliances": missing_required_appliances,
    }
    if missing_required_appliances:
        fail_types.append("MISSING_REQUIRED_ACTION")
        fail_reasons.append({
            "check": "required_appliances",
            "detail": required_appliance_detail,
        })

    # 6) abandon_appliance
    abandon_appliances = det.get("abandon_appliance", []) or []
    found_abandon_appliances = []
    for appliance_rule in abandon_appliances:
        for actual in parsed["appliances"]:
            if appliance_matches(appliance_rule, actual):
                found_abandon_appliances.append({
                    "rule": appliance_rule,
                    "matched_actual": actual,
                })

    abandon_appliance_detail = {
        "abandon_appliance": abandon_appliances,
        "observed_appliances": parsed["appliances"],
        "found_abandon_appliances": found_abandon_appliances,
    }
    if found_abandon_appliances:
        fail_types.append("WRONG_ACTION_TARGET")
        fail_reasons.append({
            "check": "abandon_appliance",
            "detail": abandon_appliance_detail,
        })

    # 7) ordering_constraints
    ordering_ok, ordering_violations = check_ordering_constraints(
        det.get("ordering_constraints", []) or [],
        parsed["flat_sequence"]
    )
    ordering_detail = {
        "ordering_constraints": det.get("ordering_constraints", []) or [],
        "flat_sequence": parsed["flat_sequence"],
        "violations": ordering_violations,
    }
    if not ordering_ok:
        fail_types.append("TRAJECTORY_VIOLATION")
        fail_reasons.append({
            "check": "ordering_constraints",
            "detail": ordering_detail,
        })

    # 8) final_answer check
    has_final_answer = "final_answer" in parsed["flat_sequence"]
    final_answer_detail = {
        "has_final_answer": has_final_answer,
        "flat_sequence": parsed["flat_sequence"],
    }
    if not has_final_answer:
        fail_types.append("MISSING_FINAL_RESPONSE")
        fail_reasons.append({
            "check": "final_answer",
            "detail": final_answer_detail,
        })

    # deduplicate fail_types
    dedup_fail_types: List[str] = []
    for ft in fail_types:
        if ft not in dedup_fail_types:
            dedup_fail_types.append(ft)

    result = "PASS" if not dedup_fail_types else "FAIL"

    return {
        "id": benchmark_case.get("id"),
        "result": result,
        "fail_types": dedup_fail_types,
        "fail_reasons": fail_reasons,
        "details": {
            "note_check": note_check_detail,
            "expected_state_check": expected_state_detail,
            "required_tool_check": required_tool_detail,
            "abandon_tool_check": abandon_tool_detail,
            "required_appliance_check": required_appliance_detail,
            "abandon_appliance_check": abandon_appliance_detail,
            "ordering_check": ordering_detail,
            "final_answer_check": final_answer_detail,
        }
    }


def run_deterministic_eval(benchmark_path: str, agent_output_path: str) -> Dict[str, Any]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    with open(agent_output_path, "r", encoding="utf-8") as f:
        agent_output_data = json.load(f)

    benchmark_by_id = {item["id"]: item for item in benchmark_data}
    output_by_id = {item["id"]: item for item in agent_output_data}

    results: List[Dict[str, Any]] = []
    missing_in_benchmark: List[str] = []
    missing_in_output: List[str] = []

    all_ids = sorted(set(benchmark_by_id.keys()) | set(output_by_id.keys()))

    for case_id in all_ids:
        benchmark_case = benchmark_by_id.get(case_id)
        output_case = output_by_id.get(case_id)

        if benchmark_case is None:
            missing_in_benchmark.append(case_id)
            continue

        if output_case is None:
            missing_in_output.append(case_id)
            results.append({
                "id": case_id,
                "result": "FAIL",
                "fail_types": ["MISSING_REQUIRED_ACTION"],
                "fail_reasons": [{
                    "check": "missing_agent_output",
                    "detail": f"No agent_output entry found for testcase id={case_id}",
                }],
                "details": {},
            })
            continue

        results.append(evaluate_single_case(benchmark_case, output_case))

    pass_count = sum(1 for r in results if r["result"] == "PASS")
    fail_count = sum(1 for r in results if r["result"] == "FAIL")

    return {
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


if __name__ == "__main__":
    benchmark_path = "./eval/eval_dataset_full.json"
    agent_output_path = "./eval/agent_output_gemini_3fewshot.json"
    output_path = "./eval/deterministic_report_gemini_3fewshot.json"

    report = run_deterministic_eval(benchmark_path, agent_output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))