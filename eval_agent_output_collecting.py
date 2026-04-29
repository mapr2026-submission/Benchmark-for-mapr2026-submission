import json
import os
from datetime import datetime
from tools import add_note
from espnode_manager.esp_communication import send_command
from local_llm import load_system_prompt
from agent import build_agent
import asyncio
import time



appliance_reset_commands = [{'espID': 1, 'device_type': 'actuator', 'device_name': 'led1', 'action': 'set', 'value': False}, {'espID': 1, 'device_type': 'actuator', 'device_name': 'motor1', 'action': 'set', 'value': 0}, {'espID': 2, 'device_type': 'actuator', 'device_name': 'led1', 'action': 'set', 'value': False}, {'espID': 2, 'device_type': 'actuator', 'device_name': 'led2', 'action': 'set', 'value': False}, {'espID': 2, 'device_type': 'actuator', 'device_name': 'motor1', 'action': 'set', 'value': 0}, {'espID': 2, 'device_type': 'actuator', 'device_name': 'motor2', 'action': 'set', 'value': 0}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'led1', 'action': 'set', 'value': False}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'led2', 'action': 'set', 'value': False}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'led3', 'action': 'set', 'value': False}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'motor1', 'action': 'set', 'value': 0}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'motor2', 'action': 'set', 'value': 0}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'servo', 'action': 'set', 'value': False}, {'espID': 3, 'device_type': 'actuator', 'device_name': 'pump', 'action': 'set', 'value': False}]


def clear_file(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)

delays = [5, 7, 9, 11, 13]
def testcase_initial_setup() -> None:
    for attempt, delay in enumerate(delays, start=1):
        try:
            for cmd in appliance_reset_commands:
                send_command(cmd, cmd["espID"] - 1)
            break  # success → exit loop
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")

            if attempt == len(delays):
                print("All retries failed.")
                raise  # or handle final failure

            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        

    clear_file("./schedule_trigger.json")
    clear_file("./note_storage.json")


def get_actual_states(expected_state, appliance_file="appliances_data.json"):
    with open(appliance_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    house = data["List_of house appliance (current values)"]
    results = []

    for exp in expected_state:
        esp_id = exp["espID"]
        device_type = exp["device_type"]
        device_name = exp["device_name"]

        found = None

        for room_name, room in house.items():
            if room["espID"] != esp_id:
                continue

            for device in room.get(device_type, []):
                if device["id"] == device_name:
                    found = device["value"]
                    break

            if found is not None:
                break

        results.append({
            "espID": esp_id,
            "device_type": device_type,
            "device_name": device_name,
            "expected": exp["value"],
            "actual": found
        })

    return results



def collect_agent_outputs(benchmark_path: str, output_path: str = "./eval/agent_output_gemini_3fewshot.json") -> None:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    model = "gemini-3-flash-preview:cloud"
    # gemini-3-flash-preview:cloud
    # gemma4:31b-cloud
    # qwen3.5:397b-cloud
    # gpt-oss:120b-cloud
    # gpt-oss:20b-cloud
    # ministral-3:14b-cloud

    print(f"[TOTAL CASES]: {len(benchmark)}")
    print(f"[MODEL]: {model}")

    # Initialize output file
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not isinstance(results, list):
                results = []
        except Exception:
            results = []
    else:
        results = []

    for case in benchmark:
        # Clear schedule_trigger and note_storage before running the benchmark
        testcase_initial_setup()

        print("[START COLLECTING]: "+ case["id"])
        # Store id
        result_item = {
            "id": case["id"]
        }
        try:
            print("Setting up precondition...")
            # Set up preconditions - Notes
            if case.get("preconditions", {}).get("notes_setup", []):
                for note in case["preconditions"]["notes_setup"]:
                    if note["date"].lower() == "today":
                        add_note(note["note_text"], [datetime.now().strftime('%Y-%m-%d')])
                    else:
                        add_note(note["note_text"], [note["date"]])
            
            # Set up preconditions - Appliance states
            if case.get("preconditions", {}).get("state_setup", []):
                for command in case["preconditions"]["state_setup"]:
                    send_command(command, command["espID"] - 1)

            # Set up agent
            role_sys_prompt = load_system_prompt('./system_prompt_doc/role.txt')
            instruction_sys_prompt = load_system_prompt('./system_prompt_doc/instruction.txt')
            parts = [role_sys_prompt, instruction_sys_prompt]
            sys_prompt = "\n\n".join([p for p in parts if p])
            agent = build_agent(sys_prompt, model=model)

            # Execution test case
            print("Running agent...")
            start_time = time.time()
            inference_ouput = asyncio.run(agent.eval_collect(case["prompt"]))
            end_time = time.time()

            print("Collecting results...")
            # Store inference memory + execution time
            result_item["inference_ouput"] = inference_ouput
            result_item["execution_time"] = end_time - start_time

            # Store actual states for deterministic check
            actual_states = get_actual_states(case["deterministic_eval"]["expected_state"])
            result_item["state_check"] = actual_states

            # Store notes after execution
            with open("./note_storage.json", "r", encoding="utf-8") as f:
                note_after = json.load(f)
            result_item["notes_after"] = note_after

            # Store schedule trigger after execution
            with open("./schedule_trigger.json", "r", encoding="utf-8") as f:
                trigger= json.load(f)
            result_item["schedule_trigger_after"] = trigger

        except Exception as e:
            result_item["error"] = str(e)

        results.append(result_item)

        # Save after each case
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("[FINISH]")
    
    # Return to initial setup
    testcase_initial_setup()
    print("\n[ALL CASES FINISHED] Agent outputs collected in: "+ output_path)





if __name__ == "__main__":

    
    benchmark_path = "./eval/eval_dataset_full.json"
    collect_agent_outputs(benchmark_path)

    # testcase_initial_setup()

    # mem = ''
    # print(mem)


    