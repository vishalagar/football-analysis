
import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000"

def log(msg, type="INFO"):
    print(f"[{type}] {msg}")

def test_rigorous_e2e():
    log("Starting Rigorous E2E API Journey Simulation...", "START")
    
    # 1. Verification of System Health
    log("Step 1: Checking System Status API", "ACTION")
    try:
        res = requests.get(f"{BASE_URL}/api/status")
        if res.status_code != 200:
            log(f"Health check failed: {res.status_code}", "CRITICAL")
            return
        log("Health check PASSED", "SUCCESS")
    except Exception as e:
        log(f"Server unreachable: {e}", "CRITICAL")
        return

    # 2. Trigger AI Dataset Analysis
    log("Step 2: Triggering Llama3 Analysis", "ACTION")
    issues = []
    start = time.time()
    try:
        res = requests.get(f"{BASE_URL}/api/analyze", timeout=120)
        if res.status_code == 200:
            data = res.json()
            log(f"Analysis successful in {time.time()-start:.2f}s", "SUCCESS")
            log(f"Agent Recommendation: {data.get('recommended_action')}", "INFO")
            
            issues = data.get('issues_list', [])
            if issues:
                log(f"Found {len(issues)} issues. Verifying issue structure...", "INFO")
                first_issue = issues[0]
                required_keys = ['file_path', 'issue_type', 'given_label', 'suggested_label', 'confidence']
                if all(k in first_issue for k in required_keys):
                    log("Issue structure is valid", "SUCCESS")
                else:
                    log("Issue structure is MISSING keys!", "WARNING")
        else:
            log(f"Analysis failed ({res.status_code}): {res.text}", "ERROR")
    except Exception as e:
        log(f"Analysis request failed: {e}", "ERROR")

    # 3. Simulate Data Cleaning (Apply a recommendation)
    if issues:
        log("Step 3: Simulating Batch Fix for Analysis results", "ACTION")
        fix_items = []
        for issue in issues[:2]:
             fix_items.append({
                 "file_path": issue['file_path'],
                 "new_label": issue['suggested_label']
             })
        
        batch_req = {"items": fix_items}
        try:
            res = requests.post(f"{BASE_URL}/api/batch_fix_suggestions", json=batch_req)
            if res.status_code == 200:
                log("Batch fix applied successfully", "SUCCESS")
            else:
                log(f"Batch fix failed: {res.text}", "ERROR")
        except Exception as e:
            log(f"Batch fix request failed: {e}", "ERROR")

    # 4. Start AutoML Benchmarking
    log("Step 4: Starting Multi-Model ResNet18 Benchmark", "ACTION")
    try:
        res = requests.post(f"{BASE_URL}/api/start_auto_training")
        if res.status_code == 200:
            log("Benchmark started", "SUCCESS")
        else:
            log(f"Benchmark start failed: {res.text}", "ERROR")
    except Exception as e:
        log(f"Benchmark start request failed: {e}", "ERROR")

    # 5. Monitor & Validate Leadboard Convergence
    log("Step 5: Monitoring Leadboard and Convergence", "ACTION")
    completed = False
    for i in range(15):
        time.sleep(10)
        try:
            res = requests.get(f"{BASE_URL}/api/auto_training_status")
            state = res.json()
            status = state.get('status')
            acc = state.get('best_acc', 0)
            log(f"Poll {i+1}: Status={status}, Best Acc={acc:.4f}", "MONITOR")
            
            if status == "completed":
                log("AutoML Journey COMPLETE!", "SUCCESS")
                completed = True
                break
            elif status == "failed":
                log(f"AutoML FAILED: {state.get('error')}", "ERROR")
                break
        except Exception as e:
            log(f"Status poll failed: {e}", "ERROR")
            break
            
    if not completed:
        log("Monitoring timed out - training still in progress or slow.", "INFO")

    log("Rigorous E2E API Journey Simulation Finished.", "END")

if __name__ == "__main__":
    test_rigorous_e2e()
