
import requests
import time
import sys
import json

BASE_URL = "http://localhost:8000"

def log(msg, type="INFO"):
    print(f"[{type}] {msg}")

def test_api():
    log("Starting Headless E2E Test...", "START")
    
    # 1. Check System Status
    try:
        res = requests.get(f"{BASE_URL}/api/status")
        if res.status_code == 200:
            log("System Status: OK", "SUCCESS")
            log(f"Stats: {json.dumps(res.json()['dataset_stats'], indent=2)}")
        else:
            log(f"System Status Failed: {res.status_code}", "ERROR")
            return
    except Exception as e:
        log(f"Connection failed: {e}", "CRITICAL")
        return

    # 2. Trigger Analysis (Mocking likely needed if Llama3 is slow/unavailable, but we try real first)
    log("Triggering Dataset Analysis (this depends on Llama3)...", "ACTION")
    start = time.time()
    try:
        res = requests.get(f"{BASE_URL}/api/analyze", timeout=60)
        if res.status_code == 200:
            data = res.json()
            log(f"Analysis Complete in {time.time()-start:.2f}s", "SUCCESS")
            log(f"Decision: {data.get('recommended_action')}", "INFO")
            log(f"Issues Found: {len(data.get('issues_list', []))}", "INFO")
        else:
            log(f"Analysis Failed: {res.text}", "ERROR")
    except Exception as e:
        log(f"Analysis Timeout/Error: {e}", "ERROR")

    # 3. Start Auto-Training Benchmarking
    log("Starting Multi-Model Benchmark...", "ACTION")
    try:
        res = requests.post(f"{BASE_URL}/api/start_auto_training")
        if res.status_code == 200:
            log("Benchmark Started Successfully", "SUCCESS")
        else:
            log(f"Benchmark Start Failed: {res.text}", "ERROR")
            # If it says "already in progress", that's fine for testing
    except Exception as e:
        log(f"Benchmark Request Error: {e}", "ERROR")

    # 4. Monitor Progress
    log("Monitoring Training Progress...", "monitor")
    for i in range(20): # Check for 20 intervals
        try:
            res = requests.get(f"{BASE_URL}/api/auto_training_status")
            state = res.json()
            status = state.get("status")
            iteration = state.get("iteration")
            config = state.get("current_config")
            acc = state.get("best_acc")
            
            log(f"Status: {status} | Iter: {iteration} | Config: {config} | Best Acc: {acc:.4f}", "STATUS")
            
            if status == "completed":
                log("Benchmarking COMPLETE!", "SUCCESS")
                results = state.get("exploration_results", {})
                log(f"Winner: {results.get('best_result', {}).get('config_name')}", "WINNER")
                break
            
            if status == "failed":
                log(f"Benchmarking FAILED: {state.get('error')}", "ERROR")
                break
                
            time.sleep(2)
        except Exception as e:
            log(f"Monitoring Error: {e}", "ERROR")
            break

if __name__ == "__main__":
    # Wait for server to be up
    log("Waiting 5s for server warmup...", "WAIT")
    time.sleep(5)
    test_api()
