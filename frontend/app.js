const API_BASE = "/api";

// Global state
let selectedIssues = new Set();
let availableClasses = [];
let allIssues = [];
let isTraining = false;

// Initial Stats & Environment Setup
async function fetchStats(shouldRedirect = true) {
    try {
        const res = await fetch(`${API_BASE}/status`);
        if (!res.ok) throw new Error("Backend not reachable");
        const data = await res.json();
        renderStats(data);

        // Auto-refresh training status if active
        if (data.training_active && !isTraining) {
            startPollingStatus();
        }
    } catch (e) {
        console.error("Failed to fetch stats:", e);
    }
}

function escapeHtml(text) {
    if (!text) return text;
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function renderStats(stats) {
    document.getElementById('stat-total').innerText = stats.total_images || 0;
    document.getElementById('stat-classes').innerText = stats.num_classes || 0;
    document.getElementById('stat-split').innerText = `${stats.train_count} / ${stats.val_count} / ${stats.test_count || 0}`;

    availableClasses = stats.classes || [];
    updateBatchDropdown();
}

function updateBatchDropdown() {
    const select = document.getElementById('batch-label-select');
    if (!select || !availableClasses.length) return;

    select.innerHTML = '<option value="" disabled selected>Choose new label...</option>';
    availableClasses.forEach(c => {
        select.innerHTML += `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`;
    });
}

// Intelligent Agent Integration
async function triggerAnalysis() {
    const outputDiv = document.getElementById('agent-output');
    outputDiv.innerHTML = `<div class="agent-response">🤖 User uploaded data. Analyzing structure and quality...</div>`;

    try {
        const res = await fetch(`${API_BASE}/analyze`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Analysis failed");
        }
        const decision = await res.json();
        displayAgentDecision(decision);

    } catch (e) {
        outputDiv.innerHTML = `<div class="agent-response" style="border-color: var(--danger-color);">❌ Error: ${e.message}</div>`;
    }
}

async function evaluateCurrentModel() {
    const evalBtn = document.getElementById('evaluate-btn');
    const logs = document.getElementById('eval-results-container');

    evalBtn.disabled = true;
    evalBtn.innerHTML = `<span class="spinner"></span> Evaluating...`;
    logs.style.display = 'block';
    logs.innerHTML = `<div class="agent-response">📊 Calculating comprehensive metrics (Miss Rate, Overkill) for current best model...</div>`;

    try {
        const res = await fetch(`${API_BASE}/evaluate_current_model`);
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Evaluation failed");
        }
        const results = await res.json();
        displayEvaluationResults(results);
    } catch (e) {
        logs.innerHTML = `<div class="agent-response" style="border-color: var(--danger-color);">❌ Error: ${e.message}</div>`;
    } finally {
        evalBtn.disabled = false;
        evalBtn.innerHTML = `📊 Evaluate Best Model`;
    }
}

function displayEvaluationResults(results) {
    const container = document.getElementById('eval-results-container');

    if (results.error) {
        container.innerHTML = `<div class="agent-response" style="border-color: var(--danger-color);">❌ Evaluation Error: ${results.error}</div>`;
        return;
    }

    const val = results.val_metrics;
    const test = results.test_metrics;

    let html = `
        <div class="agent-response" style="background: rgba(15, 23, 42, 0.9); border: 1px solid var(--accent-color);">
            <h3 style="color: var(--accent-color); margin-bottom: 10px;">📊 Model Evaluation Report</h3>
            <p style="margin-bottom: 15px; color: var(--text-secondary); font-size: 0.85rem;">
                Evaluated Model: <b>${results.model_name || 'Best Model'}</b>
            </p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <!-- Validation Stats -->
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                    <h4 style="color: var(--warning-color); margin-bottom: 10px; font-size: 0.9rem;">Validation Set</h4>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Accuracy:</span> <b>${(val.accuracy * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Miss Rate (False Neg):</span> <b>${(val.miss_rate * 100).toFixed(2)}%</b>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>Overkill (False Pos):</span> <b>${(val.overkill_rate * 100).toFixed(2)}%</b>
                    </div>
                </div>

                <!-- Test Stats -->
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                    <h4 style="color: var(--success-color); margin-bottom: 10px; font-size: 0.9rem;">Test Set</h4>
                    ${test ? `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Accuracy:</span> <b>${(test.accuracy * 100).toFixed(2)}%</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Miss Rate (False Neg):</span> <b>${(test.miss_rate * 100).toFixed(2)}%</b>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Overkill (False Pos):</span> <b>${(test.overkill_rate * 100).toFixed(2)}%</b>
                        </div>
                    ` : '<div style="color: grey; font-style: italic;">No Test Set available</div>'}
                </div>
            </div>
            
            <div style="margin-top: 15px; font-size: 0.8rem; color: var(--text-secondary); text-align: center;">
                <i>Metrics weighted by class support. Lower Miss/Overkill is better.</i>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

function displayAgentDecision(decision) {
    const container = document.getElementById('agent-output');
    const cleaningSec = document.getElementById('cleaning-section');
    const trainingSec = document.getElementById('training-section');

    container.innerHTML = `
        <div style="margin-bottom: 12px;">
            <b style="color: var(--accent-color)">AGENT INSIGHT:</b> ${decision.analysis || decision.decision}
        </div>
        <div style="padding: 10px; background: rgba(56, 189, 248, 0.1); border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b style="color: var(--success-color)">RECOMMENDATION:</b> ${decision.recommended_action.replace('_', ' ')}
            </div>
            ${decision.recommended_action === "data_cleaning" ? `<button class="btn-success" onclick="skipToBenchmark()" style="padding: 4px 10px; height: auto; min-height: unset; font-size: 0.75rem;">Skip & Continue</button>` : ''}
        </div>
    `;

    if (decision.recommended_action === "data_cleaning") {
        cleaningSec.style.display = 'block';
        cleaningSec.scrollIntoView({ behavior: 'smooth' });
        renderIssues(decision.issues_list);
    } else {
        trainingSec.style.display = 'block';
        trainingSec.scrollIntoView({ behavior: 'smooth' });
    }
}

// Data Cleaning & Batch Logic
function renderIssues(issues) {
    allIssues = issues || [];
    selectedIssues.clear();
    const container = document.getElementById('issues-container');
    container.innerHTML = "";

    document.getElementById('selected-count').innerText = "0";
    document.getElementById('select-all-box').checked = false;

    if (!issues || issues.length === 0) {
        container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--success-color);">✨ No issues detected! Data looks clean.</div>`;
        return;
    }

    issues.forEach((issue, idx) => {
        const div = document.createElement('div');
        div.className = 'issue-card';
        div.innerHTML = `
            <div style="position: relative;">
                <img src="/api/image/${encodeURIComponent(issue.path)}" class="issue-img" loading="lazy">
                <div style="position: absolute; top: 10px; left: 10px;">
                    <input type="checkbox" class="issue-checkbox" 
                           data-idx="${idx}" 
                           onchange="updateSelection(${idx}, this.checked)">
                </div>
                <div style="position: absolute; top: 10px; right: 10px;">
                   <span class="badge ${issue.issue_type}">${issue.issue_type}</span>
                </div>
            </div>
            <div class="issue-details">
                <h4>${fileName(issue.path)}</h4>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 10px;">
                    <div>📂 Current: <b>${escapeHtml(issue.label)}</b></div>
                    ${issue.suggested_label ? `<div>💡 Suggested: <b style="color: var(--accent-color)">${escapeHtml(issue.suggested_label)}</b></div>` : ''}
                    <div>⚠️ Score: ${(issue.confidence_score * 100).toFixed(1)}%</div>
                </div>
                
                <div style="display: flex; gap: 8px;">
                     ${issue.suggested_label ? `
                        <button class="btn-primary" style="flex: 1; padding: 6px; font-size: 0.75rem;" onclick="applyFixSingle(${idx}, 'relabel')">
                            Accept Fix
                        </button>
                    ` : ''}
                    <button class="btn-danger" style="flex: 1; padding: 6px; font-size: 0.75rem;" onclick="applyFixSingle(${idx}, 'delete')">
                        Delete
                    </button>
                </div>
                 <div style="margin-top: 8px; text-align: center;">
                    <button style="background: transparent; border: 1px solid rgba(255,255,255,0.1); color: var(--text-secondary); width: 100%; font-size: 0.7rem; padding: 4px;" onclick="applyFixSingle(${idx}, 'ignore')">
                        Ignore Issue
                    </button>
                </div>
            </div>
        `;
        container.appendChild(div);
    });

    // Handle batch dropdown visibility
    const actionSelect = document.getElementById('batch-action-select');
    const relabelContainer = document.getElementById('batch-relabel-container');

    actionSelect.onchange = (e) => {
        relabelContainer.style.display = e.target.value === 'relabel' ? 'block' : 'none';
        if (e.target.value === 'relabel') {
            updateBatchDropdown();
        }
    };
}

function fileName(path) {
    return path.split('\\').pop().split('/').pop();
}

function updateSelection(idx, isChecked) {
    if (isChecked) selectedIssues.add(idx);
    else selectedIssues.delete(idx);

    document.getElementById('selected-count').innerText = selectedIssues.size;

    // Update master checkbox state
    const allBox = document.getElementById('select-all-box');
    allBox.indeterminate = selectedIssues.size > 0 && selectedIssues.size < allIssues.length;
    allBox.checked = selectedIssues.size === allIssues.length && allIssues.length > 0;
}

function toggleSelectAll() {
    const isChecked = document.getElementById('select-all-box').checked;
    const checkboxes = document.querySelectorAll('.issue-checkbox');

    selectedIssues.clear();
    checkboxes.forEach(cb => {
        cb.checked = isChecked;
        if (isChecked) selectedIssues.add(parseInt(cb.dataset.idx));
    });

    document.getElementById('selected-count').innerText = selectedIssues.size;
}

async function applyBatchFix() {
    const action = document.getElementById('batch-action-select').value;
    if (!action) return alert("Please select an action first.");
    if (selectedIssues.size === 0) return alert("No items selected.");

    let targetLabel = null;
    if (action === 'relabel') {
        targetLabel = document.getElementById('batch-label-select').value;
        if (!targetLabel) return alert("Please choose a target label.");
    }

    if (!confirm(`Apply '${action}' to ${selectedIssues.size} items?`)) return;

    // Collect fixes
    const fixes = [];
    selectedIssues.forEach(idx => {
        const issue = allIssues[idx];
        fixes.push({
            path: issue.path,
            action: action === 'delete' ? 'delete' : 'move',
            new_label: action === 'relabel' ? targetLabel : null
        });
    });

    try {
        const res = await fetch(`${API_BASE}/fix_issues`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(fixes)
        });

        if (res.ok) {
            alert("Batch fix applied successfully!");
            // Refresh analysis
            triggerAnalysis();
        } else {
            alert("Batch fix failed. See console.");
        }
    } catch (e) {
        console.error(e);
        alert("Error applying batch fix.");
    }
}

function downloadCSV() {
    if (!allIssues.length) return alert("No issues to export.");

    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "FilePath,Type,CurrentLabel,SuggestedLabel,Confidence\n";

    allIssues.forEach(row => {
        csvContent += `${row.path},${row.issue_type},${row.label},${row.suggested_label || ''},${row.confidence_score}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "dataset_issues.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function autoFixAll() {
    if (!confirm("Auto-Fix will automatically apply the AI's top suggestion for ALL detected issues.\n\nAre you sure?")) return;

    const fixes = allIssues.map(issue => ({
        path: issue.path,
        action: issue.suggested_label ? 'move' : 'delete', // Default to delete if no suggestion (e.g. outlier) - logic can be refined
        new_label: issue.suggested_label
    }));

    try {
        const res = await fetch(`${API_BASE}/fix_issues`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(fixes)
        });
        if (res.ok) {
            alert("Auto-Fix Complete! Re-running analysis...");
            triggerAnalysis();
        }
    } catch (e) {
        alert("Auto-fix failed: " + e.message);
    }
}

async function applyFixSingle(idx, action) {
    const issue = allIssues[idx];
    const fix = {
        path: issue.path,
        action: action === 'relabel' ? 'move' : (action === 'delete' ? 'delete' : 'ignore'),
        new_label: action === 'relabel' ? issue.suggested_label : null
    };

    if (action === 'ignore') {
        // Just remove from UI locally
        document.querySelector(`.issue-card:nth-child(${idx + 1})`).style.opacity = '0.3';
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/fix_issues`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify([fix])
        });
        if (res.ok) {
            // Remove card or update UI
            const card = document.querySelectorAll('.issue-card')[idx]; // This might be brittle if array shifts
            // Better to refresh analysis for correctness
            triggerAnalysis();
        }
    } catch (e) {
        alert("Fix failed: " + e.message);
    }
}

function skipToBenchmark() {
    const cleaningSec = document.getElementById('cleaning-section');
    const trainingSec = document.getElementById('training-section');

    cleaningSec.style.display = 'none';
    trainingSec.style.display = 'block';

    // Auto-start training
    startTraining();
}

function forceShowTraining() {
    document.getElementById('training-section').style.display = 'block';
    document.getElementById('cleaning-section').style.display = 'none';
}


// AutoML Benchmarking Logic
async function startTraining() {
    const logs = document.getElementById('training-logs');
    document.getElementById('evaluate-btn').style.display = 'none'; // Hide eval button until done
    logs.innerHTML = `<div style="color: var(--accent-color);">🚀 Starting AutoML process...</div>`;

    try {
        const res = await fetch(`${API_BASE}/train_auto`);
        const data = await res.json();

        if (data.status === "started") {
            startPollingStatus();
        } else {
            logs.innerHTML += `<div style="color: var(--danger-color);">❌ Failed to start: ${data.message}</div>`;
        }
    } catch (e) {
        logs.innerHTML += `<div style="color: var(--danger-color);">❌ Error: ${e.message}</div>`;
    }
}

async function resetTrainingState() {
    try {
        await fetch(`${API_BASE}/reset_training_state`, { method: 'POST' });
        document.getElementById('training-logs').innerHTML = "State reset.";
        document.getElementById('leaderboard-content').innerHTML = "";
    } catch (e) {
        console.error("Reset failed", e);
    }
}

async function performSoftReset() {
    if (confirm("Are you sure you want to perform a soft reset? This will clear current progress but keep uploaded data.")) {
        try {
            const res = await fetch(`${API_BASE}/reset_system`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hard_reset: false })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Soft reset failed");
            }
            alert("Soft reset successful! Reloading page...");
            location.reload();
        } catch (e) {
            alert(`Soft Reset Failed: ${e.message}`);
        }
    }
}

async function performHardReset() {
    if (confirm("WARNING: Are you absolutely sure you want to perform a HARD reset? This will clear ALL progress, uploaded data, and cached models. This action cannot be undone.")) {
        try {
            const res = await fetch(`${API_BASE}/reset_system`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hard_reset: true })
            });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Hard reset failed");
            }
            alert("Hard reset successful! Reloading page...");
            location.reload();
        } catch (e) {
            alert(`Hard Reset Failed: ${e.message}`);
        }
    }
}

function fullSystemReset() {
    performSoftReset();
}


function startPollingStatus() {
    isTraining = true;
    const btn = document.getElementById('upload-btn'); // If exists
    const statusBadge = document.getElementById('system-status');
    // btn.disabled = true; 
    statusBadge.innerText = "Auto-Benchmarking Active";
    statusBadge.classList.add('pulse');

    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/auto_training_status`);
            const state = await res.json();

            updateAutoTrainingUI(state);

            if (state.status === "completed" || state.status === "failed") {
                clearInterval(interval);
                isTraining = false;
                statusBadge.innerText = state.status === "completed" ? "System Ready" : "System Error";
                statusBadge.classList.remove('pulse');

                // Show eval button if completed
                if (state.status === "completed") {
                    document.getElementById('evaluate-btn').style.display = 'inline-flex';
                }
            }
        } catch (e) {
            console.error("Polling error", e);
            // Don't stop polling immediately on one error, but maybe log it
        }
    }, 2000);
}

function updateAutoTrainingUI(state) {
    const logs = document.getElementById('training-logs');
    const leaderboard = document.getElementById('leaderboard-content');

    if (state.status === "exploring" || state.status === "final_training") {
        const title = state.status === "exploring" ? "🚀 AUTO-BENCHMARKING ACTIVE" : "🏋️ FINAL MODEL TRAINING";
        const subtext = state.status === "exploring"
            ? `Fine-tuning hyperparameters using Optuna (Trial ${state.iteration})...`
            : "Performing final deep fine-tuning for maximum accuracy...";

        logs.innerHTML = `
            <div style="color: var(--accent-color); font-weight: 700; margin-bottom: 10px;">${title}</div>
            <div class="progress-bar-container" style="height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; margin-bottom: 15px; overflow: hidden;">
                <div style="width: ${(state.current_config / state.total_configs) * 100}%; height: 100%; background: var(--accent-color);"></div>
            </div>
            <p>Evaluating Architecture <b>${state.current_config + 1}</b> / ${state.total_configs}</p>
            <p>Accuracy: <b>${(state.best_acc * 100).toFixed(2)}%</b> ${state.current_val_acc ? `<small>(Current: ${(state.current_val_acc * 100).toFixed(2)}%)</small>` : ''}</p>
            
            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-bottom: 5px;">
                    <span>Epoch ${state.current_epoch || 0} / ${state.total_epochs || 0}</span>
                </div>
                <div style="height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                    <div style="width: ${((state.current_epoch || 0) / (state.total_epochs || 1)) * 100}%; height: 100%; background: var(--success-color);"></div>
                </div>
            </div>

            <div style="margin-top: 20px; font-size: 0.75rem; color: var(--text-secondary);">
                ${subtext}
            </div>
        `;
    } else if (state.status === "completed") {
        const explorationResults = state.exploration_results || {};
        const best = explorationResults.best_result || null;
        const results = explorationResults.all_results || state.results || []; // Fallback

        if (best) {
            logs.innerHTML = `
                <div style="color: var(--success-color); font-weight: 700;">✅ BENCHMARK COMPLETE</div>
                <h1 style="margin: 15px 0;">${(best.val_acc * 100).toFixed(1)}% <small style="font-size: 0.5em; color: var(--text-secondary)">Val Acc</small></h1>
                
                <p><b>Winner:</b> ${best.config_name || 'Best Model'}</p>
                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 10px;">Train Acc: ${(best.train_acc * 100).toFixed(1)}%</p>

                <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 0.85rem; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
                    <div>
                        <b style="color: var(--accent-color)">Validation Set</b>
                        <div style="margin-top: 5px;">
                            <div style="display:flex; justify-content:space-between;"><span>Miss:</span> <b>${(best.miss_rate * 100).toFixed(1)}%</b></div>
                            <div style="display:flex; justify-content:space-between;"><span>Overkill:</span> <b>${(best.overkill_rate * 100).toFixed(1)}%</b></div>
                        </div>
                    </div>
                </div>
            `;

            // Check for diagnosis and append analysis
            const diagnosis = state.diagnosis || {};
            if (diagnosis.conclusion || diagnosis.dataset_analysis) {
                logs.innerHTML += `
                    <div style="margin-top: 20px; padding: 15px; background: rgba(56, 189, 248, 0.1); border-radius: 8px; font-size: 0.9rem;">
                        <p style="color: var(--accent-color); font-weight: 600; margin-bottom: 8px;">🧠 AI Analysis:</p>
                        ${diagnosis.conclusion ? `<p style="margin-bottom:8px"><b>Conclusion:</b> ${escapeHtml(diagnosis.conclusion)}</p>` : ''}
                        ${diagnosis.dataset_analysis ? `<p style="margin-bottom:8px"><b>Dataset:</b> ${escapeHtml(diagnosis.dataset_analysis)}</p>` : ''}
                         ${diagnosis.next_steps ? `
                            <div style="margin-top: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
                                ${diagnosis.next_steps.map(step =>
                    `<button class="btn-primary" style="font-size: 0.8rem; padding: 5px 10px;" onclick="handleNextStep('${escapeHtml(step.action)}')">${escapeHtml(step.label)}</button>`
                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                 `;
            }

            // Update Leaderboard with history
            if (results.length > 0) {
                leaderboard.innerHTML = results.sort((a, b) => b.val_acc - a.val_acc).map((run, i) => `
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.8rem; color: ${i === 0 ? 'var(--warning-color)' : 'inherit'}">${i + 1}. ${run.config_name || 'Model ' + (i + 1)}</span>
                        <span style="font-weight: 600;">${(run.val_acc * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            }
        }
    } else if (state.status === "failed") {
        logs.innerHTML = `<div style="color: var(--danger-color)">❌ Benchmarking failed: ${state.error}</div>`;
    }
}

async function uploadAndRun() {
    console.log("uploadAndRun triggered");
    const fileInput = document.getElementById('dataset-upload');
    const statusDiv = document.getElementById('upload-status');
    const btn = document.getElementById('upload-btn');

    if (!fileInput || !fileInput.files.length) {
        console.warn("No file selected or input not found");
        alert("Please select a zip file first.");
        return;
    }

    const file = fileInput.files[0];
    console.log(`Starting upload: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);

    const formData = new FormData();
    formData.append("file", file);

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Uploading...`;
    statusDiv.innerText = "Transferring data to server... (this may take a minute for large files)";
    statusDiv.style.color = "var(--accent-color)";

    try {
        console.log("Sending fetch request to /api/upload_dataset");
        const res = await fetch(`${API_BASE}/upload_dataset`, {
            method: 'POST',
            body: formData
        });

        console.log("Response received:", res.status, res.statusText);

        if (!res.ok) {
            let errText = "Upload failed";
            try {
                const errData = await res.json();
                errText = errData.detail || errText;
            } catch (p) {
                errText = await res.text() || errText;
            }
            throw new Error(errText);
        }

        const successData = await res.json();
        console.log("Upload Success:", successData);

        statusDiv.innerText = "✅ Upload & Extraction Complete! Analyzing...";
        statusDiv.style.color = "var(--success-color)";
        btn.innerHTML = "Success!";

        alert("Upload Successful!\nDataset has been updated. The agent will now analyze the new data.");

        // Reset and run
        setTimeout(() => {
            statusDiv.innerText = "";
            fileInput.value = "";
            btn.disabled = false;
            btn.innerHTML = "🚀 Upload & Run";
            fetchStats();
            // Trigger analysis automatically
            console.log("Triggering analysis...");
            triggerAnalysis();
        }, 1000);

    } catch (e) {
        console.error("Upload Error Details:", e);
        statusDiv.innerText = "❌ Error: " + e.message;
        statusDiv.style.color = "var(--danger-color)";
        btn.disabled = false;
        btn.innerHTML = "🚀 Upload & Run";
        alert("Upload Failed: " + e.message + "\nCheck browser console (F12) for more details.");
    }
}

async function handleNextStep(action) {
    if (action === 'filter_dataset') {
        const cleaningSec = document.getElementById('cleaning-section');
        const trainingSec = document.getElementById('training-section');
        const logs = document.getElementById('training-logs');

        // Show loading state in logs
        logs.innerHTML += `<div style="margin-top:20px; color: var(--accent-color); font-style: italic;">🔄 Running Hybrid Analysis using best model...</div>`;

        try {
            const res = await fetch(`${API_BASE}/analyze_with_model`);
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Analysis failed");
            }
            const decision = await res.json();

            // Switch views
            trainingSec.style.display = 'none';
            cleaningSec.style.display = 'block';
            cleaningSec.scrollIntoView({ behavior: 'smooth' });

            // Render new issues
            displayAgentDecision(decision);

        } catch (e) {
            alert(`Hybrid Analysis Failed: ${e.message}`);
        }
    } else if (action === 'more_tuning') {
        // Reset state and restart training
        if (confirm("This will reset the current results and start a new hyperparameter tuning session. Continue?")) {
            await resetTrainingState();
            await startTraining();
        }
    }
}


// Global scope expose
window.uploadAndRun = uploadAndRun;
window.triggerAnalysis = triggerAnalysis;
window.evaluateCurrentModel = evaluateCurrentModel;
window.startTraining = startTraining;
window.autoFixAll = autoFixAll;
window.applyBatchFix = applyBatchFix;
window.updateSelection = updateSelection;
window.toggleSelectAll = toggleSelectAll;
window.applyFixSingle = applyFixSingle;
window.skipToBenchmark = skipToBenchmark;
window.forceShowTraining = forceShowTraining;
window.handleNextStep = handleNextStep;
window.performSoftReset = performSoftReset;
window.performHardReset = performHardReset;

// Init
document.addEventListener('DOMContentLoaded', fetchStats);
