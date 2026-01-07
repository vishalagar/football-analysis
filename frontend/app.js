const API_BASE = "/api";

// Global state
let selectedIssues = new Set();
let availableClasses = [];
let allIssues = [];
let isTraining = false;

/**
 * Initial Stats & Environment Setup
 */
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        renderStats(data.dataset_stats);

        // Populate available classes for dropdowns
        const clsRes = await fetch(`${API_BASE}/get_classes`);
        const clsData = await clsRes.json();
        availableClasses = clsData.classes || [];
        updateBatchDropdown();

        // Sync Phase 4 (AutoML) training state if already running
        if (data.auto_training_state && data.auto_training_state.status !== "idle") {
            if (["exploring", "diagnosing", "waiting_user", "completed"].includes(data.auto_training_state.status)) {
                forceShowTraining();
                startPollingStatus();
            }
        }
    } catch (e) {
        console.error("Dashboard out of sync:", e);
        document.getElementById('system-status').innerText = "System Offline";
    }
}

function renderStats(stats) {
    const container = document.getElementById('stats-container');
    if (!container) return;

    let html = '';
    for (const [split, info] of Object.entries(stats)) {
        html += `
            <div class="stat-item">
                <span style="text-transform: capitalize;">${split} Set</span>
                <span>${info.count} samples</span>
            </div>
        `;
    }
    container.innerHTML = html || '<p>No data found.</p>';
}

function updateBatchDropdown() {
    const select = document.getElementById('batch-label-select');
    if (!select) return;

    // Save current value
    const curVal = select.value;
    select.innerHTML = '<option value="">Move to...</option>' +
        availableClasses.map(cls => `<option value="${cls}">${cls}</option>`).join('');
    select.value = curVal;
}

/**
 * Intelligent Agent Integration
 */
async function triggerAnalysis() {
    const output = document.getElementById('agent-output');
    const btn = document.getElementById('analyze-btn');

    output.classList.add('pulse');
    output.innerHTML = "<b>Agent is analyzing dataset gradients and label consistency...</b>";
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/analyze`);
        if (!res.ok) {
            const errBody = await res.text();
            throw new Error(`Server Error (${res.status}): ${errBody.slice(0, 100)}`);
        }
        const decision = await res.json();
        displayAgentDecision(decision);
    } catch (e) {
        output.innerHTML = `<span style="color: var(--danger-color)">Analysis Error: ${e.message}</span>`;
    } finally {
        output.classList.remove('pulse');
        btn.disabled = false;
    }
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

/**
 * Data Cleaning & Batch Logic
 */
function renderIssues(issues) {
    allIssues = issues;
    const container = document.getElementById('issues-container');
    const countPill = document.getElementById('issue-count-pill');

    countPill.innerText = issues.length;

    if (issues.length === 0) {
        container.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; border: 2px dashed var(--card-border); border-radius: 20px;">
                <h3 style="color: var(--success-color)">✨ Dataset is Clean!</h3>
                <p style="color: var(--text-secondary)">No further issues detected by the agent.</p>
                <button class="btn-primary" onclick="forceShowTraining()" style="margin: 20px auto;">Continue to Benchmarking</button>
            </div>
        `;
        return;
    }

    container.innerHTML = issues.map((issue, idx) => `
        <div class="issue-card" id="card-${idx}">
            <div style="position: relative;">
                <input type="checkbox" class="issue-checkbox" 
                       onchange="updateSelection(${idx}, this.checked)" 
                       ${selectedIssues.has(issue.file_path) ? 'checked' : ''}
                       style="position: absolute; top: 15px; left: 15px; z-index: 10;">
                <img src="/dataset/${issue.split}/${issue.given_label}/${fileName(issue.file_path)}?t=${Date.now()}" 
                     class="issue-img" loading="lazy">
            </div>
            <div class="issue-details">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; align-items: center;">
                    <span class="badge ${issue.issue_type}">${issue.issue_type.replace('_', ' ').toUpperCase()}</span>
                    <span style="font-size: 0.75rem; color: var(--accent-color)">${(issue.confidence * 100).toFixed(0)}% Conf.</span>
                </div>
                <div style="margin-bottom: 15px;">
                    <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 4px;">Current: <span style="color: var(--danger-color); font-weight: 600;">${issue.given_label}</span></p>
                    <p style="font-size: 0.8rem; color: var(--text-secondary);">Suggest: <span style="color: ${issue.suggested_label === 'delete' ? 'var(--danger-color)' : 'var(--success-color)'}; font-weight: 600;">${issue.suggested_label.toUpperCase()}</span></p>
                </div>
                <select id="select-${idx}" style="margin-bottom: 12px;">
                    ${availableClasses.map(cls => `<option value="${cls}" ${cls === issue.suggested_label ? 'selected' : ''}>${cls}</option>`).join('')}
                </select>
                <div class="actions">
                    <button class="btn-success" onclick="applyFixSingle(${idx}, 'move')" style="padding: 6px;">Move</button>
                    <button class="btn-danger" onclick="applyFixSingle(${idx}, 'delete')" style="padding: 6px;">Delete</button>
                </div>
            </div>
        </div>
    `).join('');
}

function fileName(path) {
    return path.split(/[\\/]/).pop();
}

function updateSelection(idx, isChecked) {
    const path = allIssues[idx].file_path;
    if (isChecked) selectedIssues.add(path);
    else selectedIssues.delete(path);
}

function toggleSelectAll() {
    const masterCb = document.getElementById('select-all-issues');
    const cbs = document.querySelectorAll('.issue-checkbox');

    selectedIssues.clear();
    cbs.forEach((cb, idx) => {
        cb.checked = masterCb.checked;
        if (masterCb.checked) selectedIssues.add(allIssues[idx].file_path);
    });
}

async function applyBatchFix() {
    const targetLabel = document.getElementById('batch-label-select').value;
    if (selectedIssues.size === 0) return alert("Select items first!");
    if (!targetLabel) return alert("Select a target move label!");

    try {
        const res = await fetch(`${API_BASE}/batch_fix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_paths: Array.from(selectedIssues),
                action: 'move',
                new_label: targetLabel
            })
        });

        if (res.ok) {
            allIssues = allIssues.filter(i => !selectedIssues.has(i.file_path));
            selectedIssues.clear();
            renderIssues(allIssues);
            fetchStats();
        }
    } catch (e) {
        alert("Batch fix failed: " + e.message);
    }
}

async function autoFixAll() {
    if (allIssues.length === 0) return;
    if (!confirm(`Apply all ${allIssues.length} logical suggestions?`)) return;

    try {
        const items = allIssues.map(i => ({ file_path: i.file_path, new_label: i.suggested_label }));
        const res = await fetch(`${API_BASE}/batch_fix_suggestions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ items })
        });

        if (res.ok) {
            allIssues = [];
            selectedIssues.clear();
            renderIssues([]);
            fetchStats();
        }
    } catch (e) {
        alert("Auto-fix failed");
    }
}

async function applyFixSingle(idx, action) {
    const issue = allIssues[idx];
    const newLabel = document.getElementById(`select-${idx}`).value;

    try {
        const res = await fetch(`${API_BASE}/fix_issue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: issue.file_path, action, new_label: newLabel })
        });

        if (res.ok) {
            allIssues.splice(idx, 1);
            selectedIssues.delete(issue.file_path);
            renderIssues(allIssues);
            fetchStats();
        }
    } catch (e) {
        alert("Action failed");
    }
}

function skipToBenchmark() {
    const cleaningSec = document.getElementById('cleaning-section');
    const trainingSec = document.getElementById('training-section');
    cleaningSec.style.display = 'none';
    trainingSec.style.display = 'block';
    trainingSec.scrollIntoView({ behavior: 'smooth' });
}

/**
 * AutoML Benchmarking Logic
 */
async function startTraining() {
    if (isTraining) return;

    try {
        await fetch(`${API_BASE}/start_auto_training`, { method: 'POST' });
        startPollingStatus();
    } catch (e) {
        alert("Could not initiate benchmarking.");
    }
}

function startPollingStatus() {
    isTraining = true;
    const btn = document.getElementById('train-btn');
    const statusBadge = document.getElementById('system-status');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Benchmarking...`;
    statusBadge.innerText = "Auto-Benchmarking Active";
    statusBadge.classList.add('pulse');

    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/auto_training_status`);
            const state = await res.json();

            updateAutoTrainingUI(state);

            if (["completed", "failed", "waiting_user"].includes(state.status)) {
                clearInterval(interval);
                isTraining = false;
                btn.disabled = false;
                btn.innerText = "Start Multi-Model Benchmark";
                statusBadge.innerText = "System Standby";
                statusBadge.classList.remove('pulse');
                fetchStats();
            }
        } catch (e) {
            clearInterval(interval);
            isTraining = false;
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
        // Safely handle exploration_results
        const explorationResults = state.exploration_results || {};
        const best = explorationResults.best_result || null;
        const results = explorationResults.all_results || state.results || [];

        if (best) {
            logs.innerHTML = `
                <div style="color: var(--success-color); font-weight: 700;">✅ BENCHMARK COMPLETE</div>
                <h1 style="margin: 15px 0;">${(best.val_acc * 100).toFixed(1)}% <small style="font-size: 0.5em; color: var(--text-secondary)">Acc</small></h1>
                <p><b>Winner:</b> ${best.config_name || 'Best Model'}</p>
                <div style="margin-top: 15px; font-size: 0.8rem;">
                    <p>Train Acc: ${(best.train_acc * 100).toFixed(1)}%</p>
                    <p>Miss Rate: ${(best.miss_rate * 100).toFixed(1)}%</p>
                    <p>Overkill Rate: ${(best.overkill_rate * 100).toFixed(1)}%</p>
                </div>
            `;

            // Update Leaderboard with history
            if (results.length > 0) {
                leaderboard.innerHTML = results.sort((a, b) => b.val_acc - a.val_acc).map((run, i) => `
                    <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <span style="font-size: 0.8rem; color: ${i === 0 ? 'var(--warning-color)' : 'inherit'}">${i === 0 ? '👑' : i + 1}. ${run.config_name || 'Model ' + (i + 1)}</span>
                        <span style="font-weight: 600;">${(run.val_acc * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            } else {
                leaderboard.innerHTML = '<p style="color: var(--text-secondary);">No evaluation history available.</p>';
            }
        } else {
            // Fallback if no best_result found
            logs.innerHTML = `
                <div style="color: var(--success-color); font-weight: 700;">✅ TRAINING COMPLETE</div>
                <p style="margin-top: 15px; color: var(--text-secondary);">Best model saved. Results available in logs.</p>
            `;
            leaderboard.innerHTML = '<p style="color: var(--text-secondary);">Evaluation results shown here.</p>';
        }
    } else if (state.status === "failed") {
        logs.innerHTML = `<div style="color: var(--danger-color)">❌ Benchmarking failed: ${state.error}</div>`;
    }
}

// Global scope expose
window.triggerAnalysis = triggerAnalysis;
window.startTraining = startTraining;
window.autoFixAll = autoFixAll;
window.applyBatchFix = applyBatchFix;
window.toggleSelectAll = toggleSelectAll;
window.applyFixSingle = applyFixSingle;
window.skipToBenchmark = skipToBenchmark;

// Init
document.addEventListener('DOMContentLoaded', fetchStats);
