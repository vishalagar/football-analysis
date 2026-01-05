
const API_BASE = "/api";

// Global state
let selectedIssues = new Set();
let availableClasses = [];
let allIssues = [];

async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        renderStats(data.dataset_stats);
        updateTrainingState(data.training_state);
    } catch (e) {
        console.error("Failed to fetch stats", e);
    }
}

function renderStats(stats) {
    const container = document.getElementById('stats-container');
    let html = '';

    for (const [split, info] of Object.entries(stats)) {
        html += `
            <div class="stat-item">
                <span style="text-transform: capitalize;">${split}</span>
                <span>${info.count} imgs</span>
            </div>
        `;
    }
    container.innerHTML = html;
}

function updateTrainingState(state) {
    const logs = document.getElementById('training-logs');
    if (state.status === "running") {
        document.getElementById('training-section').style.display = 'block';
        logs.innerHTML = `<div>Status: Running...</div>`;
        document.getElementById('train-btn').disabled = true;
        document.getElementById('train-btn').innerText = "Training...";
    } else if (state.status === "completed") {
        document.getElementById('training-section').style.display = 'block';
        logs.innerHTML = `
            <div style="color: var(--success-color)">Training Completed!</div>
            <div>Best Val Acc: ${(state.result.val_accuracy * 100).toFixed(2)}%</div>
            ${state.result.test_accuracy ? `<div>Test Acc: ${(state.result.test_accuracy * 100).toFixed(2)}%</div>` : ''}
            <div>Best Params: ${JSON.stringify(state.result.best_params)}</div>
        `;
        document.getElementById('train-btn').disabled = false;
        document.getElementById('train-btn').innerText = "Restart Training";
    }
}

async function triggerAnalysis() {
    const output = document.getElementById('agent-output');
    output.innerHTML = "🤔 Llama3 is thinking... (This may take a moment)";

    try {
        const res = await fetch(`${API_BASE}/analyze`);
        const decision = await res.json();

        displayAgentDecision(decision);
    } catch (e) {
        output.innerHTML = `Error: ${e.message}`;
    }
}

function displayAgentDecision(decision) {
    const output = document.getElementById('agent-output');
    const cleaningSection = document.getElementById('cleaning-section');
    const trainingSection = document.getElementById('training-section');

    // reset visibility
    cleaningSection.style.display = 'none';
    trainingSection.style.display = 'none';

    output.innerHTML = `
        <strong>Analysis:</strong> ${decision.analysis || decision.decision} <br><br>
        <strong>Recommendation:</strong> <span style="color: var(--accent-color)">${decision.recommended_action}</span>
    `;

    if (decision.recommended_action === "data_cleaning") {
        cleaningSection.style.display = 'block';
        renderIssues(decision.issues_list);
    } else {
        // Assume training or tuning
        trainingSection.style.display = 'block';
    }
}

async function renderIssues(issues) {
    const container = document.getElementById('issues-container');
    document.getElementById('issue-count').innerText = `${issues.length} Issues found`;

    allIssues = issues;
    selectedIssues.clear();

    // Fetch available classes
    try {
        const res = await fetch(`${API_BASE}/get_classes`);
        const data = await res.json();
        availableClasses = data.classes || [];
    } catch (e) {
        console.error('Failed to fetch classes', e);
        availableClasses = [];
    }

    // Batch actions header
    const headerHTML = `
        <div style="display: flex; gap: 10px; margin-bottom: 15px; align-items: center; flex-wrap: wrap; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            <button class="btn-primary" onclick="selectAll()" style="padding: 6px 12px;">Select All</button>
            <button onclick="clearSelection()" style="padding: 6px 12px;">Clear</button>
            <span id="selected-count" style="color: var(--accent-color); font-weight: 600;">0 selected</span>
            <div style="display: flex; gap: 5px; margin-left: auto; align-items: center; flex-wrap: wrap;">
                <select id="batch-label-select" style="padding: 6px; border-radius: 4px; background: rgba(255,255,255,0.1); color: var(--text-primary); border: 1px solid rgba(255,255,255,0.2);">
                    <option value="">-- Batch Move To --</option>
                    ${availableClasses.map(cls => `<option value="${cls}">${cls}</option>`).join('')}
                </select>
                <button class="btn-success" onclick="batchMove()" style="padding: 6px 12px;">Move Selected</button>
                <button class="btn-danger" onclick="batchDelete()" style="padding: 6px 12px;">Delete Selected</button>
            </div>
        </div>
    `;

    const cardsHTML = issues.map((issue, idx) => `
        <div class="issue-card" id="card-${idx}" data-path="${escapePath(issue.file_path)}">
            <input type="checkbox" 
                   class="issue-checkbox" 
                   style="position: absolute; top: 10px; left: 10px; width: 20px; height: 20px; cursor: pointer; z-index: 10;"
                   onchange="toggleSelection(${idx})"
                   id="checkbox-${idx}">
            <div style="position: relative;">
                <img src="/dataset/${issue.split}/${issue.given_label}/${fileName(issue.file_path)}" class="issue-img" onerror="this.src='https://via.placeholder.com/200?text=Error'">
                <span style="position: absolute; top: 10px; right: 10px; background: rgba(0,0,0,0.7); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;">${issue.split.toUpperCase()}</span>
            </div>
            <div class="issue-details">
                <h4>Given: <span style="color: var(--danger-color)">${issue.given_label}</span></h4>
                <h4>Suggested: <span style="color: var(--success-color)">${issue.suggested_label}</span></h4>
                <p>Conf: ${(issue.confidence * 100).toFixed(1)}%</p>
                <div style="margin: 10px 0;">
                    <label style="font-size: 0.8rem; color: var(--text-secondary); display: block; margin-bottom: 4px;">Move to:</label>
                    <select id="label-select-${idx}" style="width: 100%; padding: 6px; border-radius: 4px; background: rgba(255,255,255,0.1); color: var(--text-primary); border: 1px solid rgba(255,255,255,0.2);">
                        ${availableClasses.map(cls => `<option value="${cls}" ${cls === issue.suggested_label ? 'selected' : ''}>${cls}</option>`).join('')}
                    </select>
                </div>
                <div class="actions">
                    <button class="btn-success" onclick="moveToSelected(${idx})">Move</button>
                    <button class="btn-danger" onclick="fixIssue(${idx}, 'delete')">Delete</button>
                    <button onclick="fixIssue(${idx}, 'ignore')">Ignore</button>
                </div>
            </div>
        </div>
    `).join('');

    container.innerHTML = headerHTML + cardsHTML;
}

function toggleSelection(idx) {
    const issue = allIssues[idx];
    if (!issue) return;

    if (selectedIssues.has(issue.file_path)) {
        selectedIssues.delete(issue.file_path);
    } else {
        selectedIssues.add(issue.file_path);
    }
    document.getElementById('selected-count').innerText = `${selectedIssues.size} selected`;
}

function selectAll() {
    selectedIssues.clear();
    allIssues.forEach((issue, idx) => {
        selectedIssues.add(issue.file_path);
        const cb = document.getElementById(`checkbox-${idx}`);
        if (cb) cb.checked = true;
    });
    document.getElementById('selected-count').innerText = `${selectedIssues.size} selected`;
}

function clearSelection() {
    selectedIssues.clear();
    document.querySelectorAll('.issue-checkbox').forEach(cb => cb.checked = false);
    document.getElementById('selected-count').innerText = '0 selected';
}

async function batchMove() {
    const targetLabel = document.getElementById('batch-label-select').value;
    if (!targetLabel) {
        alert('Please select a target label');
        return;
    }
    if (selectedIssues.size === 0) {
        alert('No issues selected');
        return;
    }

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
            // Remove cards for moved items
            allIssues = allIssues.filter(issue => !selectedIssues.has(issue.file_path));
            selectedIssues.clear();
            renderIssues(allIssues);
            fetchStats();
        }
    } catch (e) {
        alert('Batch move failed: ' + e.message);
    }
}

async function batchDelete() {
    if (selectedIssues.size === 0) {
        alert('No issues selected');
        return;
    }
    if (!confirm(`Delete ${selectedIssues.size} images?`)) return;

    try {
        const res = await fetch(`${API_BASE}/batch_fix`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_paths: Array.from(selectedIssues),
                action: 'delete'
            })
        });

        if (res.ok) {
            allIssues = allIssues.filter(issue => !selectedIssues.has(issue.file_path));
            selectedIssues.clear();
            renderIssues(allIssues);
            fetchStats();
        }
    } catch (e) {
        alert('Batch delete failed: ' + e.message);
    }
}

function moveToSelected(idx) {
    const selectedLabel = document.getElementById(`label-select-${idx}`).value;
    if (!selectedLabel) {
        alert("Please select a label");
        return;
    }
    fixIssue(idx, 'move', selectedLabel);
}

async function fixIssue(idx, action, newLabel = null) {
    const issue = allIssues[idx];
    if (!issue) return;

    try {
        const res = await fetch(`${API_BASE}/fix_issue`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: issue.file_path, action, new_label: newLabel })
        });

        if (res.ok) {
            // Remove from allIssues and re-render
            allIssues = allIssues.filter((_, i) => i !== idx);
            renderIssues(allIssues);
            fetchStats();
        }
    } catch (e) {
        alert("Failed to apply fix: " + e.message);
    }
}

async function startTraining() {
    try {
        // Use new auto-training endpoint
        await fetch(`${API_BASE}/start_auto_training`, { method: 'POST' });
        document.getElementById('train-btn').disabled = true;
        document.getElementById('train-btn').innerText = "Auto-Exploring...";
        document.getElementById('training-logs').innerHTML = "🚀 Starting intelligent auto-exploration...<br>This will try multiple configurations automatically.";

        // Poll for auto-training updates
        const interval = setInterval(async () => {
            const res = await fetch(`${API_BASE}/auto_training_status`);
            const state = await res.json();

            updateAutoTrainingUI(state);

            if (state.status === 'completed' || state.status === 'failed' || state.status === 'waiting_user') {
                clearInterval(interval);
            }
        }, 3000);  // Poll every 3 seconds

    } catch (e) {
        alert("Failed to start auto-training");
    }
}

function updateAutoTrainingUI(state) {
    const logs = document.getElementById('training-logs');
    const btn = document.getElementById('train-btn');

    if (state.status === "exploring") {
        logs.innerHTML = `
            <div style="color: var(--accent-color)">🔍 Auto-Exploration in Progress...</div>
            <div>Configuration: ${state.current_config + 1} / ${state.total_configs || '?'}</div>
            <div>Best Accuracy So Far: ${(state.best_acc * 100).toFixed(2)}%</div>
            <div>Iteration: ${state.iteration}</div>
            <div style="margin-top: 10px; font-size: 0.7rem; color: var(--text-secondary);">
                Trying different hyperparameters automatically...<br>
                This may take 30-60 minutes depending on dataset size.
            </div>
        `;
        btn.disabled = true;
        btn.innerText = "Exploring...";
    } else if (state.status === "diagnosing") {
        logs.innerHTML = `
            <div style="color: var(--warning-color)">🤖 AI Agent Diagnosing...</div>
            <div>Analyzing why training underperformed...</div>
        `;
    } else if (state.status === "waiting_user") {
        // Show diagnosis and ask user
        const diagnosis = state.diagnosis;
        logs.innerHTML = `
            <div style="color: var(--warning-color)">⚠️ Agent Needs Your Help</div>
            <div style="margin-top: 10px;">
                <strong>Diagnosis:</strong> ${diagnosis.diagnosis}
            </div>
            <div style="margin-top: 5px;">
                <strong>Reasoning:</strong> ${diagnosis.reasoning}
            </div>
            <div style="margin-top: 10px;">
                <strong>Problematic Classes:</strong> ${diagnosis.problematic_classes.join(', ')}
            </div>
            <div style="margin-top: 15px; padding: 10px; background: rgba(255,193,7,0.1); border-left: 3px solid var(--warning-color);">
                <strong>Recommended Action:</strong> ${diagnosis.recommended_action}<br>
                Please clean the data and click "Re-clean Complete" below.
            </div>
        `;

        // Add action buttons
        const btnContainer = document.createElement('div');
        btnContainer.style.marginTop = '15px';
        btnContainer.style.display = 'flex';
        btnContainer.style.gap = '10px';

        btnContainer.innerHTML = `
            <button class="btn-primary" onclick="window.location.reload()">Go Back to Clean Data</button>
            <button class="btn-success" onclick="userFeedback('recleaned')">Re-clean Complete</button>
            <button onclick="userFeedback('satisfied')">I'm Satisfied</button>
        `;
        logs.appendChild(btnContainer);

        btn.disabled = false;
        btn.innerText = "Waiting for User";
    } else if (state.status === "completed") {
        const result = state.exploration_results?.best_result;
        logs.innerHTML = `
            <div style="color: var(--success-color)">✅ Auto-Training Complete!</div>
            ${result ? `
                <div>Best Val Accuracy: ${(result.val_acc * 100).toFixed(2)}%</div>
                <div>Train Accuracy: ${(result.train_acc * 100).toFixed(2)}%</div>
                <div>Best Config: ${result.config_name}</div>
                <div>Model saved at: ${result.model_path}</div>
            ` : '<div>Training completed successfully!</div>'}
        `;
        btn.disabled = false;
        btn.innerText = "Train Again";
    } else if (state.status === "failed") {
        logs.innerHTML = `
            <div style="color: var(--danger-color)">❌ Training Failed</div>
            <div>${state.error || 'Unknown error'}</div>
        `;
        btn.disabled = false;
        btn.innerText = "Retry";
    }
}

async function userFeedback(action) {
    try {
        await fetch(`${API_BASE}/user_feedback?action=${action}`, { method: 'POST' });
        if (action === 'recleaned') {
            document.getElementById('training-logs').innerHTML = "🔄 Restarting exploration with cleaned data...";
        }
    } catch (e) {
        alert("Failed to send feedback");
    }
}

// Helpers
function fileName(path) {
    return path.split(/[\\/]/).pop();
}

function cleanId(path) {
    return path.replace(/[^a-zA-Z0-9]/g, '');
}

function escapePath(path) {
    return path.replace(/\\/g, '\\\\');
}

// Init
fetchStats();
