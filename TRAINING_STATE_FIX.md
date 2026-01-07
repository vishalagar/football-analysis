# Training State Stuck Fix

## Problem
When clicking "Start Training" for the first time, getting error:
```
Failed to start training: Auto-training already in progress (status: exploring). Please wait for completion or refresh.
```

But no actual training is running.

## Root Cause
The `auto_training_state` in the backend gets stuck in `"exploring"` status from:
1. A previous training session that crashed
2. Server restart while training was in progress
3. State restoration from metrics.json setting status to partial/exploring

## Solution Implemented

### 1. **Backend: Reset State Endpoint** (`backend/routers/api.py`)
Added new endpoint `/api/reset_training_state` to manually reset stuck state:

```python
@router.post("/reset_training_state")
def reset_training_state():
    """
    Resets the auto-training state to idle. 
    Useful when the state gets stuck in 'exploring' or 'diagnosing' with no actual training running.
    """
    global auto_training_state
    
    current_status = auto_training_state["status"]
    
    # Reset to initial idle state
    auto_training_state = {
        "status": "idle",
        "current_config": 0,
        "total_configs": 0,
        # ... all other fields reset
    }
    
    return {
        "status": "reset", 
        "message": f"Training state reset from '{current_status}' to 'idle'",
        "previous_status": current_status
    }
```

### 2. **Frontend: Auto-Reset with Confirmation** (`frontend/app.js`)
Enhanced `startTraining()` to detect stuck state and offer automatic reset:

```javascript
async function startTraining() {
    // ...
    if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Unknown error' }));
        
        // Check if the error is about training already in progress
        if (res.status === 400 && errorData.detail && errorData.detail.includes('already in progress')) {
            // Offer to reset the stuck state
            const shouldReset = confirm(
                `${errorData.detail}\n\n` +
                `It seems the training state is stuck. No actual training is running.\n\n` +
                `Would you like to reset the training state and try again?`
            );
            
            if (shouldReset) {
                await resetTrainingState();
                // Try starting again after reset
                await startTraining();
            }
        } else {
            alert(`Failed to start training: ${errorData.detail || 'Unknown error'}`);
        }
        return;
    }
    // ...
}

async function resetTrainingState() {
    try {
        const res = await fetch(`${API_BASE}/reset_training_state`, { method: 'POST' });
        const data = await res.json();
        
        if (res.ok) {
            console.log(`Training state reset: ${data.message}`);
            await fetchStats();
        } else {
            alert('Failed to reset training state');
        }
    } catch (e) {
        alert(`Error resetting state: ${e.message}`);
    }
}
```

### 3. **Better Error Messages**
- Backend now includes current status in error message
- Frontend provides helpful context and automatic recovery option

### 4. **Fixed Missing Function**
Added `forceShowTraining()` function that was referenced but not defined.

## How to Use

### Automatic (Recommended)
1. Click "Start Training" button
2. If you see the "already in progress" error, a dialog will appear
3. Click "OK" to automatically reset the state and retry
4. Training will start fresh

### Manual Reset (If Needed)
Send POST request to reset endpoint:
```bash
# PowerShell
Invoke-WebRequest -Method POST -Uri http://localhost:8000/api/reset_training_state

# Or from browser console
fetch('/api/reset_training_state', { method: 'POST' }).then(r => r.json()).then(console.log)
```

## Files Modified
- `backend/routers/api.py` - Added reset endpoint + better error messages
- `frontend/app.js` - Added auto-reset logic + missing forceShowTraining function

## Testing
1. **Restart your backend server** to load the new endpoint
2. **Hard refresh your browser** (Ctrl+F5) to load new frontend code
3. Try clicking "Start Training" - you should now see the recovery dialog if state is stuck
4. Click OK to reset and retry automatically

## Prevention
To avoid this in the future:
- Always stop training gracefully before shutting down the backend
- Consider adding automatic state cleanup on backend startup
- Monitor for orphaned background threads
