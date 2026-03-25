document.addEventListener('DOMContentLoaded', () => {

    // ==========================================
    // 1. AUTH & USER IDENTITY (SELF-HEALING)
    // ==========================================
    const profileNameLabel = document.getElementById('profileName');
    const logoutBtn = document.getElementById('logoutBtn');
    const deleteBtn = document.getElementById('deleteAccountBtn');
    const logsPanel = document.querySelector('.live-logs');
    
    // Check Server Session on Load
    fetch('/api/user_info')
        .then(res => res.json())
        .then(data => {
            if (data.username) {
                localStorage.setItem('jobGuardUser', data.username);
                updateIdentityUI(data.username);
            } else {
                localStorage.removeItem('jobGuardUser');
                updateIdentityUI(null);
            }
        })
        .catch(err => console.log("Session check failed", err));

    // Fallback to LocalStorage for instant UI (Self-Healing)
    const storedUser = localStorage.getItem('jobGuardUser');
    if (storedUser) updateIdentityUI(storedUser);

    function updateIdentityUI(username) {
        if (!username) { 
            if(profileNameLabel) profileNameLabel.innerText = "OPERATOR: GUEST"; 
            return; 
        }
        if(profileNameLabel) profileNameLabel.innerText = `OPERATOR: ${username.toUpperCase()}`;
        
        // If Admin, show logs
        if (username === 'Yoge') { 
            if(logsPanel) logsPanel.style.display = 'flex'; 
            fetchGlobalSystemLogs(); 
        }
    }

    if(logoutBtn) {
        logoutBtn.addEventListener('click', () => { 
            if(confirm("TERMINATE SESSION?")) { 
                fetch('/api/logout', { method: 'POST' })
                .then(() => { 
                    localStorage.removeItem('jobGuardUser'); 
                    window.location.href = '/'; 
                }); 
            } 
        });
    }

    if(deleteBtn) {
        deleteBtn.addEventListener('click', () => { 
            if(confirm("PERMANENTLY DELETE ACCOUNT?")) { 
                fetch('/api/delete_account', { method: 'POST' })
                .then(() => { 
                    localStorage.removeItem('jobGuardUser'); 
                    window.location.href = '/'; 
                }); 
            } 
        });
    }

    // ==========================================
    // 2. LOGGING SYSTEM
    // ==========================================
    function fetchGlobalSystemLogs() {
        const logList = document.querySelector('.log-list');
        if(!logList) return;
        
        fetch('/api/system_logs')
        .then(res => res.json())
        .then(logs => { 
            if(logs.length > 0) { 
                logList.innerHTML = ""; 
                logs.forEach(log => logList.appendChild(createLogItem(log))); 
            } 
        });
    }

    function createLogItem(logText) {
        const li = document.createElement('li');
        li.style.fontFamily = "monospace"; 
        li.style.fontSize = "0.75rem"; 
        li.style.listStyle = "none"; 
        li.style.marginBottom = "4px"; 
        li.style.borderBottom = "1px solid rgba(255,255,255,0.05)"; 
        li.style.wordBreak = "break-all";

        if (logText.includes("[ERROR]") || logText.includes("BLOCK")) li.style.color = "var(--neon-pink)";
        else if (logText.includes("[WARN]")) li.style.color = "var(--neon-yellow)";
        else if (logText.includes("[SUCCESS]") || logText.includes("[INIT]")) li.style.color = "var(--neon-green)";
        else if (logText.includes("[AI]")) li.style.color = "var(--neon-cyan)";
        else li.style.color = "#888";
        
        li.innerText = logText;
        return li;
    }

    function updateLogs(logs) {
        const logList = document.querySelector('.log-list');
        if(!logList) return;
        logList.innerHTML = ""; 
        logs.forEach(log => logList.appendChild(createLogItem(log)));
    }

    const copyBtn = document.getElementById('copyLogsBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => { 
            const list = document.querySelector('.log-list'); 
            if (!list) return; 
            let txt = ""; 
            list.querySelectorAll('li').forEach(li => txt += li.innerText + "\n"); 
            navigator.clipboard.writeText(txt); 
        });
    }

    // ==========================================
    // 3. UI EFFECTS (MATRIX & TYPING)
    // ==========================================
    const canvas = document.getElementById('matrixCanvas');
    if (canvas) { 
        const ctx = canvas.getContext('2d'); 
        const alphabet = '01'; 
        const fontSize = 14; 
        let columns = 0, drops = []; 

        function initMatrix() { 
            canvas.width = window.innerWidth; 
            canvas.height = window.innerHeight; 
            columns = Math.floor(canvas.width / fontSize); 
            drops = [];
            for(let x = 0; x < columns; x++) drops[x] = Math.floor(Math.random() * -100);
        } 

        const draw = () => { 
            ctx.fillStyle = 'rgba(5, 8, 10, 0.1)'; 
            ctx.fillRect(0, 0, canvas.width, canvas.height); 
            ctx.font = fontSize + 'px monospace'; 
            
            for(let i = 0; i < drops.length; i++) { 
                ctx.fillStyle = Math.random() > 0.9 ? '#00f3ff' : '#00ff9d'; 
                ctx.fillText(alphabet.charAt(Math.floor(Math.random() * alphabet.length)), i*fontSize, drops[i]*fontSize); 
                if(drops[i]*fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0; 
                drops[i]++; 
            } 
        }; 

        initMatrix(); 
        setInterval(draw, 33); 
        window.addEventListener('resize', initMatrix); 
    }

    const jobInput = document.getElementById('jobDescription');
    const charCount = document.getElementById('charCount');
    const scanStatus = document.getElementById('scanStatus');
    const analyzeBtn = document.getElementById('analyzeBtn');
    let typingTimer;

    if (jobInput) {
        jobInput.addEventListener('input', function() { 
            const text = this.value;
            if(charCount) charCount.innerText = `BUFFER: ${text.length} chars`;
            
            clearTimeout(typingTimer);
            resetScanUI(); 
            
            scanStatus.innerText = ">> RECEIVING DATA...";
            scanStatus.style.color = "var(--neon-cyan)"; 
            scanStatus.classList.add('blink');
            
            typingTimer = setTimeout(() => { 
                scanStatus.classList.remove('blink'); 
                scanStatus.innerText = text.length > 0 ? ">> SIGNAL STANDBY" : ">> IDLE"; 
                scanStatus.style.color = text.length > 0 ? "var(--neon-green)" : "#666"; 
            }, 800);
        });
    }

    function resetScanUI() {
        if(!analyzeBtn) return;
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span>INITIALIZE SCAN</span>';
        analyzeBtn.style.border = "1px solid var(--neon-cyan)";
        analyzeBtn.style.backgroundColor = "transparent";
        analyzeBtn.style.color = "var(--neon-cyan)";
        analyzeBtn.style.boxShadow = "none";
        analyzeBtn.style.opacity = "1";
    }

    function formatText(text) { 
        if (!text) return ""; 
        return text.replace(/\*\*(.*?)\*\*/g, '<strong style="color:#fff; font-weight:bold;">$1</strong>'); 
    }

    const resultContainer = document.getElementById('resultContainer');
    const loader = document.getElementById('loader');

    // ==========================================
    // 4. SCAN LOGIC
    // ==========================================
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            const text = jobInput.value.trim();
            if (text.length < 5) { alert("DATA EMPTY"); return; }

            // VISUAL STATE: SCANNING
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span>SCANNING...</span>';
            analyzeBtn.style.color = "var(--neon-cyan)";
            analyzeBtn.style.borderColor = "var(--neon-cyan)";
            analyzeBtn.style.boxShadow = "0 0 15px var(--neon-cyan)"; 
            analyzeBtn.style.backgroundColor = "rgba(0, 243, 255, 0.1)"; 
            analyzeBtn.style.opacity = "1";
            
            scanStatus.innerText = ">> ANALYZING PACKETS...";
            scanStatus.style.color = "var(--neon-cyan)"; 
            
            resultContainer.classList.add('hidden');
            loader.classList.remove('hidden');
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            })
            .then(res => res.json())
            .then(data => {
                loader.classList.add('hidden');
                if (data.system_logs) updateLogs(data.system_logs);
                if (data.error) throw new Error(data.error);
                displayResult(data);
            })
            .catch(err => {
                loader.classList.add('hidden');
                resetScanUI();
                alert("SYSTEM ERROR: " + err.message);
            });
        });
    }

    function displayResult(data) {
        resultContainer.classList.remove('hidden');
        const prob = data.fraud_probability;
        const root = document.documentElement;
        let uiColor, titleText, rgbaColor, statusText;

        // ==========================================
        // ⚡ VERDICT LOGIC (Aligned with Backend)
        // ==========================================
        if (data.is_gibberish) {
            uiColor = "var(--neon-yellow)"; 
            titleText = "LANGUAGE ERROR"; 
            rgbaColor = "rgba(255, 255, 0, 0.1)"; 
            statusText = ">> INVALID DATA";
            
        } else if (prob <= 35) {
            // GREEN: System Clean (0-35%)
            uiColor = "var(--neon-green)"; 
            titleText = "SYSTEM CLEAN"; 
            rgbaColor = "rgba(0, 255, 157, 0.1)"; 
            statusText = ">> CLEAN SIGNAL";
            
        } else if (prob <= 55) {
            // YELLOW: Review Required (36-55%)
            uiColor = "#ffff00"; 
            titleText = "REVIEW REQUIRED"; 
            rgbaColor = "rgba(255, 255, 0, 0.1)"; 
            statusText = ">> SUSPICIOUS ACTIVITY";
            
        } else if (prob < 85) {
            // ORANGE: High Risk (56-84%)
            uiColor = "var(--neon-orange)"; 
            titleText = "HIGH RISK"; 
            rgbaColor = "rgba(255, 165, 0, 0.1)"; 
            statusText = ">> THREAT DETECTED";
            
        } else {
            // RED: Critical (85%+)
            uiColor = "var(--neon-pink)"; 
            titleText = "CRITICAL THREAT"; 
            rgbaColor = "rgba(255, 0, 153, 0.1)"; 
            statusText = ">> MALICIOUS PATTERN";
        }

        // UPDATE UI
        scanStatus.innerText = statusText;
        scanStatus.style.color = uiColor;
        root.style.setProperty('--state-color', uiColor);
        document.getElementById('verdictTitle').innerText = titleText;
        document.getElementById('verdictTitle').style.color = uiColor;

        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `<span>SCAN COMPLETE</span>`;
        analyzeBtn.style.border = `2px solid ${uiColor}`;
        analyzeBtn.style.backgroundColor = rgbaColor;
        analyzeBtn.style.color = uiColor;
        analyzeBtn.style.boxShadow = `0 0 15px ${uiColor}`;

        const fill = document.getElementById('confidenceFill');
        fill.style.width = '0%'; fill.style.backgroundColor = uiColor;
        
        // Update Score Text
        const score = document.getElementById('confidenceScore');
        score.innerText = prob + '%'; 
        score.style.color = uiColor;
        
        setTimeout(() => { fill.style.width = prob + '%'; }, 100);

        const msgBox = document.getElementById('verdictMessage');
        msgBox.innerHTML = ""; 

        // 1. ADVISORY NOTES (With Overlap Fix)
        if (data.advisory && data.advisory.length > 0) {
            const advDiv = document.createElement('div');
            
            // ⚡ FIX: Add Top Margin to prevent overlap with Percentage Bar
            advDiv.style.marginTop = "25px";
             
            advDiv.style.borderLeft = "3px solid var(--neon-cyan)"; 
            advDiv.style.background = "rgba(0, 243, 255, 0.05)"; 
            advDiv.style.padding = "10px"; 
            advDiv.style.marginBottom = "15px";
            
            let advHtml = `<h4 style="margin:0 0 5px 0; color:var(--neon-cyan); font-size:0.9rem;">ℹ️ ADVISORY NOTES</h4><ul style="margin:0; padding-left:20px; color:#ccc;">`;
            data.advisory.forEach(a => advHtml += `<li>${formatText(a.replace('ℹ️', '').trim())}</li>`);
            advHtml += `</ul>`; 
            advDiv.innerHTML = advHtml; 
            msgBox.appendChild(advDiv);
        }

        // 2. REASONS (Human/AI)
        if (data.reasons && data.reasons.length > 0) {
            const ul = document.createElement('ul'); 
            ul.style.listStyle = "none"; 
            ul.style.padding = "0";
            data.reasons.forEach(r => { 
                const li = document.createElement('li'); 
                li.innerHTML = `> ${formatText(r)}`; 
                li.style.color = "rgba(255,255,255,0.9)"; 
                li.style.marginBottom = "8px"; 
                ul.appendChild(li); 
            });
            msgBox.appendChild(ul);
        }

        // 3. TECHNICAL DETAILS (Anomaly & XAI)
        if ((data.anomaly_analysis && data.anomaly_analysis.length > 0) || (data.xai_insights && data.xai_insights.length > 0)) {
            const techDiv = document.createElement('div'); 
            techDiv.style.marginTop = "20px"; 
            techDiv.style.paddingTop = "10px"; 
            techDiv.style.borderTop = "1px dashed #444"; 
            techDiv.style.fontSize = "0.8rem"; 
            techDiv.style.fontFamily = "monospace"; 
            techDiv.style.color = "#888";
            
            let techHtml = "";
            
            if(data.anomaly_analysis && data.anomaly_analysis.length > 0) { 
                techHtml += `<div style="color:var(--neon-yellow); margin-bottom:5px;">⚠️ STRUCTURAL WARNINGS:</div>`; 
                data.anomaly_analysis.forEach(a => techHtml += `<div>- ${a}</div>`); 
            }
            
            if(data.xai_insights && data.xai_insights.length > 0) { 
                techHtml += `<div style="color:var(--neon-purple); margin-top:10px; margin-bottom:5px;">🧠 NEURAL WEIGHTS (LIME):</div>`; 
                techHtml += `<div style="display:flex; flex-wrap:wrap; gap:10px;">`; 
                data.xai_insights.forEach(x => techHtml += `<span style="background:rgba(255,255,255,0.05); padding:2px 6px; border-radius:4px;">${formatText(x)}</span>`); 
                techHtml += `</div>`; 
            }
            
            techDiv.innerHTML = techHtml; 
            msgBox.appendChild(techDiv);
        }
    }
});
