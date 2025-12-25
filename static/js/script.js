document.addEventListener('DOMContentLoaded', () => {

    // --- 1. AUTH & USER IDENTITY ---
    const profileNameLabel = document.getElementById('profileName');
    const logoutBtn = document.getElementById('logoutBtn');
    const deleteBtn = document.getElementById('deleteAccountBtn');
    const logsPanel = document.querySelector('.live-logs');
    
    // Get User
    const storedUser = localStorage.getItem('jobGuardUser') || 'UNKNOWN';
    
    // Display Operator Name
    if(profileNameLabel) profileNameLabel.innerText = `OPERATOR: ${storedUser.toUpperCase()}`;

    // --- ADMIN PRIVILEGE CHECK (MOBILE) ---
    // Rule: Only 'Yoge' can see logs on Mobile. Others see nothing.
    if (window.innerWidth < 900) {
        if (storedUser === 'Yoge') {
            // Force show logs for Admin
            if(logsPanel) {
                logsPanel.style.display = 'flex'; 
                logsPanel.style.borderTop = '1px solid var(--neon-cyan)'; // Add separator
                logsPanel.style.marginTop = '20px';
            }
        } else {
            // Ensure hidden for everyone else
            if(logsPanel) logsPanel.style.display = 'none';
        }
    }

    // Logout Logic
    if(logoutBtn) logoutBtn.addEventListener('click', () => {
        if(confirm("TERMINATE SESSION?")) fetch('/api/logout', { method: 'POST' }).then(() => { localStorage.removeItem('jobGuardUser'); window.location.href = '/'; });
    });

    // Delete Account Logic
    if(deleteBtn) deleteBtn.addEventListener('click', () => {
        if(confirm("DELETE ACCOUNT?")) fetch('/api/delete_account', { method: 'POST' }).then(() => { localStorage.removeItem('jobGuardUser'); window.location.href = '/'; });
    });

    // --- 2. TYPING STATUS (COLORS) ---
    const jobInput = document.getElementById('jobDescription');
    const charCount = document.getElementById('charCount');
    const scanStatus = document.getElementById('scanStatus');
    let typingTimer;

    if (jobInput) {
        jobInput.addEventListener('input', function() { 
            charCount.innerText = `BUFFER: ${this.value.length}`;
            clearTimeout(typingTimer);
            
            scanStatus.innerText = ">> RECEIVING DATA...";
            scanStatus.style.color = "#fff"; 
            scanStatus.classList.add('blink');

            typingTimer = setTimeout(() => {
                scanStatus.classList.remove('blink');
                if(this.value.length > 0) {
                    scanStatus.innerText = ">> SIGNAL STANDBY";
                    scanStatus.style.color = "var(--neon-green)";
                } else {
                    scanStatus.innerText = ">> IDLE";
                    scanStatus.style.color = "#666";
                }
            }, 800);
        });
    }

    // --- 3. COPY LOGS FUNCTION ---
    const copyBtn = document.getElementById('copyLogsBtn');
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const logs = document.querySelectorAll('.log-list li');
            if (logs.length === 0) return;
            let logText = "--- SYSTEM LOGS ---\n";
            logs.forEach(li => logText += li.innerText + "\n");
            navigator.clipboard.writeText(logText).then(() => {
                copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
                setTimeout(() => copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>', 2000);
            });
        });
    }

    // --- 4. MATRIX BACKGROUND ---
    const canvas = document.getElementById('matrixCanvas');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth; canvas.height = window.innerHeight;
        const alphabet = '01'; const fontSize = 14;
        const columns = canvas.width / fontSize;
        const drops = []; for(let x = 0; x < columns; x++) drops[x] = 1;
        const draw = () => {
            ctx.fillStyle = 'rgba(5, 8, 10, 0.05)'; ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.font = fontSize + 'px monospace';
            for(let i = 0; i < drops.length; i++) {
                ctx.fillStyle = Math.random() > 0.9 ? '#00f3ff' : '#00ff9d';
                ctx.fillText(alphabet.charAt(Math.floor(Math.random() * alphabet.length)), i*fontSize, drops[i]*fontSize);
                if(drops[i]*fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
                drops[i]++;
            }
        };
        setInterval(draw, 33);
        window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });
    }

    // --- 5. ANALYZE & PREDICT ---
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const loader = document.getElementById('loader');
    const logList = document.querySelector('.log-list');

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            const text = jobInput.value.trim();
            if (text.length < 5) { alert("DATA EMPTY"); return; }

            analyzeBtn.disabled = true;
            resultContainer.classList.add('hidden');
            loader.classList.remove('hidden');
            
            scanStatus.innerText = ">> ESTABLISHING UPLINK...";
            scanStatus.style.color = "var(--neon-cyan)";
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                loader.classList.add('hidden');
                analyzeBtn.disabled = false;
                
                if (data.system_logs && logList) updateLogs(data.system_logs);
                
                if (data.error) { 
                    alert("ERROR: " + data.error); 
                    scanStatus.innerText = ">> CONNECTION ERROR";
                    scanStatus.style.color = "var(--neon-pink)";
                    return; 
                }
                displayResult(data);
            })
            .catch(error => {
                loader.classList.add('hidden');
                analyzeBtn.disabled = false;
                alert("CONNECTION ERROR");
            });
        });
    }

    function updateLogs(logs) {
        if(!logList) return;
        logList.innerHTML = "";
        logs.reverse().forEach(log => {
            const li = document.createElement('li');
            li.style.fontFamily = "monospace"; li.style.fontSize = "0.75rem";
            if (log.includes("ERROR")) li.style.color = "var(--neon-pink)";
            else if (log.includes("WARN")) li.style.color = "var(--neon-yellow)";
            else if (log.includes("SUCCESS")) li.style.color = "var(--neon-green)";
            else li.style.color = "#888";
            li.innerText = log;
            logList.appendChild(li);
        });
    }

    function displayResult(data) {
        resultContainer.classList.remove('hidden');
        const prob = data.fraud_probability;
        const root = document.documentElement;

        let uiColor, titleText;
        if (data.is_gibberish) {
            uiColor = "var(--neon-yellow)"; titleText = "UNKNOWN DATA"; 
            scanStatus.innerText = ">> LANGUAGE ERROR";
        } else if (prob < 20) {
            uiColor = "var(--neon-green)"; titleText = "SYSTEM CLEAN"; 
            scanStatus.innerText = ">> SCAN COMPLETE";
        } else if (prob < 80) {
            uiColor = "var(--neon-orange)"; titleText = "HIGH RISK"; 
            scanStatus.innerText = ">> THREAT DETECTED";
        } else {
            uiColor = "var(--neon-pink)"; titleText = "CRITICAL THREAT"; 
            scanStatus.innerText = ">> THREAT DETECTED";
        }

        scanStatus.style.color = uiColor;
        root.style.setProperty('--state-color', uiColor);
        document.getElementById('verdictTitle').innerText = titleText;
        
        const fill = document.getElementById('confidenceFill');
        fill.style.width = '0%'; fill.style.backgroundColor = uiColor;
        setTimeout(() => { 
            fill.style.width = prob + '%'; 
            document.getElementById('confidenceScore').innerText = prob + '%';
            document.getElementById('confidenceScore').style.color = uiColor;
        }, 100);

        const msgBox = document.getElementById('verdictMessage');
        msgBox.innerHTML = "";
        if (data.reasons) {
            const ul = document.createElement("ul");
            ul.style.paddingLeft = "0"; ul.style.listStyle = "none";
            data.reasons.forEach(r => {
                const li = document.createElement("li");
                li.innerHTML = `> ${r.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}`;
                li.style.color = "rgba(255,255,255,0.9)"; li.style.marginBottom = "5px";
                ul.appendChild(li);
            });
            msgBox.appendChild(ul);
        }
    }
});
