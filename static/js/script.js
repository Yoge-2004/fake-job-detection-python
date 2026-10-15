document.addEventListener('DOMContentLoaded', () => {

    const ORIGINAL_TITLE = document.title || "JobGuard AI :: Neural Defense Grid";
    let isScanActive = false;
    let scanFinishedWhileHidden = false;
    let activePollInterval = null;

    // ==========================================
    // 1. THEME INITIALIZATION & TOGGLE
    // ==========================================
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const savedTheme = localStorage.getItem('jobGuardTheme') || 'dark';

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('jobGuardTheme', theme);
        if (themeToggleBtn) {
            const icon = themeToggleBtn.querySelector('i');
            if (icon) {
                if (theme === 'light') {
                    icon.className = 'fa-solid fa-moon';
                    themeToggleBtn.title = "Switch to Dark Theme";
                } else {
                    icon.className = 'fa-solid fa-sun';
                    themeToggleBtn.title = "Switch to Light Theme";
                }
            }
        }
    }

    applyTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
            showCyberToast(`INTERFACE THEME: ${newTheme.toUpperCase()}`, "info", 1800);
        });
    }

    // ==========================================
    // 2. IN-APP CYBERPUNK MODAL & TOAST SYSTEM
    // ==========================================
    const modalOverlay = document.getElementById('cyberModal');
    const modalTitle = document.getElementById('cyberModalTitle');
    const modalCloseBtn = document.getElementById('cyberModalClose');
    const modalIcon = document.getElementById('cyberModalIcon');
    const modalMessage = document.getElementById('cyberModalMessage');
    const modalCancelBtn = document.getElementById('cyberModalCancel');
    const modalConfirmBtn = document.getElementById('cyberModalConfirm');
    const modalBox = modalOverlay ? modalOverlay.querySelector('.cyber-modal-box') : null;
    const toastContainer = document.getElementById('cyberToastContainer');

    let currentModalConfirmHandler = null;
    let currentModalCancelHandler = null;

    function showCyberToast(message, type = 'info', duration = 3500) {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `cyber-toast toast-${type}`;

        let iconClass = 'fa-solid fa-circle-info';
        if (type === 'success') iconClass = 'fa-solid fa-circle-check';
        else if (type === 'warning') iconClass = 'fa-solid fa-triangle-exclamation';
        else if (type === 'error' || type === 'danger') iconClass = 'fa-solid fa-skull-crossbones';

        toast.innerHTML = `<i class="${iconClass}"></i><span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-fadeout');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    function closeCyberModal() {
        if (!modalOverlay) return;
        modalOverlay.classList.add('hidden');
        if (modalBox) {
            modalBox.classList.remove('modal-danger', 'modal-warning');
        }
        currentModalConfirmHandler = null;
        currentModalCancelHandler = null;
    }

    function showCyberModal({
        title = "// SYSTEM PROMPT",
        iconHtml = '<i class="fa-solid fa-shield-halved" style="color:var(--neon-cyan)"></i>',
        message = "",
        confirmText = "CONFIRM",
        cancelText = "ABORT",
        type = "info",
        singleButton = false,
        onConfirm = null,
        onCancel = null
    }) {
        if (!modalOverlay) return;

        modalTitle.innerText = title;
        modalIcon.innerHTML = iconHtml;
        modalMessage.innerHTML = message;

        if (modalBox) {
            modalBox.classList.remove('modal-danger', 'modal-warning');
            if (type === 'danger') modalBox.classList.add('modal-danger');
            else if (type === 'warning') modalBox.classList.add('modal-warning');
        }

        modalConfirmBtn.innerText = confirmText;
        modalConfirmBtn.className = "modal-btn modal-btn-confirm";
        if (type === 'danger') {
            modalConfirmBtn.className = "modal-btn modal-btn-danger";
        }

        if (singleButton) {
            modalCancelBtn.style.display = "none";
        } else {
            modalCancelBtn.style.display = "inline-block";
            modalCancelBtn.innerText = cancelText;
        }

        currentModalConfirmHandler = onConfirm;
        currentModalCancelHandler = onCancel;

        modalOverlay.classList.remove('hidden');
    }

    function showCyberAlert(title, message, type = 'info') {
        let iconHtml = '<i class="fa-solid fa-circle-info" style="color:var(--neon-cyan)"></i>';
        if (type === 'danger' || type === 'error') {
            iconHtml = '<i class="fa-solid fa-triangle-exclamation" style="color:var(--neon-pink)"></i>';
        } else if (type === 'warning') {
            iconHtml = '<i class="fa-solid fa-triangle-exclamation" style="color:var(--neon-orange)"></i>';
        } else if (type === 'success') {
            iconHtml = '<i class="fa-solid fa-circle-check" style="color:var(--neon-green)"></i>';
        }

        showCyberModal({
            title: title.startsWith("//") ? title : `// ${title}`,
            iconHtml,
            message,
            confirmText: "ACKNOWLEDGE",
            type,
            singleButton: true,
            onConfirm: () => {}
        });
    }

    if (modalConfirmBtn) {
        modalConfirmBtn.addEventListener('click', () => {
            const cb = currentModalConfirmHandler;
            closeCyberModal();
            if (cb) cb();
        });
    }

    if (modalCancelBtn) {
        modalCancelBtn.addEventListener('click', () => {
            const cb = currentModalCancelHandler;
            closeCyberModal();
            if (cb) cb();
        });
    }

    if (modalCloseBtn) {
        modalCloseBtn.addEventListener('click', () => {
            const cb = currentModalCancelHandler;
            closeCyberModal();
            if (cb) cb();
        });
    }

    if (modalOverlay) {
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                const cb = currentModalCancelHandler;
                closeCyberModal();
                if (cb) cb();
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modalOverlay && !modalOverlay.classList.contains('hidden')) {
            const cb = currentModalCancelHandler;
            closeCyberModal();
            if (cb) cb();
        }
    });

    // ==========================================
    // 3. AUTH & USER IDENTITY
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

    const storedUser = localStorage.getItem('jobGuardUser');
    if (storedUser) updateIdentityUI(storedUser);

    function updateIdentityUI(username) {
        if (!username) {
            if (profileNameLabel) profileNameLabel.innerText = "OPERATOR: GUEST";
            return;
        }
        if (profileNameLabel) profileNameLabel.innerText = `OPERATOR: ${username.toUpperCase()}`;

        if (username === 'Yoge') {
            if (logsPanel) logsPanel.style.display = 'flex';
            fetchGlobalSystemLogs();
        }
    }

    // Custom App Modal for Logout
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            showCyberModal({
                title: "// TERMINATE_SESSION_CONFIRMATION",
                iconHtml: '<i class="fa-solid fa-power-off" style="color:var(--neon-cyan)"></i>',
                message: "Are you sure you want to terminate operator session and disconnect from the Neural Defense Grid?",
                confirmText: "DISCONNECT",
                cancelText: "STAY CONNECTED",
                type: "info",
                onConfirm: () => {
                    fetch('/api/logout', { method: 'POST' })
                        .then(() => {
                            localStorage.removeItem('jobGuardUser');
                            localStorage.removeItem('jobGuardPendingScan');
                            sessionStorage.removeItem('jobGuardLastResult');
                            showCyberToast("SESSION TERMINATED", "info", 1500);
                            setTimeout(() => window.location.href = '/', 500);
                        })
                        .catch(() => {
                            window.location.href = '/';
                        });
                }
            });
        });
    }

    // Custom App Modal for Delete Account
    if (deleteBtn) {
        deleteBtn.addEventListener('click', () => {
            showCyberModal({
                title: "// CRITICAL_PURGE_PROTOCOL",
                iconHtml: '<i class="fa-solid fa-triangle-exclamation" style="color:var(--neon-pink)"></i>',
                message: "PERMANENT IDENTITY PURGE: You are about to permanently delete your operator credentials from the neural registry. This action is irreversible. Confirm account deletion?",
                confirmText: "PURGE ACCOUNT",
                cancelText: "ABORT",
                type: "danger",
                onConfirm: () => {
                    fetch('/api/delete_account', { method: 'POST' })
                        .then(res => res.json())
                        .then(data => {
                            if (data.success) {
                                localStorage.removeItem('jobGuardUser');
                                localStorage.removeItem('jobGuardPendingScan');
                                sessionStorage.removeItem('jobGuardLastResult');
                                showCyberToast("OPERATOR RECORD PURGED", "success", 1500);
                                setTimeout(() => window.location.href = '/', 600);
                            } else {
                                showCyberAlert("PURGE FAILED", data.error || "Unable to delete account at this time.", "danger");
                            }
                        })
                        .catch(err => {
                            showCyberAlert("SYSTEM ERROR", "Account purge failed: " + err.message, "danger");
                        });
                }
            });
        });
    }

    // ==========================================
    // 4. LOGGING SYSTEM & COPY BUTTON (VISUAL FEEDBACK)
    // ==========================================
    function fetchGlobalSystemLogs() {
        const logList = document.querySelector('.log-list');
        if (!logList) return;

        fetch('/api/system_logs')
            .then(res => res.json())
            .then(logs => {
                if (logs.length > 0) {
                    logList.innerHTML = "";
                    logs.forEach(log => logList.appendChild(createLogItem(log)));
                }
            })
            .catch(err => console.warn("Log fetch error", err));
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
        else li.style.color = "var(--text-muted)";

        li.innerText = logText;
        return li;
    }

    function updateLogs(logs) {
        const logList = document.querySelector('.log-list');
        if (!logList) return;
        logList.innerHTML = "";
        logs.forEach(log => logList.appendChild(createLogItem(log)));
    }

    const copyBtn = document.getElementById('copyLogsBtn');
    let copyResetTimer = null;

    function fallbackClipboardCopy(text, onSuccess, onError) {
        try {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.top = '0';
            textarea.style.left = '0';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            const successful = document.execCommand('copy');
            document.body.removeChild(textarea);
            if (successful) onSuccess();
            else onError(new Error("execCommand failed"));
        } catch (err) {
            onError(err);
        }
    }

    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const list = document.querySelector('.log-list');
            if (!list) return;
            const items = list.querySelectorAll('li');
            if (items.length === 0) {
                showCyberToast("BUFFER EMPTY: No logs available to copy", "warning");
                return;
            }

            let txt = "";
            items.forEach(li => txt += li.innerText + "\n");

            const handleSuccess = () => {
                clearTimeout(copyResetTimer);
                copyBtn.classList.add('copy-success');
                copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> <span class="copy-text">COPIED!</span>';
                showCyberToast("SYSTEM LOGS COPIED TO CLIPBOARD", "success", 2500);

                copyResetTimer = setTimeout(() => {
                    copyBtn.classList.remove('copy-success');
                    copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i> <span class="copy-text">COPY</span>';
                }, 2000);
            };

            const handleError = (err) => {
                console.error("Clipboard write error:", err);
                showCyberToast("COPY FAILED: Clipboard permission denied", "error");
            };

            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(txt).then(handleSuccess).catch(() => {
                    fallbackClipboardCopy(txt, handleSuccess, handleError);
                });
            } else {
                fallbackClipboardCopy(txt, handleSuccess, handleError);
            }
        });
    }

    // ==========================================
    // 5. MATRIX RAIN (POWER-SAVING & VISIBILITY AWARE)
    // ==========================================
    const canvas = document.getElementById('matrixCanvas');
    let matrixInterval = null;

    if (canvas) {
        const ctx = canvas.getContext('2d');
        const alphabet = '01';
        const fontSize = 14;
        let columns = 0;
        let drops = [];

        function initMatrix() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            columns = Math.floor(canvas.width / fontSize);
            drops = [];
            for (let x = 0; x < columns; x++) {
                drops[x] = Math.floor(Math.random() * -100);
            }
        }

        const draw = () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            ctx.fillStyle = currentTheme === 'light' ? 'rgba(240, 244, 248, 0.08)' : 'rgba(5, 8, 10, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < drops.length; i++) {
                if (currentTheme === 'light') {
                    ctx.fillStyle = Math.random() > 0.85 ? '#008fa8' : '#008751';
                } else {
                    ctx.fillStyle = Math.random() > 0.9 ? '#00f3ff' : '#00ff9d';
                }
                ctx.fillText(alphabet.charAt(Math.floor(Math.random() * alphabet.length)), i * fontSize, drops[i] * fontSize);
                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
                drops[i]++;
            }
        };

        function startMatrix() {
            if (!matrixInterval) {
                matrixInterval = setInterval(draw, 33);
            }
        }

        function stopMatrix() {
            if (matrixInterval) {
                clearInterval(matrixInterval);
                matrixInterval = null;
            }
        }

        initMatrix();
        startMatrix();
        window.addEventListener('resize', initMatrix);

        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                stopMatrix();
            } else {
                startMatrix();
            }
        });
    }

    // ==========================================
    // 6. INPUT BUFFER & UI EFFECT
    // ==========================================
    const jobInput = document.getElementById('jobDescription');
    const textareaContainer = document.querySelector('.textarea-container');
    const charCount = document.getElementById('charCount');
    const scanStatus = document.getElementById('scanStatus');
    const analyzeBtn = document.getElementById('analyzeBtn');
    let typingTimer;

    if (jobInput) {
        jobInput.addEventListener('input', function() {
            const text = this.value;
            if (charCount) charCount.innerText = `BUFFER: ${text.length} chars`;

            if (textareaContainer && textareaContainer.classList.contains('input-error')) {
                textareaContainer.classList.remove('input-error');
            }

            clearTimeout(typingTimer);
            if (!isScanActive) resetScanUI();

            scanStatus.innerText = ">> RECEIVING DATA...";
            scanStatus.style.color = "var(--neon-cyan)";
            scanStatus.classList.add('blink');

            typingTimer = setTimeout(() => {
                scanStatus.classList.remove('blink');
                scanStatus.innerText = text.length > 0 ? ">> SIGNAL STANDBY" : ">> IDLE";
                scanStatus.style.color = text.length > 0 ? "var(--neon-green)" : "var(--text-muted)";
            }, 800);
        });
    }

    function resetScanUI() {
        if (!analyzeBtn) return;
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
        return text.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--text-primary); font-weight:bold;">$1</strong>');
    }

    const resultContainer = document.getElementById('resultContainer');
    const loader = document.getElementById('loader');
    const loaderText = document.getElementById('loaderStatusText');

    // ==========================================
    // 7. ASYNCHRONOUS SCAN & RELOAD RESILIENCE
    // ==========================================
    function setScanLoadingUI(statusMessage = "DECRYPTING & RUNNING MULTI-MODAL INFERENCE...") {
        isScanActive = true;
        if (analyzeBtn) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<span>SCANNING...</span>';
            analyzeBtn.style.color = "var(--neon-cyan)";
            analyzeBtn.style.borderColor = "var(--neon-cyan)";
            analyzeBtn.style.boxShadow = "0 0 15px var(--neon-cyan)";
            analyzeBtn.style.backgroundColor = "rgba(0, 243, 255, 0.1)";
            analyzeBtn.style.opacity = "1";
        }
        if (scanStatus) {
            scanStatus.innerText = ">> ANALYZING PACKETS (BERT + AUTOENCODER)...";
            scanStatus.style.color = "var(--neon-cyan)";
        }
        if (resultContainer) resultContainer.classList.add('hidden');
        if (loader) loader.classList.remove('hidden');
        if (loaderText) loaderText.innerText = statusMessage;
    }

    function stopScanLoadingUI() {
        isScanActive = false;
        if (activePollInterval) {
            clearInterval(activePollInterval);
            activePollInterval = null;
        }
        if (loader) loader.classList.add('hidden');
        resetScanUI();
    }

    function pollScanStatus(jobId, originalText) {
        if (activePollInterval) clearInterval(activePollInterval);

        activePollInterval = setInterval(() => {
            fetch(`/api/scan_status/${jobId}`)
                .then(res => {
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    return res.json();
                })
                .then(data => {
                    if (data.status === 'completed' && data.result) {
                        stopScanLoadingUI();
                        localStorage.removeItem('jobGuardPendingScan');

                        try {
                            sessionStorage.setItem('jobGuardLastResult', JSON.stringify({
                                inputText: originalText,
                                result: data.result,
                                timestamp: Date.now()
                            }));
                        } catch (e) {
                            console.warn("Storage error", e);
                        }

                        if (data.result.system_logs) updateLogs(data.result.system_logs);
                        displayResult(data.result);

                        if (document.hidden) {
                            scanFinishedWhileHidden = true;
                            document.title = "[!] SCAN COMPLETE - JobGuard AI";
                        } else {
                            showCyberToast("ANALYSIS COMPLETE: Threat assessment ready", "success", 3000);
                            if (window.innerWidth <= 900 && resultContainer) {
                                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                            }
                        }
                    } else if (data.status === 'error') {
                        stopScanLoadingUI();
                        localStorage.removeItem('jobGuardPendingScan');
                        showCyberAlert("SYSTEM ERROR", "Inference error: " + (data.error || "Unknown"), "danger");
                        showCyberToast("SCAN FAILED: " + (data.error || "Execution fault"), "error");
                    }
                })
                .catch(err => {
                    console.warn("Poll status check:", err);
                });
        }, 850);
    }

    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            const text = jobInput.value.trim();
            if (text.length < 5) {
                if (textareaContainer) {
                    textareaContainer.classList.add('input-error');
                    setTimeout(() => textareaContainer.classList.remove('input-error'), 1800);
                }
                jobInput.focus();
                showCyberToast("BUFFER EMPTY: Please enter job description data before scanning.", "warning");
                return;
            }

            setScanLoadingUI("DECRYPTING & RUNNING MULTI-MODAL INFERENCE...");
            scanFinishedWhileHidden = false;

            // Submit async scan request
            fetch('/api/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            })
                .then(res => {
                    if (!res.ok) {
                        return res.json().then(data => { throw new Error(data.error || `HTTP ${res.status}`); });
                    }
                    return res.json();
                })
                .then(data => {
                    // Record pending scan ticket into localStorage for reload / suspend resilience
                    localStorage.setItem('jobGuardPendingScan', JSON.stringify({
                        jobId: data.job_id,
                        text: text,
                        startedAt: Date.now()
                    }));

                    if (data.status === 'completed' && data.result) {
                        // Returned from cache instantly
                        stopScanLoadingUI();
                        localStorage.removeItem('jobGuardPendingScan');
                        sessionStorage.setItem('jobGuardLastResult', JSON.stringify({
                            inputText: text,
                            result: data.result,
                            timestamp: Date.now()
                        }));
                        if (data.result.system_logs) updateLogs(data.result.system_logs);
                        displayResult(data.result);
                        showCyberToast("ANALYSIS COMPLETE (CACHED)", "success", 2500);
                    } else {
                        // Processing in background: poll for results
                        pollScanStatus(data.job_id, text);
                    }
                })
                .catch(err => {
                    stopScanLoadingUI();
                    localStorage.removeItem('jobGuardPendingScan');
                    showCyberAlert("SYSTEM ERROR", "Scan dispatch failed: " + err.message, "danger");
                    showCyberToast("SCAN ERROR: " + err.message, "error");
                });
        });
    }

    // Tab Visibility Handler: Re-trigger animation or alerts if tab was in background
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            document.title = ORIGINAL_TITLE;
            if (scanFinishedWhileHidden) {
                scanFinishedWhileHidden = false;
                showCyberToast("ANALYSIS COMPLETE: Threat assessment ready", "success", 4000);

                const fill = document.getElementById('confidenceFill');
                const score = document.getElementById('confidenceScore');
                if (fill && score && score.innerText) {
                    fill.style.width = score.innerText;
                }

                if (resultContainer && !resultContainer.classList.contains('hidden')) {
                    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }
        }
    });

    // ==========================================
    // 8. DISPLAY RESULT LOGIC
    // ==========================================
    function displayResult(data) {
        if (!resultContainer) return;
        resultContainer.classList.remove('hidden');
        const prob = data.fraud_probability;
        const root = document.documentElement;
        let uiColor, titleText, rgbaColor, statusText;

        if (data.is_gibberish) {
            uiColor = "var(--neon-yellow)";
            titleText = "LANGUAGE ERROR";
            rgbaColor = "rgba(255, 255, 0, 0.1)";
            statusText = ">> INVALID DATA";
        } else if (prob <= 35) {
            uiColor = "var(--neon-green)";
            titleText = "SYSTEM CLEAN";
            rgbaColor = "rgba(0, 255, 157, 0.1)";
            statusText = ">> CLEAN SIGNAL";
        } else if (prob <= 55) {
            uiColor = "var(--neon-yellow)";
            titleText = "REVIEW REQUIRED";
            rgbaColor = "rgba(255, 255, 0, 0.1)";
            statusText = ">> SUSPICIOUS ACTIVITY";
        } else if (prob < 85) {
            uiColor = "var(--neon-orange)";
            titleText = "HIGH RISK";
            rgbaColor = "rgba(255, 165, 0, 0.1)";
            statusText = ">> THREAT DETECTED";
        } else {
            uiColor = "var(--neon-pink)";
            titleText = "CRITICAL THREAT";
            rgbaColor = "rgba(255, 0, 153, 0.1)";
            statusText = ">> MALICIOUS PATTERN";
        }

        if (scanStatus) {
            scanStatus.innerText = statusText;
            scanStatus.style.color = uiColor;
        }
        root.style.setProperty('--state-color', uiColor);

        const verdictTitle = document.getElementById('verdictTitle');
        if (verdictTitle) {
            verdictTitle.innerText = titleText;
            verdictTitle.style.color = uiColor;
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `<span>SCAN COMPLETE</span>`;
            analyzeBtn.style.border = `2px solid ${uiColor}`;
            analyzeBtn.style.backgroundColor = rgbaColor;
            analyzeBtn.style.color = uiColor;
            analyzeBtn.style.boxShadow = `0 0 15px ${uiColor}`;
        }

        const fill = document.getElementById('confidenceFill');
        if (fill) {
            fill.style.width = '0%';
            fill.style.backgroundColor = uiColor;
        }

        const score = document.getElementById('confidenceScore');
        if (score) {
            score.innerText = prob + '%';
            score.style.color = uiColor;
        }

        setTimeout(() => {
            if (fill) fill.style.width = prob + '%';
        }, 80);

        const msgBox = document.getElementById('verdictMessage');
        if (!msgBox) return;
        msgBox.innerHTML = "";

        // 1. ADVISORY NOTES
        if (data.advisory && data.advisory.length > 0) {
            const advDiv = document.createElement('div');
            advDiv.style.marginTop = "25px";
            advDiv.style.borderLeft = "3px solid var(--neon-cyan)";
            advDiv.style.background = "rgba(0, 243, 255, 0.05)";
            advDiv.style.padding = "10px";
            advDiv.style.marginBottom = "15px";

            let advHtml = `<h4 style="margin:0 0 5px 0; color:var(--neon-cyan); font-size:0.9rem;">ℹ️ ADVISORY NOTES</h4><ul style="margin:0; padding-left:20px; color:var(--text-primary);">`;
            data.advisory.forEach(a => advHtml += `<li>${formatText(a.replace('ℹ️', '').trim())}</li>`);
            advHtml += `</ul>`;
            advDiv.innerHTML = advHtml;
            msgBox.appendChild(advDiv);
        }

        // 2. REASONS
        if (data.reasons && data.reasons.length > 0) {
            const ul = document.createElement('ul');
            ul.style.listStyle = "none";
            ul.style.padding = "0";
            data.reasons.forEach(r => {
                const li = document.createElement('li');
                li.innerHTML = `> ${formatText(r)}`;
                li.style.color = "var(--text-primary)";
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
            techDiv.style.borderTop = "1px dashed var(--hud-border)";
            techDiv.style.fontSize = "0.8rem";
            techDiv.style.fontFamily = "monospace";
            techDiv.style.color = "var(--text-muted)";

            let techHtml = "";

            if (data.anomaly_analysis && data.anomaly_analysis.length > 0) {
                techHtml += `<div style="color:var(--neon-yellow); margin-bottom:5px;">⚠️ STRUCTURAL WARNINGS:</div>`;
                data.anomaly_analysis.forEach(a => techHtml += `<div>- ${a}</div>`);
            }

            if (data.xai_insights && data.xai_insights.length > 0) {
                techHtml += `<div style="color:#d870ff; margin-top:10px; margin-bottom:5px;">🧠 NEURAL WEIGHTS (LIME):</div>`;
                techHtml += `<div style="display:flex; flex-wrap:wrap; gap:10px;">`;
                data.xai_insights.forEach(x => techHtml += `<span style="background:rgba(255,255,255,0.08); padding:2px 6px; border-radius:4px;">${formatText(x)}</span>`);
                techHtml += `</div>`;
            }

            techDiv.innerHTML = techHtml;
            msgBox.appendChild(techDiv);
        }
    }

    // ==========================================
    // 9. RESUME PROCESSING AFTER MEMORY SUSPENSION OR RELOAD
    // ==========================================
    try {
        const pendingScan = localStorage.getItem('jobGuardPendingScan');
        if (pendingScan) {
            const parsed = JSON.parse(pendingScan);
            // If the pending scan was initiated within the last 10 minutes
            if (parsed && parsed.jobId && (Date.now() - parsed.startedAt < 600000)) {
                if (jobInput && parsed.text) {
                    jobInput.value = parsed.text;
                    if (charCount) charCount.innerText = `BUFFER: ${parsed.text.length} chars`;
                }
                setScanLoadingUI("RECONNECTED: INFERENCE IN PROGRESS IN BACKEND...");
                showCyberToast("RECONNECTED TO BACKGROUND SCAN", "info", 2500);
                pollScanStatus(parsed.jobId, parsed.text);
            } else {
                localStorage.removeItem('jobGuardPendingScan');
            }
        } else {
            // Restore cached scan result from current session if available
            const cachedSession = sessionStorage.getItem('jobGuardLastResult');
            if (cachedSession) {
                const parsed = JSON.parse(cachedSession);
                if (parsed && parsed.result) {
                    if (jobInput && !jobInput.value && parsed.inputText) {
                        jobInput.value = parsed.inputText;
                        if (charCount) charCount.innerText = `BUFFER: ${parsed.inputText.length} chars`;
                    }
                    displayResult(parsed.result);
                    if (scanStatus) {
                        scanStatus.innerText = ">> BUFFER RESTORED";
                        scanStatus.style.color = "var(--neon-green)";
                    }
                }
            }
        }
    } catch (e) {
        console.warn("Could not check pending/session state", e);
    }
});
