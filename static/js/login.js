document.addEventListener('DOMContentLoaded', () => {

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
    // 2. TOAST NOTIFICATION SYSTEM
    // ==========================================
    const toastContainer = document.getElementById('cyberToastContainer');

    function showCyberToast(message, type = 'info', duration = 3500) {
        if (!toastContainer) return;
        const toast = document.createElement('div');
        toast.className = `cyber-toast toast-${type}`;

        let iconClass = 'fa-solid fa-circle-info';
        if (type === 'success') iconClass = 'fa-solid fa-circle-check';
        else if (type === 'warning') iconClass = 'fa-solid fa-triangle-exclamation';
        else if (type === 'error') iconClass = 'fa-solid fa-skull-crossbones';

        toast.innerHTML = `<i class="${iconClass}"></i><span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-fadeout');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    
    // ==========================================
    // 2.1 OAUTH REDIRECT & ERROR HANDLING
    // ==========================================
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has("error")) {
        const errParam = urlParams.get("error");
        let errMsg = "OAUTH ERROR: Google authentication aborted";
        if (errParam === "oauth_not_configured") {
            errMsg = "OAUTH DISABLED: Google Client ID/Secret not set on server";
        } else if (errParam === "google_missing_email") {
            errMsg = "OAUTH ERROR: Verified email not returned by Google";
        } else if (errParam === "oauth_failed") {
            errMsg = "OAUTH REJECTED: Authorization handshake failed";
        }
        showCyberToast(errMsg, "error", 5000);
        window.history.replaceState({}, document.title, window.location.pathname);
    }

    const googleLoginBtn = document.getElementById("googleLoginBtn");
    const googleSignupBtn = document.getElementById("googleSignupBtn");
    [googleLoginBtn, googleSignupBtn].forEach(btn => {
        if (btn) {
            btn.addEventListener("click", () => {
                showCyberToast("REDIRECTING TO GOOGLE IDENTITY PROVIDER...", "info", 2500);
            });
        }
    });

    // ==========================================
    // 3. MATRIX RAIN (RESPONSIVE & POWER-SAVING)
    // ==========================================
    const canvas = document.getElementById('matrixCanvas');
    let matrixInterval = null;

    if (canvas) {
        const ctx = canvas.getContext('2d');
        const fontSize = 14;
        const alphabet = '01';
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
            ctx.fillStyle = currentTheme === 'light' ? 'rgba(240, 244, 248, 0.08)' : 'rgba(5, 8, 10, 0.05)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.font = fontSize + 'px monospace';

            for (let i = 0; i < drops.length; i++) {
                const text = alphabet.charAt(Math.floor(Math.random() * alphabet.length));
                if (currentTheme === 'light') {
                    ctx.fillStyle = Math.random() > 0.85 ? '#008fa8' : '#008751';
                } else {
                    ctx.fillStyle = Math.random() > 0.9 ? '#00f3ff' : '#00ff9d';
                }
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
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

        // Power-saving Page Visibility handler
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                stopMatrix();
            } else {
                startMatrix();
            }
        });
    }

    // ==========================================
    // 4. UI TOGGLES
    // ==========================================
    const loginBox = document.querySelector('.login-box');
    const signupBox = document.querySelector('.signup-box');

    document.getElementById('showSignup').addEventListener('click', (e) => {
        e.preventDefault();
        loginBox.classList.add('hidden');
        signupBox.classList.remove('hidden');
    });

    document.getElementById('showLogin').addEventListener('click', (e) => {
        e.preventDefault();
        signupBox.classList.add('hidden');
        loginBox.classList.remove('hidden');
    });

    document.querySelectorAll('.password-toggle').forEach(button => {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('.password-field');
            const icon = this.querySelector('i');
            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.replace('fa-eye', 'fa-eye-slash');
            } else {
                input.type = 'password';
                icon.classList.replace('fa-eye-slash', 'fa-eye');
            }
        });
    });

    // ==========================================
    // 5. LOGIN LOGIC
    // ==========================================
    const loginForm = document.getElementById('loginForm');
    const loginUser = document.getElementById('loginUser');
    const loginPass = document.getElementById('loginPass');
    const loginBtn = loginForm.querySelector('button[type="submit"]');
    const btnText = loginBtn.querySelector('.btn-text');

    function resetLoginState() {
        if (loginBtn.disabled || loginBtn.style.borderColor === "var(--neon-pink)") {
            loginBtn.disabled = false;
            loginBtn.style.borderColor = "var(--neon-cyan)";
            loginBtn.style.background = "transparent";
            loginBtn.style.color = "var(--neon-cyan)";
            loginBtn.style.boxShadow = "none";
            btnText.innerText = "INITIATE UPLINK";
            loginBtn.style.cursor = "pointer";
        }
    }

    loginUser.addEventListener('input', resetLoginState);
    loginPass.addEventListener('input', resetLoginState);

    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const user = loginUser.value.trim();
        const pass = loginPass.value;
        const remember = document.getElementById('rememberCheck').checked;

        if (!user || !pass) {
            showCyberToast("AUTHENTICATION ERROR: Input fields required", "warning");
            return;
        }

        btnText.innerText = "AUTHENTICATING...";
        loginBtn.style.background = "rgba(0, 243, 255, 0.1)";
        loginBtn.style.color = "var(--neon-cyan)";
        loginBtn.disabled = true;
        loginBtn.style.cursor = "wait";

        fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass, remember: remember })
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                localStorage.setItem('jobGuardUser', data.username);
                loginBtn.style.borderColor = "var(--neon-green)";
                loginBtn.style.color = "var(--neon-green)";
                btnText.innerText = "ACCESS GRANTED";
                showCyberToast(`WELCOME OPERATOR [${data.username.toUpperCase()}]`, "success", 2000);
                setTimeout(() => window.location.href = '/dashboard', 600);
            } else {
                btnText.innerText = "ACCESS DENIED";
                loginBtn.style.background = "var(--neon-pink)";
                loginBtn.style.borderColor = "var(--neon-pink)";
                loginBtn.style.color = "#fff";
                loginBtn.style.boxShadow = "0 0 15px var(--neon-pink)";
                loginBtn.style.cursor = "not-allowed";
                showCyberToast(data.error || "ACCESS DENIED: Invalid credentials", "error");
            }
        })
        .catch(err => {
            console.error(err);
            btnText.innerText = "SYSTEM FAILURE";
            loginBtn.style.borderColor = "#ffaa00";
            loginBtn.style.color = "#ffaa00";
            loginBtn.disabled = false;
            showCyberToast("NETWORK ERROR: Host connection unreachable", "error");
        });
    });

    // ==========================================
    // 6. SIGNUP LOGIC & VALIDATION WITH CONSTRAINTS
    // ==========================================
    const signupForm = document.getElementById('signupForm');
    const sUser = document.getElementById('signupUser');
    const sEmail = document.getElementById('signupEmail');
    const sPass = document.getElementById('signupPass');
    const regBtn = document.getElementById('regBtn');
    const regBtnText = regBtn.querySelector('.btn-text');

    const strengthStatus = document.getElementById('strengthStatus');
    const strengthFill = document.getElementById('strengthFill');

    const ruleLength = document.getElementById('c-length');
    const ruleUpper = document.getElementById('c-upper');
    const ruleLower = document.getElementById('c-lower');
    const ruleNumber = document.getElementById('c-number');
    const ruleSpecial = document.getElementById('c-special');

    const patterns = {
        user: /^[a-zA-Z0-9_]{3,15}$/,
        email: /^[\w\.\+-]+@[\w\.-]+\.[a-zA-Z]{2,8}$/
    };

    function clearButtonStates() {
        regBtn.classList.remove('btn-loading', 'btn-error', 'btn-success');
    }

    function updateConstraintUI(element, isMet) {
        if (!element) return;
        const icon = element.querySelector('i');
        if (isMet) {
            element.classList.remove('unmet');
            element.classList.add('met');
            if (icon) {
                icon.className = 'fa-solid fa-circle-check';
            }
        } else {
            element.classList.remove('met');
            element.classList.add('unmet');
            if (icon) {
                icon.className = 'fa-solid fa-circle-xmark';
            }
        }
    }

    function evaluatePasswordStrength(password) {
        const checks = {
            length: password.length >= 8,
            upper: /[A-Z]/.test(password),
            lower: /[a-z]/.test(password),
            number: /[0-9]/.test(password),
            special: /[^a-zA-Z0-9]/.test(password)
        };

        updateConstraintUI(ruleLength, checks.length);
        updateConstraintUI(ruleUpper, checks.upper);
        updateConstraintUI(ruleLower, checks.lower);
        updateConstraintUI(ruleNumber, checks.number);
        updateConstraintUI(ruleSpecial, checks.special);

        let metCount = 0;
        if (checks.length) metCount++;
        if (checks.upper) metCount++;
        if (checks.lower) metCount++;
        if (checks.number) metCount++;
        if (checks.special) metCount++;

        if (password.length === 0) {
            strengthStatus.innerText = "WAITING...";
            strengthStatus.style.color = "var(--text-muted)";
            strengthFill.style.width = "0%";
            strengthFill.style.backgroundColor = "var(--text-muted)";
            strengthFill.style.boxShadow = "none";
            return false;
        }

        let percent = 0;
        let label = "";
        let color = "";

        switch (metCount) {
            case 1:
                percent = 20;
                label = "CRITICAL // WEAK";
                color = "var(--neon-pink)";
                break;
            case 2:
                percent = 40;
                label = "LOW // VULNERABLE";
                color = "#ff5533";
                break;
            case 3:
                percent = 60;
                label = "MODERATE // FAIR";
                color = "var(--neon-orange)";
                break;
            case 4:
                percent = 80;
                label = "HIGH // STRONG";
                color = "var(--neon-yellow)";
                break;
            case 5:
                percent = 100;
                label = "CYBER-SECURE // MAXIMUM";
                color = "var(--neon-green)";
                break;
            default:
                percent = 10;
                label = "CRITICAL // WEAK";
                color = "var(--neon-pink)";
        }

        strengthStatus.innerText = label;
        strengthStatus.style.color = color;
        strengthFill.style.width = percent + "%";
        strengthFill.style.backgroundColor = color;
        strengthFill.style.boxShadow = `0 0 10px ${color}`;

        return metCount === 5;
    }

    function validateField(field, regex, msgElement, errorMsg) {
        if (regBtn.classList.contains('btn-error')) {
            clearButtonStates();
            regBtnText.innerText = "REGISTER UNIT";
        }

        if (regex.test(field.value.trim())) {
            field.parentElement.classList.add('valid');
            field.parentElement.classList.remove('invalid');
            msgElement.innerText = ">> VALID";
            msgElement.style.color = "var(--neon-green)";
            return true;
        } else {
            field.parentElement.classList.add('invalid');
            field.parentElement.classList.remove('valid');
            msgElement.innerText = errorMsg;
            msgElement.style.color = "var(--neon-pink)";
            return false;
        }
    }

    function checkFormValidity() {
        const isUserValid = patterns.user.test(sUser.value.trim());
        const isEmailValid = patterns.email.test(sEmail.value.trim());
        const isPassValid = evaluatePasswordStrength(sPass.value);

        if (isUserValid && isEmailValid && isPassValid) {
            regBtn.disabled = false;
            regBtn.style.cursor = "pointer";
            regBtn.style.borderColor = "var(--neon-cyan)";
            regBtn.style.color = "var(--neon-cyan)";
        } else {
            regBtn.disabled = true;
            regBtn.style.cursor = "not-allowed";
            regBtn.style.borderColor = "";
            regBtn.style.color = "";
        }
    }

    if (sUser && sEmail && sPass) {
        sUser.addEventListener('input', () => {
            validateField(sUser, patterns.user, document.getElementById('userMsg'), ">> 3-15 CHARS, ALPHANUMERIC");
            checkFormValidity();
        });
        sEmail.addEventListener('input', () => {
            validateField(sEmail, patterns.email, document.getElementById('emailMsg'), ">> INVALID EMAIL FORMAT");
            checkFormValidity();
        });
        sPass.addEventListener('input', () => {
            const passMsg = document.getElementById('passMsg');
            const passValid = evaluatePasswordStrength(sPass.value);
            if (passValid) {
                sPass.parentElement.classList.add('valid');
                sPass.parentElement.classList.remove('invalid');
                passMsg.innerText = ">> CONSTRAINTS SATISFIED";
                passMsg.style.color = "var(--neon-green)";
            } else {
                sPass.parentElement.classList.add('invalid');
                sPass.parentElement.classList.remove('valid');
                passMsg.innerText = ">> REQUIRES 8+ CHARS, UPPER, LOWER, NUM & SYMBOL";
                passMsg.style.color = "var(--neon-pink)";
            }
            checkFormValidity();
        });
    }

    signupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const user = sUser.value.trim();
        const email = sEmail.value.trim();
        const pass = sPass.value;

        if (!patterns.user.test(user) || !patterns.email.test(email) || !evaluatePasswordStrength(pass)) {
            showCyberToast("VALIDATION FAILURE: Review all requirements", "warning");
            return;
        }

        clearButtonStates();
        regBtn.classList.add('btn-loading');
        regBtnText.innerText = "REGISTERING...";
        regBtn.disabled = true;

        fetch('/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, email: email, password: pass })
        })
        .then(res => res.json().then(data => ({ status: res.status, data })))
        .then(({ status, data }) => {
            if (status === 200 && data.success) {
                clearButtonStates();
                regBtn.classList.add('btn-success');
                regBtnText.innerText = "UNIT INTEGRATED";
                localStorage.setItem('jobGuardUser', user);
                showCyberToast(`OPERATOR [${user.toUpperCase()}] REGISTERED SUCCESSFULLY`, "success", 2000);
                setTimeout(() => window.location.href = '/dashboard', 800);
            } else {
                clearButtonStates();
                regBtn.classList.add('btn-error');
                const errMsg = data.error || "REGISTRATION REJECTED";
                regBtnText.innerText = errMsg.toUpperCase();
                showCyberToast(errMsg, "error");
            }
        })
        .catch(err => {
            console.error(err);
            clearButtonStates();
            regBtnText.innerText = "SYSTEM OFFLINE";
            regBtn.style.borderColor = "#ffaa00";
            regBtn.style.color = "#ffaa00";
            regBtn.disabled = false;
            showCyberToast("NETWORK FAULT: Uplink transmission failed", "error");
        });
    });
});
