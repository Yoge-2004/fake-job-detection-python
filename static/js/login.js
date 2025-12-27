document.addEventListener('DOMContentLoaded', () => {
    
    // ==========================================
    // 1. MATRIX RAIN (RESPONSIVE)
    // ==========================================
    const canvas = document.getElementById('matrixCanvas');
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
        for(let x = 0; x < columns; x++) drops[x] = Math.floor(Math.random() * -100); 
    }

    const draw = () => {
        ctx.fillStyle = 'rgba(5, 8, 10, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#0F0';
        ctx.font = fontSize + 'px monospace';

        for(let i = 0; i < drops.length; i++) {
            const text = alphabet.charAt(Math.floor(Math.random() * alphabet.length));
            ctx.fillStyle = Math.random() > 0.9 ? '#00f3ff' : '#00ff9d';
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
            if(drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
            drops[i]++;
        }
    };

    initMatrix();
    setInterval(draw, 33);
    window.addEventListener('resize', initMatrix);

    // ==========================================
    // 2. UI TOGGLES
    // ==========================================
    const loginBox = document.querySelector('.login-box');
    const signupBox = document.querySelector('.signup-box');
    
    document.getElementById('showSignup').addEventListener('click', (e) => {
        e.preventDefault(); loginBox.classList.add('hidden'); signupBox.classList.remove('hidden');
    });

    document.getElementById('showLogin').addEventListener('click', (e) => {
        e.preventDefault(); signupBox.classList.add('hidden'); loginBox.classList.remove('hidden');
    });

    document.querySelectorAll('.password-toggle').forEach(button => {
        button.addEventListener('click', function() {
            const input = this.parentElement.querySelector('.password-field');
            const icon = this.querySelector('i');
            if (input.type === 'password') { input.type = 'text'; icon.classList.replace('fa-eye', 'fa-eye-slash'); }
            else { input.type = 'password'; icon.classList.replace('fa-eye-slash', 'fa-eye'); }
        });
    });

    // ==========================================
    // 3. LOGIN LOGIC (SMART VISUALS)
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
        const user = loginUser.value;
        const pass = loginPass.value;
        const remember = document.getElementById('rememberCheck').checked;

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
            if(data.success) {
                localStorage.setItem('jobGuardUser', data.username);
                loginBtn.style.borderColor = "var(--neon-green)";
                loginBtn.style.color = "var(--neon-green)";
                btnText.innerText = "ACCESS GRANTED";
                setTimeout(() => window.location.href = '/dashboard', 500);
            } else {
                btnText.innerText = "ACCESS DENIED";
                loginBtn.style.background = "var(--neon-pink)";
                loginBtn.style.borderColor = "var(--neon-pink)";
                loginBtn.style.color = "#fff";
                loginBtn.style.boxShadow = "0 0 15px var(--neon-pink)";
                loginBtn.style.cursor = "not-allowed";
            }
        })
        .catch(err => {
            console.error(err);
            btnText.innerText = "SYSTEM FAILURE";
            loginBtn.style.borderColor = "#ffaa00";
            loginBtn.style.color = "#ffaa00";
            loginBtn.disabled = false;
        });
    });

    // ==========================================
    // 4. SIGNUP LOGIC (WITH STATE CLASSES)
    // ==========================================
    const signupForm = document.getElementById('signupForm');
    const regBtn = document.getElementById('regBtn');
    const regBtnText = regBtn.querySelector('.btn-text');

    function clearButtonStates() {
        regBtn.classList.remove('btn-loading', 'btn-error', 'btn-success');
    }

    signupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const user = document.getElementById('signupUser').value;
        const email = document.getElementById('signupEmail').value;
        const pass = document.getElementById('signupPass').value;

        // 1. Loading
        clearButtonStates();
        regBtn.classList.add('btn-loading');
        regBtnText.innerText = "REGISTERING...";
        regBtn.disabled = true;

        fetch('/api/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, email: email, password: pass })
        })
        .then(res => res.json())
        .then(data => {
            if(data.success) {
                // 2. Success
                clearButtonStates();
                regBtn.classList.add('btn-success');
                regBtnText.innerText = "UNIT INTEGRATED";
                localStorage.setItem('jobGuardUser', user);
                setTimeout(() => window.location.href = '/dashboard', 800);
            } else {
                // 3. Error
                clearButtonStates();
                regBtn.classList.add('btn-error');
                regBtnText.innerText = "USER EXISTS";
            }
        })
        .catch(err => {
            console.error(err);
            clearButtonStates();
            regBtnText.innerText = "SYSTEM OFFLINE";
            regBtn.style.borderColor = "#ffaa00";
            regBtn.style.color = "#ffaa00";
            regBtn.disabled = false;
        });
    });

    // ==========================================
    // 5. VALIDATION
    // ==========================================
    const sUser = document.getElementById('signupUser');
    const sEmail = document.getElementById('signupEmail');
    const sPass = document.getElementById('signupPass');

    const patterns = {
        user: /^[a-zA-Z0-9_]{3,15}$/,
        email: /^([a-z\d\.-]+)@([a-z\d-]+)\.([a-z]{2,8})(\.[a-z]{2,8})?$/, 
        pass: /^(?=.*[0-9])(?=.*[!@#$%^&*])[a-zA-Z0-9!@#$%^&*]{8,}$/
    };

    function validateField(field, regex, msgElement, errorMsg) {
        // RESET ERROR STATE ON TYPING
        if(regBtn.classList.contains('btn-error')) {
            clearButtonStates();
            regBtnText.innerText = "REGISTER UNIT";
        }

        if (regex.test(field.value)) {
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
        if (patterns.user.test(sUser.value) && 
            patterns.email.test(sEmail.value) && 
            patterns.pass.test(sPass.value)) {
            regBtn.disabled = false;
            regBtn.style.cursor = "pointer";
            regBtn.style.borderColor = "var(--neon-cyan)";
            regBtn.style.color = "var(--neon-cyan)";
        } else {
            regBtn.disabled = true;
            regBtn.style.cursor = "not-allowed";
        }

        regBtn.style.borderColor = "";
        regBtn.style.color = "";
    }

    if(sUser && sEmail && sPass) {
        sUser.addEventListener('input', () => {
            validateField(sUser, patterns.user, document.getElementById('userMsg'), ">> 3-15 CHARS, ALPHANUMERIC");
            checkFormValidity();
        });
        sEmail.addEventListener('input', () => {
            validateField(sEmail, patterns.email, document.getElementById('emailMsg'), ">> INVALID EMAIL FORMAT");
            checkFormValidity();
        });
        sPass.addEventListener('input', () => {
            validateField(sPass, patterns.pass, document.getElementById('passMsg'), ">> 8+ CHARS, 1 NUM, 1 SPECIAL");
            checkFormValidity();
        });
    }
});
