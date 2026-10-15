"""
JobGuard Playwright End-to-End UI Automation Test Suite
======================================================
Tests all user interface interactions, animations, theme toggling,
password constraints, async scanning, custom cyberpunk modals,
copy button feedback, mobile viewport responsiveness, and reload resilience.

Prerequisites:
  .venv/bin/python -m playwright install chromium

Execution:
  # Standalone runner:
  .venv/bin/python tests/test_ui_playwright.py

  # With PyTest:
  .venv/bin/pytest tests/test_ui_playwright.py -v -s
"""

import multiprocessing
import os
import random
import socket
import sys
import time
import pytest
from playwright.sync_api import sync_playwright, Page, expect

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, get_db

TEST_HOST = "127.0.0.1"
TEST_PORT = 7865
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


import subprocess

@pytest.fixture(scope="session", autouse=True)
def test_server():
    """Starts local Flask server in background subprocess if not already running."""
    server_process = None
    if not is_port_open(TEST_HOST, TEST_PORT):
        print(f"\n[SERVER] Starting test server on {BASE_URL}...")
        server_process = subprocess.Popen(
            [sys.executable, "-c", f"from app import app; app.run(host='{TEST_HOST}', port={TEST_PORT}, debug=False, use_reloader=False)"],
            cwd=PROJECT_ROOT,
            stdout=open("/tmp/jobguard_test_server.log", "w"),
            stderr=subprocess.STDOUT
        )

        # Wait for server ready
        for _ in range(60):
            if is_port_open(TEST_HOST, TEST_PORT):
                break
            time.sleep(0.5)

    yield BASE_URL

    if server_process and server_process.poll() is None:
        print("[SERVER] Stopping test server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()


class TestJobGuardUI:

    @pytest.fixture(autouse=True)
    def setup_operator(self):
        self.username = f"PlayOp_{random.randint(10000, 99999)}"
        self.email = f"{self.username.lower()}@guard.ai"
        self.password = "CyberSecure#2026!"
        yield
        with get_db() as conn:
            conn.execute("DELETE FROM users WHERE username LIKE 'PlayOp_%'")

    # ==========================================
    # 1. LANDING & THEME TOGGLING
    # ==========================================
    def test_01_landing_page_and_matrix_canvas(self, test_server):
        """Verify landing page loads, matrix canvas renders, and default theme is dark."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL)

            # Check title and logo
            expect(page).to_have_title("Fake Job Detector - AI Scam Protection Tool | JobGuard AI")
            expect(page.locator(".logo-area h1")).to_have_text("JOBGUARD_AI")

            # Check matrix canvas
            canvas = page.locator("#matrixCanvas")
            expect(canvas).to_be_visible()

            # Check default theme is dark
            theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
            assert theme == "dark", f"Expected dark theme by default, got: {theme}"

            browser.close()

    def test_02_theme_toggle_persistence(self, test_server):
        """Toggle from dark to light and back, verifying data-theme attribute and persistence."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL)

            theme_btn = page.locator("#themeToggleBtn")
            expect(theme_btn).to_be_visible()

            # Click to switch to light theme
            theme_btn.click()
            page.wait_for_timeout(300)

            theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
            assert theme == "light", "Theme failed to toggle to light"
            local_val = page.evaluate("localStorage.getItem('jobGuardTheme')")
            assert local_val == "light", "localStorage theme not updated to light"

            # Check toast notification appeared
            toast = page.locator(".cyber-toast")
            expect(toast).to_be_visible()

            # Click to toggle back to dark
            theme_btn.click()
            page.wait_for_timeout(300)
            theme_back = page.evaluate("document.documentElement.getAttribute('data-theme')")
            assert theme_back == "dark", "Theme failed to toggle back to dark"

            browser.close()

    # ==========================================
    # 2. SIGNUP CONSTRAINTS & STRENGTH METER
    # ==========================================
    def test_03_password_constraints_and_strength_meter(self, test_server):
        """Verify live constraint checklist and dynamic strength meter progression."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL)

            # Switch to signup form
            page.click("#showSignup")
            expect(page.locator(".signup-box")).to_be_visible()

            pass_input = page.locator("#signupPass")
            strength_val = page.locator("#strengthStatus")
            reg_btn = page.locator("#regBtn")

            # Initial state
            expect(reg_btn).to_be_disabled()

            # Type 1: weak password (<8 chars)
            pass_input.fill("abc")
            page.wait_for_timeout(200)
            expect(page.locator("#c-length")).to_have_class("constraint-item unmet")
            expect(strength_val).to_contain_text("WEAK")

            # Type 2: Meet length, upper, lower, number, special
            pass_input.fill("SecurePass123!")
            page.wait_for_timeout(200)
            expect(page.locator("#c-length")).to_have_class("constraint-item met")
            expect(page.locator("#c-upper")).to_have_class("constraint-item met")
            expect(page.locator("#c-lower")).to_have_class("constraint-item met")
            expect(page.locator("#c-number")).to_have_class("constraint-item met")
            expect(page.locator("#c-special")).to_have_class("constraint-item met")

            expect(strength_val).to_contain_text("CYBER-SECURE")

            # Fill username and email to enable button
            page.fill("#signupUser", "Operator_X")
            page.fill("#signupEmail", "operator_x@grid.org")
            page.wait_for_timeout(200)
            expect(reg_btn).to_be_enabled()

            browser.close()

    def test_04_password_visibility_toggle(self, test_server):
        """Test password eye icon toggle reveals and hides password text."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL)

            login_pass = page.locator("#loginPass")
            toggle_btn = page.locator(".login-box .password-toggle")

            assert login_pass.get_attribute("type") == "password"
            toggle_btn.click()
            assert login_pass.get_attribute("type") == "text"
            toggle_btn.click()
            assert login_pass.get_attribute("type") == "password"

            browser.close()

    # ==========================================
    # 3. REGISTRATION & LOGIN FLOW
    # ==========================================
    def test_05_registration_and_redirect_to_dashboard(self, test_server):
        """Complete new user registration and verify redirection to dashboard."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL)

            # Switch to signup
            page.click("#showSignup")
            page.fill("#signupUser", self.username)
            page.fill("#signupEmail", self.email)
            page.fill("#signupPass", self.password)

            # Submit registration
            page.click("#regBtn")

            # Redirection to /dashboard
            page.wait_for_url(f"{BASE_URL}/dashboard", timeout=8000)
            expect(page.locator("#profileName")).to_contain_text(self.username.upper())
            expect(page.locator("#jobDescription")).to_be_visible()

            browser.close()

    def test_06_login_invalid_credentials_toast(self, test_server):
        """Invalid credentials trigger in-app error toast and access denied state."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(BASE_URL)

            page.fill("#loginUser", "GhostUser999")
            page.fill("#loginPass", "WrongPassword123!")
            page.click("button[type='submit']")

            # Check toast alert
            toast = page.locator(".cyber-toast.toast-error")
            expect(toast).to_be_visible(timeout=5000)
            expect(page.locator("#loginForm .cyber-btn .btn-text")).to_have_text("ACCESS DENIED")

            browser.close()

    # ==========================================
    # 4. DASHBOARD ANALYSIS & INFERENCE EXECUTION
    # ==========================================
    def test_07_job_buffer_counter_and_typing_status(self, test_server):
        """Typing in job description updates buffer character count and status label."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Authenticate directly
            self._login_operator(page)

            job_input = page.locator("#jobDescription")
            char_count = page.locator("#charCount")
            scan_status = page.locator("#scanStatus")

            job_input.fill("Testing buffer input for fraud detection system.")
            expect(char_count).to_contain_text("BUFFER: 48 chars")
            page.wait_for_timeout(900)
            expect(scan_status).to_have_text(">> SIGNAL STANDBY")

            browser.close()

    def test_08_empty_input_warning_toast(self, test_server):
        """Clicking scan with empty input triggers in-app warning toast (no JS alert)."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login_operator(page)

            page.click("#analyzeBtn")
            toast = page.locator(".cyber-toast.toast-warning")
            expect(toast).to_be_visible()
            expect(toast).to_contain_text("BUFFER EMPTY")

            browser.close()

    def test_09_full_scan_threat_analysis_display(self, test_server):
        """Initialize scan on job description, verify loader, verdict, confidence, and advisory."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login_operator(page)

            fake_posting = (
                "Urgent hiring! Work from home Data Entry Clerk. Earn $45/hour. "
                "No experience needed. Equipment purchase check will be mailed to your home. "
                "Contact recruiter on Telegram @FastCashHR."
            )

            page.fill("#jobDescription", fake_posting)
            page.click("#analyzeBtn")

            # Check loader appears
            loader = page.locator("#loader")
            expect(loader).to_be_visible()

            # Wait for result container
            result_container = page.locator("#resultContainer")
            expect(result_container).to_be_visible(timeout=55000)

            # Check verdict title and confidence score
            verdict_title = page.locator("#verdictTitle")
            expect(verdict_title).not_to_have_text("--")

            score = page.locator("#confidenceScore")
            expect(score).to_contain_text("%")

            # Check advisory notes section rendered
            advisory = page.locator("#verdictMessage")
            expect(advisory).to_be_visible()

            browser.close()

    # ==========================================
    # 5. COPY BUTTON VISUAL FEEDBACK
    # ==========================================
    def test_10_copy_logs_button_feedback(self, test_server):
        """Clicking copy logs transitions button to COPIED! state with green styling and toast."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login_operator(page)

            copy_btn = page.locator("#copyLogsBtn")
            expect(copy_btn).to_be_visible()

            copy_btn.click()

            # Verify active success feedback
            expect(copy_btn).to_have_class("copy-btn copy-success")
            expect(copy_btn.locator(".copy-text")).to_have_text("COPIED!")

            # Verify in-app toast appeared
            toast = page.locator(".cyber-toast.toast-success")
            expect(toast).to_be_visible()

            # Verify auto-reset after 2 seconds
            page.wait_for_timeout(2300)
            expect(copy_btn).not_to_have_class("copy-success")
            expect(copy_btn.locator(".copy-text")).to_have_text("COPY")

            browser.close()

    # ==========================================
    # 6. IN-APP CONFIRMATION MODALS (LOGOUT & DELETE)
    # ==========================================
    def test_11_logout_confirmation_modal(self, test_server):
        """Clicking logout opens custom cyberpunk modal, cancel aborts, confirm disconnects."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login_operator(page)

            page.click("#logoutBtn")

            modal = page.locator("#cyberModal")
            expect(modal).to_be_visible()
            expect(page.locator("#cyberModalTitle")).to_have_text("// TERMINATE_SESSION_CONFIRMATION")

            # Cancel aborts
            page.click("#cyberModalCancel")
            expect(modal).to_be_hidden()
            expect(page).to_have_url(f"{BASE_URL}/dashboard")

            # Open again and confirm
            page.click("#logoutBtn")
            page.click("#cyberModalConfirm")

            # Redirection to landing page
            page.wait_for_url(f"{BASE_URL}/", timeout=6000)
            expect(page.locator(".login-box")).to_be_visible()

            browser.close()

    def test_12_delete_account_danger_modal(self, test_server):
        """Clicking delete opens danger modal with pink styling, confirm purges account."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login_operator(page)

            page.click("#deleteAccountBtn")

            modal = page.locator("#cyberModal")
            expect(modal).to_be_visible()
            expect(page.locator(".cyber-modal-box")).to_have_class("cyber-modal-box modal-danger")
            expect(page.locator("#cyberModalTitle")).to_have_text("// CRITICAL_PURGE_PROTOCOL")

            # Confirm purge
            page.click("#cyberModalConfirm")

            # Redirection to landing page
            page.wait_for_url(f"{BASE_URL}/", timeout=6000)
            browser.close()

    # ==========================================
    # 7. MOBILE VIEWPORT & RELOAD RESILIENCE
    # ==========================================
    def test_13_mobile_viewport_responsiveness(self, test_server):
        """Verify layout stacks properly and toasts adapt on mobile screens (375x812)."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 375, "height": 812})
            self._login_operator(page)

            # Scanner panel visible
            scanner = page.locator(".scanner-panel")
            expect(scanner).to_be_visible()

            # Header user profile
            profile = page.locator(".user-profile")
            expect(profile).to_be_visible()

            browser.close()

    def test_14_page_reload_result_retention(self, test_server):
        """If mobile browser refreshes, stored session output is restored without losing work."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            self._login_operator(page)

            # Inject a mock completed result into sessionStorage
            page.evaluate("""() => {
                sessionStorage.setItem('jobGuardLastResult', JSON.stringify({
                    inputText: 'Restored job test posting...',
                    result: {
                        fraud_probability: 91.5,
                        is_gibberish: false,
                        reasons: ['Suspicious payment request'],
                        advisory: ['Email not listed'],
                        system_logs: ['[SYSTEM] Restored']
                    },
                    timestamp: Date.now()
                }));
            }""")

            # Reload page
            page.reload()

            # Verify result container is restored automatically
            result_container = page.locator("#resultContainer")
            expect(result_container).to_be_visible()
            expect(page.locator("#confidenceScore")).to_have_text("91.5%")
            expect(page.locator("#scanStatus")).to_have_text(">> BUFFER RESTORED")

            browser.close()

    # --- Helper Method ---
    def _login_operator(self, page: Page):
        """Helper to establish registration and navigate to dashboard."""
        page.goto(BASE_URL)
        page.click("#showSignup")
        page.fill("#signupUser", self.username)
        page.fill("#signupEmail", self.email)
        page.fill("#signupPass", self.password)
        page.click("#regBtn")
        page.wait_for_url(f"{BASE_URL}/dashboard", timeout=8000)


def run_standalone():
    """Allows running tests directly via `python tests/test_ui_playwright.py`."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_standalone()
