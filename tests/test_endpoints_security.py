"""
JobGuard Security & Comprehensive Endpoints Test Suite
======================================================
Tests all Flask endpoints against extensive edge cases, attack vectors,
authentication boundaries, SQL injection, XSS inputs, and functional flows.

Usage:
  .venv/bin/pytest tests/test_endpoints_security.py -v
  .venv/bin/python tests/test_endpoints_security.py
"""

import json
import os
import random
import sys
import unittest

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app, get_db


class TestEndpointsAndSecurity(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        self.test_password = "CyberSecure#2026!"
        self.test_user = f"SecOp_{random.randint(10000, 99999)}"
        self.test_email = f"{self.test_user.lower()}@security.org"

    def tearDown(self):
        # Clean up any test users created
        with get_db() as conn:
            conn.execute("DELETE FROM users WHERE username LIKE 'SecOp_%' OR username LIKE 'Attacker%'")

    # ==========================================
    # 1. VIEW ROUTE ACCESS CONTROL
    # ==========================================
    def test_01_home_unauthenticated(self):
        """Unauthenticated user visiting / receives login page (200)."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"JOBGUARD_AI", res.data)
        self.assertIn(b"IDENTIFY YOURSELF", res.data)

    def test_02_dashboard_unauthenticated_redirects(self):
        """Unauthenticated user visiting /dashboard must be redirected to / (302)."""
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 302)
        self.assertTrue(res.headers["Location"].endswith("/"))

    def test_03_dashboard_authenticated(self):
        """Authenticated operator can access /dashboard successfully."""
        with self.client.session_transaction() as sess:
            sess["user"] = self.test_user
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"MANUAL_OVERRIDE_INPUT", res.data)
        self.assertIn(b"cyberModal", res.data)

    def test_04_home_authenticated_redirects_to_dashboard(self):
        """Authenticated operator visiting / is redirected to /dashboard."""
        with self.client.session_transaction() as sess:
            sess["user"] = self.test_user
        res = self.client.get("/")
        self.assertEqual(res.status_code, 302)
        self.assertTrue(res.headers["Location"].endswith("/dashboard"))

    # ==========================================
    # 2. SIGNUP SECURITY & VALIDATION
    # ==========================================
    def test_05_signup_missing_payload(self):
        """Empty payload to /api/signup returns 400 Bad Request."""
        res = self.client.post("/api/signup", json={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_06_signup_invalid_usernames(self):
        """Invalid username patterns (short, long, invalid characters) rejected with 400."""
        invalid_users = ["ab", "a" * 20, "user@name", "user name", "user#$"]
        for bad_u in invalid_users:
            res = self.client.post("/api/signup", json={
                "username": bad_u, "email": "valid@email.com", "password": self.test_password
            })
            self.assertEqual(res.status_code, 400, f"Failed to reject invalid username: {bad_u}")

    def test_07_signup_invalid_emails(self):
        """Malformed emails rejected with 400."""
        bad_emails = ["notanemail", "missing@domain", "@nodomain.com", "test@", "test@.com"]
        for bad_e in bad_emails:
            res = self.client.post("/api/signup", json={
                "username": "ValidUser", "email": bad_e, "password": self.test_password
            })
            self.assertEqual(res.status_code, 400, f"Failed to reject invalid email: {bad_e}")

    def test_08_signup_password_constraint_violations(self):
        """Passwords failing complexity rules (length, uppercase, lowercase, number, symbol) rejected."""
        weak_passwords = [
            "short",              # < 8 chars
            "nocapital123!",      # no uppercase
            "NOLOWERCASE123!",    # no lowercase
            "NoNumbersHere!",     # no numeric digit
            "NoSpecialChars123"   # no special symbol
        ]
        for wp in weak_passwords:
            res = self.client.post("/api/signup", json={
                "username": f"Op_{random.randint(100, 999)}",
                "email": f"op_{random.randint(100, 999)}@test.com",
                "password": wp
            })
            self.assertEqual(res.status_code, 400, f"Failed to reject weak password: {wp}")

    def test_09_signup_sql_injection_defense(self):
        """SQL injection attacks in username and email are safely parameterized and neutralized."""
        sqli_payloads = [
            "' OR '1'='1",
            "admin'--",
            "'; DROP TABLE users;--",
            "' UNION SELECT 1, 'admin', 'a@b.com', 'hash'--"
        ]
        for payload in sqli_payloads:
            res = self.client.post("/api/signup", json={
                "username": payload, "email": "safe@email.com", "password": self.test_password
            })
            self.assertEqual(res.status_code, 400, f"SQLi payload was not rejected by validation: {payload}")

    def test_10_signup_xss_prevention(self):
        """Cross-Site Scripting (XSS) in username rejected by input validation."""
        xss_payloads = ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>"]
        for xss in xss_payloads:
            res = self.client.post("/api/signup", json={
                "username": xss, "email": "valid@email.com", "password": self.test_password
            })
            self.assertEqual(res.status_code, 400, f"XSS payload allowed through: {xss}")

    def test_11_signup_success_and_duplicate_prevention(self):
        """Valid registration succeeds, but duplicate username or email returns 409 Conflict."""
        res = self.client.post("/api/signup", json={
            "username": self.test_user,
            "email": self.test_email,
            "password": self.test_password
        })
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json().get("success"))

        # Duplicate username attempt
        res_dup_u = self.client.post("/api/signup", json={
            "username": self.test_user,
            "email": "unique_email@test.com",
            "password": self.test_password
        })
        self.assertEqual(res_dup_u.status_code, 409)

        # Duplicate email attempt
        res_dup_e = self.client.post("/api/signup", json={
            "username": "UniqueOpName",
            "email": self.test_email,
            "password": self.test_password
        })
        self.assertEqual(res_dup_e.status_code, 409)

    # ==========================================
    # 3. LOGIN & AUTHENTICATION FLOWS
    # ==========================================
    def test_12_login_invalid_credentials(self):
        """Wrong username or password returns 401 Unauthorized."""
        res1 = self.client.post("/api/login", json={"username": "NonExistentUser", "password": "WrongPassword1!"})
        self.assertEqual(res1.status_code, 401)

        # Register user first
        self.client.post("/api/signup", json={
            "username": self.test_user, "email": self.test_email, "password": self.test_password
        })
        # Attempt login with wrong password
        res2 = self.client.post("/api/login", json={"username": self.test_user, "password": "WrongPassword999!"})
        self.assertEqual(res2.status_code, 401)

    def test_13_login_success(self):
        """Valid credentials return 200 and establish session."""
        self.client.post("/api/signup", json={
            "username": self.test_user, "email": self.test_email, "password": self.test_password
        })
        res = self.client.post("/api/login", json={"username": self.test_user, "password": self.test_password})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("username"), self.test_user)

    # ==========================================
    # 4. USER PROFILE & ACCOUNT DELETION
    # ==========================================
    def test_14_user_info_authorization(self):
        """GET /api/user_info requires authenticated session."""
        res_anon = self.client.get("/api/user_info")
        self.assertEqual(res_anon.status_code, 401)

        with self.client.session_transaction() as sess:
            sess["user"] = self.test_user
        res_auth = self.client.get("/api/user_info")
        self.assertEqual(res_auth.status_code, 200)
        self.assertEqual(res_auth.get_json()["username"], self.test_user)

    def test_15_delete_account_lifecycle(self):
        """DELETE /api/delete_account clears database record and invalidates session."""
        # 1. Register
        self.client.post("/api/signup", json={
            "username": self.test_user, "email": self.test_email, "password": self.test_password
        })
        # 2. Delete account
        res = self.client.post("/api/delete_account")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json().get("success"))

        # 3. Verify session is wiped
        res_check = self.client.get("/api/user_info")
        self.assertEqual(res_check.status_code, 401)

        # 4. Verify login no longer works
        res_login = self.client.post("/api/login", json={"username": self.test_user, "password": self.test_password})
        self.assertEqual(res_login.status_code, 401)

    def test_16_system_logs_rbac(self):
        """Only admin operator ('Yoge') can access system logs; others receive empty list."""
        # Guest
        self.assertEqual(self.client.get("/api/system_logs").get_json(), [])

        # Regular operator
        with self.client.session_transaction() as sess:
            sess["user"] = "RegularUser"
        self.assertEqual(self.client.get("/api/system_logs").get_json(), [])

        # Admin operator
        with self.client.session_transaction() as sess:
            sess["user"] = "Yoge"
        logs = self.client.get("/api/system_logs").get_json()
        self.assertIsInstance(logs, list)

    def test_17_logout(self):
        """POST /api/logout clears session."""
        with self.client.session_transaction() as sess:
            sess["user"] = self.test_user
        res = self.client.post("/api/logout")
        self.assertEqual(res.status_code, 200)
        res_check = self.client.get("/api/user_info")
        self.assertEqual(res_check.status_code, 401)

    # ==========================================
    # 5. ASYNC & SYNC INFERENCE PIPELINE
    # ==========================================
    def test_18_predict_input_length_validation(self):
        """Predictions with < 5 characters return 400 Bad Request."""
        res = self.client.post("/predict", json={"text": "abc"})
        self.assertEqual(res.status_code, 400)

    def test_19_predict_gibberish_handling(self):
        """Random unparseable gibberish returns is_gibberish: True."""
        res = self.client.post("/predict", json={"text": "asdfkjhqweroiuyzxcvbnm1234567890!@#$%"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("is_gibberish"))

    def test_20_async_scan_lifecycle(self):
        """POST /api/scan dispatches job and GET /api/scan_status/<job_id> returns result."""
        job_text = "Senior Python Engineer required at Stripe. Minimum 5 years experience with Flask, AWS, PostgreSQL."
        # Submit async scan
        res_submit = self.client.post("/api/scan", json={"text": job_text})
        self.assertEqual(res_submit.status_code, 200)
        data_submit = res_submit.get_json()
        job_id = data_submit["job_id"]
        self.assertIsNotNone(job_id)

        # Check status
        res_status = self.client.get(f"/api/scan_status/{job_id}")
        self.assertEqual(res_status.status_code, 200)
        data_status = res_status.get_json()
        self.assertIn(data_status["status"], ["processing", "completed"])

    # ==========================================
    # 6. SYSTEM HEALTH & SWAGGER SPEC
    # ==========================================
    def test_21_health_probe(self):
        """GET /api/health returns healthy system status and uptime."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["models_loaded"])
        self.assertGreaterEqual(data["uptime_seconds"], 0)

    def test_22_swagger_documentation_endpoint(self):
        """Swagger UI (/docs) and OpenAPI specs (/apispec_1.json) are accessible."""
        res_docs = self.client.get("/docs")
        self.assertEqual(res_docs.status_code, 200)
        res_spec = self.client.get("/apispec_1.json")
        self.assertEqual(res_spec.status_code, 200)
        spec = res_spec.get_json()
        self.assertIn("paths", spec)
        self.assertIn("/predict", spec["paths"])
        self.assertIn("/api/scan", spec["paths"])
        self.assertIn("/api/health", spec["paths"])



    # ==========================================
    # 7. GOOGLE OAUTH 2.0 WORKFLOW & SECURITY
    # ==========================================
    def test_23_google_oauth_login_unconfigured_redirect(self):
        """GET /api/auth/google/login without credentials redirects to /?error=oauth_not_configured."""
        orig_id = os.environ.pop("GOOGLE_CLIENT_ID", None)
        orig_secret = os.environ.pop("GOOGLE_CLIENT_SECRET", None)
        try:
            res = self.client.get("/api/auth/google/login")
            self.assertEqual(res.status_code, 302)
            self.assertIn("error=oauth_not_configured", res.headers.get("Location", ""))
        finally:
            if orig_id:
                os.environ["GOOGLE_CLIENT_ID"] = orig_id
            if orig_secret:
                os.environ["GOOGLE_CLIENT_SECRET"] = orig_secret

    def test_24_google_oauth_login_unconfigured_json(self):
        """GET /api/auth/google/login with Accept: application/json returns 503."""
        orig_id = os.environ.pop("GOOGLE_CLIENT_ID", None)
        orig_secret = os.environ.pop("GOOGLE_CLIENT_SECRET", None)
        try:
            res = self.client.get("/api/auth/google/login", headers={"Accept": "application/json"})
            self.assertEqual(res.status_code, 503)
            self.assertIn("Google OAuth credentials not configured", res.get_json().get("error", ""))
        finally:
            if orig_id:
                os.environ["GOOGLE_CLIENT_ID"] = orig_id
            if orig_secret:
                os.environ["GOOGLE_CLIENT_SECRET"] = orig_secret

    def test_25_google_oauth_callback_new_user(self):
        """Google callback successfully provisions and logs in a new user."""
        from unittest.mock import patch
        mock_userinfo = {
            "email": "agent_alpha@cyberdefense.org",
            "name": "Alpha Agent"
        }
        with patch("app.oauth.google.authorize_access_token", return_value={"userinfo": mock_userinfo}):
            res = self.client.get("/api/auth/google/callback")
            self.assertEqual(res.status_code, 302)
            self.assertTrue(res.headers.get("Location", "").endswith("/dashboard"))

            # Check that user is in session
            res_user = self.client.get("/api/user_info")
            self.assertEqual(res_user.status_code, 200)
            self.assertIn("Alpha_Agent", res_user.get_json()["username"])

            # Clean up user
            with get_db() as conn:
                conn.execute("DELETE FROM users WHERE email = ?", ("agent_alpha@cyberdefense.org",))

    def test_26_google_oauth_callback_existing_user(self):
        """Google callback successfully authenticates an existing user matching by email."""
        from unittest.mock import patch
        existing_email = f"existing_{random.randint(1000, 9999)}@corp.com"
        existing_username = f"Exist_{random.randint(1000, 9999)}"
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (existing_username, existing_email, "dummy_hash")
            )

        try:
            mock_userinfo = {"email": existing_email, "name": "Existing User"}
            with patch("app.oauth.google.authorize_access_token", return_value={"userinfo": mock_userinfo}):
                res = self.client.get("/api/auth/google/callback")
                self.assertEqual(res.status_code, 302)
                self.assertTrue(res.headers.get("Location", "").endswith("/dashboard"))

                res_user = self.client.get("/api/user_info")
                self.assertEqual(res_user.status_code, 200)
                self.assertEqual(res_user.get_json()["username"], existing_username)
        finally:
            with get_db() as conn:
                conn.execute("DELETE FROM users WHERE email = ?", (existing_email,))


if __name__ == "__main__":
    unittest.main()
