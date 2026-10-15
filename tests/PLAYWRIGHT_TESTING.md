# JobGuard - Playwright End-to-End UI Testing Suite

This document outlines the Playwright End-to-End (E2E) testing suite for the JobGuard application. The suite thoroughly tests frontend functionalities, UI components, responsive layouts, theme switching, security forms, async scan polling, and state resilience across browser reloads and memory suspensions.

---

## 1. Prerequisites & Environment Setup

Ensure the virtual environment is active and required test dependencies are installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install test dependencies (playwright, pytest)
pip install playwright pytest

# Install Chromium browser binaries for Playwright
playwright install chromium
```

---

## 2. Test Execution

The test suite automatically handles spinning up an isolated, background JobGuard Flask server instance on port `7865` with a temporary, isolated SQLite test database, and tears it down cleanly upon completion.

### Run All Playwright UI Tests:
```bash
.venv/bin/pytest tests/test_ui_playwright.py -v -s
```

### Run a Specific UI Test Case:
```bash
.venv/bin/pytest tests/test_ui_playwright.py -k "test_03_theme_switcher" -v -s
```

---

## 3. Test Scenarios Covered

The test file `tests/test_ui_playwright.py` contains 14 comprehensive test scenarios:

| # | Test Name | Target View / Component | Scenario Description |
|---|-----------|-------------------------|----------------------|
| **1** | `test_01_landing_page_load` | `/login` (Cyber Landing) | Verifies initial page load, dark theme default (`data-theme="dark"`), Matrix canvas visibility, Cyber Auth header, and login form presence. |
| **2** | `test_02_swagger_docs_accessible` | `/docs` (OpenAPI Swagger) | Verifies Swagger UI loads, document title contains "Swagger UI", and JobGuard endpoints are exposed. |
| **3** | `test_03_theme_switcher_toggle_and_persistence` | Universal Theme Switcher | Validates toggle switch to Cyber-Day (Light Mode) and back, verifying root attribute `data-theme="light"` / `data-theme="dark"` and `localStorage` persistence. |
| **4** | `test_04_signup_tab_switch_and_password_strength_meter` | Signup Modal / Form | Switches from Login to Signup view, inputs varying passwords, and validates dynamic score progression from Weak to Cyber-Secure. |
| **5** | `test_05_signup_constraints_checklist` | Security Panel Checklist | Validates real-time green/red visual indicators on length, uppercase, lowercase, numbers, and special symbols as user types. |
| **6** | `test_06_signup_validation_toast_error` | Custom Toast System | Submits invalid signup data (e.g. mismatched passwords) and asserts in-app cyber toast displays error without native JS browser alert dialogs. |
| **7** | `test_07_user_registration_and_redirect` | End-to-End Registration | Completes valid user registration, checks for in-app success toast notification, and verifies auto-switch back to login form. |
| **8** | `test_08_user_login_and_dashboard_navigation` | Authentication Flow | Submits registered credentials, verifies successful authentication and automated redirect to the main Dashboard (`/`). |
| **9** | `test_09_dashboard_scrollbar_and_elements` | Cyber Dashboard (`/`) | Inspects dashboard layout: verified custom scrollbars on job description textarea and logs terminal, and presence of all metrics cards. |
| **10** | `test_10_copy_logs_button_feedback` | Cyber Terminal Logs | Clicks "Copy Logs" button, asserts active feedback state (`COPIED!`, checkmark icon, neon-green glow) and in-app confirmation toast. |
| **11** | `test_11_custom_modal_logout_dialog` | `#cyberModal` Dialog System | Clicks Logout, verifies custom cyberpunk confirmation modal appears with backdrop blur, tests cancellation (modal dismisses), and confirmed logout. |
| **12** | `test_12_custom_modal_delete_dialog` | `#cyberModal` Dialog System | Loads dashboard with scanned results, clicks "Clear Results / Delete", confirms custom cyberpunk modal appears, and verifies clearing upon confirmation. |
| **13** | `test_13_mobile_viewport_and_power_saving` | Mobile Viewport (iPhone 13) | Emulates mobile resolution (390x844), verifies responsive layout scaling, navigation bar layout, and Matrix animation pause logic on tab backgrounding. |
| **14** | `test_14_mobile_memory_suspension_reload_resilience` | Mobile Async Scan Persistence | Dispatches an asynchronous ML scan (`POST /api/scan`), simulates sudden mobile browser reload/memory suspension, and verifies scan ticket reconnects and completes. |

---

## 4. Key Architectural Patterns Tested

1. **Dark Default & Cyber-Day Light Mode:**
   - Dark theme is strictly the default across both pages.
   - Light theme custom properties (`--bg-primary`, `--cyber-cyan`, `--text-main`) provide clean high-contrast readability without breaking matrix aesthetic.
2. **Zero Native JS Dialogs:**
   - Every `alert()` and `confirm()` has been replaced with `#cyberModal` and `#cyberToastContainer`.
   - Modals support async Promise resolution (`showCyberConfirm('title', 'msg')`).
3. **Mobile Resilience:**
   - Memory suspension and background tab reload reload recovery stores active scan tickets in `localStorage.setItem('jobGuardPendingScan', JSON.stringify({jobId, text}))`.
   - When refreshed, dashboard detects active scan, resumes polling `/api/scan_status/<jobId>`, and displays ML prediction metrics without re-running inference.
