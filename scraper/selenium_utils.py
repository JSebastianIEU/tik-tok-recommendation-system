from __future__ import annotations

import time

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def dismiss_communication_banner(driver: WebDriver, timeout: int = 5) -> None:
    """Best-effort dismissal of the first communication banner, if present."""
    try:
        wait = WebDriverWait(driver, timeout)
        first = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, '//*[@id="pns-communication-service"]/div/div/div/div[2]/div/button')
            )
        )
        first.click()
        time.sleep(0.3)
    except Exception:
        # Non-fatal
        return


def dismiss_cookie_banner(driver: WebDriver, timeout: int = 8) -> None:
    """
    Dismiss TikTok's cookie banner, including Shadow DOM implementation.

    Tries a JS-based Shadow DOM traversal first, then falls back to a regular
    DOM XPath search for buttons that include "Decline optional cookies".
    """
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tiktok-cookie-banner")))
        time.sleep(0.5)
    except Exception:
        # Banner may not exist; fall through to best-effort fallback
        pass

    try:
        clicked = driver.execute_script(
            """
            var host = document.querySelector('tiktok-cookie-banner');
            if (!host || !host.shadowRoot) return false;
            var root = host.shadowRoot;
            var buttons = root.querySelectorAll('button');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.trim().indexOf('Decline optional cookies') !== -1) {
                    buttons[i].click();
                    return true;
                }
            }
            return false;
            """
        )
        if clicked:
            time.sleep(0.5)
            return
    except Exception:
        # Shadow DOM structure may have changed; fall back to DOM search
        pass

    try:
        btn = driver.find_element(By.XPATH, "//button[contains(., 'Decline optional cookies')]")
        btn.click()
        time.sleep(0.5)
    except Exception:
        return


def click_shadow_button_by_text(driver: WebDriver, text: str, timeout: int = 8) -> bool:
    """
    Click a button-like element whose visible text matches `text`,
    searching both the main DOM and nested Shadow DOMs.

    Returns True if a click was attempted.
    """
    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception:
        return False

    try:
        clicked = driver.execute_script(
            """
            function findInShadowRoot(root, matcher) {
                var el = matcher(root);
                if (el) return el;
                var nodes = root.querySelectorAll('*');
                for (var i = 0; i < nodes.length; i++) {
                    if (nodes[i].shadowRoot) {
                        el = findInShadowRoot(nodes[i].shadowRoot, matcher);
                        if (el) return el;
                    }
                }
                return null;
            }
            var matcher = function(root) {
                if (!root.querySelectorAll) return null;
                var buttons = root.querySelectorAll('button, a, [role=\"button\"]');
                for (var j = 0; j < buttons.length; j++) {
                    if (buttons[j].textContent.trim() === arguments[0]) return buttons[j];
                }
                return null;
            };
            var direct = matcher(document);
            if (direct) { direct.click(); return true; }
            var btn = findInShadowRoot(document.body, matcher.bind(null));
            if (btn) { btn.click(); return true; }
            return false;
            """,
            text,
        )
        return bool(clicked)
    except Exception:
        return False
