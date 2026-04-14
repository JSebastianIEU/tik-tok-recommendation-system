// Lightweight focus listener to detect when mobile keyboard is likely open.
// Adds `keyboard-open` class to <body> while input/textarea/select/contenteditable are focused.

function isTextInput(el: unknown): boolean {
  if (!el || typeof el !== "object") return false;
  const node = el as HTMLElement;
  const tag = node.tagName?.toLowerCase();
  if (!tag) return false;
  if (tag === "input" || tag === "textarea" || tag === "select") return true;
  if (node.isContentEditable) return true;
  const role = node.getAttribute?.("role");
  if (role === "textbox") return true;
  return false;
}

function onFocusIn(e: FocusEvent) {
  if (isTextInput(e.target)) {
    document.body.classList.add("keyboard-open");
  }
}

function onFocusOut(e: FocusEvent) {
  if (isTextInput(e.target)) {
    // Delay remove slightly to allow related focus moves
    setTimeout(() => document.body.classList.remove("keyboard-open"), 50);
  }
}

document.addEventListener("focusin", onFocusIn, true);
document.addEventListener("focusout", onFocusOut, true);

export {};

// ===== Social pill visibility on small screens =====
// Keep the social pill fixed, then hide it only after meaningful downward scroll.
// It stays hidden until the user scrolls up or returns near the top.
(() => {
  if (typeof window === "undefined") return;
  const m = window.matchMedia("(max-width: 900px)");
  const HIDE_AFTER_SCROLL_PX = 220;
  const SHOW_NEAR_TOP_PX = 96;
  const UP_SCROLL_TO_SHOW_PX = 16;
  let lastY = 0;
  let socialHidden = false;

  function setSocialHidden(hide: boolean) {
    if (hide === socialHidden) return;
    socialHidden = hide;
    if (hide) document.body.classList.add("social-hidden");
    else document.body.classList.remove("social-hidden");
  }

  function getScrollY(): number {
    const rootY = window.scrollY || document.documentElement.scrollTop || document.body.scrollTop || 0;
    const appMainY = (document.querySelector(".app-main") as HTMLElement | null)?.scrollTop || 0;
    const formPanelY = (document.querySelector(".form-panel") as HTMLElement | null)?.scrollTop || 0;
    const reportY = (document.querySelector(".report-scroll-container") as HTMLElement | null)?.scrollTop || 0;
    return Math.max(rootY, appMainY, formPanelY, reportY);
  }

  function onScroll() {
    const y = getScrollY();
    if (!m.matches) {
      lastY = y;
      setSocialHidden(false);
      return;
    }

    const dy = y - lastY;
    if (y <= SHOW_NEAR_TOP_PX) {
      setSocialHidden(false);
    } else if (y >= HIDE_AFTER_SCROLL_PX && dy > 0) {
      setSocialHidden(true);
    } else if (dy < -UP_SCROLL_TO_SHOW_PX) {
      setSocialHidden(false);
    }

    lastY = y;
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  document.addEventListener("scroll", onScroll, { passive: true, capture: true });
  // if viewport flips (resize/orientation) ensure class removed when not small
  window.addEventListener("resize", onScroll, { passive: true });
  onScroll();
})();

// ===== Detect upload layout presence and make bottom nav scroll with page =====
(() => {
  if (typeof window === "undefined") return;

  const UPLOAD_SELECTOR = ".panel-transition-wrapper.layout-upload";
  const BODY_CLASS = "upload-layout-active";

  function updateUploadClass() {
    const exists = !!document.querySelector(UPLOAD_SELECTOR);
    if (exists) document.body.classList.add(BODY_CLASS);
    else document.body.classList.remove(BODY_CLASS);
  }

  // Observe mutations on the app shell (or body fallback) to catch dynamic route changes
  const root = document.querySelector(".app-shell") || document.body;
  try {
    const mo = new MutationObserver(() => updateUploadClass());
    mo.observe(root, { childList: true, subtree: true });
  } catch (err) {
    // If MutationObserver not available, fall back to polling
    setInterval(updateUploadClass, 500);
  }

  // Initial check in case the layout is already present
  updateUploadClass();

  // Also listen for simple navigation (popstate) events
  window.addEventListener("popstate", updateUploadClass);
})();
