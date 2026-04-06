"""
ComicForge Backend Server
=========================
Bridges the ComicForge HTML frontend with:
  - Gemini (via Playwright automation) for image generation  (AI 2 + AI 3)
  - OpenAI API for JSON generation                           (AI 1)

Run Chrome first:
  "C:\Program Files\Google\Chrome\Application\chrome.exe" ^
      --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug"

Install deps:
  pip install flask flask-cors playwright openai
  playwright install chromium

Then run:
  python server.py
"""

import asyncio
import base64
import time
import threading
import urllib.request
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ── Config ────────────────────────────────────────────────────────────────────
CHROME_DEBUG_URL = "http://localhost:9222"
IMAGES_DIR       = Path("comicforge_images")
MAX_WAIT_SEC     = 240
POLL_INTERVAL    = 2
PORT             = 5000
# ─────────────────────────────────────────────────────────────────────────────

IMAGES_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
CORS(app)  # Allow requests from the HTML file opened locally

# Shared asyncio event loop running in a background thread
# (Flask is sync; Playwright is async — we bridge them)
_loop: asyncio.AbstractEventLoop = None
_browser = None
_playwright_lock = threading.Lock()


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYWRIGHT HELPERS  (all async, run on the background loop)
# ══════════════════════════════════════════════════════════════════════════════

async def get_browser():
    """Return a connected browser instance, reconnecting if needed."""
    global _browser
    try:
        if _browser and _browser.is_connected():
            return _browser
    except Exception:
        pass

    log("Connecting to Chrome on port 9222...")
    pw = await async_playwright().start()
    _browser = await pw.chromium.connect_over_cdp(CHROME_DEBUG_URL)
    log(f"Connected: {_browser.version}")
    return _browser


async def get_or_open_gemini_page():
    """Find existing Gemini tab or open a new one."""
    browser = await get_browser()
    for ctx in browser.contexts:
        for pg in ctx.pages:
            if "gemini.google.com" in pg.url:
                return pg
    ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
    page = await ctx.new_page()
    await page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=30_000)
    await asyncio.sleep(2)
    return page


async def ensure_new_chat(page):
    """Always force-navigate to a brand new chat by going to /app directly.
    This guarantees a fresh conversation every single time, regardless of
    what was open before.
    """
    log("  Navigating to new chat...")
    await page.goto("https://gemini.google.com/app", wait_until="domcontentloaded", timeout=20_000)
    await asyncio.sleep(2)

    # Wait for the prompt input to be ready — confirms we are on a clean new chat
    try:
        await page.locator('[aria-label="Enter a prompt for Gemini"]').wait_for(state="visible", timeout=10_000)
        log("  New chat ready.")
    except PWTimeout:
        log("  Warning: prompt input not visible after navigation — continuing anyway.")


async def type_into_quill(page, text: str):
    """Inject text directly into Quill's internal API — no keyboard/clipboard events.
    This is the only method that guarantees the full prompt is inserted as one
    atomic operation without any Enter/newline triggering a premature send.
    """
    selector = '[aria-label="Enter a prompt for Gemini"]'

    # Flatten ALL whitespace variants to a single space
    import re
    flat = re.sub(r'[\r\n\t]+', ' ', text)
    flat = re.sub(r' {2,}', ' ', flat).strip()
    log(f"  Injecting prompt ({len(flat)} chars)...")

    editor = page.locator(selector)
    await editor.wait_for(state="visible", timeout=15_000)
    await editor.click()
    await asyncio.sleep(0.3)

    # Inject via Quill's own API — completely bypasses keyboard/clipboard events
    injected = await page.evaluate(f"""
        () => {{
            const el = document.querySelector('{selector}');
            if (!el) return false;

            // Try Quill instance first (most reliable)
            const quill = el.__quill || el._quill;
            if (quill) {{
                quill.setText('');
                quill.insertText(0, {repr(flat)});
                quill.setSelection(quill.getLength(), 0);
                return true;
            }}

            // Fallback: set innerText directly then fire input event
            el.focus();
            el.innerText = {repr(flat)};
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            // Move cursor to end
            const range = document.createRange();
            const sel = window.getSelection();
            range.selectNodeContents(el);
            range.collapse(false);
            sel.removeAllRanges();
            sel.addRange(range);
            return true;
        }}
    """)
    await asyncio.sleep(0.6)

    # Verify
    actual = (await editor.inner_text()).strip()
    if actual:
        log(f"  Injected OK ({len(actual)} chars).")
    else:
        log("  Warning: editor appears empty after injection.")


async def wait_for_image(page) -> list[str]:
    """Poll until a generated image appears. Returns list of src URLs."""
    start = time.time()
    while time.time() - start < MAX_WAIT_SEC:
        await asyncio.sleep(POLL_INTERVAL)

        # Still generating?
        is_loading = await page.evaluate("""
            () => {
                const stop = document.querySelector('[aria-label="Stop response"]');
                return stop !== null && stop.offsetWidth > 0;
            }
        """)
        if is_loading:
            continue

        images = await page.evaluate("""
            () => {
                const imgs = [];
                // Search in response containers first
                const responses = document.querySelectorAll(
                    'model-response, .model-response-text, response-container, [data-test-id="response-container"]'
                );
                if (responses.length > 0) {
                    const last = responses[responses.length - 1];
                    last.querySelectorAll('img').forEach(img => {
                        if (img.naturalWidth > 100 &&
                            !img.src.includes('avatar') && !img.src.includes('icon')) {
                            imgs.push(img.src);
                        }
                    });
                }
                // Fallback: all large images
                if (imgs.length === 0) {
                    document.querySelectorAll('img').forEach(img => {
                        if (img.naturalWidth > 200 &&
                            !img.src.includes('avatar') && !img.src.includes('icon') &&
                            !img.src.includes('logo')) {
                            imgs.push(img.src);
                        }
                    });
                }
                return imgs;
            }
        """)

        if images:
            return images

        # Check for error text in the response
        error = await page.evaluate("""
            () => {
                const responses = document.querySelectorAll('model-response, response-container');
                if (responses.length === 0) return null;
                const last = responses[responses.length - 1];
                const text = (last.innerText || '').toLowerCase();
                const hasError = text.includes("can't generate") ||
                                 text.includes("cannot generate") ||
                                 text.includes("unable to create") ||
                                 text.includes("i'm not able") ||
                                 text.includes("i can't") ||
                                 text.includes("error");
                return hasError ? last.innerText.slice(0, 300) : null;
            }
        """)
        if error:
            raise ValueError(f"Gemini refused or errored: {error}")

    raise TimeoutError(f"No image after {MAX_WAIT_SEC}s")


async def save_image_from_page(page, src: str, out_path: Path) -> bool:
    """Save image using XHR (avoids CORS) with element screenshot fallback."""

    # Strategy 1: XHR inside page context
    if src.startswith("blob:") or src.startswith("http"):
        try:
            data = await page.evaluate(f"""
                async () => {{
                    return new Promise((resolve, reject) => {{
                        const xhr = new XMLHttpRequest();
                        xhr.open('GET', {repr(src)}, true);
                        xhr.responseType = 'arraybuffer';
                        xhr.onload = function() {{
                            if (xhr.status === 0 || xhr.status === 200) {{
                                const bytes = new Uint8Array(xhr.response);
                                let binary = '';
                                const chunk = 8192;
                                for (let i = 0; i < bytes.length; i += chunk)
                                    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
                                resolve(btoa(binary));
                            }} else reject('HTTP ' + xhr.status);
                        }};
                        xhr.onerror = () => reject('XHR error');
                        xhr.send();
                    }});
                }}
            """)
            out_path.write_bytes(base64.b64decode(data))
            return True
        except Exception:
            pass

    # Strategy 2: Screenshot the img element
    try:
        img_el = await page.evaluate_handle(f"""
            () => {{
                for (const img of document.querySelectorAll('img'))
                    if (img.src === {repr(src)} || img.currentSrc === {repr(src)}) return img;
                // Fallback: largest image in last response
                const responses = document.querySelectorAll(
                    'model-response, response-container, [data-test-id="response-container"]'
                );
                if (responses.length > 0) {{
                    const imgs = [...responses[responses.length-1].querySelectorAll('img')]
                        .filter(i => i.naturalWidth > 100);
                    if (imgs.length) return imgs[0];
                }}
                return null;
            }}
        """)
        if img_el:
            el = img_el.as_element()
            await el.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)
            await el.screenshot(path=str(out_path))
            if out_path.stat().st_size > 1000:
                return True
    except Exception:
        pass

    return False


async def delete_conversation(page):
    """Delete current conversation via sidebar 3-dot menu."""
    try:
        # ── Step 1: Open sidebar if collapsed ─────────────────────────────
        conv_visible = await page.locator('a[data-test-id="conversation"]').first.is_visible()
        if not conv_visible:
            log("  Sidebar collapsed — opening via dispatch...")
            toggle = page.locator('[data-test-id="side-nav-menu-button"]').first
            await toggle.wait_for(state="visible", timeout=5_000)
            await toggle.dispatch_event("click")  # bypass overlapping elements
            await asyncio.sleep(0.8)

        # ── Step 2: Find conversation to delete ───────────────────────────
        # Try selected first, fall back to first in list
        conv = None
        try:
            selected = page.locator('a[data-test-id="conversation"].selected').first
            await selected.wait_for(state="visible", timeout=4_000)
            conv = selected
            log("  Found selected conversation.")
        except PWTimeout:
            try:
                first = page.locator('a[data-test-id="conversation"]').first
                await first.wait_for(state="visible", timeout=4_000)
                conv = first
                log("  No selected conv — using first in list.")
            except PWTimeout:
                log("  No conversations visible — nothing to delete.")
                return

        # ── Step 3: Hover to reveal 3-dot menu ────────────────────────────
        await conv.hover()
        await asyncio.sleep(0.5)

        # ── Step 4: Click 3-dot menu (dispatch to avoid overlay blocking) ──
        menu = page.locator('[data-test-id="actions-menu-button"]').first
        await menu.wait_for(state="visible", timeout=5_000)
        await menu.dispatch_event("click")
        await asyncio.sleep(0.4)

        # ── Step 5: Click Delete ───────────────────────────────────────────
        del_btn = page.locator('[data-test-id="delete-button"]').first
        await del_btn.wait_for(state="visible", timeout=5_000)
        await del_btn.click()
        await asyncio.sleep(0.4)

        # ── Step 6: Confirm dialog ─────────────────────────────────────────
        try:
            confirm = page.locator(
                'mat-dialog-container button:has-text("Delete"), [role="dialog"] button:has-text("Delete")'
            ).first
            await confirm.wait_for(state="visible", timeout=5_000)
            await confirm.click()
            await asyncio.sleep(0.8)
            log("  Conversation deleted.")
        except PWTimeout:
            log("  No confirm dialog — deleted silently.")

    except Exception as e:
        log(f"  Delete warning: {e}")

async def run_gemini_generation(prompt: str) -> dict:
    """
    Full flow: new chat → type prompt → send → wait for image → save → delete.
    Returns {"success": True, "image_b64": "...", "filename": "..."}
         or {"success": False, "error": "...", "retriable": True/False}
    """
    try:
        page = await get_or_open_gemini_page()
        await page.bring_to_front()
        await page.wait_for_load_state("domcontentloaded")
        await ensure_new_chat(page)

        log(f"Sending prompt: {prompt[:80]}...")
        await type_into_quill(page, prompt)

        # Send
        send_btn = page.locator('[aria-label="Send message"]').first
        try:
            await send_btn.wait_for(state="visible", timeout=8_000)
            await send_btn.click()
        except PWTimeout:
            await page.locator('[aria-label="Enter a prompt for Gemini"]').press("Enter")

        await asyncio.sleep(1)

        # Wait for image
        try:
            srcs = await wait_for_image(page)
        except ValueError as e:
            log(f"Gemini refused: {e}")
            await delete_conversation(page)
            return {"success": False, "error": str(e), "retriable": True}
        except TimeoutError as e:
            log(f"Timeout: {e}")
            await delete_conversation(page)
            return {"success": False, "error": str(e), "retriable": True}

        log(f"Found {len(srcs)} image(s). Saving...")

        # Save first image
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = IMAGES_DIR / f"cf_{ts}.png"

        saved = False
        last_err = ""
        for attempt, src in enumerate(srcs):
            log(f"  Save attempt {attempt+1}: src type={src[:30]}")
            try:
                saved = await save_image_from_page(page, src, out_path)
                if saved:
                    log(f"  Saved OK: {out_path} ({out_path.stat().st_size} bytes)")
                    break
            except Exception as e:
                last_err = str(e)
                log(f"  Save attempt {attempt+1} failed: {e}")

        # Delete conversation regardless of save success
        await delete_conversation(page)

        if not saved or not out_path.exists() or out_path.stat().st_size < 500:
            return {
                "success": False,
                "error": f"Image found in Gemini but could not be saved. Last error: {last_err}",
                "retriable": True
            }

        # Return as base64 so the browser can display it directly
        img_bytes = out_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        log(f"Returning image: {len(img_bytes)} bytes as base64")
        return {
            "success": True,
            "image_b64": img_b64,
            "filename": out_path.name,
            "path": str(out_path.resolve())
        }

    except Exception as e:
        import traceback
        log(f"Generation error: {e}")
        log(traceback.format_exc())
        return {"success": False, "error": str(e), "retriable": False}


def run_async(coro):
    """Run a coroutine on the shared background event loop from a sync Flask thread."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=MAX_WAIT_SEC + 60)


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
@app.route("/app")
def serve_frontend():
    """Serve the ComicForge HTML frontend."""
    return send_from_directory(Path(__file__).parent, "comicforge.html")


@app.route("/")
def index():
    """Serve the ComicForge UI — always open via http://localhost:5000"""
    return send_from_directory(".", "comicforge.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "chrome": CHROME_DEBUG_URL})


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    """
    Body: { "prompt": "...", "mode": "asset"|"scene" }
    Returns: { "success": true, "image_b64": "data:image/png;base64,..." }
          or { "success": false, "error": "...", "retriable": true }
    """
    try:
        data = request.get_json()
        if not data or not data.get("prompt"):
            return jsonify({"success": False, "error": "Missing prompt"}), 400

        prompt = data["prompt"]
        log(f"[API] generate-image request: {prompt[:60]}...")

        with _playwright_lock:
            result = run_async(run_gemini_generation(prompt))

        if result.get("success"):
            # Prefix with data URI header for direct use in <img src="...">
            result["image_b64"] = f"data:image/png;base64,{result['image_b64']}"

        return jsonify(result)

    except Exception as e:
        import traceback
        log(f"[API ERROR] /api/generate-image: {e}")
        log(traceback.format_exc())
        return jsonify({"success": False, "error": str(e), "retriable": True}), 200


@app.route("/api/generate-json", methods=["POST"])
def generate_json_endpoint():
    """
    Body: { "prompt": "...", "type": "character"|"style"|..., "model": "gpt-4o", "api_key": "sk-..." }
    Returns: { "success": true, "json_data": {...}, "name": "..." }
    """
    import json, re
    try:
        from openai import OpenAI
    except ImportError:
        return jsonify({"success": False, "error": "openai package not installed. Run: pip install openai"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No body"}), 400

    prompt   = data.get("prompt", "")
    obj_type = data.get("type", "character")
    model    = data.get("model", "gpt-4o")
    api_key  = data.get("api_key", "")

    if not prompt:
        return jsonify({"success": False, "error": "Missing prompt"}), 400
    if not api_key:
        return jsonify({"success": False, "error": "Missing api_key"}), 400

    system = f"""You are a comic book asset definition assistant. Generate detailed JSON for AI image generation consistency.

For a {obj_type}, return ONLY a valid JSON object (no markdown, no preamble) with:
- "name": short unique identifier
- "type": "{obj_type}"
- "description": 1-2 sentence visual summary
- "appearance": object with detailed visual traits (colors, features, build, clothing, marks)
- "art_style": object with style guidance (line_weight, shading, color_palette, rendering_style)
- "generation_notes": special tips for the image generator to ensure consistency

Be very specific. Return ONLY valid JSON."""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1400,
            temperature=0.7,
        )
        raw = response.choices[0].message.content or "{}"
        clean = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)
        name = parsed.get("name", f"{obj_type}_{int(time.time())}")
        return jsonify({"success": True, "json_data": parsed, "name": name})

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"Could not parse JSON from AI response: {e}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════════

def start_event_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


if __name__ == "__main__":
    # Start the asyncio loop in a daemon background thread
    t = threading.Thread(target=start_event_loop, daemon=True)
    t.start()
    # Give the loop a moment to start
    time.sleep(0.5)

    print("\n" + "="*60)
    print("  ComicForge Backend Server")
    print("="*60)
    print(f"  Listening on  : http://localhost:{PORT}")
    print(f"  Images saved  : {IMAGES_DIR.resolve()}")
    print(f"  Chrome debug  : {CHROME_DEBUG_URL}")
    print("="*60)
    print("\n  Make sure Chrome is running with --remote-debugging-port=9222")
    print("  Then open comicforge.html in your browser.\n")

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)