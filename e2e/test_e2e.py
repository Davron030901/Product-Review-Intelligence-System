"""End-to-end browser tests against a real backend and a real production build.

These cover what jsdom cannot: chart rendering, CSS-driven responsive
behaviour, actual layout overflow, and the frontend talking to the live API.

Prerequisites -- both must be running:

    # terminal 1
    cd backend && PYTHONPATH=$PWD ALLOWED_ORIGINS=http://localhost:4173 \
        uvicorn src.api.main:app --port 8000

    # terminal 2
    cd frontend && VITE_USE_MOCK=false VITE_API_BASE_URL=http://localhost:8000 \
        npm run build && npx vite preview --port 4173

    # then
    pip install playwright && playwright install chromium
    python e2e/test_e2e.py
"""
from playwright.sync_api import sync_playwright, expect
import sys

fails = []
def check(name, fn):
    try:
        fn(); print(f"PASS  {name}")
    except Exception as e:
        fails.append(name); print(f"FAIL  {name}: {str(e)[:200]}")

with sync_playwright() as p:
    b = p.chromium.launch()
    errs = []
    pg = b.new_page(viewport={"width":1440,"height":950})
    pg.on("console", lambda m: errs.append(m.text) if m.type=="error" else None)
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.goto("http://localhost:4173/", wait_until="networkidle")
    pg.wait_for_timeout(1500)

    # 1. real backend analyze
    def t1():
        pg.fill("#review-text", "Arrived two weeks late and the box was completely crushed. Returning it.")
        pg.get_by_role("button", name="Analyze review").click()
        expect(pg.get_by_text("Negative").first).to_be_visible(timeout=10000)
        expect(pg.get_by_text("delivery").first).to_be_visible()
    check("analyze against real API", t1)

    # 2. model version proves it's the real backend, not mock
    def t2():
        expect(pg.get_by_text("model v1-baseline")).to_be_visible()
    check("served by real model (not mock)", t2)

    # 3. low confidence stamp
    def t3():
        pg.fill("#review-text", "meh")
        pg.get_by_role("button", name="Analyze review").click()
        expect(pg.get_by_text("Why this was flagged")).to_be_visible(timeout=10000)
    check("low-confidence stamp appears", t3)

    # 4. empty input -> button disabled
    def t4():
        pg.fill("#review-text", "")
        expect(pg.get_by_role("button", name="Analyze review")).to_be_disabled()
    check("empty input disables submit", t4)

    # 5. dashboard renders charts
    def t5():
        pg.get_by_role("button", name="Overview").first.click()
        pg.wait_for_timeout(1200)
        expect(pg.get_by_text("Reviews analyzed")).to_be_visible()
        assert pg.locator("svg.recharts-surface").count() >= 2, "charts missing"
    check("dashboard charts render", t5)

    # 6. filter interaction
    def t6():
        pg.select_option("#filter-sentiment", "negative")
        pg.wait_for_timeout(600)
        expect(pg.get_by_text("matching this filter.")).to_be_visible()
    check("dashboard filter works", t6)

    # 7. queue resolve
    def t7():
        pg.get_by_role("button", name="Review queue").first.click()
        pg.wait_for_timeout(800)
        tab = pg.get_by_role("tab", name="Waiting")
        before = int("".join(c for c in tab.inner_text() if c.isdigit()))
        pg.locator("article button").filter(has_text="Negative").first.click()
        pg.wait_for_timeout(600)
        after = int("".join(c for c in pg.get_by_role("tab", name="Waiting").inner_text() if c.isdigit()))
        assert after == before-1, f"{before} -> {after}"
    check("queue resolution works", t7)

    # 8. keyboard focus visible
    def t8():
        pg.get_by_role("button", name="Analyze", exact=False).first.click()
        pg.wait_for_timeout(500)
        pg.keyboard.press("Tab"); pg.keyboard.press("Tab")
        assert pg.evaluate("document.activeElement.tagName") not in ("BODY",)
    check("keyboard navigation reaches controls", t8)

    # 9. no horizontal scroll at mobile
    def t9():
        m = b.new_page(viewport={"width":360,"height":800})
        m.goto("http://localhost:4173/", wait_until="networkidle"); m.wait_for_timeout(1500)
        for label in ["Overview","Review queue","Analyze"]:
            m.get_by_role("button", name=label).first.click(); m.wait_for_timeout(1000)
            sw = m.evaluate("document.documentElement.scrollWidth")
            cw = m.evaluate("document.documentElement.clientWidth")
            assert sw <= cw+2, f"{label}: horizontal overflow {sw}>{cw}"
        m.close()
    check("no horizontal overflow at 360px", t9)

    # 10. touch targets
    def t10():
        m = b.new_page(viewport={"width":360,"height":800})
        m.goto("http://localhost:4173/", wait_until="networkidle"); m.wait_for_timeout(1500)
        # Only measure what is actually rendered: the desktop rail is in the
        # DOM at this width but display:none, so it reports height 0.
        small = m.evaluate("""() => [...document.querySelectorAll('nav button')]
            .filter(e => e.getBoundingClientRect().height > 0)
            .map(e=>({t:e.innerText.trim(), h:e.getBoundingClientRect().height}))
            .filter(x=>x.h<44)""")
        assert not small, f"small nav targets: {small}"
        m.close()
    check("mobile nav touch targets >= 44px", t10)

    print("\nconsole errors:", [e for e in errs if "favicon" not in e.lower()][:5])
    b.close()

print("\n" + ("ALL E2E PASSED" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
