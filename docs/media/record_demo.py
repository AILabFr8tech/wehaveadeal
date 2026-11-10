"""Re-record the README demo (docs/media/live-negotiation.gif and .mp4).

Maintainer tool, not part of the package. It drives the local dashboard in a
headless browser and records the session.

    pip install playwright imageio-ffmpeg && python -m playwright install chromium
    cp examples/linreg_carrier.py agents/ && cp -r examples/linreg_carrier_model agents/
    wehaveadeal serve                                   # terminal 1
    cd dashboard && npm run build && npm run start      # terminal 2 (production build: no dev badge)
    python docs/media/record_demo.py docs/media/_raw    # writes a .webm

Then convert (ffmpeg ships with imageio-ffmpeg):

    ffmpeg -ss 0.8 -i RAW.webm -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart -an live-negotiation.mp4
    ffmpeg -ss 0.8 -i RAW.webm -vf "setpts=PTS/1.5,fps=8,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" live-negotiation.gif

Session: 3 shippers (OPT, ML2, RL2) x 3 carriers (LINREGCARRIER, OPT2, RL), pinned to
seed 21, a long negotiation with no walk-aways that LINREGCARRIER wins.
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1])
BASE = "http://localhost:3000"
W, H = 1280, 800

CURSOR_JS = """
window.addEventListener('DOMContentLoaded', () => {
  const c = document.createElement('div');
  c.id = '__demo_cursor';
  Object.assign(c.style, {
    position: 'fixed', left: '0px', top: '0px', width: '18px', height: '18px', marginLeft: '-9px', marginTop: '-9px',
    borderRadius: '50%', background: 'rgba(204, 90, 18, 0.55)', border: '2px solid rgba(74, 27, 184, 0.9)',
    zIndex: 2147483647, pointerEvents: 'none', transition: 'transform 120ms ease', boxShadow: '0 0 6px rgba(0,0,0,0.25)'
  });
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', () => { c.style.transform = 'scale(0.7)'; }, true);
  document.addEventListener('mouseup', () => { c.style.transform = 'scale(1)'; }, true);
});
"""


def glide_click(page, locator, pause=0.45):
    locator.scroll_into_view_if_needed()
    box = locator.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)
    time.sleep(0.15)
    page.mouse.down(); page.mouse.up()  # a real click at the cursor's position
    time.sleep(pause)


def smooth_scroll_to(page, y, duration=1.2):
    page.evaluate(
        """([y, ms]) => new Promise(res => {
             const y0 = window.scrollY, t0 = performance.now();
             const step = t => { const k = Math.min(1, (t - t0) / ms), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;
               window.scrollTo(0, y0 + (y - y0) * e); k < 1 ? requestAnimationFrame(step) : res(); };
             requestAnimationFrame(step); })""",
        [y, int(duration * 1000)],
    )


def engine_button(page, side, label):
    return page.locator(
        f"xpath=//h3[contains(., '{side} engines')]/following-sibling::div//button[normalize-space()='{label}']"
    )


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="light",
                              record_video_dir=str(OUT), record_video_size={"width": W, "height": H})
    ctx.add_init_script(CURSOR_JS)
    page = ctx.new_page()
    # Pin the market: seed 21 gives a long negotiation (77 offers, no walk-aways) that LINREGCARRIER wins.
    page.route("**/api/live/negotiation?*", lambda route: route.continue_(url=route.request.url + "&seed=21"))
    page.goto(f"{BASE}/live")
    page.wait_for_selector("text=market scenario")
    engine_button(page, "carrier", "LINREGCARRIER").wait_for()
    page.mouse.move(W / 2, 120)
    time.sleep(1.5)

    # 4 -> 3 shippers with the slider (keyboard keeps it exact)
    slider = page.locator("input[type=range]").first
    glide_click(page, slider, pause=0.2)
    while page.locator("input[type=range]").first.input_value() != "3":
        page.keyboard.press("ArrowLeft"); time.sleep(0.25)
    time.sleep(0.5)

    for label in ["OPT", "ML2", "RL2"]:
        glide_click(page, engine_button(page, "shipper", label))
    for label in ["LINREGCARRIER", "OPT2", "RL"]:
        glide_click(page, engine_button(page, "carrier", label))
    page.wait_for_selector("text=shipper engines — pick 1 to 3 (3 picked")
    page.wait_for_selector("text=carrier engines — pick 1 to 3 (3 picked")
    time.sleep(0.6)

    glide_click(page, page.get_by_role("button", name="▶ start live negotiation"), pause=0.2)
    page.wait_for_selector("text=bilateral tables — building turn by turn")
    time.sleep(0.8)

    section = page.locator("h2", has_text="bilateral tables").first
    smooth_scroll_to(page, section.evaluate("e => e.getBoundingClientRect().top + window.scrollY - 70"))
    page.mouse.move(W - 60, H - 60, steps=10)
    page.wait_for_selector("text=turn 7/12", timeout=120_000)  # first row has closed by turn 5
    time.sleep(1.0)
    # follow the action: rows 2-3 keep negotiating until turn 12
    row2 = page.locator("section div.grid > div").nth(3)
    smooth_scroll_to(page, row2.evaluate("e => e.getBoundingClientRect().top + window.scrollY - 16"), 1.4)
    page.wait_for_selector("text=status: finished", timeout=120_000)
    time.sleep(2.5)  # all deals closed

    matching = page.locator("h2", has_text="Gale–Shapley matching")
    smooth_scroll_to(page, matching.evaluate("e => e.getBoundingClientRect().top + window.scrollY - 40"), 1.6)
    time.sleep(4.5)

    video = page.video
    ctx.close()
    browser.close()
    print("recorded:", video.path())
