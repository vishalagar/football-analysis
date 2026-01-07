
import os
import re

def log_test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return passed

def run_ui_audit():
    print("\n--- Starting Headless UI/UX Structural Audit ---\n")
    
    frontend_dir = "frontend"
    index_html = os.path.join(frontend_dir, "index.html")
    style_css = os.path.join(frontend_dir, "style.css")
    
    results = []

    # 1. Check index.html for key components
    if os.path.exists(index_html):
        with open(index_html, 'r', encoding='utf-8') as f:
            html = f.read()
            results.append(log_test("HTML: Agent Analysis Section", "agent-card" in html))
            results.append(log_test("HTML: Cleaning Section", "cleaning-section" in html))
            results.append(log_test("HTML: Benchmark Section", "training-section" in html))
            results.append(log_test("HTML: Leaderboard", "leaderboard-content" in html))
            results.append(log_test("HTML: Modern Font", "Outfit" in html))
    else:
        log_test("HTML Exists", False, "index.html not found")

    # 2. Check style.css for Premium Design Markers
    if os.path.exists(style_css):
        with open(style_css, 'r', encoding='utf-8') as f:
            css = f.read()
            
            # Glassmorphism
            has_glass = "backdrop-filter" in css or "rgba" in css
            results.append(log_test("CSS: Glassmorphism Style", has_glass, "Detected backdrop-filter/rgba"))
            
            # Dark Mode / Modern Palette
            # Look for dark bg colors and accent colors
            has_dark_mode = "#0f172a" in css.lower() or "#1e293b" in css.lower() or "background: #000" in css.lower()
            results.append(log_test("CSS: Dark Mode Palette", has_dark_mode, "Detected deep slate/blue backgrounds"))
            
            # Glow effects (box-shadow)
            has_glow = "box-shadow" in css and "rgba" in css
            results.append(log_test("CSS: Glow/Shadow Effects", has_glow))
            
            # Animations
            has_animations = "@keyframes" in css or "transition" in css
            results.append(log_test("CSS: Smooth Animations", has_animations))

            # ResNet18 focus in labels/text (though usually in HTML)
            results.append(log_test("CSS: Variable Usage", "var(--" in css))
    else:
        log_test("CSS Exists", False, "style.css not found")

    total = len(results)
    passed = sum(1 for r in results if r)
    print(f"\n--- Audit Summary: {passed}/{total} markers detected ---\n")
    return passed == total

if __name__ == "__main__":
    success = run_ui_audit()
    if not success:
        exit(1)
