---
mode: testing
url: https://patrickfield152-del.github.io/refit/?demo=1&at=100
tags: [smoke, desktop, gate4, hud]
max_steps: 20
timeout: 120
headless: true
---

# refit Gate 4 Live HUD & Closed-Loop Proof (Desktop Chrome)

## Open the Gate 4 Kane Drawer and verify initial GREEN proof + clean DevTools
Open https://patrickfield152-del.github.io/refit/?demo=1&at=100 and wait for the refit HUD cards and top status bar to render.
Click the "Gate 4 · Kane" button in the top status bar (perform the real click — never pass by reading button text alone).
Store the displayed verdict badge text as 'initial_verdict'.
Verify the drawer opens and displays "PASSED", the Antigravity MCP budget "12 / 100 cap", and the README Truth matrix showing "TRUE" entries.
Verify there are zero browser console errors and no failed 4xx or 5xx document/script requests.

## Cycle the Closed-Loop Red -> Repair -> Green trace and assert live DOM transitions
Click the "Cycle Red -> Repair -> Green" button inside the Gate 4 drawer.
Verify the verdict badge changes to "BUG_740", the step trace shows "AGENT_INFO", and the README Truth matrix displays "LIE".
Click the "Cycle Red -> Repair -> Green" button a second time.
Verify the verdict badge changes to "REPAIRED" and all README Truth rows return to "TRUE".
Click the "Close" button and verify the Gate 4 drawer closes cleanly with zero browser console errors.
