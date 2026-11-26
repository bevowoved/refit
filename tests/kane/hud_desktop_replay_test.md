---
mode: testing
url: https://bevowoved.github.io/refit/?demo=1&at=100
tags: [smoke, desktop, gate4, hud]
max_steps: 20
timeout: 120
headless: true
---

# refit Gate 4 Live HUD & Closed-Loop Proof (Desktop Chrome)

## Open the Gate 4 Kane Drawer and verify initial GREEN proof
Open https://bevowoved.github.io/refit/?demo=1&at=100 and wait for the page to finish loading.
Click the "Gate 4 · Kane" button in the top navigation bar.
Verify the drawer opens and displays "PASSED".
Verify the drawer displays "12 / 100 cap" and "TRUE".

## Cycle the Closed-Loop Red -> Repair -> Green trace and close cleanly
Click the "Cycle Red -> Repair -> Green" button inside the drawer.
Verify the drawer displays "BUG_740" and "LIE".
Click the "Cycle Red -> Repair -> Green" button a second time.
Verify the drawer displays "REPAIRED".
Click the "Close" button and verify the drawer closes.
