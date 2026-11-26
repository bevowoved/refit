---
mode: testing
url: https://bevowoved.github.io/refit/?demo=1&at=100&tab=kane
tags: [smoke, mobile, gate4]
max_steps: 15
timeout: 120
headless: true
---

# refit Gate 4 Mobile Responsive Proof (390x844)

## Verify Gate 4 Kane drawer and closed-loop cycle on mobile viewport
Open https://bevowoved.github.io/refit/?demo=1&at=100&tab=kane and wait for the Gate 4 drawer to render cleanly.
Store the displayed verdict badge text as 'mobile_initial_verdict'.
Verify the Gate 4 drawer displays "PASSED" and the "Cycle Red -> Repair -> Green" button is visible and clickable without page crash.
Click the "Cycle Red -> Repair -> Green" button and verify the verdict badge updates to "BUG_740" with zero console errors.
