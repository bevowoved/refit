# refit Live Refactoring HUD & README Truth Requirements

## Target Application
Live URL: https://patrickfield152-del.github.io/refit/?demo=1&at=100

## UC-1: Interactive Replay HUD & Gate 4 Kane Inspector (Desktop Chrome)
1. Opening the live URL renders the refit Live Refactoring HUD with the "refit" wordmark, Replay Controls bar, Topological Wavefront grid, and the "Gate 4 · Kane" button in the top status bar.
2. Clicking "Gate 4 · Kane" opens the 2-tab MVCC & Gate 4 drawer directly to the "Gate 4 · Kane & README Truth" tab showing the "PASSED" verdict badge, the Antigravity MCP tool-cap comparator ("12 / 100 cap" vs "92 / 100 cap"), Wikimedia Commons archival specimen plates, and the Dual-Layer README Truth Matrix with "TRUE" statuses.
3. Zero uncaught browser console errors and zero 4xx/5xx page network responses occur during load and interaction.

## UC-2: Closed-Loop Red -> Repair -> Green Cycle & Drawer Navigation
1. Clicking "Cycle Red -> Repair -> Green" transitions the Gate 4 verdict badge from "PASSED" to "BUG_740", updates the step trace to show "FAIL Step 2 [AGENT_INFO · result_code 740]", and flags the broken UI claim as "LIE" in the README Truth Matrix.
2. Clicking "Cycle Red -> Repair -> Green" a second time transitions the verdict badge to "REPAIRED", updates the step trace to "PASS Step 2 [Kane CLI Re-Verify]", and restores all README Truth Matrix rows to "TRUE".
3. Clicking "Close" dismisses the drawer cleanly with zero console exceptions.

## UC-3: Mobile Viewport Responsiveness (390x844) & Cloud Mobile Spec
1. Loading the live URL with `?tab=kane` at a 390x844 mobile viewport renders the Gate 4 drawer cleanly within the viewport without horizontal clipping or page crash.
2. Clicking "Cycle Red -> Repair -> Green" on mobile updates the verdict badge to "BUG_740".
