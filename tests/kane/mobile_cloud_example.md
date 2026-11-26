# Native Mobile Cloud Spec Template (TestMu AI Cloud Real Devices / Emulators)
# Intentionally named without `_test.md` suffix so desktop `kane-cli testrun run` never triggers `mobile_remote_mixed`.
# Copy to `tests/kane/mobile_android_test.md` and run from any OS (Windows/Linux/macOS):
#   kane-cli testrun run tests/kane/mobile_android_test.md --remote --device-name "Pixel 7" --os-version 14 --dry-run --agent
#
# Valid Kane CLI mobile frontmatter:
# ---
# mode: testing
# target: android
# device_name: Pixel 7
# os_version: "14"
# url: https://patrickfield152-del.github.io/refit/?demo=1&at=100&tab=kane
# tags: [mobile, cloud, gate4]
# timeout: 180
# ---
#
# ## Verify refit Gate 4 HUD on Android Real Device
# Open https://patrickfield152-del.github.io/refit/?demo=1&at=100&tab=kane on the mobile browser.
# Tap "Cycle Red -> Repair -> Green" and verify the verdict updates from "PASSED" to "BUG_740".
