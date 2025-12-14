Place your precompiled .xcframework here:

- image_recognition.xcframework/

The .xcframework should contain binaries for both device and simulator:
- ios-arm64 (device)
- ios-arm64-simulator (simulator)
- ios-x86_64-simulator (simulator for Intel Macs)

This directory is configured in image_recognition.podspec to be included as a vendored framework.
