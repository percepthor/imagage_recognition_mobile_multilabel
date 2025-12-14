#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint image_recognition.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'image_recognition'
  s.version          = '0.0.1'
  s.summary          = 'Flutter FFI wrapper for multi-label image recognition engine.'
  s.description      = <<-DESC
A Flutter FFI plugin that provides a high-performance interface to a native
image recognition engine using persistent isolates and zero-copy data transfer.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }

  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  # Precompiled native framework (statically linked)
  # Place your image_recognition.xcframework in the Frameworks directory
  s.vendored_frameworks = 'Frameworks/image_recognition.xcframework'

  # If your engine uses C++ internally, uncomment this:
  # s.libraries = 'c++'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'
end
