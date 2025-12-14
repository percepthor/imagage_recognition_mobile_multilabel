# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(image_recognition_library VERSION 0.0.1 LANGUAGES C)

add_library(image_recognition SHARED
  "image_recognition.c"
)

set_target_properties(image_recognition PROPERTIES
  PUBLIC_HEADER image_recognition.h
  OUTPUT_NAME "image_recognition"
)

target_compile_definitions(image_recognition PUBLIC DART_SHARED_LIB)
