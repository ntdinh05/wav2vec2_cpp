cmake_minimum_required(VERSION 3.14)
project(Wav2VecCpp)

set(CMAKE_CXX_STANDARD 17)

# --- QT SETTINGS (REQUIRED FOR GUI) ---
# These automatically handle Qt's meta-object system (MOC) for signals/slots
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# --- 1. Setup Paths ---
set(LIBS_DIR ${CMAKE_SOURCE_DIR}/libs)
set(ONNX_DIR ${LIBS_DIR}/onnxruntime)


# --- 2. Include Directories ---
include_directories(
    ${LIBS_DIR}
    ${ONNX_DIR}/include
)

# --- 3. Link Directories ---
link_directories(
    ${ONNX_DIR}/lib
)

# --- 4. Find Packages ---
# Find Qt6 for the new GUI. If you installed Qt via Homebrew, this usually works automatically.
# If it fails, you may need to add: -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qt"
find_package(Qt6 REQUIRED COMPONENTS Core Gui Qml Quick)

# --- 5. Find ONNX Library ---
file(GLOB ONNX_LIB "${ONNX_DIR}/lib/libonnxruntime.*")


# =========================================================
# EXISTING TARGETS (Console Apps)
# =========================================================

# --- Auto Resample (The Backend for GUI) ---
add_executable(auto_resample src/realtime_autoresample.cpp)
target_link_libraries(auto_resample PRIVATE ${ONNX_LIB})
if(APPLE)
    target_link_libraries(auto_resample PRIVATE "-framework CoreAudio" "-framework AudioToolbox" "-framework CoreFoundation")
    add_custom_command(TARGET auto_resample POST_BUILD
        COMMAND install_name_tool -add_rpath "@executable_path/../libs/onnxruntime/lib" $<TARGET_FILE:auto_resample>)
elseif(UNIX)
    target_link_libraries(auto_resample PRIVATE pthread dl)
endif()


# =========================================================
# NEW GUI TARGET (Qt C++)
# =========================================================

# 1. Create the executable from the new C++ GUI file
# Ensure you placed 'gui_main.cpp' in your 'src' folder
add_executable(InferenceRunner app/gui_main.cpp)

# 2. Link Qt Libraries
target_link_libraries(InferenceRunner PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Qml
    Qt6::Quick
)

# 3. Copy main.qml to the build directory
# This assumes you moved 'main.qml' into your 'src' folder.
# If it is still in 'app/', change "src/main.qml" to "app/main.qml" below.
file(COPY app/main.qml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})