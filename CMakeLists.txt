cmake_minimum_required(VERSION 3.14)
project(Wav2VecCpp)

set(CMAKE_CXX_STANDARD 17)

# --- 1. Setup Paths ---
# Define where our libs are
set(LIBS_DIR ${CMAKE_SOURCE_DIR}/libs)
set(ONNX_DIR ${LIBS_DIR}/onnxruntime)

# --- 2. Include Directories ---
# Tell C++ where to find the .h files
include_directories(
    ${LIBS_DIR}                  # For AudioFile.h and json.hpp
    ${ONNX_DIR}/include          # For onnxruntime_cxx_api.h
)

# --- 3. Link Directories ---
# Tell C++ where to find the .dylib / .so / .dll files
link_directories(
    ${ONNX_DIR}/lib
)

# --- 4. Create Executable ---
add_executable(voice_model src/main.cpp)

# --- 5. Link Libraries ---
# Find the specific library file inside the onnxruntime/lib folder
# Note: The name might vary slightly by OS (e.g., libonnxruntime.dylib or libonnxruntime.so)
file(GLOB ONNX_LIB "${ONNX_DIR}/lib/libonnxruntime.*") 

target_link_libraries(voice_model PRIVATE ${ONNX_LIB})

# --- 6. Post-Build Command (MacOS/Linux) ---
# This ensures your executable can find the ONNX library when you run it
if(APPLE)
    add_custom_command(TARGET voice_model POST_BUILD
        COMMAND install_name_tool -add_rpath "@executable_path/../libs/onnxruntime/lib" $<TARGET_FILE:voice_model>)
endif()