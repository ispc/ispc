cmake_minimum_required(VERSION 3.16)
project(ISPCDriverExample VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ISPC binary and derive paths from it
find_program(ISPC_EXECUTABLE
    NAMES ispc
    PATHS
        /usr/local/bin
        /usr/bin
        /opt/ispc/bin
        ${CMAKE_PREFIX_PATH}/bin
    DOC "ISPC compiler executable"
)

if(NOT ISPC_EXECUTABLE)
    message(FATAL_ERROR "ISPC executable not found. Please install ISPC or add it to PATH")
endif()

# Get the directory containing the ISPC binary
get_filename_component(ISPC_BIN_DIR ${ISPC_EXECUTABLE} DIRECTORY)

# Derive lib and include directories from bin directory
get_filename_component(ISPC_ROOT_DIR ${ISPC_BIN_DIR} DIRECTORY)
set(ISPC_LIB_DIR "${ISPC_ROOT_DIR}/lib")
set(ISPC_INCLUDE_DIR "${ISPC_ROOT_DIR}/include")

# Verify the directories exist
if(NOT EXISTS ${ISPC_LIB_DIR})
    message(FATAL_ERROR "ISPC library directory not found: ${ISPC_LIB_DIR}")
endif()

if(NOT EXISTS ${ISPC_INCLUDE_DIR})
    message(FATAL_ERROR "ISPC include directory not found: ${ISPC_INCLUDE_DIR}")
endif()

message(STATUS "Found ISPC executable: ${ISPC_EXECUTABLE}")
message(STATUS "Using ISPC lib directory: ${ISPC_LIB_DIR}")
message(STATUS "Using ISPC include directory: ${ISPC_INCLUDE_DIR}")

# Create the executable
add_executable(simple simple.cpp)

# Add include directory
target_include_directories(simple PRIVATE ${ISPC_INCLUDE_DIR})

# Add library directory and link ISPC library
target_link_directories(simple PRIVATE ${ISPC_LIB_DIR})
target_link_libraries(simple ispc)

# Set RPATH so the executable can find libispc at runtime
set_target_properties(simple PROPERTIES
    INSTALL_RPATH ${ISPC_LIB_DIR}
    BUILD_RPATH ${ISPC_LIB_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Copy simple.ispc to build directory so the example can find it
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/simple.ispc ${CMAKE_CURRENT_BINARY_DIR}/simple.ispc COPYONLY)