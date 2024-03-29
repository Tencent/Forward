cmake_minimum_required (VERSION 3.12)
project (HDF5 C)

if (POLICY CMP0074)
  cmake_policy (SET CMP0074 NEW)
endif ()

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 OLD)
endif()

if (POLICY CMP0083)
  cmake_policy (SET CMP0083 NEW)
endif ()

#-----------------------------------------------------------------------------
# Instructions for use : Sub-Project Build
#
# To include HDF5 as a sub-project within another project.
# Set HDF5_EXTERNALLY_CONFIGURED to 1 in the parent project and
# supply values for the following variables...
#
# HDF5_EXPORTED_TARGETS :
#   Set this to the name of the targets variable which controls exports
#   If unset (because parent project does not support/use the
#   install (EXPORT target...) syntax), then targets are not configured
#   for export during install.
#
# HDF5_LIB_DEPENDENCIES :
#   If the build of HDF5 libs is being customized, then rules for the
#   dependencies of the HDF5 libs may be 'incomplete', add additional
#   dependencies to this variable so that external projects pick them up
#
#option (HDF5_EXTERNAL_LIB_PREFIX "Use prefix for custom library naming." "")
set (HDF5_EXTERNAL_LIB_PREFIX "" CACHE STRING "Use prefix for custom library naming.")
mark_as_advanced (HDF5_EXTERNAL_LIB_PREFIX)
# HDF5_EXTERNAL_LIB_PREFIX :
#   If the parent project needs to install hdf libraries, but avoid
#   name conflicts with system versions, then a prefix may be added
#   to ensure that the correct versions configured are used.
#
# HDF5_INSTALL_BIN_DIR, HDF5_INSTALL_LIB_DIR, HDF5_INSTALL_INCLUDE_DIR, HDF5_INSTALL_DATA_DIR :
#   Customize the 'bin', 'lib', 'include', and 'share' installation directories.
#
# HDF5_INSTALL_NO_DEVELOPMENT :
#   Set to true to skip installation of headers and CMake package files.
#
# Consider this example from the ParaView project, it builds its own zlib
# library and tells HDF5 to add it as a dependency - this ensures that
# any project making use of this build of HDF5 will use the correct zlib
#
#   # Tell hdf5 that we are manually overriding certain settings
#   set (HDF5_EXTERNALLY_CONFIGURED 1)
#   # Avoid duplicating names of installed libraries
#   set (HDF5_EXTERNAL_LIB_PREFIX "vtk")
#   # Export configuration to this export variable
#   set (HDF5_EXPORTED_TARGETS "paraview-targets")
#
#   # Setup all necessary overrides for zlib so that HDF5 uses our
#   # internally compiled zlib rather than any other version
#   if (HDF5_ENABLE_Z_LIB_SUPPORT)
#     # We must tell the main HDF5 library that it depends on our zlib
#     set (HDF5_LIB_DEPENDENCIES vtkzlib)
#     # Override the zlib header file
#     if (VTK_USE_SYSTEM_ZLIB)
#       set (H5_ZLIB_HEADER "zlib.h")
#     else ()
#       set (H5_ZLIB_HEADER "vtk_zlib.h")
#       # Set vars that FindZlib would have set if used in sub project
#       set (ZLIB_INCLUDE_DIRS "${VTK_ZLIB_INCLUDE_DIRS}")
#       set (ZLIB_LIBRARIES vtkzlib)
#     endif ()
#   endif ()
#
#   # Add the sub project
#   add_subdirectory (Utilities/hdf5-1.8)
#-----------------------------------------------------------------------------
string (TIMESTAMP CONFIG_DATE "%Y-%m-%d")

#-----------------------------------------------------------------------------
# Set the core names of all the libraries
#-----------------------------------------------------------------------------
set (HDF5_LIB_CORENAME              "hdf5")
set (HDF5_CPP_LIB_CORENAME          "hdf5_cpp")

#-----------------------------------------------------------------------------
# Set the true names of all the libraries if customized by external project
#-----------------------------------------------------------------------------
set (HDF5_LIB_NAME              "${HDF5_EXTERNAL_LIB_PREFIX}${HDF5_LIB_CORENAME}")
set (HDF5_CPP_LIB_NAME          "${HDF5_EXTERNAL_LIB_PREFIX}${HDF5_CPP_LIB_CORENAME}")

#-----------------------------------------------------------------------------
# Set the target names of all the libraries
#-----------------------------------------------------------------------------
set (HDF5_LIB_TARGET              "${HDF5_LIB_CORENAME}")
set (HDF5_CPP_LIB_TARGET          "${HDF5_CPP_LIB_CORENAME}")

#-----------------------------------------------------------------------------
# Define some CMake variables for use later in the project
#-----------------------------------------------------------------------------
set (HDF_RESOURCES_DIR       ${HDF5_SOURCE_DIR}/config/cmake)
set (HDF_RESOURCES_EXT_DIR   ${HDF5_SOURCE_DIR}/config/cmake_ext_mod)
set (HDF5_SRC_DIR            ${HDF5_SOURCE_DIR}/src)
set (HDF5_CPP_SRC_DIR        ${HDF5_SOURCE_DIR}/c++)

#-----------------------------------------------------------------------------
# parse the full version number from H5public.h and include in H5_VERS_INFO
#-----------------------------------------------------------------------------
file (READ ${HDF5_SRC_DIR}/H5public.h _h5public_h_contents)
string (REGEX REPLACE ".*#define[ \t]+H5_VERS_MAJOR[ \t]+([0-9]*).*$"
    "\\1" H5_VERS_MAJOR ${_h5public_h_contents})
string (REGEX REPLACE ".*#define[ \t]+H5_VERS_MINOR[ \t]+([0-9]*).*$"
    "\\1" H5_VERS_MINOR ${_h5public_h_contents})
string (REGEX REPLACE ".*#define[ \t]+H5_VERS_RELEASE[ \t]+([0-9]*).*$"
    "\\1" H5_VERS_RELEASE ${_h5public_h_contents})
string (REGEX REPLACE ".*#define[ \t]+H5_VERS_SUBRELEASE[ \t]+\"([0-9A-Za-z._]*)\".*$"
    "\\1" H5_VERS_SUBRELEASE ${_h5public_h_contents})
#message (STATUS "VERSION: ${H5_VERS_MAJOR}.${H5_VERS_MINOR}.${H5_VERS_RELEASE}-${H5_VERS_SUBRELEASE}")

# Basic HDF5 stuff here
#-----------------------------------------------------------------------------
set (HDF5_PACKAGE "hdf5")
set (HDF5_PACKAGE_NAME "HDF5")
set (HDF5_PACKAGE_VERSION "${H5_VERS_MAJOR}.${H5_VERS_MINOR}.${H5_VERS_RELEASE}")
set (HDF5_PACKAGE_VERSION_MAJOR "${H5_VERS_MAJOR}.${H5_VERS_MINOR}")
set (HDF5_PACKAGE_VERSION_MINOR "${H5_VERS_RELEASE}")
if (H5_VERS_SUBRELEASE)
  set (HDF5_PACKAGE_VERSION_STRING "${HDF5_PACKAGE_VERSION}-${H5_VERS_SUBRELEASE}")
else ()
  set (HDF5_PACKAGE_VERSION_STRING "${HDF5_PACKAGE_VERSION}")
endif ()
if (H5_CXX_SOVERS_EXISTS)
  set (HDF5_CXX_PACKAGE_SOVERSION "${H5_CXX_SOVERS_MAJOR}.${H5_CXX_SOVERS_RELEASE}.${H5_CXX_SOVERS_MINOR}")
  set (HDF5_CXX_PACKAGE_SOVERSION_MAJOR "${H5_CXX_SOVERS_MAJOR}")
else ()
  set (HDF5_CXX_PACKAGE_SOVERSION "${H5_LIB_SOVERS_MAJOR}.${H5_LIB_SOVERS_RELEASE}.${H5_LIB_SOVERS_MINOR}")
  set (HDF5_CXX_PACKAGE_SOVERSION_MAJOR "${H5_LIB_SOVERS_MAJOR}")
endif ()
set (HDF5_PACKAGE_STRING "${HDF5_PACKAGE_NAME} ${HDF5_PACKAGE_VERSION_STRING}")
set (HDF5_PACKAGE_TARNAME "${HDF5_PACKAGE}${HDF_PACKAGE_EXT}")
set (HDF5_PACKAGE_URL "http://www.hdfgroup.org")
set (HDF5_PACKAGE_BUGREPORT "help@hdfgroup.org")

#-----------------------------------------------------------------------------
# Include some macros for reusable code
#-----------------------------------------------------------------------------
include (${HDF_RESOURCES_EXT_DIR}/HDFMacros.cmake)

HDF_DIR_PATHS(${HDF5_PACKAGE_NAME})

include (${HDF_RESOURCES_EXT_DIR}/HDFLibMacros.cmake)
include (${HDF_RESOURCES_DIR}/HDF5Macros.cmake)

#-----------------------------------------------------------------------------
# Targets built within this project are exported at Install time for use
# by other projects.
#-----------------------------------------------------------------------------
if (NOT HDF5_EXPORTED_TARGETS)
  set (HDF5_EXPORTED_TARGETS "hdf5-targets")
endif ()

#-----------------------------------------------------------------------------
# To include a library in the list exported by the project AT BUILD TIME,
# add it to this variable. This is NOT used by Make Install, but for projects
# which include hdf5 as a sub-project within their build tree
#-----------------------------------------------------------------------------
set_global_variable (HDF5_LIBRARIES_TO_EXPORT "")
set_global_variable (HDF5_UTILS_TO_EXPORT "")

set (EXTERNAL_HEADER_LIST "")
set (EXTERNAL_LIBRARY_LIST "")
set (EXTERNAL_LIBRARYDLL_LIST "")

#-----------------------------------------------------------------------------
# Run all the CMake configuration tests for our build environment
#-----------------------------------------------------------------------------
# include (${HDF_RESOURCES_DIR}/ConfigureChecks.cmake)

set (CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

#-----------------------------------------------------------------------------
# Option to Build Shared and Static libs, default is both
#-----------------------------------------------------------------------------
option (BUILD_STATIC_LIBS "Build Static Libraries" ON)
set (H5_ENABLE_STATIC_LIB NO)
option (BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
set (H5_ENABLE_SHARED_LIB NO)
option (ONLY_SHARED_LIBS "Only Build Shared Libraries" OFF)
mark_as_advanced (ONLY_SHARED_LIBS)

if (BUILD_STATIC_LIBS)
  set (H5_ENABLE_STATIC_LIB YES)
endif ()
if (BUILD_SHARED_LIBS)
  set (H5_ENABLE_SHARED_LIB YES)
endif ()

# Force only shared libraries if all OFF
if (NOT BUILD_STATIC_LIBS AND NOT BUILD_SHARED_LIBS)
  set (ONLY_SHARED_LIBS ON CACHE BOOL "Only Build Shared Libraries" FORCE)
endif ()

if (ONLY_SHARED_LIBS)
  set (H5_ENABLE_STATIC_LIB NO)
  set (BUILD_SHARED_LIBS ON CACHE BOOL "Build Shared Libraries")
endif ()

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT BUILD_SHARED_LIBS)
  set (tgt_file_ext "")
else ()
  set (tgt_file_ext "-shared")
endif ()

#-----------------------------------------------------------------------------
# Option to Build Static executables
#-----------------------------------------------------------------------------
option (BUILD_STATIC_EXECS "Build Static Executables" OFF)
if (BUILD_STATIC_EXECS)
  if (NOT WIN32)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
  endif ()
endif ()

if (HDF5_ENABLE_ANALYZER_TOOLS)
  include (${HDF5_SOURCE_DIR}/config/sanitizer/tools.cmake)
endif ()
if (HDF5_ENABLE_SANITIZERS)
  include (${HDF5_SOURCE_DIR}/config/sanitizer/sanitizers.cmake)
endif ()

#-----------------------------------------------------------------------------
# Option to use code coverage
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_COVERAGE "Enable code coverage for Libraries and Programs" OFF)
if (HDF5_ENABLE_COVERAGE)
  include (${HDF5_SOURCE_DIR}/config/sanitizer/code-coverage.cmake)
  if(CODE_COVERAGE AND CODE_COVERAGE_ADDED)
    add_code_coverage () # Adds instrumentation to all targets
  else ()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 --coverage -fprofile-arcs -ftest-coverage")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g --coverage -O0 -fprofile-arcs -ftest-coverage")
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
      set (LDFLAGS "${LDFLAGS} -fprofile-arcs -ftest-coverage")
      link_libraries (gcov)
    else ()
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif ()
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Option to indicate using dmalloc
#-----------------------------------------------------------------------------
# option (HDF5_ENABLE_USING_DMALLOC "Indicate that dmalloc is used" OFF)
# if (HDF5_ENABLE_USING_DMALLOC)
#   find_package (DMALLOC)
#   set (H5_HAVE_DMALLOC DMALLOC_FOUND)
# endif ()

#-----------------------------------------------------------------------------
# Option to indicate using a memory checker
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_USING_MEMCHECKER "Indicate that a memory checker is used" OFF)
if (HDF5_ENABLE_USING_MEMCHECKER)
  set (H5_USING_MEMCHECKER 1)
endif ()

#-----------------------------------------------------------------------------
# Option to indicate internal memory allocation sanity checks are enabled
#-----------------------------------------------------------------------------
option (HDF5_MEMORY_ALLOC_SANITY_CHECK "Indicate that internal memory allocation sanity checks are enabled" OFF)
if (HDF5_MEMORY_ALLOC_SANITY_CHECK)
  set (H5_MEMORY_ALLOC_SANITY_CHECK 1)
endif ()

#-----------------------------------------------------------------------------
# Option to enable/disable using pread/pwrite for VFDs
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_PREADWRITE "Use pread/pwrite in sec2/log/core VFDs in place of read/write (when available)" OFF)
mark_as_advanced (HDF5_ENABLE_PREADWRITE)
if (HDF5_ENABLE_PREADWRITE AND H5_HAVE_PREAD AND H5_HAVE_PWRITE)
  set (H5_HAVE_PREADWRITE 1)
endif ()

#-----------------------------------------------------------------------------
# Option to use deprecated public API symbols
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_DEPRECATED_SYMBOLS "Enable deprecated public API symbols" ON)
if (HDF5_ENABLE_DEPRECATED_SYMBOLS)
  set (H5_NO_DEPRECATED_SYMBOLS 0)
else ()
  set (H5_NO_DEPRECATED_SYMBOLS 1)
endif ()

#-----------------------------------------------------------------------------
# When building utility executables that generate other (source) files :
# we make use of the following variables defined in the root CMakeLists.
# Certain systems may add /Debug or /Release to output paths
# and we need to call the executable from inside the CMake configuration
#-----------------------------------------------------------------------------
set (EXE_EXT "")
if (WIN32 OR MINGW)
  set (EXE_EXT ".exe")
  add_compile_definitions (_BIND_TO_CURRENT_VCLIBS_VERSION=1 _CRT_SECURE_NO_WARNINGS _CONSOLE)
endif ()

if (MSVC)
  set (CMAKE_MFC_FLAG 0)
  set (WIN_COMPILE_FLAGS "")
  set (WIN_LINK_FLAGS "")
endif ()

set (MAKE_SYSTEM)
if (CMAKE_BUILD_TOOL MATCHES "make")
  set (MAKE_SYSTEM 1)
endif ()

set (CFG_INIT "/${CMAKE_CFG_INTDIR}")
if (MAKE_SYSTEM)
  set (CFG_INIT "")
endif ()

# Do not generate test programs by default
option (HDF5_BUILD_GENERATORS "Build Test Generators" OFF)

#-----------------------------------------------------------------------------
# Add some definitions for Debug Builds
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_TRACE "Enable API tracing capability" OFF)
mark_as_advanced (HDF5_ENABLE_TRACE)
if (${HDF_CFG_NAME} MATCHES "Debug")
  # Enable instrumenting of the library's internal operations
  option (HDF5_ENABLE_INSTRUMENT "Instrument The library" OFF)
  if (HDF5_ENABLE_INSTRUMENT)
    set (H5_HAVE_INSTRUMENTED_LIBRARY 1)
  endif ()
  mark_as_advanced (HDF5_ENABLE_INSTRUMENT)
endif ()

#-----------------------------------------------------------------------------
# Option to embed library info into executables
#-----------------------------------------------------------------------------
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set (H5_HAVE_EMBEDDED_LIBINFO 0)
else ()
  option (HDF5_ENABLE_EMBEDDED_LIBINFO "embed library info into executables" ON)
  if (HDF5_ENABLE_EMBEDDED_LIBINFO)
    set (H5_HAVE_EMBEDDED_LIBINFO 1)
  endif ()
endif ()

include (${HDF_RESOURCES_DIR}/HDFCompilerFlags.cmake)
set (CMAKE_MODULE_PATH ${HDF_RESOURCES_DIR} ${HDF_RESOURCES_EXT_DIR} ${CMAKE_MODULE_PATH})

#-----------------------------------------------------------------------------
# Option to Enable HDFS
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_HDFS "Enable HDFS" OFF)
if (HDF5_ENABLE_HDFS)
  find_package(JNI REQUIRED)
  if (JNI_FOUND)
    set (H5_HAVE_LIBJVM 1)
  endif ()
  find_package(HDFS REQUIRED)
  if (HDFS_FOUND)
    set (H5_HAVE_LIBHDFS 1)
    set (H5_HAVE_HDFS_H 1)
    if (NOT MSVC)
      list (APPEND LINK_LIBS -pthread)
    endif ()
  else ()
    message (FATAL_ERROR "Set to use libhdfs library, but could not find or use libhdfs. Please verify that the path to HADOOP_HOME is valid, and/or reconfigure without HDF5_ENABLE_HDFS")
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Option to Enable MPI Parallel
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_PARALLEL "Enable parallel build (requires MPI)" OFF)
if (HDF5_ENABLE_PARALLEL)
  find_package(MPI REQUIRED)
  if (MPI_C_FOUND)
    set (H5_HAVE_PARALLEL 1)
    # MPI checks, only do these if MPI_C_FOUND is true, otherwise they always fail
    # and once set, they are cached as false and not regenerated
    set (CMAKE_REQUIRED_LIBRARIES "${MPI_C_LIBRARIES}")
    set (CMAKE_REQUIRED_INCLUDES "${MPI_C_INCLUDE_DIRS}")
    # Used by Fortran + MPI
    CHECK_SYMBOL_EXISTS (MPI_Comm_c2f "mpi.h"  H5_HAVE_MPI_MULTI_LANG_Comm)
    CHECK_SYMBOL_EXISTS (MPI_Info_c2f "mpi.h"  H5_HAVE_MPI_MULTI_LANG_Info)

    # Used by Parallel Compression feature
    set (PARALLEL_FILTERED_WRITES ON)
    CHECK_SYMBOL_EXISTS (MPI_Mprobe "mpi.h" H5_HAVE_MPI_Mprobe)
    CHECK_SYMBOL_EXISTS (MPI_Imrecv "mpi.h" H5_HAVE_MPI_Imrecv)
    if (NOT H5_HAVE_MPI_Mprobe OR NOT H5_HAVE_MPI_Imrecv)
      message (WARNING "The MPI_Mprobe and/or MPI_Imrecv functions could not be located.
               Parallel writes of filtered data will be disabled.")
      set (PARALLEL_FILTERED_WRITES OFF)
    endif ()

    # Used by big I/O feature
    set (LARGE_PARALLEL_IO ON)
    CHECK_SYMBOL_EXISTS (MPI_Get_elements_x "mpi.h" H5_HAVE_MPI_Get_elements_x)
    CHECK_SYMBOL_EXISTS (MPI_Type_size_x "mpi.h" H5_HAVE_MPI_Type_size_x)
    if (NOT H5_HAVE_MPI_Get_elements_x OR NOT H5_HAVE_MPI_Type_size_x)
      message (WARNING "The MPI_Get_elements_x and/or MPI_Type_size_x functions could not be located.
               Reading/Writing >2GB of data in a single parallel I/O operation will be disabled.")
      set (LARGE_PARALLEL_IO OFF)
    endif ()
  else ()
    message (FATAL_ERROR "Parallel libraries not found")
  endif ()
endif ()

# Parallel IO usage requires MPI to be Linked and Included
if (H5_HAVE_PARALLEL)
  set (LINK_LIBS ${LINK_LIBS} ${MPI_C_LIBRARIES})
  if (MPI_C_LINK_FLAGS)
    set (CMAKE_EXE_LINKER_FLAGS "${MPI_C_LINK_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
  endif ()
endif ()

#option (DEFAULT_API_VERSION "Enable v1.12 API (v16, v18, v110, v112)" "v112")
set (DEFAULT_API_VERSION "v112" CACHE STRING "Enable v1.12 API (v16, v18, v110, v112)")
set_property (CACHE DEFAULT_API_VERSION PROPERTY STRINGS v16 v18 v110 v112)
#-----------------------------------------------------------------------------
# Option to use 1.6.x API
#-----------------------------------------------------------------------------
set (H5_USE_16_API_DEFAULT 0)
if (DEFAULT_API_VERSION MATCHES "v16")
  set (H5_USE_16_API_DEFAULT 1)
endif ()

#-----------------------------------------------------------------------------
# Option to use 1.8.x API
#-----------------------------------------------------------------------------
set (H5_USE_18_API_DEFAULT 0)
if (DEFAULT_API_VERSION MATCHES "v18")
  set (H5_USE_18_API_DEFAULT 1)
endif ()

#-----------------------------------------------------------------------------
# Option to use 1.10.x API
#-----------------------------------------------------------------------------
set (H5_USE_110_API_DEFAULT 0)
if (DEFAULT_API_VERSION MATCHES "v110")
  set (H5_USE_110_API_DEFAULT 1)
endif ()

#-----------------------------------------------------------------------------
# Option to use 1.12.x API
#-----------------------------------------------------------------------------
if (NOT DEFAULT_API_VERSION)
  set (DEFAULT_API_VERSION "v112")
endif ()
set (H5_USE_112_API_DEFAULT 0)
if (DEFAULT_API_VERSION MATCHES "v112")
  set (H5_USE_112_API_DEFAULT 1)
endif ()

#-----------------------------------------------------------------------------
# Include user macros
#-----------------------------------------------------------------------------
include (UserMacros.cmake)

#-----------------------------------------------------------------------------
# Include filter (zlib, szip, etc.) macros
#-----------------------------------------------------------------------------
include (CMakeFilters.cmake)

#-----------------------------------------------------------------------------
# Option for external libraries on windows
#-----------------------------------------------------------------------------
option (HDF5_PACKAGE_EXTLIBS "CPACK - include external libraries" OFF)
if (NOT HDF5_EXTERNALLY_CONFIGURED)
  if (HDF5_PACKAGE_EXTLIBS)
    set (HDF5_NO_PACKAGES OFF CACHE BOOL "CPACK - Disable packaging" FORCE)
    if (HDF5_ENABLE_Z_LIB_SUPPORT AND ZLIB_FOUND)
      PACKAGE_ZLIB_LIBRARY (${HDF5_ALLOW_EXTERNAL_SUPPORT})
    endif ()

    if (HDF5_ENABLE_SZIP_SUPPORT AND SZIP_FOUND)
      PACKAGE_SZIP_LIBRARY (${HDF5_ALLOW_EXTERNAL_SUPPORT})
    endif ()
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Option to use threadsafe
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_THREADSAFE "Enable thread-safety" OFF)
if (HDF5_ENABLE_THREADSAFE)
  # check for unsupported options
  if (WIN32 OR MINGW)
    message (STATUS " **** thread-safety option not supported with static library **** ")
    message (STATUS " **** thread-safety option will not be used building static library **** ")
  endif ()
  if (HDF5_ENABLE_PARALLEL)
    if (NOT ALLOW_UNSUPPORTED)
      message (FATAL_ERROR " **** parallel and thread-safety options are not supported **** ")
    else ()
      message (STATUS " **** Allowing unsupported parallel and thread-safety options **** ")
    endif ()
  endif ()
  if (HDF5_BUILD_FORTRAN)
    if (NOT ALLOW_UNSUPPORTED)
      message (FATAL_ERROR " **** Fortran and thread-safety options are not supported **** ")
    else ()
      message (STATUS " **** Allowing unsupported Fortran and thread-safety options **** ")
    endif ()
  endif ()
  if (HDF5_BUILD_CPP_LIB)
    if (NOT ALLOW_UNSUPPORTED)
      message (FATAL_ERROR " **** C++ and thread-safety options are not supported **** ")
    else ()
      message (STATUS " **** Allowing unsupported C++ and thread-safety options **** ")
    endif ()
  endif ()
  if (HDF5_BUILD_HL_LIB)
    if (NOT ALLOW_UNSUPPORTED)
      message (FATAL_ERROR " **** HL and thread-safety options are not supported **** ")
    else ()
      message (STATUS " **** Allowing unsupported HL and thread-safety options **** ")
    endif ()
  endif ()
  if (H5_HAVE_IOEO)
    message (STATUS " **** Win32 threads requires WINVER>=0x600 (Windows Vista/7/8) **** ")
    set (H5_HAVE_WIN_THREADS 1)
  else ()
    if (NOT H5_HAVE_PTHREAD_H)
      message (FATAL_ERROR " **** thread-safe option requires Win32 threads or Pthreads **** ")
    endif ()
  endif ()
  set (THREADS_PREFER_PTHREAD_FLAG ON)
  find_package (Threads REQUIRED)
  if (Threads_FOUND)
    set (H5_HAVE_THREADSAFE 1)
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Option to build the map API
#-----------------------------------------------------------------------------
option (HDF5_ENABLE_MAP_API "Build the map API" OFF)
if (HDF5_ENABLE_MAP_API)
  set (H5_HAVE_MAP_API 1)
endif ()

#-----------------------------------------------------------------------------
# Add the HDF5 Library Target to the build
#-----------------------------------------------------------------------------
add_subdirectory (src)

if (HDF5_ALLOW_EXTERNAL_SUPPORT MATCHES "GIT" OR HDF5_ALLOW_EXTERNAL_SUPPORT MATCHES "TGZ")
  if (ZLIB_FOUND AND ZLIB_USE_EXTERNAL)
    if (NOT ONLY_SHARED_LIBS)
      add_dependencies (${HDF5_LIB_TARGET} ZLIB)
    endif ()
    if (BUILD_SHARED_LIBS)
      add_dependencies (${HDF5_LIBSH_TARGET} ZLIB)
    endif ()
  endif ()
  if (SZIP_FOUND AND SZIP_USE_EXTERNAL)
    if (NOT ONLY_SHARED_LIBS)
      add_dependencies (${HDF5_LIB_TARGET} SZIP)
    endif ()
    if (BUILD_SHARED_LIBS)
      add_dependencies (${HDF5_LIBSH_TARGET} SZIP)
    endif ()
  endif ()
endif ()

#-----------------------------------------------------------------------------
# Option to build HDF5 C++ Library
#-----------------------------------------------------------------------------
include (${HDF_RESOURCES_EXT_DIR}/HDFUseCXX.cmake)
if (CMAKE_NO_STD_NAMESPACE)
  set (H5_NO_STD 1)
endif ()
add_subdirectory (c++)

#-----------------------------------------------------------------------------
# Generate the H5pubconf.h file containing user settings needed by compilation
#-----------------------------------------------------------------------------
configure_file (${HDF5_SOURCE_DIR}/include/H5pubconf.h ${HDF5_BINARY_DIR}/H5pubconf.h @ONLY)

#-----------------------------------------------------------------------------
# Options for use by cross compiling and toolchains
#-----------------------------------------------------------------------------
option (HDF5_USE_PREGEN "Use pre-generated Files" OFF)
option (HDF5_BATCH_H5DETECT "Use a batch command for running h5detect" OFF)

include (CMakeInstallation.cmake)
