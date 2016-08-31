# - Try to find WarpCTC
#
# The following variables are optionally searched for defaults
#  WARP_CTC_ROOT_DIR:            Base directory where all WARP_CTC components are found
#
# The following are set after configuration is done:
#  WARP_CTC_FOUND
#  WARP_CTC_INCLUDE_DIRS
#  WARP_CTC_LIBRARIES
#  WARP_CTC_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

set(WARP_CTC_ROOT_DIR "" CACHE PATH "Folder contains WarpCTC")

if(WIN32)
    find_path(WARP_CTC_INCLUDE_DIR warp_ctc/ctcpp.h ctcpp.h
        PATHS ${WARP_CTC_ROOT_DIR}/src/windows)
else()
    find_path(WARP_CTC_INCLUDE_DIR warp_ctc/ctcpp.h ctcpp.h
        PATHS ${WARP_CTC_ROOT_DIR})
endif()

if(MSVC)
    find_library(WARP_CTC_LIBRARY_RELEASE libwarpctc
        PATHS ${WARP_CTC_ROOT_DIR}
        PATH_SUFFIXES Release)

    find_library(WARP_CTC_LIBRARY_DEBUG libwarpctc
        PATHS ${WARP_CTC_ROOT_DIR}
        PATH_SUFFIXES Debug)

    set(WARP_CTC_LIBRARY optimized ${WARP_CTC_LIBRARY_RELEASE} debug ${WARP_CTC_LIBRARY_DEBUG})
else()
    find_library(WARP_CTC_LIBRARY warpctc
        PATHS ${WARP_CTC_ROOT_DIR}
        PATH_SUFFIXES lib lib64)
endif()

find_package_handle_standard_args(WarpCTC DEFAULT_MSG WARP_CTC_INCLUDE_DIR WARP_CTC_LIBRARY)

if(WarpCTC_FOUND)
  set(WARP_CTC_INCLUDE_DIRS ${WARP_CTC_INCLUDE_DIR})
  set(WARP_CTC_LIBRARIES ${WARP_CTC_LIBRARY})
  message(STATUS "Found WarpCTC (include: ${WARP_CTC_INCLUDE_DIR}, library: ${WARP_CTC_LIBRARY})")
  mark_as_advanced(WARP_CTC_ROOT_DIR WARP_CTC_LIBRARY_RELEASE WARP_CTC_LIBRARY_DEBUG
                   WARP_CTC_LIBRARY WARP_CTC_INCLUDE_DIR)
endif()
