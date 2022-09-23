##############################################
###    An example on how Arshin.KFusion    ###
### should be integrated with your project ###
##############################################

cmake_minimum_required(VERSION 3.18)
if(MSVC AND NOT "${CMAKE_GENERATOR_PLATFORM}" STREQUAL "x64")
  message(FATAL_ERROR "We support only x64 platform")
endif()

project(Arshin.KFusion LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/bin")
  file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
endif()

macro(discover_cpp include_dir src_dir headers src)
  file(GLOB_RECURSE ${headers} ${include_dir}/*.h)
  file(GLOB_RECURSE hpp        ${include_dir}/*.hpp)
  list(APPEND ${headers} ${hpp})
  file(GLOB_RECURSE ${src}     ${src_dir}/*.cpp)
  file(GLOB_RECURSE cu         ${src_dir}/*.cu)
  list(APPEND ${src} ${cu})

  # Configure filters for VS project
  if(MSVC)
    foreach(_source IN ITEMS ${${headers}})
      get_filename_component(_source_path "${_source}" PATH)
      file(RELATIVE_PATH _source_path_rel "${include_dir}" "${_source_path}")
      string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
      source_group("Header Files\\${_group_path}" FILES "${_source}")
    endforeach()

    foreach(_source IN ITEMS ${${src}})
      get_filename_component(_source_path "${_source}" PATH)
      file(RELATIVE_PATH _source_path_rel "${src_dir}" "${_source_path}")
      string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
      source_group("Source Files\\${_group_path}" FILES "${_source}")
    endforeach()
  endif()
endmacro()

discover_cpp("${CMAKE_SOURCE_DIR}/include/kfusion" "${CMAKE_SOURCE_DIR}/src"
             KFUSION_HEADERS KFUSION_SRC)

add_library(kfusion STATIC ${KFUSION_HEADERS} ${KFUSION_SRC})
target_include_directories(kfusion
                           PRIVATE "${CMAKE_SOURCE_DIR}/src"
                                   "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                           PUBLIC  "${CMAKE_SOURCE_DIR}/include")
set_target_properties(kfusion PROPERTIES
                      CXX_STANDARD 14
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      DEBUG_POSTFIX "d")

add_custom_command(TARGET kfusion POST_BUILD
                   COMMAND ${CMAKE_COMMAND}
                   -E copy $<TARGET_FILE:kfusion> "${CMAKE_SOURCE_DIR}/bin")
