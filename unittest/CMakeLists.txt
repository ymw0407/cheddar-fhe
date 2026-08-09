include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(jsoncpp)

add_compile_definitions(
  PARAM_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

add_executable(basic_test BasicTest.cpp)
target_link_libraries(basic_test
                      PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)

if (ENABLE_EXTENSION)
  add_executable(boot_test Bootstrapping.cpp)
  target_link_libraries(boot_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json)
endif()

# ---- ResNet-20 workload (ported from cheddar-ae; see /PORTING.md) ----------
if (ENABLE_EXTENSION)
  FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy
    GIT_TAG        master
  )
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.3.9
  )
  FetchContent_MakeAvailable(Eigen)
  FetchContent_MakeAvailable(cnpy)

  add_executable(resnet resnet/ResNet.cpp)
  add_executable(memresnet resnet/MemResNet.cpp)
  add_executable(ops_test resnet/OpsTest.cpp)
  target_compile_definitions(ops_test PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_include_directories(ops_test PRIVATE ${cnpy_SOURCE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/resnet)
  target_link_libraries(ops_test
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json
                                cnpy)
  target_compile_definitions(resnet PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_include_directories(resnet PRIVATE ${cnpy_SOURCE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/resnet)
  target_link_libraries(resnet
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json
                                cnpy eigen)
  target_compile_definitions(memresnet PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_include_directories(memresnet PRIVATE ${cnpy_SOURCE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/resnet)
  target_link_libraries(memresnet
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json
                                cnpy eigen)

  add_executable(memresnet_wide resnet/MemResNetWide.cpp)
  target_compile_definitions(memresnet_wide
                             PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_include_directories(memresnet_wide PRIVATE ${cnpy_SOURCE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/resnet)
  target_link_libraries(memresnet_wide
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json
                                cnpy eigen)

  add_executable(memresnet_bank resnet/MemResNetBank.cpp)
  target_compile_definitions(memresnet_bank
                             PRIVATE PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
  target_include_directories(memresnet_bank PRIVATE ${cnpy_SOURCE_DIR}
                             ${CMAKE_CURRENT_SOURCE_DIR}/resnet)
  target_link_libraries(memresnet_bank
                        PRIVATE cheddar gtest_main nlohmann_json::nlohmann_json
                                cnpy eigen)

  configure_file(${CMAKE_SOURCE_DIR}/parameters/resnetparam_30.json
                 ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_30.json COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/parameters/resnetparam_35.json
                 ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_35.json COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/parameters/resnetparam_40.json
                 ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_40.json COPYONLY)
  configure_file(${CMAKE_SOURCE_DIR}/parameters/resnetparam_48.json
                 ${CMAKE_CURRENT_BINARY_DIR}/resnetparam_48.json COPYONLY)
endif()

configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_30.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_30.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_35.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_35.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_40.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40.json COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/parameters/bootparam_40_64bit.json
               ${CMAKE_CURRENT_BINARY_DIR}/bootparam_40_64bit.json COPYONLY)
