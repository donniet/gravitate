cmake_minimum_required(VERSION 3.13)
project(gravitate)              
set(CMAKE_CXX_STANDARD 20)   

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 


find_package(Boost 1.71.0 REQUIRED) 
find_package(OpenCL REQUIRED)

include_directories(include ${Boost_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}) 

enable_testing()
add_executable(block_test src/block_test.cpp)
target_link_libraries(block_test GTest::gtest_main)


include(GoogleTest)
gtest_discover_tests(block_test)


add_executable(gravitate src/main.cpp)
# disable the deprecated sprintf warnings in boost::lexical_cast
#TODO: figure out how to eliminate this warning without disabling
target_compile_options(gravitate PRIVATE -Wno-deprecated-declarations)
target_link_libraries(gravitate ${Boost_LIBRARIES} ${OpenCL_LIBRARIES})
