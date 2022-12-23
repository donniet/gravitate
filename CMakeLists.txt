cmake_minimum_required(VERSION 3.13)
project(gravitate)              
set(CMAKE_CXX_STANDARD 20)            

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 


find_package(Boost 1.79.0 REQUIRED) 
find_package(OpenCL REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(gravitate src/main.cpp)
    # disable the deprecated sprintf warnings in boost::lexical_cast
    #TODO: figure out how to eliminate this warning without disabling
    target_compile_options(gravitate PRIVATE -Wno-deprecated-declarations)
    target_link_libraries(gravitate ${Boost_LIBRARIES} OpenCL::OpenCL)
endif()
