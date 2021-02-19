cmake_minimum_required( VERSION 2.8 )
project( g2o_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

#list( APPEND CMAKE_MODULE_PATH /home/hri/ThirdParty/g2o/cmake_modules )


# 添加cmake模块以使用ceres库
#list( APPEND CMAKE_MODULE_PATH /home/god/CLionProjects/slambook-master/3rdparty/g2o/cmake_modules )
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# 寻找G2O
#set(G2O_ROOT /usr/local/include/g2o)
find_package( G2O REQUIRED )
if (G2O_FOUND)
    message(" G2O DIR " ${G2O_INCLUDE_DIRS})
    else(G2O_FOUND)
    message(FATAL_ERROR " G2O DIR NOT FOUND")
endif (G2O_FOUND)
include_directories(
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3"
)

message("g2o: %%%%%%%%%%%%%%%"  ${G2O_INCLUDE_DIRS})

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting main.cpp )
# 与G2O和OpenCV链接
target_link_libraries( curve_fitting 
    ${OpenCV_LIBS}
    g2o_core g2o_stuff
)
