cmake_minimum_required( VERSION 2.8 )
project( vo1 )

#set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 

find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )

# some tests for the args
# OpenCV
if (OpenCV_FOUND)
    message("We have OpenCV .h files at " ${OpenCV_INCLUDE_DIRS})
    message("We have OpenCV .a or .so files at " ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
    message(FATAL_ERROR " No OpenCV lib available")
endif (OpenCV_FOUND)
# G2O
#if (G2O_FOUND)
#    message("We have G2O .h files at " ${G2O_INCLUDE_DIRS})
#    message("We have G2O .a or .so files at " ${G2O_LIBRARIES})
#else(G2O_FOUND)
#    message(FATAL_ERROR " No G2O lib available")
#endif (G2O_FOUND)
# CSparse
#if (CSparse_FOUND)
#    message("We have CSparse .h files at " ${CSparse_INCLUDE_DIRS})
#    message("We have CSparse .a or .so files at " ${CSparse_LIBRARIES})
#else(CSparse_FOUND)
#    message(FATAL_ERROR " No CSparse lib available")
#endif (CSparse_FOUND)

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

message("g2o: %%%%%%%%%%%%%%%%%%" ${G2O_INCLUDE_DIRS})
message("csparse: %%%%%%%%%%%%%%%%" ${CSPARSE_INCLUDE_DIR})

add_executable( feature_extraction feature_extraction.cpp  )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2 
add_executable( triangulation triangulation.cpp )
target_link_libraries( triangulation ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
)
