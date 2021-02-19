cmake_minimum_required( VERSION 2.8 )
project( useSophus )

# 为使用 sophus，您需要使用find_package命令找到它
find_package( Sophus REQUIRED )

if (Sophus_FOUND)
    message("We have Sophus .h files at " ${Sophus_INCLUDE_DIRS})
    message("We have Sophus .a or .so files at " ${Sophus_LIBRARIES})

else(Sophus_FOUND)
    message(FATAL_ERROR " No Sophus lib avaliable")
endif (Sophus_FOUND)

include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable( useSophus useSophus.cpp )
target_link_libraries( useSophus ${Sophus_LIBRARIES} )