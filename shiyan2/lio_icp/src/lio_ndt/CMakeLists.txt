cmake_minimum_required(VERSION 3.0.2)
project(lio_ndt)

# ===================== 编译配置 =====================
set(CMAKE_BUILD_TYPE "Debug") 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# ===================== 依赖包 =====================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  geometry_msgs
  tf
  eigen_conversions
)

# PCL依赖
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ===================== 外部依赖（glog/eigen/geographic） =====================
set(ALL_TARGET_LIBRARIES "")
include(cmake/glog.cmake)
include(cmake/pcl.cmake)
include(cmake/eigen.cmake)
include(cmake/geographic.cmake)

# ===================== 头文件路径 =====================
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

include(cmake/global_defination.cmake)

catkin_package()

# ===================== 源文件收集 =====================
# 收集所有 .cpp 文件
file(GLOB_RECURSE ALL_SRCS "src/*.cpp")

# 单独收集 node 文件（节点可执行程序）
file(GLOB_RECURSE NODE_SRCS "src/*_node.cpp")

# 第三方源文件（若有）
file(GLOB_RECURSE THIRD_PARTY_SRCS "third_party/*.cpp")

# 在 ALL_SRCS 剔除 NODE 与 THIRD_PARTY
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS})
list(REMOVE_ITEM ALL_SRCS ${THIRD_PARTY_SRCS})

# ===================== 新增：确保配准方法被包含 =====================
# （如果使用 file(GLOB_RECURSE ALL_SRCS) 已经覆盖，可省略）
set(METHOD_SRCS
    src/method/pcl_icp_method.cpp
    src/method/pcl_ndt_method.cpp
    src/method/registration_method.cpp
)
list(APPEND ALL_SRCS ${METHOD_SRCS})

# ===================== 可执行文件定义 =====================
add_executable(test_frame_node src/test_frame_node.cpp ${ALL_SRCS})

target_link_libraries(test_frame_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${ALL_TARGET_LIBRARIES}
)

add_executable(front_end_node src/front_end_node.cpp ${ALL_SRCS})
target_link_libraries(front_end_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${ALL_TARGET_LIBRARIES}
)



# ===================== 调试输出 =====================
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "PCL include: ${PCL_INCLUDE_DIRS}")
message(STATUS "Number of source files: ${ALL_SRCS}")
