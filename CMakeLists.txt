SET(PROJECT_NAME midas)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Path to the Torch library
SET(CMAKE_PREFIX_PATH $ENV{HOME}/libtorch)

# Find package
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python numpy REQUIRED)

# Show PyTorch libraries and directories
MESSAGE("-- TORCH_LIBRARIES : " ${TORCH_LIBRARIES})
MESSAGE("-- TORCH_INCLUDE_DIRS : " ${TORCH_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(
    include
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

ADD_LIBRARY(${PROJECT_NAME}_lib SHARED src/midas.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}_lib ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})

ADD_EXECUTABLE(${PROJECT_NAME}_inference src/inference_midas.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}_inference ${PROJECT_NAME}_lib)

##### Uncomment this section for a Boost.Python project ##############################
## Without this, any build libraries automatically have names "lib{x}.so"
# set(CMAKE_SHARED_MODULE_PREFIX "")

# ADD_LIBRARY(midas MODULE module/midas.cpp)
# TARGET_LINK_LIBRARIES(midas ${PROJECT_NAME}_lib)
# SET_TARGET_PROPERTIES(midas PROPERTIES
# LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python
# )
#######################################################################################
