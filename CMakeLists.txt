cmake_minimum_required(VERSION 3.10)
project(MIDDLEPROJECT_COMPUTERVISION)

find_package(OpenCV REQUIRED)

add_executable(main code/src/main.cpp 
    code/src/Utils.cpp 
    code/src/Label.cpp 
    code/src/Dataset.cpp 
    code/src/ObjectDetector/ObjectDetector.cpp 
    code/src/ObjectDetector/FeaturePipeline/FeaturePipeline.cpp 
    code/src/ObjectDetector/FeaturePipeline/FeatureDetector.cpp 
    code/src/ObjectDetector/FeaturePipeline/FeatureMatcher.cpp 
    code/src/DetectionAccuracy.cpp
    code/src/Logger.cpp
    code/src/ImageFilter.cpp
    code/src/ObjectDetector/ViolaJones/ViolaJones.cpp)


add_executable(main_image_generator 
    code/src/MainGenerator.cpp 
    code/src/ObjectDetector/ViolaJones/ImageDatasetGenerator.cpp 
    code/src/ObjectDetector/ViolaJones/ViolaJonesTraining.cpp
    code/src/Utils.cpp
    code/src/Label.cpp)


target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main_image_generator ${OpenCV_LIBS})

