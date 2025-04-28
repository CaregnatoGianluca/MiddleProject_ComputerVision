
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

int main() {
    // Load the trained cascade model
    cv::CascadeClassifier cascade;
    //std::string cascadePath = "../Image_generated/cascade_mustard_bottle24/cascade.xml";
    //std::string cascadePath = "../Image_generated/cascade_mustard_bottle60/cascade.xml";
    //std::string cascadePath = "../Image_generated/cascade_sugar_box60/cascade.xml";
    //std::string cascadePath = "../Image_generated/cascade_power_drill/cascade.xml";
    
    std::string cascadePath = "../Image_generated/cascade_sugar_box_gray40/cascade.xml";
    //std::string cascadePath = "../Image_generated/cascade_mustard_bottle_gray40/cascade.xml";
    //std::string cascadePath = "../Image_generated/cascade_power_drill_gray40/cascade.xml";
    if (!cascade.load(cascadePath)) {
        std::cerr << "Error loading cascade classifier from " << cascadePath << std::endl;
        return -1;
    }

    // Get the list of test images (adjust extension if needed)
    std::vector<cv::String> imageFiles;
    //std::string testDir = "../dataset/006_mustard_bottle/test_images/*.jpg";
    //std::string testDir = "../Image_generated/Mustard_bottle_Generated/*.png";
    //std::string testDir = "../dataset/035_power_drill/test_images/*.jpg";
    //std::string testDir = "../Image_generated/Power_drill_Generated/*.png";
    std::string testDir = "../dataset/004_sugar_box/test_images/*.jpg";
    cv::glob(testDir, imageFiles, false);
    if (imageFiles.empty()) {
        std::cerr << "No images found in " << testDir << std::endl;
        return -1;
    }

    // Process each test image
    for (const auto& imageFile : imageFiles) {
        cv::Mat image = cv::imread(imageFile);
        if (image.empty()) {
            std::cerr << "Could not load image: " << imageFile << std::endl;
            continue;
        }
                            
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        //cv::equalizeHist(gray, gray);
        cv::Mat blurred_gray;
        cv::GaussianBlur(gray, blurred_gray, cv::Size(3, 3), 0); // Kernel size (must be odd), sigmaX
        // Then potentially apply CLAHE or equalizeHist to blurred_gray
        // Mat image_to_detect = blurred_gray;
        //imshow("Gray", blurred_gray);

        // Detect objects using the cascade classifier
        std::vector<cv::Rect> detections;
        cascade.detectMultiScale(blurred_gray, detections, 1.0005, 3, 3, cv::Size(5, 5)); 
        // Option 1a: Slightly more sensitive scaling
        //cascade.detectMultiScale(blurred_gray, detections, 1.05, 0, 0, cv::Size(5, 5));
        // If any detection, output details
        if (!detections.empty()) {
            std::cout << "Object detected in image: " << imageFile << std::endl;
        } else {
            std::cout << "No object detected in image: " << imageFile << std::endl;
        }

        // Draw bounding boxes for each detection and print coordinates
        for (const auto& rect : detections) {
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
            std::cout << "Detection coordinates (x, y, width, height): (" 
                      << rect.x << ", " << rect.y << ", " 
                      << rect.width << ", " << rect.height << ")" << std::endl;
        }

        // Display the image with detections
        cv::imshow("Detection", image);
        std::cout << "Showing image: " << imageFile << ". Press any key to continue, ESC to exit." << std::endl;
        int key = cv::waitKey(0);
        if (key == 27) break;
    }

    return 0;
}