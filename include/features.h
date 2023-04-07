#pragma once
/*author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 3: Real-time 2-D Object Recognition
*/
#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//function for thresholding img
Mat thresholding(cv::Mat& img);

//function for cleaning up img using morphological filtering
Mat cleanup_images(cv::Mat& img);

//function for get regions
Mat segment(cv::Mat& img, cv::Mat& label_Region, cv::Mat& stats, cv::Mat& centroid, vector<int>& top_NLabel);

//function for getting bounding box
RotatedRect bounding_box(cv::Mat& region, double x, double y, double theta);

//function for drawing line
void show_arrow(cv::Mat& img, double x, double y, double theta, Scalar color);

//function for drawing bounding box
void draw_bb(cv::Mat& img, RotatedRect bb, Scalar color);

//function for calculating HU moments
void find_HuMoments(Moments mo, vector<double>& huMoments);

//function for distance metric
double euclid_Dist(vector<double> feat_a, vector<double> feat_b);

//n-N_classifier
string N_classifier(vector<vector<double>> feature_Vect, vector<string> class_Name, vector<double> current_Feat);

//KNN-Classifier
string KNN_classifier(vector<vector<double>> feature_Vect, vector<string> class_Name, vector<double> current_Feat, int K);

//function for getting class name
string find_ClassName(char c);

#endif //FEATURES_H
