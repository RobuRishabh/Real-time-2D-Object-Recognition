#ifndef Real_time_2D_Object_Recognition_PROCESSORS_H
#define Real_time_2D_Object_Recognition_PROCESSORS_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
 * The function first converts an image to grayscale,
 * and then set any pixel with a value less or equal to 130 as foreground, and other pixels as background
 *
 * @parameter image: the input image
 * @return an 8 bit single-channel threshold image as a Mat
 */
Mat threshold(Mat &image);

/*
 * The function applies dilation on an image and then applies erosion on it
 *
 * @parameter image: the input image
 * @return the cleaned up image as a Mat
 */
Mat cleanup(Mat &image);

/*
 * The function extracts the largest three regions of a given image, and writes the attributes into related Mats.
 *
 * @parameter image: the given image
 * @parameter labeledRegions: a Mat to store the label of each pixel
 * @parameter stats: a Mat to store the attributes of each labeled region
 * @parameter centroids: a Mat to store the centroids of each labeled region
 * @parameter topNLabels: a vector to store the labels of the largest 3 regions
 * @return an image of the largest three regions as a Mat
 */
Mat getRegions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels);

/*
 * The function computes the rotated bounding box of a given region
 *
 * @parameter region: the given region
 * @parameter x: the x-axis value of the centroid of the region
 * @parameter y: the y-axis value of the centroid of the region
 * @parameter alpha: the angle between the x-axis and least central x-axis
 * @return a rotated bounding box of the given region
 */
RotatedRect getBoundingBox(Mat &region, double x, double y, double alpha);

/*
 * This function draws a line of 100 pixels given a starting point and an angle
 *
 * @parameter image: the image where the line is drawn
 * @parameter x: the value of x-axis of the starting point
 * @parameter y: the value of y-axis of the starting point
 * @parameter alpha: the given angle
 * @parameter color: the color of the line to be drawn
 */
void drawLine(Mat &image, double x, double y, double alpha, Scalar color);

/*
 * This function draws a rectangle on a given image
 *
 * @parameter image: the given image
 * @parameter boundingBox: the rectangle to be drawn
 * @parameter color: the color of the rectangle
 */
void drawBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color);

/*
 * This function calculates the HU Moments according to the given central moments
 *
 * @parameter mo: the given Moments contains the central moments
 * @parameter huMoments: a vector to store the 7 attributes of HU Moments
 */
void calcHuMoments(Moments mo, vector<double> &huMoments);

/*
 * This function calculates the normalized Euclidean distance between two vectors
 *
 * @parameter feature1: the first vector
 * @parameter feature2: the second vector
 *
 * @return the normalized distance as a double
 */
double euclideanDistance(vector<double> features1, vector<double> features2);

/*
 * Given some data and a feature vector, this function gets the class name of the given feature vector
 * Infers based on the nearest neighbor, and use normalized euclidean distance as distance metric
 *
 * @parameter featureVectors: a vector to store the feature vectors of known objects
 * @parameter classNames: a vector to store the class names of known objects
 * @parameter currentFeature: the feature vector of the object needed to be inferred
 * @return the inferred class name as a string
 */
string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature);

/*
 * Given some data and a feature vector, this function gets the name of the given feature vector
 * Infers based on K-Nearest-Neighbor, and use normalized euclidean distance as distance metric
 *
 * @parameter featureVectors: a vector to store the feature vectors of known objects
 * @parameter classNames: a vector to store the class names of known objects
 * @parameter currentFeature: the feature vector of the object needed to be inferred
 * @parameter K: the k value in KNN
 * @return the inferred class name as a string
 */
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K);

/*
 * This function returns the corresponding class name given a code
 *
 * @parameter c: the code for each class name
 * @return the class name as a string
 */
string getClassName(char c);

#endif //Real_time_2D_Object_Recognition_PROCESSORS_H
