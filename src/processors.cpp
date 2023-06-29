#include <stdlib.h>
#include <map>
#include <float.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include"/home/roburishabh/PRCV_Projects/Real_time_2D_Object_Recognition/include/processors.h"
#include"/home/roburishabh/PRCV_Projects/Real_time_2D_Object_Recognition/include/csv_util.h"

using namespace cv;
using namespace std;

/*
The function first converts an image to grayscale,
and then set any pixel with a value less or equal to 130 as foreground and other pixels as background

@parameter image: thr input image
@return an 8 bit single-channel threshold image as a Mat
*/

cv::Mat threshold(Mat &image){
	// threshold used for classifying pixels as foreground or background.
	int THRESHOLD = 130;
	cv::Mat processedImage, grayscale;
	//create a new Mat object named processedImage with the same size as the input image. 
	//The image.size() function returns the size of the input image. 
	//The CV_8UC1 argument specifies the type of the Mat object, indicating that it is an 8-bit single-channel image.
	processedImage = Mat(image.size(), CV_8UC1);
	//convert input image to grayscale
	cvtColor(image, grayscale, COLOR_BGR2GRAY);
	//iterates over each pixel in the grayscale image. 
	for(int i = 0; i < grayscale.rows; i++){
		for(int j = 0; j < grayscale.cols; j++){
			//perform the thresholding operation. 
			//The intensity value of the current pixel in the grayscale image is accessed using grayscale.at<uchar>(i, j). 
			//If the intensity value is less than or equal to the threshold (THRESHOLD), the corresponding pixel in the processedImage is set to 255 (foreground). 
			//Otherwise, it is set to 0 (background).
			if(grayscale.at<uchar>(i,j) <= THRESHOLD){
				processedImage.at<uchar>(i, j) = 255;
			}
			else{
				processedImage.at<uchar>(i, j) = 0;
			}
		}
	}
	return processedImage;
}

/*
The function applies dilation on an image and then applies erosion on it

@parameter image : the inpute image
@return the cleaned up image as a Mat
*/
cv::Mat cleanup(Mat &image){
	cv::Mat processedImage;
	//creates a structuring element, which is essentially a small matrix used for morphological operations. 
	//The function getStructuringElement is called with the parameters MORPH_CROSS and Size(25, 25) to create a cross-shaped structuring element with a size of 25x25 pixels.
	//The resulting structuring element is stored in the kernel variable.
	const Mat kernel = getStructuringElement(MORPH_CROSS, Size(25, 25));
	//performs a morphological operation called closing on the image using the specified kernel. 
	//The morphologyEx function is called with the parameters image (the input image), processedImage (the output image), MORPH_CLOSE (indicating that we want to perform the closing operation), 
	//and kernel (the structuring element to be used for the operation). The result of the operation is stored in the processedImage variable.
	morphologyEx(image, processedImage, MORPH_CLOSE, kernel);
	return processedImage;
}




/*
The function extracts the largest 3 regions of a given image, and writes the attributes into related Mats

@parameter image : the given image
@parameter lebelledRegions : a Mat to store the label of each pixel
@parameter stats : a Mat to store the attributes of each labelled regions
@parameter centroids : a Mat to store the centroids of each labelled regions
@parameter topNLabels: a vector to store the labels of the largest 3 regions
@return an image of the largest three regions as a Mat
*/
cv::Mat getRegions(Mat &image, Mat &labeledRegions, Mat &stats, Mat &centroids, vector<int> &topNLabels){
	cv::Mat processedImage;
	// performs connected components analysis on the image using the connectedComponentsWithStats function from OpenCV. 
	//It assigns the total number of labels found in the image to the variable nLabels. The resulting labeled regions, 
	//along with their statistics and centroids, are stored in the corresponding cv::Mat variables.
	int nLabels = connectedComponentsWithStats(image, labeledRegions, stats, centroids);

	//store the areas of each labeled region.
	cv::Mat areas = Mat::zeros(1, nLabels-1, CV_32S);
	cv::Mat sortedIdx;
	for(int i=1; i<nLabels; i++){
		//areas are extracted from the stats matrix, specifically from the column CC_STAT_AREA
		int area = stats.at<int>(i, CC_STAT_AREA);
		areas.at<int>(i-1) = area;
	}
	//sortIdx function is then used to sort the areas in descending order, and the resulting indices are stored in the sortedIdx matrix.
	if(areas.cols > 0){
		sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
	}
	// declares a vector named colors of type Vec3b, which will be used for label-to-color mapping. 
	// The vector is initialized with nLabels elements, each initialized to the RGB color (0, 0, 0).
	vector<Vec3b> colors(nLabels, Vec3b(0, 0, 0));
	// determine the number of regions to consider (N) by either using the value 3 or the number of sorted indices (sortedIdx.cols), whichever is smaller.
	int N=3;
	N = (N < sortedIdx.cols) ? N : sortedIdx.cols;
	int THRESHOLD = 5000;
	// iterates over the top N sorted indices
	for(int i=0; i<N; i++){
		//retrieves the corresponding label, and checks if the region's area (obtained from stats) is greater than a threshold (THRESHOLD)
		int label = sortedIdx.at<int>(i)+1;
		if(stats.at<int>(label, CC_STAT_AREA) > THRESHOLD){
			//If it is, a random RGB color is assigned to that label in the colors vector, and the label is added to the topNLabels vector.
			colors[label] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
			topNLabels.push_back(label);
		}
	}
	//create an image named processedImage with the same size as labeledRegions, initialized with all zeros. 
	processedImage = Mat::zeros(labeledRegions.size(), CV_8UC3);
	// iterates over each pixel in processedImage
	for(int i=0; i<processedImage.rows; i++){
		for(int j=0; j<processedImage.cols; j++){
			//retrieves the label from the corresponding position in labeledRegions
			int label = labeledRegions.at<int>(i,j);
			//assigns the corresponding color from the colors vector to that pixel.
			processedImage.at<Vec3b>(i,j) = colors[label];
		}
	}
	//returns the resulting image of the largest three regions as a cv::Mat
	return processedImage;
}

/*
 * The function computes the rotated bounding box of a given region
 *
 * @parameter region: the given region
 * @parameter x: the x-axis value of the centroid of the region
 * @parameter y: the y-axis value of the centroid of the region
 * @parameter alpha: the angle between the x-axis and least central x-axis
 * @return a rotated bounding box of the given region
 */

cv::RotatedRect getBoundingBox(Mat &region, double x, double y, double alpha) {
    int maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX;
    //iterate over each pixel of the region image
    for (int i = 0; i < region.rows; i++) {
        for (int j = 0; j < region.cols; j++) {
        	//checks if the pixel value at (i, j) = 255, which typically represents the foreground or object in binary images.
            if (region.at<uchar>(i, j) == 255) {
            	//Calculate the projected coordinates of each pixel (i, j) with respect to the specified rotation and translation parameters (x, y, alpha)
                int projectedX = (i - x) * cos(alpha) + (j - y) * sin(alpha);
                int projectedY = -(i - x) * sin(alpha) + (j - y) * cos(alpha);
                //update the maxX, minX, maxY, and minY variables to keep track of the maximum and minimum transformed coordinates encountered during the loop.
                maxX = max(maxX, projectedX);
                minX = min(minX, projectedX);
                maxY = max(maxY, projectedY);
                minY = min(minY, projectedY);
            }
        }
    }
    //calculate the lengths of the bounding box along the x-axis (lengthX) and y-axis (lengthY) 
    //based on the differences between the maximum and minimum transformed coordinates.
    int lengthX = maxX - minX;
    int lengthY = maxY - minY;

    // Point object centroid with coordinates (x, y) representing the center of the bounding box
    Point centroid = Point(x, y);
    //Size object size with dimensions lengthX and lengthY representing the width and height of the bounding box.
    Size size = Size(lengthX, lengthY);

    //creates and returns a RotatedRect object with the calculated centroid, size, and rotation angle in degrees (alpha * 180.0 / CV_PI)
    return RotatedRect(centroid, size, alpha * 180.0 / CV_PI);
}

/*
 * This function draws a line of 100 pixels given a starting point and an angle
 *
 * @parameter image: the image where the line is drawn
 * @parameter x: the value of x-axis of the starting point
 * @parameter y: the value of y-axis of the starting point
 * @parameter alpha: the given angle
 * @parameter color: the color of the line to be drawn
 */
 void drawLine(Mat &image, double x, double y, double alpha, Scalar color) {
    double length = 100.0;
    //calculates two edge lengths, edge1 and edge2, using the given alpha angle.
    double edge1 = length * sin(alpha);
    double edge2 = sqrt(length * length - edge1 * edge1);
    //calculates the coordinates xPrime and yPrime of the end point of the line segment based on the given x, y, and the edge lengths.
    double xPrime = x + edge2, yPrime = y + edge1;
    //It takes the starting point coordinates (x, y), the ending point coordinates (xPrime, yPrime), 
    //the specified color, and a line thickness of 3 as parameters. 
    //The function draws a line segment with an arrowhead from the starting point to the ending point on the image.
    arrowedLine(image, Point(x, y), Point(xPrime, yPrime), color, 3);
}
/*
 * This function draws a rectangle on a given image
 *
 * @parameter image: the given image
 * @parameter boundingBox: the rectangle to be drawn
 * @parameter color: the color of the rectangle
 */
//This function drawBoundingBox takes a Mat object image as the input image,
//a RotatedRect object boundingBox representing the rotated bounding box, 
//and a Scalar object color specifying the color of the bounding box. 
 void drawBoundingBox(Mat &image, RotatedRect boundingBox, Scalar color) {
 	////It declares an array of Point2f objects rect_points to store the four corner points of the bounding box.
    Point2f rect_points[4];
    //The boundingBox.points() function is then used to retrieve the four corner points of the rotated bounding box and store them in the rect_points array.
    boundingBox.points(rect_points);
    //This for loop iterates four times, corresponding to the four sides of the bounding box. It draws each side of the bounding box as a line segment on the image using the line function from OpenCV. The starting point of the line is given by rect_points[i], and the ending point is given by rect_points[(i + 1) % 4]. The modulus operator % is used to ensure that the last point connects back to the first point, thus completing the closed shape of the bounding box. The specified color and a line thickness of 3 are used as parameters for drawing the lines.
    for (int i = 0; i < 4; i++) {
        line(image, rect_points[i], rect_points[(i + 1) % 4], color, 3);
    }
}
/*
 * This function calculates the HU Moments according to the given central moments
 *
 * @parameter mo: the given Moments contains the central moments
 * @parameter huMoments: a vector to store the 7 attributes of HU Moments
 */

void calcHuMoments(Moments mo, vector<double> &huMoments) {
    double hu[7]; // HuMoments require the parameter type to be double[]
    HuMoments(mo, hu);

    //These lines convert the hu array of Hu moments into a vector format and store them in the huMoments vector. 
    //It iterates over each element d in the hu array using a range-based for loop and appends each element to the huMoments vector using the push_back method. 
    //This ensures that the computed Hu moments are stored in the huMoments vector.
    for (double d : hu) {
        huMoments.push_back(d);
    }
    //Finally, the return statement signifies the end of the function.
    return;
}


/*
 * This function calculates the normalized Euclidean distance between two vectors
 *
 * @parameter feature1: the first vector
 * @parameter feature2: the second vector
 *
 * @return the normalized distance as a double
 */
double euclideanDistance(vector<double> features1, vector<double> features2) {
	//These lines initialize three variables: sum1, sum2, and sumDifference. 
	//sum1 and sum2 will store the sum of squares of elements in features1 and features2, respectively
	//sumDifference will store the sum of squares of the differences between corresponding elements in features1 and features2.
    double sum1 = 0, sum2 = 0, sumDifference;
    //iterates over the elements of features1 and features2
    for (int i = 0; i < features1.size(); i++) {
    	//calculates the squared difference between the corresponding elements in features1 and features2, and adds it to sumDifference
        sumDifference += (features1[i] - features2[i]) * (features1[i] - features2[i]);
        //calculates the sum of squares of the elements in features1 and features2 and adds them to sum1 and sum2, respectively
        sum1 += features1[i] * features1[i];
        sum2 += features2[i] * features2[i];
    }
    //calculates the Euclidean distance by dividing the square root of sumDifference by the sum of the square roots of sum1 and sum2. 
    //The division is performed to normalize the Euclidean distance. The result is returned as the output of the function.
    return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2));
}

/*
 * Given some data and a feature vector, this function gets the class name of the given feature vector
 * Infers based on the nearest neighbor, and use normalized euclidean distance as distance metric
 *
 * @parameter featureVectors: a vector to store the feature vectors of known objects
 * @parameter classNames: a vector to store the class names of known objects
 * @parameter currentFeature: the feature vector of the object needed to be inferred
 * @return the inferred class name as a string
 */
string classifier(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature) {
    //THRESHOLD variable sets a threshold value to determine how close a feature vector needs to be to a known object's feature vector for classification.
    double THRESHOLD = 0.15;
    // distance variable is initialized to the maximum value of a double. 
    double distance = DBL_MAX;
    //The className variable is initialized with an empty string.
    string className = " ";
    //iterates over the known feature vectors stored in featureVectors. 
    //It retrieves the feature vector and class name for each known object and calculates the Euclidean distance between the current feature vector (currentFeature) and the known feature vector (dbFeature) using the euclideanDistance function.
    for (int i = 0; i < featureVectors.size(); i++) { 
        vector<double> dbFeature = featureVectors[i];
        string dbClassName = classNames[i];
        double curDistance = euclideanDistance(dbFeature, currentFeature);
        //If the calculated curDistance is smaller than the previous distance value and 
        //also smaller than the specified threshold (THRESHOLD), 
        //the className and distance variables are updated with the class name and distance of the current known object.
        if (curDistance < distance && curDistance < THRESHOLD) {
            className = dbClassName;
            distance = curDistance;
        }
    }
    //Finally, the inferred class name (the class name of the closest known object based on the feature vector's distance) is returned as a string.
    return className;
}

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
string classifierKNN(vector<vector<double>> featureVectors, vector<string> classNames, vector<double> currentFeature, int K) {
    //threshold distance below which feature vectors are considered similar.
    double THRESHOLD = 0.15;
    // compute the distances of current feature vector with all the feature vectors in DB
    vector<double> distances;
    //iterates over each feature vector in featureVectors
    for (int i = 0; i < featureVectors.size(); i++) {
        vector<double> dbFeature = featureVectors[i];
        //calculates the Euclidean distance between dbFeature and currentFeature using a function euclideanDistance
        double distance = euclideanDistance(dbFeature, currentFeature);
        //if the distance is below the THRESHOLD, it adds the distance to the distances vector.
        if (distance < THRESHOLD) {
            distances.push_back(distance);
        }
    }
    //initialize the className variable as an empty string
    string className = " ";
    if (distances.size() > 0) {
        // sort the distances in ascending order
        vector<int> sortedIdx;
        //If there are any distances stored in the distances vector
        //it creates an empty sortedIdx vector and sorts the indices of the distances in ascending order using the sortIdx function.
        sortIdx(distances, sortedIdx, SORT_EVERY_ROW + SORT_ASCENDING);

        // get the first K class name, and count the number of each name
        vector<string> firstKNames;
        int s = sortedIdx.size();
        map<string, int> nameCount;
        int range = min(s, K); // if less than K classnames, get all of them
        //iterates over the range (the smaller of s and K), 
        for (int i = 0; i < range; i++) {
            //retrieves the class name corresponding to the sortedIdx[i] index from classNames, and increases its count in nameCount.
            string name = classNames[sortedIdx[i]];
            if (nameCount.find(name) != nameCount.end()) {
                nameCount[name]++;
            } else {
                nameCount[name] = 1;
            }
        }

        // get the class name that appear the most times in the K nearest neighbors
        int count = 0;
        for (map<string ,int>::iterator it = nameCount.begin(); it != nameCount.end(); it++) {
            if (it->second > count) {
                className = it->first;
                count = it->second;
            }
        }
    }
    return className;
}

/*
 * This function returns the corresponding class name given a code
 *
 * @parameter c: the code for each class name
 * @return the class name as a string
 */
string getClassName(char c) {
    std::map<char, string> myMap {
            {'p', "pen"}, {'c', "cell phone"}, {'n', "notebook"}, {'g', "glasses"},
            {'a', "apple"}, {'h', "hat"}, {'b', "bottle"}, {'k', "key"},
            {'m', "mouse"}, {'x', "wire"},
            {'w', "watch"}, {'s', "speaker"}, {'t', "Television Screen"} , {'l', "lamp"}
    };
    return myMap[c];
}


