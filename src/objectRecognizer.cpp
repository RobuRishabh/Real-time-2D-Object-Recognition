#include<stdio.h>
#include<opencv2/opencv.hpp>
#include"/home/roburishabh/PRCV_Projects/Real_time_2D_Object_Recognition/include/processors.h"
#include"/home/roburishabh/PRCV_Projects/Real_time_2D_Object_Recognition/include/csv_util.h"

using namespace cv;
using namespace std;

/*
 * Takes two inputs
 * The first is the path to the csv file store the class name and feature vector for each known object
 * The second is the classifier type ('n' for the nearest neighbor, 'k' for KNN)
 *
 * This program will threshold and clean up the images of the objects in live video,
 * and then computes features of each region.
 * If in training mode, this program allows users to label each object and save the data in database.
 * If in inference mode, this program infers and displays the class name of each object on the live video
 */
int main(int argc, char *argv[]) {
    // check for sufficient arguments
    if (argc < 3) {
        cout << "Wrong input." << endl;
        exit(-1);
    }

    // featuresDB and classNamesDB are used to save the feature vectors of known objects
    // featuresDB.size() == classNamesDB.size()
    // featuresDB[i] is the i-th object's feature vector, classNamesDB[i] is the i-th object's class name
    vector<string> classNamesDB;
    vector<vector<double>> featuresDB;

    // load existing data from csv file to featuresDB and classNameDB
    loadFromCSV(argv[1], classNamesDB, featuresDB);

    // open the video device
    VideoCapture *capdev;
    capdev = new VideoCapture(2);
    if (!capdev->isOpened()) {
        cout << "Unable to open video device\n";
        return -1;
    }

    // identify window
    namedWindow("Original Video", 1);

    Mat frame;
    bool training = false; // whether the system is in training mode

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        char key = waitKey(10); // see if there is a waiting keystroke for the video

        // switch between training mode and inference mode
        if (key == 't') {
            training = !training;
            if (training) {
                cout << "Training Mode" << endl;
            } else {
                cout << "Inference Mode" << endl;
            }
        }

        // threshold the image, thresholdFrame is single-channel
        Mat thresholdFrame = threshold(frame);

        // clean up the image
        Mat cleanupFrame = cleanup(thresholdFrame);

        // get the largest 3 regions
        Mat labeledRegions, stats, centroids;
        vector<int> topNLabels;
        Mat regionFrame = getRegions(cleanupFrame, labeledRegions, stats, centroids, topNLabels);

        // for each region, get bounding box and calculate HuMoments
        for (int n = 0; n < topNLabels.size(); n++) {
            int label = topNLabels[n];
            Mat region;
            region = (labeledRegions == label);

            // calculate central moments, centroids, and alpha
            Moments m = moments(region, true);
            double centroidX = centroids.at<double>(label, 0);
            double centroidY = centroids.at<double>(label, 1);
            double alpha = 1.0 / 2.0 * atan2(2 * m.mu11, m.mu20 - m.mu02);

            // get the least central axis and bounding box of this region
            RotatedRect boundingBox = getBoundingBox(region, centroidX, centroidY, alpha);
            drawLine(frame, centroidX, centroidY, alpha, Scalar(0, 0, 255));
            drawBoundingBox(frame, boundingBox, Scalar(0, 255, 0));

            // calculate hu moments of this region
            vector<double> huMoments;
            calcHuMoments(m, huMoments);

            if (training) {
                // in training mode
                // display current region in binary form
                namedWindow("Current Region", WINDOW_AUTOSIZE);
                imshow("Current Region", region);

                // ask the user for a class name
                cout << "Input the class for this object." << endl;
                // get the code for each class name
                char k = waitKey(0);
                string className = getClassName(k); //see the function for a detailed mapping

                // update the DB
                featuresDB.push_back(huMoments);
                classNamesDB.push_back(className);

                // after labeling all the objects,
                // switch back to inference mode and destroy all the windows created in training mode
                if (n == topNLabels.size() - 1) {
                    training = false;
                    cout << "Inference Mode" << endl;
                    destroyWindow("Current Region");
                }
            } else {
                // in inference mode
                // classify the object
                string className;
                if (!strcmp(argv[2], "n")) { // nearest neighbor
                    className = classifier(featuresDB, classNamesDB, huMoments);
                } else if (!strcmp(argv[2], "k")) { // KNN
                    className = classifierKNN(featuresDB, classNamesDB, huMoments, 5);
                }
                // overlay classname to the video
                putText(frame, className, Point(centroids.at<double>(label, 0), centroids.at<double>(label, 1)), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 3);
            }
        }

        imshow("Original Video", frame); // display the video

        // if user types 'q', quit.
        if (key == 'q') {
            // when quit, add data in classNamesDB and featuresDB to csv file
            writeToCSV(argv[1], classNamesDB, featuresDB);
            break;
        }
    }
    return 0;
}