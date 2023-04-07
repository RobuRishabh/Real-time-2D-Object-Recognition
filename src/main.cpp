/*author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 3: Real-time 2-D Object Recognition
*/
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "features.h"
#include "csv_util.h"

using namespace cv;
using namespace std;


int main(){
//int main(int argc, char* argv[]) {
    // check for sufficient arguments
    
    /*
    if (argc < 3) {
        cout << "<exectutable> <path to csv file> <classifier>" << endl;
        exit(-1);
    }
    */

    /*

MODIFY COMMAND LINE ARGUMENTS HERE
*/
    std::string argv0 = "a.out";
    std::string argv1 = "E:\\PRCV Assignments\\Real-time 2D Object Recognition\\output.csv";
    std::string argv2 = "n";

    int argc = 3;
    char* argv[3];
    argv[0] = _strdup(argv0.c_str());
    argv[1] = _strdup(argv1.c_str());
    argv[2] = _strdup(argv2.c_str());

    vector<string> classNamesDB;
    vector<vector<double>> featuresDB;


    from_CSV(argv[1], classNamesDB, featuresDB);

    // open the video device
    VideoCapture* capdev;
    capdev = new VideoCapture(1);
    if (!capdev->isOpened()) {
        cout << "Unable to open the video device\n";
        return -1;
    }


    namedWindow("Original Video", 1);

    Mat frame;
    bool training = false;

    while (true) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty()) {
            cout << "frame is empty\n";
            break;
        }

        char key = waitKey(10);

        // switch between training mode and inference mode
        if (key == 't') {
            training = !training;
            if (training) {
                cout << "Training Mode" << endl;
            }
            else {
                cout << "Inference Mode" << endl;
            }
        }


        cv::Mat thresh_img = thresholding(frame);
        cv::Mat cleanup_img = cleanup_images(thresh_img);

        cv::Mat label_Region, stats, centroid;
        vector<int> top_NLabel;
        cv::Mat regionFrame = segment(cleanup_img, label_Region, stats, centroid, top_NLabel);


        for (int n = 0; n < top_NLabel.size(); n++) {
            int label = top_NLabel[n];
            cv::Mat region;
            region = (label_Region == label);


            Moments m = moments(region, true);
            double X_centroid = centroid.at<double>(label, 0);
            double Y_centroid = centroid.at<double>(label, 1);
            double theta = 1.0 / 2.0 * atan2(2 * m.mu11, m.mu20 - m.mu02);


            RotatedRect bb = bounding_box(region, X_centroid, Y_centroid, theta);
            show_arrow(frame, X_centroid, Y_centroid, theta, Scalar(0, 0, 255));
            draw_bb(frame, bb, Scalar(0, 255, 0));


            vector<double> huMoments;
            find_HuMoments(m, huMoments);

            if (training) {

                cv::namedWindow("Currently Identified Region", WINDOW_AUTOSIZE);
                cv::imshow("Currently Identified Region", region);


                cout << "Input the class for this object." << endl;
                char k = waitKey(0);
                string className = find_ClassName(k);


                featuresDB.push_back(huMoments);
                classNamesDB.push_back(className);

                if (n == top_NLabel.size() - 1) {
                    training = false;
                    cout << "Inference Mode" << endl;
                    cv::destroyWindow("Currently Identified Region");
                }
            }
            else {
                string className;
                if (!strcmp(argv[2], "n")) {
                    className = N_classifier(featuresDB, classNamesDB, huMoments);
                }
                else if (!strcmp(argv[2], "k")) {
                    className = KNN_classifier(featuresDB, classNamesDB, huMoments, 5);
                }

                putText(frame, className, Point(centroid.at<double>(label, 0), centroid.at<double>(label, 1)), FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 0, 255), 3);
            }
        }

        cv::imshow("Original Video", frame);


        if (key == 'q') {
            to_CSV(argv[1], classNamesDB, featuresDB);
            break;
        }
    }
    return 0;
}
