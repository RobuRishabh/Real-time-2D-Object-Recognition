/*author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 3: Real-time 2-D Object Recognition
*/
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <map>
#include "features.h"
#include "csv_util.h"

using namespace cv;
using namespace std;


Mat thresholding(cv::Mat& img) {
    int Thres = 135;
    cv::Mat Thresh_Image, grayscale;
    Thresh_Image = cv::Mat(img.size(), CV_8UC1);

    cvtColor(img, grayscale, COLOR_BGR2GRAY);

    int i = 0;
    while (i < grayscale.rows) {
        int j = 0;
        while (j < grayscale.cols) {
            if (grayscale.at<uchar>(i, j) <= Thres) {
                Thresh_Image.at<uchar>(i, j) = 255;
            }
            else {
                Thresh_Image.at<uchar>(i, j) = 0;
            }
            j++;
        }
        i++;
    }

    return Thresh_Image;
}


Mat cleanup_images(cv::Mat& img) {
    cv::Mat cleaned_Image;
    const Mat element = getStructuringElement(MORPH_CROSS, Size(25, 25));
    morphologyEx(img, cleaned_Image, MORPH_CLOSE, element);
    return cleaned_Image;
}


Mat segment(cv::Mat& img, cv::Mat& label_Reg, cv::Mat& stats, cv::Mat& centroid, vector<int>& top_N_Labels) {
    cv::Mat regions;
    int N_Labels = connectedComponentsWithStats(img, label_Reg, stats, centroid);


    cv::Mat areas = cv::Mat::zeros(1, N_Labels - 1, CV_32S);
    cv::Mat sortedIdx;
    int x = 1;
    while (x < N_Labels) {
        int area = stats.at<int>(x, CC_STAT_AREA);
        areas.at<int>(x - 1) = area;
        x++;
    }
    if (areas.cols > 0) {
        sortIdx(areas, sortedIdx, SORT_EVERY_ROW + SORT_DESCENDING);
    }


    vector<Vec3b> colors(N_Labels, cv::Vec3b(0, 0, 0));

    int N = 3;
    N = (N < sortedIdx.cols) ? N : sortedIdx.cols;
    int THRESH = 4000;
    int y = 0;
    while (y < N) {
        int label = sortedIdx.at<int>(y) + 1;
        if (stats.at<int>(label, CC_STAT_AREA) > THRESH) {
            colors[label] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
            top_N_Labels.push_back(label);
        }
        y++;
    }


    regions = cv::Mat::zeros(label_Reg.size(), CV_8UC3);
    int p = 0;
    while (p < regions.rows) {
        int q = 0;
        while (q < regions.cols) {
            int label = label_Reg.at<int>(p, q);
            regions.at<cv::Vec3b>(p, q) = colors[label];
            q++;
        }
        p++;
    }

    return regions;
}


RotatedRect bounding_box(cv::Mat& region, double x, double y, double theta) {
    int X_max = INT_MIN, X_min = INT_MAX, Y_max = INT_MIN, Y_min = INT_MAX;
    int a = 0;
    while (a < region.rows) {
        int b = 0;
        while (b < region.cols) {
            if (region.at<uchar>(a, b) == 255) {
                int projectedX = (a - x) * cos(theta) + (b - y) * sin(theta);
                int projectedY = -(a - x) * sin(theta) + (b - y) * cos(theta);
                X_max = max(X_max, projectedX);
                X_min = min(X_min, projectedX);
                Y_max = max(Y_max, projectedY);
                Y_min = min(Y_min, projectedY);
            }
            b++;
        }
        a++;
    }

    int X_len = X_max - X_min;
    int Y_len = Y_max - Y_min;

    Point centroid = Point(x, y);
    Size size = Size(X_len, Y_len);

    return RotatedRect(centroid, size, theta * 180.0 / CV_PI);
}


void show_arrow(cv::Mat& img, double x, double y, double theta, Scalar color) {
    double len = 100.0;
    double side_a = len * sin(theta);
    double side_b = sqrt(len * len - side_a * side_a);
    double x_cap = x + side_b;
    double y_cap = y + side_a;

    arrowedLine(img, Point(x, y), Point(x_cap, y_cap), color, 3);
}


void draw_bb(cv::Mat& img, RotatedRect bound_box, Scalar color) {
    Point2f rectangular_pt[4];
    bound_box.points(rectangular_pt);
    int w = 0;
    while (w < 4) {
        line(img, rectangular_pt[w], rectangular_pt[(w + 1) % 4], color, 3);
        w++;
    }

}


void find_HuMoments(Moments mo, vector<double>& huMoments) {
    double hu[7];
    HuMoments(mo, hu);


    for (double d : hu) {
        huMoments.push_back(d);
    }
    return;
}


double euclid_Dist(vector<double> feat_a, vector<double> feat_b) {
    double sum1 = 0, sum2 = 0;
    double sumDifference = 0;
    int temp = 0;
    while (temp < feat_a.size()) {
        sumDifference += (feat_a[temp] - feat_b[temp]) * (feat_a[temp] - feat_b[temp]);
        sum1 += feat_a[temp] * feat_a[temp];
        sum2 += feat_b[temp] * feat_b[temp];
        temp++;
    }
    return sqrt(sumDifference) / (sqrt(sum1) + sqrt(sum2));

}


string N_classifier(vector<vector<double>> feature_Vect, vector<string> class_Name, vector<double> current_Feat) {
    double THRESH = 0.15;
    double dist = DBL_MAX;
    string class_Nm = " ";
    for (int i = 0; i < feature_Vect.size(); i++) {
        vector<double> dbFeature = feature_Vect[i];
        string db_ClassName = class_Name[i];
        double current_Dist = euclid_Dist(dbFeature, current_Feat);
        if (current_Dist < dist && current_Dist < THRESH) {
            class_Nm = db_ClassName;
            dist = current_Dist;
        }
    }
    return class_Nm;
}


string KNN_classifier(vector<vector<double>> feature_Vect, vector<string> class_Name, vector<double> current_Feat, int node) {
    double THRESH = 0.15;

    vector<double> dist;
    int z = 0;
    while (z < feature_Vect.size()) {
        vector<double> dbFeature = feature_Vect[z];
        double distance = euclid_Dist(dbFeature, current_Feat);
        if (distance < THRESH) {
            dist.push_back(distance);
        }
        z++;
    }

    string className = " ";
    if (dist.size() > 0) {

        vector<int> sort_index;
        sortIdx(dist, sort_index, SORT_EVERY_ROW + SORT_ASCENDING);


        vector<string> top_Knames;
        int s = sort_index.size();
        map<string, int> nameCount;
        int range = min(s, node);
        for (int i = 0; i < range; i++) {
            string name = class_Name[sort_index[i]];
            if (nameCount.find(name) != nameCount.end()) {
                nameCount[name]++;
            }
            else {
                nameCount[name] = 1;
            }
        }


        int count = 0;
        map<string, int>::iterator it = nameCount.begin();
        while (it != nameCount.end()) {
            if (it->second > count) {
                className = it->first;
                count = it->second;
            }
            it++;
        }

    }
    return className;
}

string find_ClassName(char c) {
    string className;
    switch (c) {
    case 'p':
        className = "pen";
        break;
    case 'g':
        className = "glasses";
        break;
    case 'k':
        className = "key";
        break;
    case 'w':
        className = "watch";
        break;
    case 'h':
        className = "Headphones";
        break;
    default:
        className = "";
        break;
    }
    return className;
}


