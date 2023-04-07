#pragma once
/*author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 3: Real-time 2-D Object Recognition
*/
#ifndef CSV_UTILS_H
#define CSV_UTILS_H

using namespace std;


//function for loading feature value from csv file
void from_CSV(string filename, vector<string>& classNamesDB, vector<vector<double>>& featuresDB);

//function for writing feature value to csv file
void to_CSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB);


#endif //CSV_UTILS_H
