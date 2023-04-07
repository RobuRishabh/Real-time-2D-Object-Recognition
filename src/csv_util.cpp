/*author: HRITHIK KANOJE, RISHABH SINGH
Class: CS5330 Pattern Recog & Computer Vision
Prof: Bruce Maxwell
Project 3: Real-time 2-D Object Recognition
*/

#include <fstream>
#include <string>
#include <vector>
#include "csv_util.h"

using namespace std;


void from_CSV(string filename, vector<string>& classNamesDB, vector<vector<double>>& featuresDB) {

    ifstream csvFile(filename);
    if (csvFile.is_open()) {

        string line;
        while (getline(csvFile, line)) {
            vector<string> currLine;
            int pos = 0;
            string token;
            while ((pos = line.find(",")) != string::npos) {
                token = line.substr(0, pos);
                currLine.push_back(token);
                line.erase(0, pos + 1);
            }
            currLine.push_back(line);

            vector<double> currFeature;
            if (currLine.size() != 0) {
                classNamesDB.push_back(currLine[0]);
                for (int i = 1; i < currLine.size(); i++) {
                    currFeature.push_back(stod(currLine[i]));
                }
                featuresDB.push_back(currFeature);
            }
        }
    }
}

void to_CSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB) {

    ofstream csvFile;
    csvFile.open(filename, ofstream::trunc);


    for (int i = 0; i < classNamesDB.size(); i++) {

        csvFile << classNamesDB[i] << ",";

        for (int j = 0; j < featuresDB[i].size(); j++) {
            csvFile << featuresDB[i][j];
            if (j != featuresDB[i].size() - 1) {
                csvFile << ",";
            }
        }
        csvFile << "\n";
    }
}



