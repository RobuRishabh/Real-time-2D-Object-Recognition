#include <fstream>
#include <string>
#include <vector>
#include"/home/roburishabh/PRCV_Projects/Real_time_2D_Object_Recognition/include/csv_util.h"

using namespace std;

/*
 * Given a filename, a 1 x M vector classNameDB and an M x N vector featuresDB,
 * the function will delete the current contents in csv file and write new data into it.
 *
 * classNamesDB[i] is written to the first position of row i as a string,
 * and featuresDB[i] is written to the following positions of row i as double.
 */
void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB) {
    // create an output filestream object
    ofstream csvFile;
    csvFile.open(filename, ofstream::trunc);

    // send data to the stream
    for (int i = 0; i < classNamesDB.size(); i++) {
        // add class name
        csvFile << classNamesDB[i] << ",";
        // add features
        for (int j = 0; j < featuresDB[i].size(); j++) {
            csvFile << featuresDB[i][j];
            if (j != featuresDB[i].size() - 1) {
                csvFile << ","; // no comma at the end of line
            }
        }
        csvFile << "\n";
    }
}

/*
 * Given a filename, a 1 x M vector classNameDB and an M x N vector featuresDB,
 * the function will load the data from csv file to the vectors.
 *
 * the value in the first position of row i will be the value of classNamesDB[i],
 * and the values in the following positions of row i will be the values of featuresDB[i].
 */
void loadFromCSV(string filename, vector<string> &classNamesDB, vector<vector<double>> &featuresDB) {
    // create an input filestream object
    ifstream csvFile(filename);
    if (csvFile.is_open()) {
        // read data line by line
        string line;
        while (getline(csvFile, line)) {
            vector<string> currLine; // all the values from current line
            int pos = 0;
            string token;
            while ((pos = line.find(",")) != string::npos) {
                token = line.substr(0, pos);
                currLine.push_back(token);
                line.erase(0, pos + 1);
            }
            currLine.push_back(line);

            vector<double> currFeature; // all the values except the first one from current line
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