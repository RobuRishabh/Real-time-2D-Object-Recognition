#ifndef Real_time_2D_Object_Recognition_CSV_UTILS_H
#define Real_time_2D_Object_Recognition_CSV_UTILS_H

using namespace std;

/*
 * Given a filename, a 1 x M vector classNameDB and an M x N vector featuresDB,
 * the function will delete the current contents in csv file and write new data into it.
 *
 * classNamesDB[i] is written to the first position of row i as a string,
 * and featuresDB[i] is written to the following positions of row i as double.
 */
void writeToCSV(string filename, vector<string> classNamesDB, vector<vector<double>> featuresDB);

/*
 * Given a filename, a 1 x M vector classNameDB and an M x N vector featuresDB,
 * the function will load the data from csv file to the vectors.
 *
 * the value in the first position of row i will be the value of classNamesDB[i],
 * and the values in the following positions of row i will be the values of featuresDB[i].
 */
void loadFromCSV(string filename, vector<string> &classNamesDB, vector<vector<double>> &featuresDB);

#endif //Real_time_2D_Object_Recognition_CSV_UTILS_H

