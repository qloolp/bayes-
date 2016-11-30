#ifndef BAYES_H
#define BAYES_H

#include <iostream>
#include <stdio.h>
#include <strstream>
#include <vector>
#include <algorithm>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;


const string trainPrefix = "MNIST\\trainImages\\";
const string testPrefix = "MNIST\\testImages\\";

/*
best blockThreshold = 0.15;
max correct rate = 0.738

*/



class bayes
{
private:
	int blockRows;
	int blockCols;
	double blockThreshold;
	int blockTotal;


	int* trainSet;
	double trainSum;
	int** attributes; //attributes[10][blockTotal]
	double* pwi;        //p(Wi)  pwi[10]



	vector<Mat> slice(const string& path, int rows, int cols);
	void getPictureAttributes(vector<Mat>& picture, bool* attributes);
public:
	//blockRows,blockCols,blockThreshold,trainSet[10] = {90,90,......}
	bayes(int br, int bc, double bt, const int* ts);

	//训练集文件前缀;"train\\"后接"i\\i_j.bmp"
	void train(const string& prefix);

	//检测单幅图片类型，path为问件路径
	int checkAPicture(const string& path);

	//检测测试集所有图片类型：文件前缀：“test\\"后接"i\\i_j.bmp",testSet[10] = {10, 10,10, ......}
	//返回检测正确率
	float checkAllPictures(const string& prefix, const int* testSet);

	//寻找最佳阈值
	float findBestThreshold(const int* testPic);
};

#endif