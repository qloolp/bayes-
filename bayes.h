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

	//ѵ�����ļ�ǰ׺;"train\\"���"i\\i_j.bmp"
	void train(const string& prefix);

	//��ⵥ��ͼƬ���ͣ�pathΪ�ʼ�·��
	int checkAPicture(const string& path);

	//�����Լ�����ͼƬ���ͣ��ļ�ǰ׺����test\\"���"i\\i_j.bmp",testSet[10] = {10, 10,10, ......}
	//���ؼ����ȷ��
	float checkAllPictures(const string& prefix, const int* testSet);

	//Ѱ�������ֵ
	float findBestThreshold(const int* testPic);
};

#endif