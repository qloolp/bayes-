#include "bayes.h"



const int trainPic[10] = { 592, 671, 581, 608, 623, 514, 608, 651, 551, 601};
const int testPic[10] = {53,73,64,62,67,56,52,57,52,64};


int main()
{


	//bayes(int br, int bc, double bt, int* ts);
	bayes b(5,5,0.15,trainPic);
	b.train(trainPrefix);
	b.checkAllPictures(testPrefix, testPic);
//	b.findBestThreshold(testPic);   //ʹ��ʱ��ע�͵�train��checkAllPictures�е�������
	cin.get();
	return 0;
}