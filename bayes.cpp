#include "bayes.h"

bayes::bayes(int br, int bc, double bt, const int* ts)
{
	blockRows = br;
	blockCols = bc;
	blockThreshold = bt;
	blockTotal = br*bc;

	trainSet = new int[10];


	for (int i = 0; i < 10; i++)
		trainSet[i] = ts[i];
	attributes = new int*[10];
	trainSum = 0.0;
	pwi = new double[10];
	for (int i = 0; i < 10; i++)
	{
		attributes[i] = new int[blockTotal];
		for (int j = 0; j < blockTotal; j++)
		{
			attributes[i][j] = 1;
		}
		trainSum += trainSet[i];
	}
}

vector<Mat> bayes::slice(const string & path, int rows, int cols)
{

	//返回的结果
	vector<Mat> result;

	Mat image;
	//打开文件
	image = imread(path, 0);
	if (image.empty())
	{
		cout << " can not open the file:" << path << endl;
		throw;
	}
	//二值化图像
	threshold(image, image, 127, 255, THRESH_BINARY);

	//得到归一化数字的位置
	int x_min = image.cols;      //数字最左像素x值
	int y_min = image.rows;      //数字顶部像素y值
	int x_max = 0;      //数字最右像素x值
	int y_max = 0;      //数字底部像素y值


	for (int y = 0; y<image.rows; y++)
	{
		for (int x = 0; x<image.cols; x++)
		{
			if (image.at<uchar>(y, x))
			{
				if (x < x_min)
					x_min = x;

				if (x > x_max)
					x_max = x;

				if (y < y_min)
					y_min = y;

				if (y > y_max)
					y_max = y;
			}
		}
	}
	//    cout << "x_min=" << x_min << endl;
	//    cout << "x_max=" << x_max << endl;
	//    cout << "y_min=" << y_min << endl;
	//    cout << "y_max=" << y_max << endl;

	//计算裁剪比例
	int x_begin;        //切割出的图片最左x坐标
	int x_end;          //切割出的图片最右x坐标
	int y_begin;        //切割出的图片顶部y坐标
	int y_end;          //切割出的图片底部y坐标
	int vary_center;    //需要比例变化长或宽的中心
	int vary_size;      //等比例变化后的长或宽

	


	if (((double)(y_max - y_min) / (x_max - x_min)) > (double)(image.rows / image.cols))
	{
		vary_center = (x_max + x_min + 1) / 2;
		vary_size = ((double)(y_max - y_min) * image.cols + image.rows / 2) / image.rows;
		x_begin = vary_center - ((vary_size + 1) / 2);
		x_end = vary_center + ((vary_size + 1) / 2);
		y_begin = y_min;
		y_end = y_max + 1;
	}
	else
	{
		vary_center = (y_max + y_min + 1) / 2;
		vary_size = ((double)(x_max - x_min) * image.rows + image.cols / 2) / image.cols;
		x_begin = x_min;
		x_end = x_max + 1;
		y_begin = vary_center - ((vary_size + 1) / 2);
		y_end = vary_center + ((vary_size + 1) / 2);
	}

//	x_begin = x_begin > x_min ? x_begin : x_min;
	x_end = x_end > x_max ? x_max : x_end;
//	y_begin = y_begin > y_min ? y_begin : y_min;
//	y_end = y_end > y_max ? y_max : y_end;


	//裁剪得到归一化图像
	Mat imageNorm(image, Range(y_begin, y_end), Range(x_begin, x_end));





	int x_step = (imageNorm.cols + cols / 2) / cols;
	int y_step = (imageNorm.rows + rows / 2) / rows;

	for (int i = 0; i<rows; i++)
	{
		for (int j = 0; j<cols; j++)
		{
			Mat part(imageNorm.clone(),
				Range(i*y_step, i != (rows - 1) ? (i + 1)*y_step : imageNorm.rows),
				Range(j*x_step, j != (cols - 1) ? (j + 1)*x_step : imageNorm.cols));
			result.push_back(part);
		}
	}

	return result;
}

void bayes::getPictureAttributes(vector<Mat>& picture, bool * attributes)
{
	double count;
	for (int i=0; i < picture.size(); i++)
	{
		MatIterator_<uchar> it;
		count = 0.0;
		for (it = picture[i].begin<uchar>(); it != picture[i].end<uchar>(); it++)
		{
			count += (*it > 127 ? 1 : 0);
		}
		if (count / picture[i].total() > blockThreshold)
		{
			attributes[i] = true;
		}
		else
		{
			attributes[i] = false;
		}
	}
}

void bayes::train(const string& prefix)
{
	stringstream path;
	bool* pictureAttributes = new bool[blockTotal];
	double sum = 0.0;

	for (int i = 0; i < 10; i++)
	{
		cout << "train:" << i << endl;
		pwi[i] = 0.0;
		for (int j = 0; j < blockTotal; j++)
			attributes[i][j] = 1;



		for (int j = 1; j <= trainSet[i]; j++)
		{
			path.str("");
			path << i << "\\" << i << '_' << j << ".bmp";
			vector<Mat> picture = slice(prefix + path.str(), blockRows, blockCols);
			getPictureAttributes(picture, pictureAttributes);
			for (int k = 0; k < blockTotal; k++)
			{
				if (pictureAttributes[k])
					attributes[i][k]++;
			}
		}
		pwi[i] = trainSet[i] / trainSum;
	}
	delete[] pictureAttributes;
}

int bayes::checkAPicture(const string & path)
{
	vector<Mat> picture = slice(path, blockRows, blockCols);
	bool* pictureAttributes = new bool[blockTotal];
	getPictureAttributes(picture, pictureAttributes);
	double pay;        //p(ai|Y）
	double pyx;        //p(Y|X)
	double max = 0;
	int index = 0;
	for (int i = 0; i < 10; i++)
	{
		pay = 1.0;
		for (int j = 0; j < blockTotal; j++)
		{
			if (pictureAttributes[j] == true)
			{
				pay *= (double)attributes[i][j] / (double)trainSet[i];
			}
			else
			{ 
				pay *= 1- (double)attributes[i][j] / (double)trainSet[i];
			}
		}
		pyx = pay*pwi[i];
		if (pyx > max)
		{
			max = pyx;
			index = i;
		}
	}
	
	delete[] pictureAttributes;
	return index;
}

float bayes::checkAllPictures(const string & prefix, const int* testSet)
{
	cout << "Check all pictures:"<<endl;
	stringstream path;
	int type;
	int correct;
	int sum = 0;
	int testSum = 0;
	for (int i = 0; i < 10; i++)
	{
		correct = 0;
		for (int j = 1; j <= testSet[i]; j++)
		{
			path.str("");
			path << i << "\\" << i << '_' << j << ".bmp";
			type = checkAPicture(prefix + path.str());
			if (type == i)
				correct++;
		}
		sum += correct;
		testSum += testSet[i];
		cout << "Num:" << i<<"\ttotal:"<< testSet[i] << "\tcorrect:" << correct << "\trate:" << (double)correct / (double)testSet[i] << endl;
	}
	cout << "Total:"<<testSum<<"\tcorrect sum:" << sum << "\t rate:" << (double)sum / (double)testSum << endl;
	return (double)sum / (double)testSum;
}

float bayes::findBestThreshold(const int* testPic)
{
	float best=0;
	int index;
	float rate;
	blockThreshold = 0.0;
	for (int i = 1; i <= 100; i++)
	{
		train(trainPrefix);
		rate = checkAllPictures(testPrefix, testPic);
		if (rate > best)
		{
			best = rate;
			index = i;
		}
		cout << "index:"<<i<<"\tthreshold:" << blockThreshold << "\trate:" << rate << endl;
		blockThreshold += 0.01;
	}
	cout << "best threshold:" << best << endl;
	return best;
}

























































































