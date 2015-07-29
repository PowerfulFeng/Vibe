#if 0
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "objdetect\objdetect.hpp"
#include "video\background_segm.hpp"
#include "vector"
using namespace std;
using namespace cv;
//++++++++++++++++++++++++++++++++++++++++++++
const int IMAGE_WIDTH = 500;
const int IMAGE_HEIGHT = 300;
Size scale(IMAGE_WIDTH, IMAGE_HEIGHT);
const int shift_xy[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };
const int defaultRadius = 20;//前景判定半径
const int defaultReqMatches = 2;	//#min指数
const int defaultfactor = 16; //概率p
const int NB = 50; //前景计数器
const int defaultsamples = 20;//每个像素点都有一个长度为20的数组存放他的背景样本点
float backmodel[IMAGE_HEIGHT][IMAGE_WIDTH][defaultsamples + 1];
//********************************************
void initialbm(Mat &M){
	if (M.empty())
	{
		cout << "initial failed" << endl;
		return;
	}
	RNG rng;
	//每个像素点都要初始化
	int x, y, k;
	for (y = 0; y<M.rows; y++)
	{
		for (x = 0; x<M.cols; x++)
		{
			backmodel[y][x][defaultsamples] = 0;//前景点计数方法：对像素点进行统计，如果摸个像素点连续K次被检测为前景，则将其更新为背景点。
			for (k = 0; k < defaultsamples; k++)//对图像样本集进行初始化，即从其邻居中随机选择一个点进行初始化，重复defaultsamples = 20次
			{
				int c;
				int r;
				int s_x = rng.uniform(0, 9);
				int s_y = rng.uniform(0, 9);
				if (x>0)
				{
					if (x < M.cols - 1)
					{
						c = x + shift_xy[s_x];
					}
					else
					{
						c = M.cols - 1;
					}
				}
				else
				{
					c = 0;
				}

				if (y>0)
				{
					if (y<M.rows - 1)
					{
						r = y + shift_xy[s_y];
					}
					else
					{
						r = M.rows - 1;
					}
				}
				else
				{
					r = 0;
				}
				backmodel[y][x][k] = M.at<uchar>(r, c);
			}
		}
	}
}
void updatebm(Mat dst, Mat fgmask)
{
	//对每个像素进行更新背景模型
	int hitcount;//记录与背景模型相符的个数
	RNG rng;
	int a, b;
	for (int y = 0; y<dst.rows; y++)
	{
		for (int x = 0; x<dst.cols; x++)
		{
			int k = 0;
			hitcount = 0;
			while (k < defaultsamples && hitcount<defaultReqMatches)
			{
				b = dst.at<uchar>(y, x);
				if (abs(backmodel[y][x][k] - b) < defaultRadius)//defaultRadius是当前点与样本点的距离的阈值，即判断样本集(共defaultsamples = 20个)中有多少个点与当前点距离过近
					hitcount++;
				k++;
			}
			if (hitcount >= defaultReqMatches)//如果与当前点距离过近的点的数量超过defaultReqMatches，则表示当前点应该为背景
			{//则判断为背景像素；
				//首先 将 计数值设为0；
				backmodel[y][x][defaultsamples] = 0;
				//此时将有p的概率去更新 模板中 随机的一个样本点
				fgmask.at<uchar>(y, x) = 0;
				a = rng.uniform(0, defaultfactor);//产生一个随机数0-defaultfactor=16，即等概率产生0-15这16个数字
				if (a == 0)//如果这个数字为0(是0-15都可以，只是以1/16的概率更新当前点的样本集)，则选择一个随机的点进行更新
				{
					a = rng.uniform(0, defaultsamples);
					backmodel[y][x][a] = b;
					//cout<<"x="<<x<<",y="<<y<<",更新自己背景b="<<b<<endl;
				}
				a = rng.uniform(0, defaultfactor);//以1/16的概率更新邻居点的样本集
				if (a == 0){//中，去更新邻居背景
					//cout<<"x="<<x<<",y="<<y<<",更新邻居背景b="<<b<<endl;
					int s_x = rng.uniform(0, 9);
					int s_y = rng.uniform(0, 9);

					if (x>0){
						if (x < dst.cols - 1){
							s_x = x + shift_xy[s_x];
						}
						else{
							s_x = dst.cols - 1;
						}
					}
					else{
						s_x = 0;
					}
					if (y>0){
						if (y<dst.rows - 1){
							s_y = y + shift_xy[s_y];
						}
						else{
							s_y = dst.rows - 1;
						}
					}
					else{
						s_y = 0;
					}
					a = rng.uniform(0, defaultsamples);
					backmodel[s_y][s_x][a] = b;//选择一个邻居更新其随机的一个样本集
				}
			}
			else{//判断为前景像素
				fgmask.at<uchar>(y, x) = 255;
				backmodel[y][x][defaultsamples]++;//前景计数
				if (backmodel[y][x][defaultsamples]>NB){//如果该点被记为前景的次数过多，说明有新东西覆盖了该点，而且该物体长时间静止，将其更新为背景
					//连续计数超过NB 次 须将至更新为背景点
					fgmask.at<uchar>(y, x) = 0;
					a = rng.uniform(0, defaultfactor);
					if (a == 0){//更新当前点的样本集
						a = rng.uniform(0, defaultsamples);
						backmodel[y][x][a] = b;
					}
				}
			}
		}

	}
}
void findRect(Mat mask,Mat &src)
{

	vector<vector<Point> > vecRect;//前景轮廓的矩形框
	findContours(mask, vecRect, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);//找轮廓，返回的框保存在vecRect中
	if (vecRect.empty())return;
	double area = 0;
	const double MIN_AREA = 50;
	for (int i = 0; i < vecRect.size(); ++i)
	{
		if (contourArea(vecRect[i]) < MIN_AREA)//如果矩形框的面积小于MIN_AREA则丢弃，表示他是一个细小的干扰点
			continue;
		else
		{
			Rect roiRect = boundingRect(vecRect[i]);
			Mat objectImg = src(roiRect);
			rectangle(src, roiRect, Scalar(0, 0, 255), 3);
			//imshow("objectImg",objectImg);
			//waitKey(0);
		}
	}
}

void main(){
	char *path = "d:\\dgq.avi";
	VideoCapture cap;
	cap.open(path);
	if (!cap.isOpened())
	{
		cout << "no file" << endl;
	}
	Mat img, dst;
	dst.create(scale, CV_32FC1);
	cap >> img;
	resize(img, img, scale);//将原图做尺寸转换，否则在原图上做会太慢
	cvtColor(img, dst, CV_BGR2GRAY);//前景图像，表示前景像素
	Mat fgmask(dst.size(), CV_8UC1, Scalar(0));
	clock_t t1 = clock();
	initialbm(dst);//初始化背景模型backmodel
	clock_t t2 = clock();
	cout << t2 - t1 << endl;//建模时间

	float fps = cap.get(CV_CAP_PROP_FPS);
	float vps = 1000 / fps;
	do
	{
		cap >> img;
		if (img.empty())
		{
			break;
		}
		resize(img, img, scale);
		cvtColor(img, dst, CV_BGR2GRAY);
		updatebm(dst, fgmask);
		erode(fgmask, fgmask, Mat(), Point(), 1);//腐蚀
		dilate(fgmask, fgmask, Mat(), Point(), 5);//膨胀
		erode(fgmask, fgmask, Mat(), Point(), 4);
		Mat tempFgmask = fgmask.clone();

		findRect(tempFgmask, img);
		imshow("dst", dst);
		imshow("fgmask", fgmask);
		imshow("img", img);
		if (waitKey(vps) > 0)
		{
			break;
		}
	} while (!img.empty());

}
#endif
