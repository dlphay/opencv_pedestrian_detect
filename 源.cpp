#include <cv.h>   
#include <highgui.h>     
#include <string>   
#include <iostream>   
#include <algorithm>   
#include <iterator>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <iostream>  


#include <stdio.h>  
#include <string.h>  
#include <ctype.h>  

using namespace cv;
using namespace std;

void help()
{
	printf(
		"\nDemonstrate the use of the HoG descriptor using\n"
		"  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
		"Usage:\n"
		"./peopledetect (<image_filename> | <image_list>.txt)\n\n");
}

int AAAA = 0;
int main()
{
	Mat img;
	FILE* f = 0;
	char _filename[1024];

	//打开视频文件：其实就是建立一个VideoCapture结构 
	VideoCapture capture("C:\\Users\\Administrator\\Desktop\\detector\\demo\\demo\\video.mp4");

	//检测是否正常打开:成功打开时，isOpened返回ture  
	if (!capture.isOpened())
		cout << "fail to open!" << endl;
	//获取整个帧数  
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "整个视频共" << totalFrameNumber << "帧" << endl;

	//设置开始帧()  
	long frameToStart = 310;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	cout << "从第" << frameToStart << "帧开始读" << endl;


	//设置结束帧  
	int frameToStop = totalFrameNumber;

	if (frameToStop < frameToStart)
	{
		cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
		return -1;
	}
	else
	{
		cout << "结束帧为：第" << frameToStop << "帧" << endl;
	}

	//获取帧率  
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "帧率为:" << rate << endl;

	//定义一个用来控制读取视频循环结束的变量  
	bool stop = false;
	//承载每一帧的图像  
	Mat frame;
	//显示每一帧的窗口  
	// namedWindow("Extracted frame");
	//两帧间的间隔时间:  
	//int delay = 1000/rate;  
	int delay = 1000 / rate;


	//利用while循环读取帧  
	//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量  
	long currentFrame = frameToStart;


	//滤波器的核  
	int kernel_size = 3;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);


	// hog行人检测
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器  




	Mat Final_display_mat;


	while (!stop)
	{
		//读取下一帧  
		if (!capture.read(frame))
		{
			cout << "读取视频失败" << endl;
			return -1;
		}

	    // 最终显示图片的初始化
		Final_display_mat = frame;
		/*********************************************************************/
		/****   行人检测处理程序   *****/
		Mat input_xingren;
		Mat input_luzhangxian;

		input_luzhangxian = frame;

		//取部分帧
		input_xingren = frame;

		int width = input_xingren.size().width;
		int height = input_xingren.size().height;
		Range R1;
		R1.start = width / 2;  // 320
		R1.end = width;      // 640
		Mat mask = Mat::Mat(input_xingren, Range::all(), R1);
		imshow("mask", mask);

		vector<Rect> found, found_filtered;
		double t = (double)getTickCount();

		/****   路沿边线检测处理程序   *****/

		Mat CannyImg;
		Canny(input_luzhangxian, CannyImg, 140, 250, 3);

		Mat DstImg;
		//cvtColor(frame, DstImg, CV_GRAY2BGR);

		vector<Vec4i> Lines;
		HoughLinesP(CannyImg, Lines, 1, CV_PI / 360, 170, 30, 15);
		for (size_t i = 0; i < Lines.size(); i++)
		{
			//line(input_luzhangxian, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 0, 255), 2, 8);
			line(Final_display_mat, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 0, 255), 2, 8);
		}



		// run the detector with default parameters. to get a higher hit-rate  
		// (and more false alarms, respectively), decrease the hitThreshold and  
		// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).  

		//Mat roi(input_xingren, Rect(input_xingren.cols - 100, 0, input_xingren.cols, input_xingren.rows));

		if (AAAA++%1 == 0)
		{
			hog.detectMultiScale(mask, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
			size_t i, j;
			for (i = 0; i < found.size(); i++)
			{
				Rect r = found[i];
				for (j = 0; j < found.size(); j++)
					if (j != i && (r & found[j]) == r)
						break;
				if (j == found.size())
					found_filtered.push_back(r);
			}
			for (i = 0; i < found_filtered.size(); i++)
			{
				Rect r = found_filtered[i];
				// the HOG detector returns slightly larger rectangles than the real objects.  
				// so we slightly shrink the rectangles to get a nicer output.  
				r.x += cvRound(r.width*0.1);
				r.width = cvRound(r.width*0.6);
				r.y += cvRound(r.height*0.07);
				r.height = cvRound(r.height*0.8);

				Rect rxxx = found_filtered[i];
				// the HOG detector returns slightly larger rectangles than the real objects.  
				// so we slightly shrink the rectangles to get a nicer output.  
				rxxx.x += cvRound(rxxx.width*0.1);
				rxxx.width = cvRound(rxxx.width*0.6);
				rxxx.y += cvRound(rxxx.height*0.055);
				rxxx.height = cvRound(rxxx.height*0.8);


				r.x += width / 2;
				rxxx.x +=( width / 2 + 53);

				//rectangle(input_xingren, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
				rectangle(Final_display_mat, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
				if (currentFrame == 344)  	rectangle(Final_display_mat, rxxx.tl(), rxxx.br(), cv::Scalar(0, 255, 0), 3);
			}
		}

		/**********************************************************************/

		//这里加滤波程序  
		imshow("Canny", CannyImg);
		//filter2D(frame, frame, -1, kernel);

		imshow("output", Final_display_mat);
		cout << "正在读取第" << currentFrame << "帧" << endl;



		//waitKey(int delay=0)当delay ≤ 0时会永远等待；当delay>0时会等待delay毫秒  
		//当时间结束前没有按键按下时，返回值为-1；否则返回按键  


		// FPS
		int c = waitKey(1);
		//按下ESC或者到达指定的结束帧后退出读取视频  
		if ((char)c == 27 || currentFrame > frameToStop)
		{
			stop = true;
		}
		//按下按键后会停留在当前帧，等待下一次按键  
		if (c >= 0)
		{
			waitKey(0);
		}
		currentFrame++;

	}
	//关闭视频文件  
	capture.release();
	waitKey(0);
	return 0;
	//img = imread("C:\\Users\\Administrator\\Desktop\\detector\\demo\\demo\\123.jpg");

	//if (img.data)
	//{
	//	strcpy(_filename, "C:\\Users\Administrator\Desktop\detector\demo\demo\123.jpg");
	//}
	//else
	//{
	//	f = fopen("C:\\Users\Administrator\Desktop\detector\demo\demo\123.jpg", "rt");
	//	if (!f)
	//	{
	//		fprintf(stderr, "ERROR: the specified file could not be loaded\n");
	//		return -1;
	//	}
	//}

	//// 开始逐帧分析！！！
	//while (true)
	//{
	//	Mat frame;
	//	Mat edge;
	//	capture >> frame;

	//	if (frame.empty())
	//		break;

	//	cvtColor(frame, edge, COLOR_BGR2GRAY);

	//	blur(edge, edge, Size(7, 7));

	//	Canny(edge, edge, 10, 30);

	//	imshow("Video", frame);
	//	imshow("After canny", edge);

	//	//等待50ms，如果从键盘输入的是q、Q、或者是Esc键，则退出
	//	int key = waitKey(50);
	//	if (key == 'q' || key == 'Q' || key == 27)
	//		break;
	//}




	//HOGDescriptor hog;
	//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器  
	//namedWindow("people detector", 1);

	//for (;;)
	//{
	//	char* filename = _filename;
	//	if (f)
	//	{
	//		if (!fgets(filename, (int)sizeof(_filename) - 2, f))
	//			break;
	//		//while(*filename && isspace(*filename))  
	//		//  ++filename;  
	//		if (filename[0] == '#')
	//			continue;
	//		int l = strlen(filename);
	//		while (l > 0 && isspace(filename[l - 1]))
	//			--l;
	//		filename[l] = '\0';
	//		img = imread(filename);
	//	}
	//	printf("%s:\n", filename);
	//	if (!img.data)
	//		continue;

	//	fflush(stdout);
	//	vector<Rect> found, found_filtered;
	//	double t = (double)getTickCount();
	//	// run the detector with default parameters. to get a higher hit-rate  
	//	// (and more false alarms, respectively), decrease the hitThreshold and  
	//	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).  
	//	hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	//	t = (double)getTickCount() - t;
	//	printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
	//	size_t i, j;
	//	for (i = 0; i < found.size(); i++)
	//	{
	//		Rect r = found[i];
	//		for (j = 0; j < found.size(); j++)
	//			if (j != i && (r & found[j]) == r)
	//				break;
	//		if (j == found.size())
	//			found_filtered.push_back(r);
	//	}
	//	for (i = 0; i < found_filtered.size(); i++)
	//	{
	//		Rect r = found_filtered[i];
	//		// the HOG detector returns slightly larger rectangles than the real objects.  
	//		// so we slightly shrink the rectangles to get a nicer output.  
	//		r.x += cvRound(r.width*0.1);
	//		r.width = cvRound(r.width*0.8);
	//		r.y += cvRound(r.height*0.07);
	//		r.height = cvRound(r.height*0.8);
	//		rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
	//	}
	//	imshow("people detector", img);
	//	int c = waitKey(0) & 255;
	//	if (c == 'q' || c == 'Q' || !f)
	//		break;
	//}
	//if (f)
	//	fclose(f);
	//return 0;
}