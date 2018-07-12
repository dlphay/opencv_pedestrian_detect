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

	//����Ƶ�ļ�����ʵ���ǽ���һ��VideoCapture�ṹ 
	VideoCapture capture("C:\\Users\\Administrator\\Desktop\\detector\\demo\\demo\\video.mp4");

	//����Ƿ�������:�ɹ���ʱ��isOpened����ture  
	if (!capture.isOpened())
		cout << "fail to open!" << endl;
	//��ȡ����֡��  
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "������Ƶ��" << totalFrameNumber << "֡" << endl;

	//���ÿ�ʼ֡()  
	long frameToStart = 310;
	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	cout << "�ӵ�" << frameToStart << "֡��ʼ��" << endl;


	//���ý���֡  
	int frameToStop = totalFrameNumber;

	if (frameToStop < frameToStart)
	{
		cout << "����֡С�ڿ�ʼ֡��������󣬼����˳���" << endl;
		return -1;
	}
	else
	{
		cout << "����֡Ϊ����" << frameToStop << "֡" << endl;
	}

	//��ȡ֡��  
	double rate = capture.get(CV_CAP_PROP_FPS);
	cout << "֡��Ϊ:" << rate << endl;

	//����һ���������ƶ�ȡ��Ƶѭ�������ı���  
	bool stop = false;
	//����ÿһ֡��ͼ��  
	Mat frame;
	//��ʾÿһ֡�Ĵ���  
	// namedWindow("Extracted frame");
	//��֡��ļ��ʱ��:  
	//int delay = 1000/rate;  
	int delay = 1000 / rate;


	//����whileѭ����ȡ֡  
	//currentFrame����ѭ�����п��ƶ�ȡ��ָ����֡��ѭ�������ı���  
	long currentFrame = frameToStart;


	//�˲����ĺ�  
	int kernel_size = 3;
	Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);


	// hog���˼��
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//�õ������  




	Mat Final_display_mat;


	while (!stop)
	{
		//��ȡ��һ֡  
		if (!capture.read(frame))
		{
			cout << "��ȡ��Ƶʧ��" << endl;
			return -1;
		}

	    // ������ʾͼƬ�ĳ�ʼ��
		Final_display_mat = frame;
		/*********************************************************************/
		/****   ���˼�⴦�����   *****/
		Mat input_xingren;
		Mat input_luzhangxian;

		input_luzhangxian = frame;

		//ȡ����֡
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

		/****   ·�ر��߼�⴦�����   *****/

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

		//������˲�����  
		imshow("Canny", CannyImg);
		//filter2D(frame, frame, -1, kernel);

		imshow("output", Final_display_mat);
		cout << "���ڶ�ȡ��" << currentFrame << "֡" << endl;



		//waitKey(int delay=0)��delay �� 0ʱ����Զ�ȴ�����delay>0ʱ��ȴ�delay����  
		//��ʱ�����ǰû�а�������ʱ������ֵΪ-1�����򷵻ذ���  


		// FPS
		int c = waitKey(1);
		//����ESC���ߵ���ָ���Ľ���֡���˳���ȡ��Ƶ  
		if ((char)c == 27 || currentFrame > frameToStop)
		{
			stop = true;
		}
		//���°������ͣ���ڵ�ǰ֡���ȴ���һ�ΰ���  
		if (c >= 0)
		{
			waitKey(0);
		}
		currentFrame++;

	}
	//�ر���Ƶ�ļ�  
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

	//// ��ʼ��֡����������
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

	//	//�ȴ�50ms������Ӽ����������q��Q��������Esc�������˳�
	//	int key = waitKey(50);
	//	if (key == 'q' || key == 'Q' || key == 27)
	//		break;
	//}




	//HOGDescriptor hog;
	//hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//�õ������  
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