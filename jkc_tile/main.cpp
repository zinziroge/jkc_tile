#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

#include "jkc_tile.hpp"

// Visual Studio 2013
#ifdef _DEBUG
    //Debugモードの場合
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_core2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_imgproc2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_highgui2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_objdetect2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_contrib2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_features2d2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_flann2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_gpu2410d.lib")
    //#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_haartraining_engined.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_legacy2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_ts2410d.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_video2410d.lib")
#else
    //Releaseモードの場合
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_core2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_imgproc2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_highgui2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_objdetect2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_contrib2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_features2d2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_flann2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_gpu2410.lib")
    //#pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_haartraining_engined.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_legacy2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_ts2410.lib")
    #pragma comment(lib,"C:\\opencv\\build\\x86\\vc12\\lib\\opencv_video2410.lib")
#endif

void on_mouse(int event, int x, int y, int flags, void *param = NULL);

IplImage *img1;
IplImage *img2;
int sw_img = 0;

void on_mouse(int event, int x, int y, int flags, void *param) {
	static int line = 0;
	const int max_line = 15, w = 15, h = 30;

	// (4)マウスイベントを取得
	switch (event) {
	case CV_EVENT_LBUTTONDOWN:
		if (sw_img == 0) {
			cvShowImage("Proc", img1);
		}
		else {
			cvShowImage("Proc", img2);
		}
		sw_img = (sw_img + 1) % 2;
		printf("(%d,%d) %s\n", x, y, "LBUTTON_DOWN");
		break;
	}
}
int main(int argc, char** argv) {
	char c;
	FILE *fp_dbg_out;

	fp_dbg_out = fopen("dbg_out.txt", "w");
	if (fp_dbg_out == NULL) {
		fprintf(stderr, "Error: Can't open %s.\n", "dbg_out.txt");
		exit(1);
	}
	img1 = cvLoadImage(argv[1], 1);
	img2 = cvCreateImage(cvSize(img1->width, img1->height), img1->depth, img1->nChannels);

	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE); 
	cvNamedWindow("Proc", CV_WINDOW_AUTOSIZE); 
	cvSetMouseCallback("Proc", on_mouse);
	//
	//cvCvtColor(img1, img1, CV_BGR2HSV);
	//cvCvtColor(img1, img1, CV_HSV2BGR);
	cvShowImage("Original", img1);

	//cv_CartoonFilter(img1, img2, 11, 5.0, 5, 30, 5);
	//cv_JKC_Tile(img1, img2, 1/8.);
	//cv_JKC_Tile_2(img1, img2, 15, 15);
	//cv_JKC_Tile_3(img1, img2, 15, 15);
	cv_JKC_Tile_4(img1, img2, fp_dbg_out, 15, 15);
	cvShowImage("Proc", img2);

	printf("Enter any key?\n");
	// (3)'Esc'キーが押された場合に終了する
	while (1) {
		c = cvWaitKey(0);
		if (c == '\x1b')
			return 1;
	}

	// local resource
	fclose(fp_dbg_out);
	cvDestroyWindow("Original");
	cvDestroyWindow("Proc");
	
	// global resource
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);

	return 0;
}