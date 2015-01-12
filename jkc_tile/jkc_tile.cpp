#define _USE_MATH_DEFINES

#include <math.h>
#include "jkc_tile.hpp"

void cv_JKC_Tile(
	const IplImage* src_img, IplImage* dst_img, 
	const double down_ratio, 
	const int BlockSize, 
	const double ContThre, 
	const int MedianLevel) {

	// block
	IplImage* down_img  = cvCreateImage(
		cvSize(src_img->width*down_ratio, src_img->height*down_ratio), 
		src_img->depth, 
		src_img->nChannels
		);

	cvResize(src_img, down_img, CV_INTER_NN);
	cvResize(down_img, dst_img, CV_INTER_NN);
	cvReleaseImage(&down_img);

    ////////////////////////////////////////////////////////////////
    //
    // 輪郭画像の作成
    //
    ////////////////////////////////////////////////////////////////

    IplImage* contour_gray  = cvCreateImage (cvGetSize (src_img), IPL_DEPTH_8U, 1);
    IplImage* contour_color = cvCreateImage (cvGetSize (src_img), IPL_DEPTH_8U, 3);

    // カラー⇒モノクロ変換
    cvCvtColor (src_img, contour_gray, CV_BGR2GRAY);
    // 適応的二値化
    cvAdaptiveThreshold (contour_gray, contour_gray, 255, 
        CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, BlockSize, ContThre);
    // ノイズ除去
    cvSmooth(contour_gray, contour_gray, CV_MEDIAN, MedianLevel);
    // モノクロ変換⇒カラー
    cvCvtColor (contour_gray, contour_color, CV_GRAY2BGR);
    // 解放
    cvReleaseImage(&contour_gray);

    ////////////////////////////////////////////////////////////////
    //
    // 輪郭画像 ＋ 減色画像の作成
    //
    ////////////////////////////////////////////////////////////////
	cvAnd(dst_img, contour_color, dst_img);

    // 解放
    cvReleaseImage(&contour_color);

}

static void HSV2BGR(int h, int s, int v, CvScalar* rgb){
	float f;
	int i, p, q, t;

	i = (int)cvFloor(h*2 / 60.0f) % 6;
	f = (float)(h / 60.0f) - (float)cvFloor(h / 60.0f);
	p = (int)cvRound(v * (1.0f - (s / 255.0f)));
	q = (int)cvRound(v * (1.0f - (s / 255.0f) * f));
	t = (int)cvRound(v * (1.0f - (s / 255.0f) * (1.0f - f)));

	switch (i){ // BGR
	case 0: rgb->val[2] = v; rgb->val[1] = t; rgb->val[0] = p; break;
	case 1: rgb->val[2] = q; rgb->val[1] = v; rgb->val[0] = p; break;
	case 2: rgb->val[2] = p; rgb->val[1] = v; rgb->val[0] = t; break;
	case 3: rgb->val[2] = p; rgb->val[1] = q; rgb->val[0] = v; break;
	case 4: rgb->val[2] = t; rgb->val[1] = p; rgb->val[0] = v; break;
	case 5: rgb->val[2] = v; rgb->val[1] = p; rgb->val[0] = q; break;
	}
	rgb->val[3] = 0;
}

/*
void cv_JKC_Tile_2(
	const IplImage* src_img, IplImage* dst_img, 
	const int blk_width, const int blk_height,
	const int BlockSize, 
	const double ContThre, 
	const int MedianLevel) {

	CvRect roi;
	int n_div_x = src_img->width/blk_width;
	int n_div_y = src_img->height/blk_height;
	IplImage* blk_img  = cvCreateImage(cvSize(blk_width, blk_height), src_img->depth, src_img->nChannels);
	int blk_x, blk_y;
	CvHistogram *hist;
    IplImage* h_plane = cvCreateImage( cvGetSize(src_img), 8, 1 );
    IplImage* s_plane = cvCreateImage( cvGetSize(src_img), 8, 1 );
    IplImage* v_plane = cvCreateImage( cvGetSize(src_img), 8, 1 );
    IplImage* planes[] = { h_plane, s_plane, v_plane };
	IplImage* hsv = cvCreateImage( cvGetSize(src_img), 8, 3 );
	IplImage* mask = cvCreateImage( cvGetSize(src_img), 8, 1 );
	IplImage* dot = cvCreateImage(cvSize(1, 1), 8, 3);
	float h_range[] = { 0, 180 };
	float s_range[] = { 0, 256 };
	float v_range[] = { 0, 256 };
	float *ranges[] = { h_range, s_range, v_range };
	int h_bins = 30, s_bins = 32, v_bins = 32;
	int hist_size[] = { h_bins, s_bins, v_bins };
	float max_value = 0;
	int h, s, v;
	CvScalar max_hsv_col, max_bgr_col;

	printf("hist\n");
	hist = cvCreateHist (3, hist_size, CV_HIST_ARRAY, ranges, 1);
	cvCvtColor( src_img, hsv, CV_BGR2HSV );
	cvCvtPixToPlane(hsv, h_plane, s_plane, v_plane, 0);
	//cvCvtColor(dst_img, dst_img, CV_BGR2HSV);
	
	for (blk_y = 0; blk_y < n_div_y; blk_y++) {
		for (blk_x = 0; blk_x < n_div_x; blk_x++) {
			roi.x = blk_width * blk_x;
			roi.y = blk_height * blk_y;
			roi.width = blk_width;
			roi.height = blk_height;

			//cvSetImageROI (dst_img, roi);
			cvZero(mask); 
			cvRectangle(
				mask, 
				cvPoint(roi.x, roi.y), cvPoint(roi.x + blk_width, roi.y + blk_height), 
				cvScalarAll(255), CV_FILLED, 8, 0
				);
			cvCalcHist(planes, hist, 0, mask);
			cvGetMinMaxHistValue (hist, 0, &max_value, 0, 0);
			
			float max_bin_val = 0;
			for (h = 0; h < h_bins; h++) {
				for (s = 0; s < s_bins; s++) {
					for (v = 0; v < v_bins; v++) {
						float bin_val = cvQueryHistValue_3D(hist, h, s, v);
						if (max_bin_val < bin_val) {
							max_bin_val = bin_val;
							max_hsv_col = cvScalar(h, s, v);
						}
					}
				}
			}

			//printf("blk_x=%d,blk_y=%d,h=%f,s=%f,v=%f\n", blk_x, blk_y, max_hsv_col.val[0], max_hsv_col.val[1], max_hsv_col.val[2]);
			dot->imageData[0] = max_hsv_col.val[0]*6;
			dot->imageData[1] = max_hsv_col.val[1]*8;
			dot->imageData[2] = max_hsv_col.val[2]*8;
			dot->imageData[3] = 0;
			cvCvtColor(dot, dot, CV_HSV2BGR);
			max_bgr_col.val[0] = dot->imageData[0];
			max_bgr_col.val[1] = dot->imageData[1];
			max_bgr_col.val[2] = dot->imageData[2];
			max_bgr_col.val[3] = 0;

			//HSV2BGR(max_hsv_col.val[0], max_hsv_col.val[1], max_hsv_col.val[2], &max_bgr_col);
			cvRectangle(
				dst_img,
				cvPoint(roi.x, roi.y), cvPoint(roi.x + blk_width, roi.y + blk_height),
				max_bgr_col, CV_FILLED, 8, 0
				);

			//cvResetImageROI (dst_img);
		}
	}

	//cvCvtColor(dst_img, dst_img, CV_HSV2BGR);
	cvReleaseImage(&h_plane);
	cvReleaseImage(&s_plane);
	cvReleaseImage(&v_plane);
	cvReleaseImage(&hsv);
	cvReleaseImage(&mask);
	cvReleaseImage(&dot);
}

void cv_JKC_Tile_3(
	IplImage* src_img, IplImage* dst_img,
	const int blk_width, const int blk_height,
	const int BlockSize,
	const double ContThre,
	const int MedianLevel
	) {

	int n_div_x = src_img->width / blk_width;
	int n_div_y = src_img->height / blk_height;
	IplImage* blk_img = cvCreateImage(cvSize(blk_width, blk_height), src_img->depth, src_img->nChannels);
	int blk_x, blk_y;
	CvHistogram *hist;
	IplImage* b_plane = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* g_plane = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* r_plane = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* planes[] = { b_plane, g_plane, r_plane };
	IplImage* bgr = cvCreateImage(cvGetSize(src_img), 8, 3);
	IplImage* mask = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* dot = cvCreateImage(cvSize(1, 1), 8, 3);
	float b_range[] = { 0, 256 };
	float g_range[] = { 0, 256 };
	float r_range[] = { 0, 256 };
	float *ranges[] = { b_range, g_range, r_range };
	int r_bins = 32, g_bins = 32, b_bins = 32;
	int hist_size[] = { r_bins, g_bins, b_bins };
	float fst_bin_val = 0, snd_bin_val = 0;
	int max_index;
	int b, g, r;
	CvScalar fst_bgr_col = cvScalar(0,0,0,0), snd_bgr_col;
	int radius;

	hist = cvCreateHist(3, hist_size, CV_HIST_ARRAY, ranges, 1);
	//cvCvtColor(src_img, hsv, CV_BGR2HSV);
	//cvCvtPixToPlane(hsv, h_plane, s_plane, v_plane, 0);
	//cvCvtPixToPlane(src_img, b_plane, g_plane, r_plane, 0);
	cvCvtPixToPlane(dst_img, b_plane, g_plane, r_plane, 0);
	//cvCvtColor(dst_img, dst_img, CV_BGR2HSV);

	for (blk_y = 0; blk_y < n_div_y; blk_y++) {
		for (blk_x = 0; blk_x < n_div_x; blk_x++) {
			roi.x = blk_width * blk_x;
			roi.y = blk_height * blk_y;
			roi.width = blk_width;
			roi.height = blk_height;

			//cvSetImageROI (dst_img, roi);
			cvZero(mask);
			cvRectangle(
				mask,
				cvPoint(roi.x, roi.y), cvPoint(roi.x + blk_width, roi.y + blk_height),
				cvScalarAll(255), CV_FILLED, 8, 0
				);
			cvCalcHist(planes, hist, 0, mask);
			cvGetMinMaxHistValue(hist, 0, &fst_bin_val, 0, 0);

			fst_bin_val = 0;
			snd_bin_val = 0;
			for (b = 0; b < b_bins; b++) {
				for (g = 0; g < g_bins; g++) {
					for (r = 0; r < r_bins; r++) {
						float bin_val = cvQueryHistValue_3D(hist, b, g, r);
						if (fst_bin_val < bin_val) {
							snd_bin_val = fst_bin_val;
							snd_bgr_col = fst_bgr_col;
							fst_bin_val = bin_val;
							fst_bgr_col = cvScalar(b * 4, g * 4, r * 4);
						}
					}
				}
			}

			cvRectangle(
				dst_img,
				cvPoint(roi.x, roi.y), cvPoint(roi.x + blk_width, roi.y + blk_height),
				fst_bgr_col, CV_FILLED, 8, 0
				);
			// blk_widht*blk_height : radius^2*pi = fst_bin_val : snd_bin_val
			radius = (int)sqrt(blk_width*blk_height * snd_bin_val / fst_bin_val / M_PI);
			cvCircle(
				dst_img,
				cvPoint(roi.x + blk_width/2, roi.y + blk_height/2), radius,
				snd_bgr_col, CV_FILLED, 8, 0
				);
		}
	}

	////////////////////////////////////////////////////////////////
	//
	// 輪郭画像の作成
	//
	////////////////////////////////////////////////////////////////

	IplImage* contour_gray = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 1);
	IplImage* contour_color = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 3);

	// カラー⇒モノクロ変換
	cvCvtColor(src_img, contour_gray, CV_BGR2GRAY);
	// 適応的二値化
	cvAdaptiveThreshold(contour_gray, contour_gray, 255,
		CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, BlockSize, ContThre);
	// ノイズ除去
	cvSmooth(contour_gray, contour_gray, CV_MEDIAN, MedianLevel);
	// モノクロ変換⇒カラー
	cvCvtColor(contour_gray, contour_color, CV_GRAY2BGR);
	// 解放
	cvReleaseImage(&contour_gray);

	////////////////////////////////////////////////////////////////
	//
	// 輪郭画像 ＋ 減色画像の作成
	//
	////////////////////////////////////////////////////////////////
	cvAnd(dst_img, contour_color, dst_img);
	//cvCopy(contour_color, dst_img, NULL);

	//cvCvtColor(dst_img, dst_img, CV_HSV2BGR);
	cvReleaseImage(&b_plane);
	cvReleaseImage(&g_plane);
	cvReleaseImage(&r_plane);
	cvReleaseImage(&bgr);
	cvReleaseImage(&mask);
	cvReleaseImage(&dot);
}
*/

void cv_JKC_Tile_addEdge(const IplImage *src_img, IplImage *dst_img, const int BlockSize, const int ContThre, const int MedianLevel) {
	////////////////////////////////////////////////////////////////
	//
	// 輪郭画像の作成
	//
	////////////////////////////////////////////////////////////////

	IplImage* contour_gray = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 1);
	IplImage* contour_color = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 3);

	// カラー⇒モノクロ変換
	cvCvtColor(src_img, contour_gray, CV_BGR2GRAY);
	// 適応的二値化
	cvAdaptiveThreshold(contour_gray, contour_gray, 255,
		CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, BlockSize, ContThre);
	// ノイズ除去
	cvSmooth(contour_gray, contour_gray, CV_MEDIAN, MedianLevel);
	// モノクロ変換⇒カラー
	cvCvtColor(contour_gray, contour_color, CV_GRAY2BGR);
	// 解放
	cvReleaseImage(&contour_gray);

	////////////////////////////////////////////////////////////////
	//
	// 輪郭画像 ＋ 減色画像の作成
	//
	////////////////////////////////////////////////////////////////
	cvAnd(dst_img, contour_color, dst_img);
	//cvCopy(contour_color, dst_img, NULL);
	
	cvReleaseImage(&contour_color);
	
}

void cv_JKC_Tile_addCannyEdge(IplImage* src_img, IplImage* dst_img, double lowThresh, double highThresh, int apertureSize)
{
	////////////////////////////////////////////////////////////////
	//
	// 輪郭画像の作成
	//
	////////////////////////////////////////////////////////////////

	IplImage* contour_gray = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 1);
	IplImage* contour_color = cvCreateImage(cvGetSize(src_img), IPL_DEPTH_8U, 3);

	// カラー⇒モノクロ変換
	cvCvtColor(src_img, contour_gray, CV_BGR2GRAY);
	// CannyEdge
	cvCanny(contour_gray, contour_gray, lowThresh, highThresh, apertureSize);
	//　反転
	cvNot(contour_gray, contour_gray);
	// ノイズ除去
	//cvSmooth(contour_gray, contour_gray, CV_MEDIAN, MedianLevel);
	// モノクロ変換⇒カラー
	cvCvtColor(contour_gray, contour_color, CV_GRAY2BGR);
	// 解放
	cvReleaseImage(&contour_gray);

	////////////////////////////////////////////////////////////////
	//
	// 輪郭画像 ＋ 減色画像の作成
	//
	////////////////////////////////////////////////////////////////
	cvAnd(dst_img, contour_color, dst_img);
	//cvCopy(contour_color, dst_img, NULL);

	cvReleaseImage(&contour_color);
}

void cv_JKC_Tile_DecreseColor(
	IplImage* src_img, IplImage* dst_img,
	const int PyrLevel, 
	double SegmentThre,
	const int MedianLevel
	) {
	////////////////////////////////////////////////////////////////
	//
	// 減色画像の作成
	//
	////////////////////////////////////////////////////////////////

	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *comp = 0;
	CvRect roi;

	// 領域分割のためにROIをセットする
	roi.x = roi.y = 0;
	roi.width = src_img->width & -(1 << PyrLevel);
	roi.height = src_img->height & -(1 << PyrLevel);
	cvSetImageROI(src_img, roi);
	cvSetImageROI(dst_img, roi);
	// 画像ピラミッドを使った領域分割
	cvPyrSegmentation(src_img, dst_img, storage, &comp, PyrLevel, 255.0, SegmentThre);
	// ノイズ除去
	cvSmooth(dst_img, dst_img, CV_MEDIAN, MedianLevel);
	// 解放
	cvReleaseMemStorage(&storage);
	cvResetImageROI(src_img);
	cvResetImageROI(dst_img);
}

// http://opencv.jp/opencv2-x-samples/k-means_clustering
void cv_JKC_Tile_DecreaseColor_2(
	IplImage* src_img, IplImage* dst_img, const int max_clusters
	)
{
	int i, size;
	CvMat *clusters, *points;
	CvMat *count = cvCreateMat(max_clusters, 1, CV_32SC1);
	CvMat *centers = cvCreateMat(max_clusters, 3, CV_32FC1);

	size = src_img->width * src_img->height;
	//dst_img = cvCloneImage(src_img);
	clusters = cvCreateMat(size, 1, CV_32SC1);
	points = cvCreateMat(size, 1, CV_32FC3);

	// (2)reshape the image to be a 1 column matrix 
	for (i = 0; i<size; i++) {
		points->data.fl[i * 3 + 0] = (uchar)src_img->imageData[i * 3 + 0];
		points->data.fl[i * 3 + 1] = (uchar)src_img->imageData[i * 3 + 1];
		points->data.fl[i * 3 + 2] = (uchar)src_img->imageData[i * 3 + 2];
	}

	// (3)run k-means clustering algorithm to segment pixels in RGB color space
	cvKMeans2(points, max_clusters, clusters,
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		1, 0, 0, centers, 0);

	// (4)make a each centroid represent all pixels in the cluster
	for (i = 0; i<size; i++) {
		int idx = clusters->data.i[i];
		dst_img->imageData[i * 3 + 0] = (char)centers->data.fl[idx * 3 + 0];
		dst_img->imageData[i * 3 + 1] = (char)centers->data.fl[idx * 3 + 1];
		dst_img->imageData[i * 3 + 2] = (char)centers->data.fl[idx * 3 + 2];
	}

	cvReleaseMat(&clusters);
	cvReleaseMat(&points);
	cvReleaseMat(&centers);
	cvReleaseMat(&count);
}

void cv_JKC_Tile_4(
	IplImage* src_img, IplImage* dst_img,
	FILE* fp_dbg_out,
	const int blk_width, const int blk_height,
	const int BlockSize,
	const double ContThre,
	const int PyrLevel,
	double SegmentThre,
	const int MedianLevel
	) {

	int n_div_x = src_img->width / blk_width;
	int n_div_y = src_img->height / blk_height;
	IplImage* blk_img = cvCreateImage(cvSize(blk_width, blk_height), src_img->depth, src_img->nChannels);
	int blk_x, blk_y;
	CvHistogram *hist;
	IplImage* b_plane = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* g_plane = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* r_plane = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* planes[] = { b_plane, g_plane, r_plane };
	IplImage* bgr = cvCreateImage(cvGetSize(src_img), 8, 3);
	IplImage* mask = cvCreateImage(cvGetSize(src_img), 8, 1);
	IplImage* dot = cvCreateImage(cvSize(1, 1), 8, 3);
	float b_range[] = { 0, 256 };
	float g_range[] = { 0, 256 };
	float r_range[] = { 0, 256 };
	float *ranges[] = { b_range, g_range, r_range };
	int r_bins = 32, g_bins = 32, b_bins = 32;
	int hist_size[] = { r_bins, g_bins, b_bins };
	float fst_bin_val = 0, snd_bin_val = 0;
	int max_index;
	int b, g, r;
	CvScalar fst_bgr_col = cvScalar(0, 0, 0, 0), snd_bgr_col;
	int radius;
	CvRect roi;
	float cnt;
	float w;

	////////////////////////////////////////////////////////////////
	cvCvtColor(src_img, src_img, CV_BGR2HSV);

	////////////////////////////////////////////////////////////////
	// 減色画像の作成
	//cv_JKC_Tile_DecreseColor(src_img, dst_img, PyrLevel, SegmentThre, MedianLevel);
	cv_JKC_Tile_DecreaseColor_2(src_img, dst_img, 8);

	cvCvtColor(dst_img, dst_img, CV_HSV2BGR);

	////////////////////////////////////////////////////////////////
	// 
	hist = cvCreateHist(3, hist_size, CV_HIST_ARRAY, ranges, 1);
	cvCvtPixToPlane(dst_img, b_plane, g_plane, r_plane, 0);

	fprintf(fp_dbg_out, "max_bin=%d\n", blk_width*blk_height);
	fprintf(fp_dbg_out, "blk_x,blk_y,fst_bin_val,snd_bin_val\n");
	for (blk_y = 0; blk_y < n_div_y; blk_y++) {
		for (blk_x = 0; blk_x < n_div_x; blk_x++) {
			roi.x = blk_width * blk_x;
			roi.y = blk_height * blk_y;
			roi.width = blk_width;
			roi.height = blk_height;

			// 該当ブロック領域のヒストを取得
			//cvSetImageROI (dst_img, roi);
			cvZero(mask);
			cvRectangle(
				mask,
				cvPoint(roi.x, roi.y), cvPoint(roi.x + blk_width - 1, roi.y + blk_height - 1),
				cvScalarAll(255), CV_FILLED, 8, 0
				);
			cvCalcHist(planes, hist, 0, mask);
			cvNormalizeHist(hist, 1.0);
			//cvGetMinMaxHistValue(hist, 0, &fst_bin_val, 0, 0);

			// 最大binとその次に大きいbinの取得
			fst_bin_val = 0;
			snd_bin_val = 0;
			cnt = 0.0;
			for (b = 0; b < b_bins; b++) {
				for (g = 0; g < g_bins; g++) {
					for (r = 0; r < r_bins; r++) {
						float bin_val = cvQueryHistValue_3D(hist, b, g, r);
						cnt += bin_val;
						if (snd_bin_val < bin_val) {
							if (fst_bin_val < bin_val) {
								snd_bin_val = fst_bin_val;
								snd_bgr_col = fst_bgr_col;
								fst_bin_val = bin_val;
								fst_bgr_col = cvScalar(b * 8, g * 8, r * 8); // 8 = 256/32
							}
							else {
								snd_bin_val = bin_val;
								snd_bgr_col = cvScalar(b * 8, g * 8, r * 8); // 8 = 256/32
							}
						}
					}
				}
			}

			fprintf(fp_dbg_out, "%3d,%3d,%f,%1.4f,%1.4f,%1.4f,", 
				blk_x, blk_y, fst_bin_val, snd_bin_val, fst_bin_val+snd_bin_val, cnt);
			
			if (fst_bin_val < 1.8) {
				fprintf(fp_dbg_out, "0,");
			} else if (	snd_bin_val / fst_bin_val > 0.25 ) { // 境界
					fprintf(fp_dbg_out, "1,");
			} else {
				fprintf(fp_dbg_out, "2,");
				cvRectangle(
					dst_img,
					cvPoint(roi.x, roi.y), cvPoint(roi.x + blk_width, roi.y + blk_height),
					fst_bgr_col, CV_FILLED, 8, 0
					);
				// fst_bin_val : snd_bin_val = (blk_width - w) * (blk_height - h) : w*h
				// blk_width=blk_height, w=hとすると
				// (blk_width^2 -2*w*blk_width + w^2) * snd_bin_val = w^2*fst_bin_val
				// (fst_bin_val-snd_bin_val)*w^2 + 2*snd_bin_val*blk_width*w - snd_bin_val*blk_width^2 = 0
				// w = -2*snd_bin_val*blk_width +- sqrt(4*snd_bin_val^2*blk_width^2 + 4*(fst_bin_val-snd_bin_val)*snd_bin_val*blk_width^2)) / (2*(fst_bin_val-snd_bin_val))
				w = (-2 * snd_bin_val * blk_width + sqrt(4 * snd_bin_val*snd_bin_val*blk_width*blk_width + 4 * (fst_bin_val - snd_bin_val)*snd_bin_val*blk_width*blk_width)) /
					(2.0*(fst_bin_val - snd_bin_val));
				cvRectangle(
					dst_img,
					cvPoint(roi.x + blk_width / 2 - w, roi.y + blk_height/2 - w), 
					cvPoint(roi.x + blk_width / 2 + w, roi.y + blk_height/2 + w),
					snd_bgr_col, CV_FILLED, 8, 0
					);
				// blk_widht*blk_height : radius^2*pi = fst_bin_val : snd_bin_val
				/*
				radius = (int)sqrt(blk_width*blk_height * snd_bin_val / fst_bin_val / M_PI);
				cvRectangle(
					dst_img,
					cvPoint(roi.x + blk_width / 2, roi.y + blk_height / 2), radius,
					snd_bgr_col, CV_FILLED, 8, 0
					);
				*/
			}

			fprintf(fp_dbg_out, "\n");
		}
	}

	////////////////////////////////////////////////////////////////
	// エッジ追加
	//cv_JKC_Tile_addEdge(dst_img, dst_img, BlockSize, ContThre, MedianLevel);
	cv_JKC_Tile_addCannyEdge(src_img, dst_img, 50, 200, 3);

	////////////////////////////////////////////////////////////////
	// リリース
	//cvCvtColor(dst_img, dst_img, CV_HSV2BGR);
	cvReleaseImage(&b_plane);
	cvReleaseImage(&g_plane);
	cvReleaseImage(&r_plane);
	cvReleaseImage(&bgr);
	cvReleaseImage(&mask);
	cvReleaseImage(&dot);
}
