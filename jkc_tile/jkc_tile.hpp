#ifndef __JKC_TILE_H__
#define	__JKC_TILE_H__

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>


void cv_JKC_Tile(
	const IplImage* src_img, IplImage* dst_img, 
	const double down_ratio, 
	const int BlockSize=11, 
	const double ContThre=3.0, 
	const int MedianLevel=5);


void cv_JKC_Tile_2(
	const IplImage* src_img, IplImage* dst_img, 
	const int blk_width, const int blk_height,
	const int BlockSize=11, 
	const double ContThre=3.0, 
	const int MedianLevel=5);

void cv_JKC_Tile_3(
	IplImage* src_img, IplImage* dst_img,
	const int blk_width, const int blk_height,
	const int BlockSize = 11,
	const double ContThre = 3.0,
	const int MedianLevel = 5
	);

void cv_JKC_Tile_4(
	IplImage* src_img, IplImage* dst_img,
	const int blk_width, const int blk_height,
	const int BlockSize = 11,
	const double ContThre = 3.0,
	const int PyrLevel = 5,
	double SegmentThre = 30.0,
	const int MedianLevel = 5
	);


#endif // __JKC_TILE_H__