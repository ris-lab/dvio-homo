/*
Copyright 2014 Alberto Crivellaro, Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
alberto.crivellaro@epfl.ch

terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include "Utilities.hpp"
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;

namespace HomoAlign{

void ConvertImageToFloat(Mat & image)
{
	//image.convertTo(image, CV_32F);
	double min,max, average;

	minMaxLoc(image,&min,&max);
	//printf("min, max: %f %f\n", min, max);
	const float v = 1.0/(max - min);
	//image.convertTo(image, CV_32F);
	image.convertTo(image, CV_32F, v, -min * v);
	//std::cout << image << std::endl;
	//float *img = (float*)image.data;
	//printf("image: %f %f %f\n", img[0], img[1], img[2]);
	assert(image.isContinuous());
}


void ComputeImageDerivatives(const Mat & image, int x, int y, float &dx, float &dy)
{
	assert(image.type() == 5);
	const float *pimg = image.ptr<float>();
	//int idata = y*image.cols+x;
	int iup, idown, ileft, iright;
	if( y == 0 )
		iup = y*image.cols+x;
	else
		iup = (y-1)*image.cols+x;
	if( x == 0)
		ileft = y*image.cols+x; 
	else
		ileft = y*image.cols+x-1;
	if( x == image.cols - 1)
		iright = y*image.cols+x;
	else
		iright = y*image.cols+x+1;
		
	if( y == image.cols - 1)
		idown = y*image.cols+x; 
	else
		idown = (y+1)*image.cols+x; 

	dx = (pimg[iright] - pimg[ileft])*.5f;
	dy = (pimg[idown] - pimg[iup])*.5f;
}

void ComputeImageDerivatives(const Mat & image, Mat & imageDx, Mat &imageDy)
{
	int ddepth = -1; //same image depth as source
	double scale = 1/32.0;// normalize wrt scharr mask for having exact gradient
	double delta = 0;

	Scharr(image, imageDx, ddepth, 1, 0, scale, delta, BORDER_REFLECT );
	Scharr(image, imageDy, ddepth, 0, 1, scale, delta, BORDER_REFLECT );
}

Mat SmoothImage(const float sigma, const Mat &im)
{
	Mat smoothedImage;
	int s = max(3, 2*int(sigma)+1);
	Size kernelSize(s, s);
	GaussianBlur(im, smoothedImage, kernelSize, sigma, sigma, BORDER_REFLECT);
	return smoothedImage;
}

vector<Mat> SmoothDescriptorFields(const float sigma, const vector<Mat> & descriptorFields)
{
	vector<Mat> smoothedDescriptorFields(descriptorFields.size());

#pragma omp parallel for
	for(int iChannel = 0; iChannel < descriptorFields.size(); ++iChannel){
		smoothedDescriptorFields[iChannel] = SmoothImage(sigma, descriptorFields[iChannel]);}

	return smoothedDescriptorFields;
}



void NormalizeImage(Mat &image)
{
	Scalar mean, stddev;
	meanStdDev(image, mean, stddev);
	image = (image - mean)/stddev[0];
}

}

