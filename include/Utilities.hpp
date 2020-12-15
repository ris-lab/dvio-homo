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

#ifndef UTLITIES_HPP_
#define UTLITIES_HPP_

#include <vector>
#include <string>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
namespace HomoAlign{
using namespace cv;
using namespace std;
void ConvertImageToFloat(Mat & image);
Mat SmoothImage(const float sigma, const Mat & im);
void ComputeImageDerivatives(const Mat & image, Mat & imageDx, Mat &imageDy);
void ComputeImageDerivatives(const Mat & image, int x, int y, float &dx, float &dy);
void NormalizeImage(Mat &image);
}
#endif /* UTLITIES_HPP_*/
