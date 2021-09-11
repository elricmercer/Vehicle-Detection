#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/saliency/saliencySpecializedClasses.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <fstream>
#include <string>

using namespace cv;

// saliency
const int STEP = 8;
const int GABOR_R = 8;
float WEIGHT_I = 0.333f;
float WEIGHT_C = 0.333f;
float WEIGHT_O = 0.333f;
int call = 0; //global


void KMeans(Mat src, int clusterCount);
Mat CombineSaliency(Mat itti, Mat FG, Mat SR);
void Binary(Mat src);
void findContours(Mat src);
void roi(Mat src);


//"Center-surround" for two images with different scales Input image path to be calculated 
Mat operateCenterSurround(const Mat& center, const Mat& surround) {

    Mat csmap(center.size(), center.type());
    std::cout << "Call: " << ++call << std::endl;
    std::cout << "surround.size(): " << surround.size() << "\t";
    std::cout << "center.size(): " << center.size() << "\t";
    std::cout << "csmap.size(): " << csmap.size() << "\n";
    resize(surround, csmap, csmap.size()); //Enlarge the Surround image to the same size as the Center image 
    std::cout << "csmap.size(): " << csmap.size() << "\n\n";
    csmap = abs(csmap - center);
    return csmap;
}

//Build a "center-surround" pyramid from various pyramids 
std::vector<Mat> buildCenterSurroundPyramid(const std::vector<Mat>& pyramid) {
    //surround = center+delta, center={2,3,4}, delta={3,4} <- total 6 Pieces
    std::vector<Mat> cspyr(6);
    cspyr[0] = operateCenterSurround(pyramid[2], pyramid[5]);
    cspyr[1] = operateCenterSurround(pyramid[2], pyramid[6]);
    cspyr[2] = operateCenterSurround(pyramid[3], pyramid[6]);
    cspyr[3] = operateCenterSurround(pyramid[3], pyramid[7]);
    cspyr[4] = operateCenterSurround(pyramid[4], pyramid[7]);
    cspyr[5] = operateCenterSurround(pyramid[4], pyramid[8]);
    return cspyr;
}

//Normalize the dynamic range of the image to [0,1] 
void normalizeRange(Mat& image) {
    double minval, maxval;
    minMaxLoc(image, &minval, &maxval);

    image -= minval;
    if (minval < maxval)
        image /= maxval - minval;
}

//Normalization operator N (?): Single peak enhancement and multi-peak suppression
//The extremum calculation was created based on [Reference Material 4] because there is no specific procedure description.
void trimPeaks(Mat& image, int step) {
    const int w = image.cols;
    const int h = image.rows;

    const double M = 1.0;
    normalizeRange(image); //Functio does not exist?
    double m = 0.0;
    for (int y = 0; y < h - step; y += step)			//The end is left over by (h% step) 
        for (int x = 0; x < w - step; x += step) {	//The end is left over by (w% step) 
            Mat roi(image, Rect(x, y, step, step));
            double minval = 0.0;
            double maxval = 0.0;
            minMaxLoc(roi, &minval, &maxval);
            m += maxval;
        }
    m /= (w / step - (w % step ? 0 : 1)) * (h / step - (h % step ? 0 : 1)); //Divide by the number of blocks to calculate the average 
    image *= (M - m) * (M - m);
}

//Calculate the saliency map 
Mat calcSaliencyMap(const Mat& image0) {
    const Mat_<Vec3f> image = image0 / 255.0f; //Dynamic range normalization

    //Pre-generated Gabor Kernel
    //For each parameter of the gabor filter
    //Since there are no specific values or procedures in the thesis, it is decided by personal preference.

    const Size ksize = Size(GABOR_R + 1 + GABOR_R, GABOR_R + 1 + GABOR_R);
    const double sigma = GABOR_R / CV_PI;
    const double lambda = GABOR_R + 1;
    const double deg45 = CV_PI / 4.0;

    Mat gabor000 = getGaborKernel(ksize, sigma, deg45 * 0, lambda, 1.0, 0.0, CV_32F);
    Mat gabor045 = getGaborKernel(ksize, sigma, deg45 * 1, lambda, 1.0, 0.0, CV_32F);
    Mat gabor090 = getGaborKernel(ksize, sigma, deg45 * 2, lambda, 1.0, 0.0, CV_32F);
    Mat gabor135 = getGaborKernel(ksize, sigma, deg45 * 3, lambda, 1.0, 0.0, CV_32F);

    const int NUM_SCALES = 9;

    std::vector<Mat> pyramidI(NUM_SCALES);	//Lumaniance Pyramid
    std::vector<Mat> pyramidRG(NUM_SCALES);	//Hue RB Pyramid
    std::vector<Mat> pyramidBY(NUM_SCALES);	//Hue BY Pyramid
    std::vector<Mat> pyramid000(NUM_SCALES);	//Direction 0 Pyramid
    std::vector<Mat> pyramid045(NUM_SCALES);	//Direction 45 Pyramid
    std::vector<Mat> pyramid090(NUM_SCALES);	//Direction 90 Pyramid
    std::vector<Mat> pyramid135(NUM_SCALES);	//Direction 135 Pyramid

    //Building a trait map pyramid
    Mat scaled = image; //The first scale is the original image 
    for (int s = 0; s < NUM_SCALES; ++s) {
        const int w = scaled.cols;
        const int h = scaled.rows;

        //Luminance Map Generation:
        std::vector<Mat_<float>> colours;
        split(scaled, colours);
        Mat_<float> imageI = (colours[0] + colours[1] + colours[2]) / 3.0f;
        //colours[0-2] contain the same image.
        //imshow("colours[0]", colours[0]);
        //imshow("colours[1]", colours[1]);
        //imshow("colours[2]", colours[2]);
        pyramidI[s] = imageI;

        //Calculation of normalized rgb values 
        double minval, maxval;
        minMaxLoc(imageI, &minval, &maxval);
        Mat_<float> r(h, w, 0.0f);
        Mat_<float> g(h, w, 0.0f);
        Mat_<float> b(h, w, 0.0f);
        for (int j = 0; j < h; ++j) {
            for (int i = 0; i < w; ++i) {
                if (imageI(j, i) < 0.1f * maxval) //Calculation of normalized rgb value Excludes pixels less than 1/10 of the maximum peak 
                    continue;
                r(j, i) = colours[2](j, i) / imageI(j, i);
                g(j, i) = colours[1](j, i) / imageI(j, i);
                b(j, i) = colours[0](j, i) / imageI(j, i);
            }
        }

        //imshow("b: colours[0]", b);
        //imshow("g: colours[1]", g);
        //imshow("r: colours[2]", r);

        //Hue map generation (negative value clamped to 0)
        Mat R = max(0.0f, r - (g + b) / 2);
        Mat G = max(0.0f, g - (b + r) / 2);
        Mat B = max(0.0f, b - (r + g) / 2);
        Mat Y = max(0.0f, (r + g) / 2 - abs(r - g) / 2 - b);
        pyramidRG[s] = R - G;
        pyramidBY[s] = B - Y;

        //Direction map generation 
        filter2D(imageI, pyramid000[s], -1, gabor000);
        filter2D(imageI, pyramid045[s], -1, gabor045);
        filter2D(imageI, pyramid090[s], -1, gabor090);
        filter2D(imageI, pyramid135[s], -1, gabor135);

        pyrDown(scaled, scaled); //Scale down for the next octave 
    }

    //Center-surround operation
    /*	It was pointed out that the subtraction order may be reversed
        for the Center Surround calculation of hue components.

        In order to be more faithful to treatise, it is necessary to
        obtain cspyrRG and cspyrBY by reversing the order in which
        only the hue components are subtracted.*/

    std::cout << "CenterSurrondPyramid: pyramidI" << std::endl;
    std::vector<Mat> cspyrI = buildCenterSurroundPyramid(pyramidI);
    /*for (int i = 0; i < cspyrI.size(); i++) {
        String name = "cspyr[" + to_string(i) + "]: ";
        imshow(name, cspyrI[i]);
    }
    waitKey();*/
    std::cout << "CenterSurrondPyramid: pyramidRG" << std::endl;
    std::vector<Mat> cspyrRG = buildCenterSurroundPyramid(pyramidRG);
    std::cout << "CenterSurrondPyramid: pyramidBY" << std::endl;
    std::vector<Mat> cspyrBY = buildCenterSurroundPyramid(pyramidBY);
    std::cout << "CenterSurrondPyramid: pyramid000" << std::endl;
    std::vector<Mat> cspyr000 = buildCenterSurroundPyramid(pyramid000);
    std::cout << "CenterSurrondPyramid: pyramid045" << std::endl;
    std::vector<Mat> cspyr045 = buildCenterSurroundPyramid(pyramid045);
    std::cout << "CenterSurrondPyramid: pyramid090" << std::endl;
    std::vector<Mat> cspyr090 = buildCenterSurroundPyramid(pyramid090);
    std::cout << "CenterSurrondPyramid: pyramid135" << std::endl;
    std::vector<Mat> cspyr135 = buildCenterSurroundPyramid(pyramid135);

    /*	Aggregate all the scales for charactersitc map

        Here, it was processed according to the original iamge size

        Apparenetly the correct answer is to unify the sie of sigma 4
        (original image 16x16pix -> corresponding to saliency map 1x1pix

        For enlargement when the sizes do not match, using nearest neighbor
        interpolation is a method faithtul*/

    Mat_<float> temp(image.size());
    Mat_<float> conspI(image.size(), 0.0f);
    Mat_<float> conspC(image.size(), 0.0f);
    Mat_<float> consp000(image.size(), 0.0f);
    Mat_<float> consp045(image.size(), 0.0f);
    Mat_<float> consp090(image.size(), 0.0f);
    Mat_<float> consp135(image.size(), 0.0f);

    for (int t = 0; t<int(cspyrI.size()); ++t) { //About each layer of S Pyramid
        //Addition to the brightness feature map
        trimPeaks(cspyrI[t], STEP);  //STEP is constant 8.
        resize(cspyrI[t], temp, image.size());
        conspI += temp;

        trimPeaks(cspyrRG[t], STEP);
        resize(cspyrRG[t], temp, image.size());
        conspC += temp;

        trimPeaks(cspyrBY[t], STEP);
        resize(cspyrBY[t], temp, image.size());
        conspC += temp;

        //Additional Directional Feature Map
        trimPeaks(cspyr000[t], STEP);
        resize(cspyr000[t], temp, image.size());
        consp000 += temp;

        trimPeaks(cspyr045[t], STEP);
        resize(cspyr045[t], temp, image.size());
        consp045 += temp;

        trimPeaks(cspyr090[t], STEP);
        resize(cspyr090[t], temp, image.size());
        consp090 += temp;

        trimPeaks(cspyr135[t], STEP);
        resize(cspyr135[t], temp, image.size());
        consp135 += temp;
    }

    trimPeaks(consp000, STEP);
    trimPeaks(consp045, STEP);
    trimPeaks(consp090, STEP);
    trimPeaks(consp135, STEP);
    Mat_<float> conspO = consp000 + consp045 + consp090 + consp135;

    //Get a saliency map by aggregating each characteristic map 
    trimPeaks(conspI, STEP);
    trimPeaks(conspC, STEP);
    trimPeaks(conspO, STEP);

    //imshow("Intensity", conspI);
    //imshow("Colour", conspC);
    //imshow("Orientation", conspO);

    cv::saliency::StaticSaliencyFineGrained fg;
    cv::saliency::StaticSaliencySpectralResidual sr;

    Mat fGrain;
    Mat sResidual;

    fg.computeSaliency(image0, fGrain);
    sr.computeSaliency(image0, sResidual);
    //imshow("fGrain", fGrain);
    //imshow("sResidual", sResidual);

    WEIGHT_C = 1.0f;
    WEIGHT_I = 1.0f;
    WEIGHT_O = 1.0f;
    float WEIGHT_FG = 0.00f;
    float WEIGHT_SR = 0.00f;
    Mat saliency = WEIGHT_I * conspI + WEIGHT_C * conspC + WEIGHT_O * conspO; // +WEIGHT_FG * fGrain + sResidual * WEIGHT_SR;
    normalizeRange(saliency);
    Mat comb = CombineSaliency(saliency, fGrain, sResidual);
    return saliency;
}

Mat CombineSaliency(Mat itti, Mat FG, Mat SR)
{
    Mat comb = Mat::zeros(itti.size(), CV_32F);
    comb = (1 * itti) + (1.5 * FG) + (0.5 * SR);
    //imshow("Saliency combined", comb);
    KMeans(comb, 3);//4,7 
    return comb;
}

//K Means
void KMeans(Mat src, int clusterCount) {
    
    const unsigned int singleLineSize = src.rows * src.cols;
    Mat data = src.reshape(1, singleLineSize);
    data.convertTo(data, CV_32F);
    std::vector<int> labels;
    Mat1f colors;
    int MAX_ITERATIONS = 5;
    kmeans(data, clusterCount, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.), MAX_ITERATIONS, KMEANS_PP_CENTERS, colors);
    for (unsigned int i = 0; i < singleLineSize; i++)
    {
        data.at<float>(i, 0) = colors(labels[i], 0);
        data.at<float>(i, 1) = colors(labels[i], 1);
        data.at<float>(i, 2) = colors(labels[i], 2);
    }
    Mat outputImage = data.reshape(1, src.rows);
    //imshow("KMeans", outputImage);
    Binary(outputImage);
}

//Kmeans -> binary
void Binary(Mat src)
{
    Mat binaryImage;
    std::vector<float> bin;
    src.convertTo(binaryImage, CV_32F);
    for (int i = 0; i < src.rows*src.cols; i++)
    {
        bin.push_back(src.at<float>(i));
    }
    sort(bin.begin(), bin.end());
    std::vector<float>::iterator it;
    it = unique(bin.begin(), bin.end());
    bin.resize(distance(bin.begin(), it));
    for (int i = 0; i < binaryImage.rows * binaryImage.cols; i++)
    {
        for (int j = 0; j < bin.size()-1; j++)
        {
            if (binaryImage.at<float>(i) == bin.at(j))
                binaryImage.at<float>(i) = 0;
        }
    }
    //imshow("Show Binary", binaryImage);
    findContours(binaryImage);
}

//Find contours
void findContours(Mat src)
{
    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    src.convertTo(src, CV_8UC3);
    findContours(src, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat imgWithContours = Mat::zeros(src.rows, src.cols, CV_8UC3);
    RNG rng(12345);
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(imgWithContours, contours, i, color, 1, 8, hierarchy, 0);
    }
    imshow("Contours", imgWithContours);
    roi(imgWithContours);
}

//ROI removal
void roi(Mat src)
{
    Mat reg(src, Rect(Point(0, 0), Point(1550, 320)));
    //imshow("test", reg);
    cvtColor(reg, reg, COLOR_BGR2GRAY);
    //std::cout << "src channel: " << src.channels();
    //std::cout << "grey channel: " << reg.channels();
    for (int i = 0; i < reg.rows; i++)
    {
        for (int j = 0; j < reg.cols; j++)
        {
            reg.at<uchar>(i, j) = 0;
        }
    }
    cvtColor(reg, reg, COLOR_GRAY2BGR);
    //std::cout << "bgr channel: " << reg.channels();
    //Mat smallImg = reg.clone();
    //smallImg.copyTo(src(reg));
    Mat insertImg(src, Rect(0, 0, 1550, 320));
    reg.copyTo(insertImg);
    imshow("test", src);

    //-----------------------------------------------------------
    /*Mat Grey;
    cvtColor(src, Grey, COLOR_BGR2GRAY);
    for (int i = 0; i <= 1000; i++)
    {
        for (int j = 0; j <=300; j++)
        {
            Grey.at<uchar>(i, j) = 0;
        }
    }
    cvtColor(Grey, src, COLOR_GRAY2BGR);
    imshow("test", src);*/
}

//RGB to Grey
Mat RGB2Grey(Mat RGB)
{
    Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);
    // conversion from RGb to Grey
    // Grey = (R+G+B)/3;
    for (int i = 0; i < RGB.rows; i++)
    {
        for (int j = 0; j < RGB.cols * 3; j = j + 3)
        {
            Grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
        }
    }
    return Grey;
}

//Grey to Binary
Mat Grey2Binary(Mat Grey)
{
    Mat Binary = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            if (Grey.at<uchar>(i, j) >= 128)
                Binary.at<uchar>(i, j) = 255;
        }
    }
    return Binary;
}

// Inversion 
Mat inversion(Mat Grey)
{
    Mat InvertImg = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++)
    {
        for (int j = 0; j < Grey.cols; j++)
        {
            InvertImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
        }
    }

    return InvertImg;

}

// Darken
Mat Darken(Mat Grey, int th)
{
    Mat Dark = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 0; i < Grey.rows; i++)
        for (int j = 0; j < Grey.cols; j++)
        {
            if (Grey.at<uchar>(i, j) <= th)
                Dark.at<uchar>(i, j) = Grey.at<uchar>(i, j);
            else
                Dark.at<uchar>(i, j) = th;
        }
    return Dark;
}

//Average
Mat Average(Mat Grey, int neigh)
{
    Mat avg = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = neigh; i < Grey.rows-neigh;i++)
        for (int j = neigh; j < Grey.cols - neigh; j++)
        {
            int sum = 0, c = 0;
            for(int ii=-neigh;ii<= neigh;ii++)
                for (int jj = -neigh; jj <= neigh; jj++)
                {
                    sum += Grey.at<uchar>(i + ii, j + jj);
                    c++;
                } 
            avg.at<uchar>(i, j) = sum / c;
        }
    return avg;
}

//Max
Mat Max(Mat Grey, int neigh)
{
    Mat max = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = neigh; i < Grey.rows - neigh; i++)
        for (int j = neigh; j < Grey.cols - neigh; j++)
        {
            int imax = 0;
            for (int ii = -neigh; ii <= neigh; ii++)
                for (int jj = -neigh; jj <= neigh; jj++)
                {
                    if (Grey.at<uchar>(i + ii, j + jj) > imax)
                        imax = Grey.at<uchar>(i + ii, j + jj);
                }
            max.at<uchar>(i, j) = imax;
        }
    return max;
}

//Min
Mat Min(Mat Grey, int neigh)
{
    Mat min = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = neigh; i < Grey.rows - neigh; i++)
        for (int j = neigh; j < Grey.cols - neigh; j++)
        {
            int imin = 256;
            for (int ii = -neigh; ii <= neigh; ii++)
                for (int jj = -neigh; jj <= neigh; jj++)
                {
                    if (Grey.at<uchar>(i + ii, j + jj) < imin)
                        imin = Grey.at<uchar>(i + ii, j + jj);
                }
            min.at<uchar>(i, j) = imin;
        }
    return min;
}

//Laplacian
Mat Laplacian(Mat Grey)
{
    Mat lap = Mat::zeros(Grey.size(), CV_8UC1);
    for (int i = 1; i < Grey.rows - 1; i++)
        for (int j = 1; j < Grey.cols - 1; j++)
        {
            int iValue = -4 *Grey.at<uchar>(i, j) + Grey.at<uchar>(i - 1, j) + Grey.at<uchar>(i + 1, j) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i, j + 1);
            if (iValue < 0)
                iValue = 0;
            if (iValue > 255)
                iValue = 255;
            lap.at<uchar>(i, j) = iValue;
        }
    return lap;
}

//Median
Mat Median(Mat Grey, int neigh)
{
    Mat medi = Mat::zeros(Grey.size(), CV_8UC1);
    int arr[100];
    for (int i = neigh; i < Grey.rows - neigh; i++)
        for (int j = neigh; j < Grey.cols - neigh; j++)
        {
            int c = 0;
            for (int ii = -neigh; ii <= neigh; ii++)
                for (int jj = -neigh; jj <= neigh; jj++)
                {
                    arr[c] = Grey.at<uchar>(i + ii, j + jj);
                    c++;
                }
            int k, l, temp;
            for (k = 0; k < c; k++)
                for (l = 0; l < c - k - 1; l++)
                {
                    if (arr[l] > arr[l + 1])
                    {
                        temp = arr[l];
                        arr[l] = arr[l + 1];
                        arr[l + 1] = temp;
                    }
                }
            medi.at<uchar>(i, j) = arr[(c / 2)-1];
        }
    return medi;
}

//Dynamic Convolution
Mat DynamicCon(Mat Grey,std::vector<int> mask,bool avg)
{
    Mat dynImg = Mat::zeros(Grey.size(), CV_8UC1);
    double sqMask = sqrt(mask.size());
    if (floor(sqMask) != sqMask || int(sqMask) % 2 != 1 || sqMask < 3)
    {
        std::cout << "Invalid vector";
        return dynImg;
    }
    int neigh = (sqMask - 1) / 2;
    std::vector<int> pix;
    int newPix;
    int totalMaskValue = 0;
    if (avg)
    {
        for (int i = 0; i < mask.size(); i++)
            totalMaskValue = totalMaskValue + mask[i];
        if (totalMaskValue == 0)
        {
            std::cout << "Divide by zero condition";
            return dynImg;
        }
    }
    for(int i=neigh;i<Grey.rows;i++)
        for (int j = neigh; j < Grey.cols; j++)
        {
            for (int ii = -neigh; ii <= neigh; ii++)
                for (int jj = -neigh; jj <= neigh; jj++)
                {
                    pix.push_back(Grey.at<uchar>(i + ii, j + jj));
                }
            newPix = 0;
            if (avg)
            {
                for (int k = 0; k < mask.size(); k++)
                    newPix = newPix + (mask[k] * pix[k]);
                double average = newPix / totalMaskValue;
                newPix = round(average);
            }
            else
            {
                for (int k = 0; k < mask.size(); k++)
                    newPix = newPix + (mask[k] * pix[k]);
            }
            if (newPix > 255)
                newPix = 255;
            if (newPix < 0)
                newPix = 0;
            dynImg.at<uchar>(i, j) = newPix;
            pix.clear();
        }
    return dynImg;
}

//Equalize Histogram
Mat EQHist(Mat Grey)
{
    Mat eq = Mat::zeros(Grey.size(), CV_8UC1);
    int count[256] = { 0 };
    int pixelNo;
    for(int i=0;i<Grey.rows;i++)
        for (int j = 0; j < Grey.cols; j++)
            count[Grey.at<uchar>(i, j)]++;
    double prob[256] = { 0.0 };
    for (int i = 0; i < 256; i++)
        prob[i] = (double)count[i] / (double)(Grey.rows * Grey.cols);
    double accprob[256] = { 0.0 };
    accprob[0] = prob[0];
    for (int i = 1; i < 256; i++)
        accprob[i] = prob[i] + accprob[i - 1];
    int newPixel[256] = { 0 };
    for (int i = 0; i < 256; i++)
        newPixel[i] = 255 * accprob[i];
    for(int i=0;i<Grey.rows;i++)
        for (int j = 0; j < Grey.cols; j++)
            eq.at<uchar>(i, j) = newPixel[Grey.at<uchar>(i, j)];
    return eq;
}

//Fourier Transform
Mat Fourier(Mat Grey)
{
    Mat padded;
    int m = getOptimalDFTSize(Grey.rows);
    int n = getOptimalDFTSize(Grey.cols);
    copyMakeBorder(Grey, padded, 0, m - Grey.rows, 0, n - Grey.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex, DFT_COMPLEX_OUTPUT);
    return complex;
}

void fftShift(Mat magI) 
{
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));   
    Mat q1(magI, Rect(cx, 0, cx, cy));  
    Mat q2(magI, Rect(0, cy, cx, cy));  
    Mat q3(magI, Rect(cx, cy, cx, cy)); 

    Mat tmp;                            
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

Mat lowPassFilter(Mat Grey, int maskSize);
Mat highPassFilter(Mat Grey, int maskSize);
void magnitudeSpectrum(Mat complex)
{
    Mat magI;
    Mat planes[] =
    {
        Mat::zeros(complex.size(), CV_32F),
        Mat::zeros(complex.size(), CV_32F)
    };
    split(complex, planes);
    magnitude(planes[0], planes[1], magI);
    magI += Scalar::all(1);
    log(magI, magI);
    fftShift(magI);
    normalize(magI, magI, 1, 0, NORM_INF);
    imshow("Fourier Spectrum", magI);
    Mat LPF = lowPassFilter(magI, 9);
    Mat HPF = highPassFilter(magI, 9);
    imshow("Fourier Spectrum LPF", LPF);
    imshow("Fourier Spectrum HPF", HPF);
}

void phaseAngle(Mat complex)
{
    Mat angle;
    Mat planes[] =
    {
        Mat::zeros(complex.size(), CV_32F),
        Mat::zeros(complex.size(), CV_32F)
    };
    split(complex, planes);
    phase(planes[0], planes[1], angle);
    angle += Scalar::all(1);
    log(angle, angle);
    fftShift(angle);
    normalize(angle, angle, 1, 0, NORM_INF);
    imshow("Phase Angle", angle);
    Mat LPF = lowPassFilter(angle, 9);
    Mat HPF = highPassFilter(angle, 9);
    imshow("Phase Angle LPF", LPF);
    imshow("Phase Angle HPF", HPF);
}

//Low Pass Filter
Mat lowPassFilter(Mat Grey, int maskSize)
{
    Scalar intensity1 = 0;
    Mat img;
    img = Grey.clone();
    for (int i = 0; i < Grey.rows - maskSize; i++)
    {
        for (int j = 0; j < Grey.cols - maskSize; j++)
        {
            Scalar intensity2;
            for (int k = 0; k < maskSize; k++)
            {
                for (int l = 0; l < maskSize; l++)
                {
                    intensity1 = Grey.at<uchar>(i + k, j + l);
                    intensity2.val[0] += intensity1.val[0];
                }
            }
            img.at<uchar>(i + (maskSize - 1) / 2, j + (maskSize - 1) / 2) = intensity2.val[0] / (maskSize * maskSize);
        }
    }
    return img;
}

//High Pass Filter
Mat highPassFilter(Mat Grey, int maskSize)
{
    Mat img;
    Scalar intensity1 = 0;
    img = Grey.clone();
    for (int i = 0; i < Grey.rows - maskSize; i++)
    {
        for (int j = 0; j < Grey.cols - maskSize; j++)
        {
            Scalar intensity2 = 0;
            for (int k = 0; k < maskSize; k++)
            {
                for (int l = 0; l < maskSize; l++)
                {
                    intensity1 = Grey.at<uchar>(i + k, j + l);
                    if ((k == (maskSize - 1) / 2) && (l == (maskSize - 1) / 2))
                        intensity2.val[0] += (maskSize * maskSize - 1) * intensity1.val[0];
                    else
                        intensity2.val[0] += (-1) * intensity1.val[0];
                }
            }
            img.at<uchar>(i + (maskSize - 1) / 2, j + (maskSize - 1) / 2) = intensity2.val[0] / (maskSize * maskSize);
        }
    }
    return img;
}




int main()
{

    //Mat img;
    //img = imread("D:\\uni\\05. 5th semester\\IPPR\\download.jpg");
    ////img = imread("D:\\uni\\05. 5th semester\\IPPR\\test2.png");
    ////img = imread("D:\\uni\\05. 5th semester\\IPPR\\test3.png");
    //imshow("RGB Image", img);

    // Grey converion
    //Mat GreyImg = RGB2Grey(img);
    //imshow("Grey Image", GreyImg);

    //// Binary conversion
    //Mat BinaryImg = Grey2Binary(GreyImg);
    //imshow("Binary Image", BinaryImg);

    //// Inversion
    //Mat InvertImg = inversion(GreyImg);
    //imshow("Inverted", InvertImg);

    ////Darken
    //Mat DarkImg = Darken(GreyImg,80);
    //imshow("Darken Img", DarkImg);

    ////Blurr
    //Mat AvgImg = Average(GreyImg,1);
    //imshow("Average Img", AvgImg);

    ////Max
    //Mat MaxImg = Max(GreyImg, 1);
    //imshow("Max Img", MaxImg);

    ////Min
    //Mat MinImg = Min(GreyImg, 1);
    //imshow("Min Img", MinImg);

    ////Laplacian
    //Mat LapImg = Laplacian(GreyImg);
    //imshow("Lap Img", LapImg);

    ////Median
    //Mat MediImg = Median(GreyImg, 1);
    //imshow("Median Img", MediImg);

    ////Dynamic Convolution
    //std::vector<int>mask;
    //bool avg = false;
    //mask = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
    //Mat DynConImg = DynamicCon(GreyImg, mask, avg);
    //imshow("Dynamic Convolution IMG", DynConImg);

    ////Equalize Histogram
    //Mat EQHistImg = EQHist(GreyImg);
    //imshow("Equalize Histogram Img", EQHistImg);

    ////Fourier Transform
    //Mat DFT = Fourier(GreyImg);
    //magnitudeSpectrum(DFT);
    //phaseAngle(DFT);

    //// Low Pass Filter
    //Mat LPF = lowPassFilter(GreyImg,9);
    //imshow("Low Pass Filter Img", LPF);

    ////High Pass Filter
    //Mat HPF = highPassFilter(GreyImg, 9);
    //imshow("High Pass Filter", HPF);

    
    
    ////K Means
    //Mat KM = KMeans(img, 3);
    //imshow("KMeans img", KM);

    Mat image0 = imread("D:\\uni\\05. 5th semester\\IPPR\\car 1.png");
    //std::cout << image0.channels();
    //imshow("Image", image0);
    Mat saliency = calcSaliencyMap(image0);
    //imshow("itti", saliency);

    waitKey();
}