#include <iostream>

#include "gaussianNoise.h"
#include "jacobi.cpp"

using namespace cv;
using namespace std;

#define NOISE_ITER 15

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
                                "{@input   |img/lena.jpg|input image}");
    parser.printMessage();

    String imageName = parser.get<String>("@input");
    string image_path = samples::findFile(imageName);
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    Mat mColorNoise(img.size(),img.type());

    for(int i = 0; i < NOISE_ITER; ++i)
    {
        AddGaussianNoise(img,mColorNoise,0,30.0);
        if(i < (NOISE_ITER -1))
        {
            uint8_t* tmp = img.data;
            img.data = mColorNoise.data;
            mColorNoise.data = tmp;
        }
    }

    //AddGaussianNoise_Opencv(img,mColorNoise,10,30.0);//I recommend to use this way!

    Mat mJacobi_seq(img.size(),img.type());

    // We will use the jacobi_cpu_openmp function
    jacobi_sequential(mColorNoise, mJacobi_seq, 1);

    /*
    uint8_t* pixelPtr = (uint8_t*)img.data;
    int cn = img.channels();

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            // bgrPixel.val[0] = 255; //B
            uint8_t b = pixelPtr[i*img.cols*cn + j*cn + 0]; // B
            uint8_t g = pixelPtr[i*img.cols*cn + j*cn + 1]; // G
            uint8_t r = pixelPtr[i*img.cols*cn + j*cn + 2]; // R
            uint8_t grey = r * 0.299 + g * 0.587 + b * 0.114;

            pixelPtr[i*img.cols*cn + j*cn + 0] = grey; // B
            pixelPtr[i*img.cols*cn + j*cn + 1] = grey; // G
            pixelPtr[i*img.cols*cn + j*cn + 2] = grey; // R
        }
    }*/

    fprintf(stdout, "Writting the output image of size %dx%d...\n", img.rows, img.cols);

    //imwrite("res/grey_res.jpg", img);
    imwrite("res/noised_res.jpg", mColorNoise);
    imwrite("res/jacobi_seq.jpg", mJacobi_seq);
    
    return 0;
}
