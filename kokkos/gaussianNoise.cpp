#include "gaussianNoise.h"
inline uchar Clamp(int n)
{
    n = n>255 ? 255 : n;
    return n<0 ? 0 : n;
}

bool AddGaussianNoise(const Mat mSrc, Mat &mDst,double Mean, double StdDev)
{
    if(mSrc.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }

    Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);
    randn(mGaussian_noise,Scalar::all(Mean),Scalar::all(StdDev));

    for (int Rows = 0; Rows < mSrc.rows; Rows++)
    {
        for (int Cols = 0; Cols < mSrc.cols; Cols++)
        {
            Vec3b Source_Pixel= mSrc.at<Vec3b>(Rows,Cols);
            Vec3b &Des_Pixel= mDst.at<Vec3b>(Rows,Cols);
            Vec3s Noise_Pixel= mGaussian_noise.at<Vec3s>(Rows,Cols);

            for (int i = 0; i < 3; i++)
            {
                int Dest_Pixel= Source_Pixel.val[i] + Noise_Pixel.val[i];
                Des_Pixel.val[i]= Clamp(Dest_Pixel);
            }
        }
    }

    return true;
}

bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean, double StdDev)
{
    if(mSrc.empty())
    {
        cout<<"[Error]! Input Image Empty!";
        return 0;
    }
    Mat mSrc_16SC;
    Mat mGaussian_noise = Mat(mSrc.size(),CV_16SC3);

    randn(mGaussian_noise,Scalar::all(Mean), Scalar::all(StdDev));

    mSrc.convertTo(mSrc_16SC,CV_16SC3);
    addWeighted(mSrc_16SC, 1.0, mGaussian_noise, 1.0, 0.0, mSrc_16SC);
    mSrc_16SC.convertTo(mDst,mSrc.type());

    return true;
}
