#include <iostream>
#include <unistd.h>

#include "gaussianNoise.h"

using namespace cv;
using namespace std;

#define NOISE_ITER 15
#define PADDING 1

#include <Kokkos_Core.hpp>

// Jacobi iteration function for GPU
void jacobi_iteration(Kokkos::View<double ***> A, Kokkos::View<double ***> A_new, int iterations)
{
    int N = A.extent(0);
    int M = A.extent(1);
    for (int iter = 0; iter < iterations; ++iter)
    {
        Kokkos::parallel_for("Jacobi", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, 3}), KOKKOS_LAMBDA(int i, int j, int c) {
            if (i > 0 && i < N-1 && j > 0 && j < M-1) {
                A_new(i,j,c) = 0.2 * (A(i,j,c) + A(i+1,j,c) + A(i-1,j,c) + A(i,j+1,c) + A(i,j-1,c));
            } 
        });
        //Kokkos::deep_copy(A, A_new);
        Kokkos::fence();
        printf("Iteration %d\n", iter);
    }
}

void initialize(Kokkos::View<double ***> A, Kokkos::View<double ***> A_new, uint8_t *pixelPtr, int cn)
{
    int N = A.extent(0);
    int M = A.extent(1);
    Kokkos::parallel_for("Initialize", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, 3}), KOKKOS_LAMBDA(int i, int j, int c) {
        A(i, j, c) = pixelPtr[i * M * cn + j * cn + c];
        A_new(i, j, c) = 0;
    });
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv,
                             "{@input   |../../img/lena.jpg|input image}");
    parser.printMessage();

    String imageName = parser.get<String>("@input");
    string image_path = samples::findFile(imageName);
    Mat img = imread(image_path, IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    Mat mColorNoise(img.size(), img.type());

    for (int i = 0; i < NOISE_ITER; ++i)
    {
        AddGaussianNoise(img, mColorNoise, 0, 30.0);
        if (i < (NOISE_ITER - 1))
        {
            uint8_t *tmp = img.data;
            img.data = mColorNoise.data;
            mColorNoise.data = tmp;
        }
    }

    // AddGaussianNoise_Opencv(img,mColorNoise,10,30.0);//I recommend to use this way!

    uint8_t *pixelPtr = (uint8_t *)img.data;
    int cn = img.channels();

    ///////////////////////////////////////////////
    // Manage Kokkos allocations
    ///////////////////////////////////////////////

    Kokkos::initialize(argc, argv);
    {
        int iterations = 1;
        Kokkos::View<double ***> A("A", img.rows, img.cols, 3);
        Kokkos::View<double ***> A_new("A_new", img.rows, img.cols, 3);

        initialize(A, A_new, pixelPtr, cn);

        Kokkos::Timer timer;

        jacobi_iteration(A, A_new, iterations);

        double time = timer.seconds();

        // Calculate bandwidth.
        double Gbytes = 1.0e-9 * double(sizeof(uint8_t) * (img.rows * img.cols * cn * 4));

        // Print results (problem size, time and bandwidth in GB/s).
        printf("time( %g s ) bandwidth( %g GB/s )\n",
               time, Gbytes / time);

        int N = A.extent(0);
        int M = A.extent(1);
        Kokkos::parallel_for("output_image", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, 3}), KOKKOS_LAMBDA(int i, int j, int c) {
            pixelPtr[i * M * cn + j * cn + c] = A_new(i, j, c);
        });
    }
    Kokkos::finalize();

    fprintf(stdout, "Writting the output image of size %dx%d...\n", img.rows, img.cols);

    imwrite("../../res/test.jpg", img);
    imwrite("../../res/test_noise.jpg", mColorNoise);
    return 0;
}