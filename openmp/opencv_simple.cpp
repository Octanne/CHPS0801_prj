#include <iostream>

#include "gaussianNoise.h"
#include "jacobi.h"
#include "gauss_seidel.h"

#include <omp.h>

using namespace cv;
using namespace std;

#define NOISE_ITER 15

int main(int argc, char** argv)
{
    const String keys =
    "{input |img/lena.jpg|input image}"
    "{filter |jacobi|filter to apply}"
    "{help h usage ? | | print this message}"
    "{iteration |1|number of iterations for the filter}"
    "{cpu |8|number of threads for the filter}"
    ;
    CommandLineParser parser(argc, argv, keys);

    // We print the help message if the user asks for it
    if(parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String imageName = parser.get<String>("input");
    String filterName = parser.get<String>("filter");
    int iterations = parser.get<int>("iteration");
    int num_threads = parser.get<int>("cpu");
    string image_path = samples::findFile(imageName, false, true);
    // Vérification de l'existence de l'image
    if(image_path.empty())
    {
        std::cout << "Could not read the image: " << imageName << std::endl;
        return 1;
    }
    // Lecture de l'image
    Mat img = imread(image_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    // On set le nombre de threads
    omp_set_num_threads(num_threads);

    // Affichage des parametres de l'execution
    cout << "Filter to apply: " << filterName << endl;
    cout << "Number of iterations: " << iterations << endl;

    // Bruitage de l'image
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

    // On creer une matrice pour stocker le resultat
    Mat m_result_filter(img.size(),img.type());

    // On creer le nom du fichier de sortie
    string filter_file_name = "res/" + filterName + "_res.jpg";

    // On marque le temps de debut
    double start = getTickCount();

    // On affiche le nombre de threads si le filtre contient le mot CPU
    if(filterName.find("cpu") != string::npos)
    {
        cout << "Number of threads: " << omp_get_max_threads() << "/" << omp_get_num_procs() << endl;
    }

    // On applique le filtre suivant la valeur de filterName   
    // Jacobi
    if(filterName == "jacobi")
    {
        jacobi_sequential(mColorNoise, m_result_filter, iterations);
    }    
    else if(filterName == "jacobi_cpu")
    {
        jacobi_parallel_cpu(mColorNoise, m_result_filter, iterations);
    }    
    else if(filterName == "jacobi_gpu")
    {
        jacobi_parallel_gpu(mColorNoise, m_result_filter, iterations);
    }
    // Gauss Seidel
    else if(filterName == "gauss_seidel")
    {
        gauss_seidel_sequential(mColorNoise, m_result_filter, iterations);
    }    
    else if(filterName == "gauss_seidel_front_cpu")
    {
        gauss_seidel_parallel_fronts_cpu(mColorNoise, m_result_filter, iterations);
    }
    else if(filterName == "gauss_seidel_rb_cpu")
    {
        gauss_seidel_parallel_fronts_cpu(mColorNoise, m_result_filter, iterations);
    }    
    else if(filterName == "gauss_seidel_front_gpu")
    {
        gauss_seidel_parallel_fronts_gpu(mColorNoise, m_result_filter, iterations);
    }
    else if(filterName == "gauss_seidel_rb_gpu")
    {
        gauss_seidel_parallel_fronts_gpu(mColorNoise, m_result_filter, iterations);
    }
    // Non trouvé
    else
    {
        cout << "Filter not found" << endl;
        return 1;
    }

    // On marque le temps de fin
    double end = getTickCount();
    double time = (end - start) / getTickFrequency();
    cout << "Execution time: " << time << "s" << endl;

    fprintf(stdout, "Writting the output image of size %dx%d...\n", img.rows, img.cols);

    imwrite("res/noised_res.jpg", mColorNoise);
    imwrite(filter_file_name, m_result_filter);
    
    return 0;
}

/**
 * Old code
*/
void old() {
    /*
    //AddGaussianNoise_Opencv(img,mColorNoise,10,30.0);//I recommend to use this way!

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

    //imwrite("res/grey_res.jpg", img);
}
