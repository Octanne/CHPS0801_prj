#include <opencv2/opencv.hpp>
#include <Kokkos_Core.hpp>
#include "gaussianNoise.h"
#include "jacobi.h"
#include "gauss_seidel.h"

using namespace cv;
using namespace std;

#define NOISE_ITER 15

int main(int argc, char** argv)
{
    const String keys =
    "{input |img/lena.jpg|input image}"
    "{filter |jacobi|filter to apply (jacobi - jacobi_cpu - jacobi_gpu - gauss_seidel - gauss_seidel_fronts_cpu - gauss_seidel_fronts_gpu - gauss_seidel_rb_cpu - gauss_seidel_rb_gpu)}"
    "{help h usage ? | | print this message}"
    "{iteration |1|number of iterations for the filter}"
    "{cpu |8|number of threads for the filter}"
    "{noise_iter |15|number of iterations of the noise maker}"
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
    int noise_iter = parser.get<int>("noise_iter");
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

    // Affichage des paramètres de l'exécution
    cout << "Filter to apply: " << filterName << endl;
    cout << "Number of iterations: " << iterations << endl;
    cout << "Number of threads: " << num_threads << endl;

    // Bruitage de l'image
    Mat mColorNoise(img.size(), img.type());
    for(int i = 0; i < noise_iter; ++i)
    {
        AddGaussianNoise(img, mColorNoise, 0, 30.0);
        if(i < (noise_iter - 1))
        {
            uint8_t* tmp = img.data;
            img.data = mColorNoise.data;
            mColorNoise.data = tmp;
        }
    }

    // On crée une matrice pour stocker le résultat
    Mat m_result_filter(img.size(), img.type());

    // On crée le nom du fichier de sortie
    string filter_file_name = "res/" + filterName + "_res.jpg";

    // On marque le temps de début
    double start = getTickCount();

    // Configuration des variables d'environnement pour OpenMP
    omp_set_num_threads(num_threads);
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);

    // Initialisation de Kokkos
    Kokkos::initialize();
    {
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
        else if (filterName == "jacobi_gpu")
        {
            jacobi_parallel_gpu(mColorNoise, m_result_filter, iterations);
        }
        // Gauss Seidel
        else if(filterName == "gauss_seidel")
        {
            gauss_seidel_sequential(mColorNoise, m_result_filter, iterations);
        }    
        else if(filterName == "gauss_seidel_fronts_cpu")
        {
            gauss_seidel_parallel_fronts_cpu(mColorNoise, m_result_filter, iterations);
        }
        else if(filterName == "gauss_seidel_fronts_gpu")
        {
            gauss_seidel_parallel_fronts_gpu(mColorNoise, m_result_filter, iterations);
        }
        else if(filterName == "gauss_seidel_rb_cpu")
        {
            gauss_seidel_rb_parallel_cpu(mColorNoise, m_result_filter, iterations);
        }
        else if(filterName == "gauss_seidel_rb_gpu")
        {
            gauss_seidel_rb_parallel_gpu(mColorNoise, m_result_filter, iterations);
        }
        // Non trouvé
        else
        {
            cout << "Filter not found" << endl;
            Kokkos::finalize();
            return 1;
        }
    }
    Kokkos::finalize();

    // On marque le temps de fin
    double end = getTickCount();
    double time = (end - start) / getTickFrequency();
    cout << "Execution time: " << time << "s" << endl;

    fprintf(stdout, "Writting the output image of size %dx%d...\n", img.rows, img.cols);

    imwrite("res/noised_res.jpg", mColorNoise);
    imwrite(filter_file_name, m_result_filter);

    return 0;
}
