#include "jacobi.h"
#include <iostream>
#include <Kokkos_Core.hpp>

using namespace std;

/**
 * Explications :
    1. Boucle d'itération : La boucle principale pour les itérations de Jacobi.
    2. Mise à jour des pixels : Chaque pixel est mis à jour en séquentiel.
       La nouvelle valeur est calculée comme la moyenne des pixels voisins.
    3. Copie des résultats : Après chaque itération, A_new est copié dans A.
 */
void jacobi_sequential(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Make a copy of the input image with a padding of 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width+2,A.size().height+2), A.type());
    // Make black the border
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;
    uint8_t* pixelPtr = (uint8_t*)A_copy.data;

    // Apply the Jacobi iteration
    for (int iter = 0; iter < iterations; ++iter) {
        // Iterate over the interior pixels
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                for (int c = 0; c < cn; ++c) {
                    // Current pixel0
                    uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                    // Top pixel
                    uint8_t p_top = pixelPtr[(i-1)*M*cn + j*cn + c];
                    // Bottom pixel
                    uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                    // Left pixel
                    uint8_t p_left = pixelPtr[i*M*cn + (j-1)*cn + c];
                    // Right pixel
                    uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                    // Compute the new pixel value
                    // We multiply by 0.2 to avoid division by 5 which is slower
                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);

                    new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
                }
            }
        }

        // Update the image for the next iteration
        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copy the middle of the image to the output image
    A_iter(cv::Rect(1, 1, A_iter.cols-2, A_iter.rows-2)).copyTo(A_new);
}

/**
 * Explications :
    1. Initialisation de Kokkos : Kokkos est initialisé pour l'exécution parallèle.
    2. Boucle d'itération : La boucle principale pour les itérations de Jacobi.
    3. Parallélisation sur CPU : Utilisation de Kokkos pour exécuter la mise à jour des pixels en parallèle sur CPU.
    4. Mise à jour des pixels : Pour chaque pixel, la nouvelle valeur est calculée 
       comme la moyenne des pixels voisins.
    5. Copie des résultats : Après chaque itération, A_new est copié dans A.
 */
void jacobi_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Make a copy of the input image with a padding of 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;
    uint8_t* pixelPtr = (uint8_t*)A_copy.data;

    // Apply the Jacobi iteration
    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("jacobi_parallel", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, M-1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int c = 0; c < cn; ++c) {
                uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                uint8_t p_top = pixelPtr[(i-1)*M*cn + j*cn + c];
                uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                uint8_t p_left = pixelPtr[i*M*cn + (j-1)*cn + c];
                uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
            }
        });

        // Update the image for the next iteration
        A_iter.copyTo(A_copy);
    }

    // Copy the middle of the image to the output image
    A_copy(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2)).copyTo(A_new);
}

/**
 * Explications :
    1. Initialisation de Kokkos : Kokkos est initialisé pour l'exécution parallèle.
    2. Allocation de la mémoire GPU : Utilisation de Kokkos::View pour allouer de la mémoire sur le GPU.
    3. Copie des données vers le GPU : Les données de la matrice A sont copiées vers le GPU.
    4. Boucle d'itération : La boucle principale pour les itérations de Jacobi.
    5. Parallélisation sur GPU : Utilisation de Kokkos pour exécuter la mise à jour des pixels en parallèle sur GPU.
    6. Mise à jour des pixels : Pour chaque pixel, la nouvelle valeur est calculée 
       comme la moyenne des pixels voisins.
    7. Copie des résultats : Après chaque itération, les données sont copiées de d_A_new vers d_A.
    8. Copie des résultats vers le CPU : À la fin des itérations, les résultats sont copiés vers A.
 */
void jacobi_parallel_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Make a copy of the input image with a padding of 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    Kokkos::View<uint8_t***> d_A("A", N, M, cn);
    Kokkos::View<uint8_t***> d_A_new("A_new", N, M, cn);

    // Copie des données vers le GPU
    Kokkos::parallel_for("copy_to_device", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, cn}),
    KOKKOS_LAMBDA(const int i, const int j, const int c) {
        d_A(i, j, c) = A_copy.data[i * M * cn + j * cn + c];
    });

    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("jacobi_gpu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, M-1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int c = 0; c < cn; ++c) {
                uint8_t p_current = d_A(i, j, c);
                uint8_t p_top = d_A(i-1, j, c);
                uint8_t p_bottom = d_A(i+1, j, c);
                uint8_t p_left = d_A(i, j-1, c);
                uint8_t p_right = d_A(i, j+1, c);
                uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                d_A_new(i, j, c) = new_pixel;
            }
        });
        Kokkos::deep_copy(d_A, d_A_new);
    }

    // Copie des données de Kokkos::View vers cv::Mat
    auto h_A = Kokkos::create_mirror_view(d_A);
    Kokkos::deep_copy(h_A, d_A);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int c = 0; c < cn; ++c) {
                A_copy.data[i * M * cn + j * cn + c] = h_A(i, j, c);
            }
        }
    }

    // Copy the middle of the image to the output image
    A_copy(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2)).copyTo(A_new);
}
