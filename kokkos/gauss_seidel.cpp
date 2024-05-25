#include "gauss_seidel.h"
#include <Kokkos_Core.hpp>

using namespace std;

/**
 * Explications :
    1. Parcours séquentiel : Chaque pixel est mis à jour en séquentiel.
    2. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.
    3. Mise à jour des pixels : Pour chaque pixel, la nouvelle valeur est calculée 
       comme la moyenne des pixels voisins.
    4. Copie des résultats : Après chaque itération, A_new est copié dans A.
 */
void gauss_seidel_sequential(cv::Mat& A, cv::Mat& A_new, int iterations) {
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

    // Apply the Gauss-Seidel iteration
    for (int iter = 0; iter < iterations; ++iter) {
        // Iterate over the interior pixels
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                for (int c = 0; c < cn; ++c) {
                    // Current pixel0
                    uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                    // Top pixel
                    uint8_t p_top = new_pixelPtr[(i-1)*M*cn + j*cn + c];
                    // Bottom pixel
                    uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                    // Left pixel
                    uint8_t p_left = new_pixelPtr[i*M*cn + (j-1)*cn + c];
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
    2. Allocation de la mémoire pour les copies de matrices.
    3. Utilisation de Kokkos pour la mise à jour parallèle des pixels.
    4. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.
    5. Copie des résultats après chaque itération pour l'utilisation dans la suivante.
 */
void gauss_seidel_parallel_fronts_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Make a copy of the input image with a padding of 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;
    uint8_t* pixelPtr = (uint8_t*)A_copy.data;

    // Apply the Gauss-Seidel iteration
    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("gauss_seidel_parallel", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, M-1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int c = 0; c < cn; ++c) {
                uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                uint8_t p_top = new_pixelPtr[(i-1)*M*cn + j*cn + c];
                uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                uint8_t p_left = new_pixelPtr[i*M*cn + (j-1)*cn + c];
                uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
            }
        });

        // Update the image for the next iteration
        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copy the middle of the image to the output image
    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);
}

/**
 * Explications :
    1. Initialisation de Kokkos pour l'exécution GPU.
    2. Allocation de la mémoire sur le GPU.
    3. Utilisation de Kokkos pour la mise à jour parallèle des pixels sur le GPU.
    4. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.
    5. Copie des résultats après chaque itération pour l'utilisation dans la suivante.
 */
void gauss_seidel_parallel_fronts_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    Kokkos::View<uint8_t***> d_A("A", N, M, cn);
    Kokkos::View<uint8_t***> d_A_new("A_new", N, M, cn);

    Kokkos::parallel_for("copy_to_device", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, cn}),
    KOKKOS_LAMBDA(const int i, const int j, const int c) {
        d_A(i, j, c) = A_copy.data[i * M * cn + j * cn + c];
    });

    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("gauss_seidel_gpu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N - 1, M - 1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            for (int c = 0; c < cn; ++c) {
                uint8_t p_current = d_A(i, j, c);
                uint8_t p_top = d_A_new(i - 1, j, c);
                uint8_t p_bottom = d_A(i + 1, j, c);
                uint8_t p_left = d_A_new(i, j - 1, c);
                uint8_t p_right = d_A(i, j + 1, c);
                uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                d_A_new(i, j, c) = new_pixel;
            }
        });
        Kokkos::deep_copy(d_A, d_A_new);
    }

    Kokkos::parallel_for("copy_to_host", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, cn}),
    KOKKOS_LAMBDA(const int i, const int j, const int c) {
        A_copy.data[i * M * cn + j * cn + c] = d_A(i, j, c);
    });

    A_copy(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2)).copyTo(A_new);
}

/**
 * Explications :
    1. Initialisation de Kokkos : Kokkos est initialisé pour l'exécution parallèle.
    2. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.
    3. Mise à jour des points rouges et noirs : Les points rouges sont mis à jour en premier, suivis des points noirs.
    4. Synchronisation : Utilisation de Kokkos::fence pour synchroniser les threads.
    5. Copie des résultats après chaque itération pour l'utilisation dans la suivante.
 */
void gauss_seidel_rb_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;
    uint8_t* pixelPtr = (uint8_t*)A_copy.data;

    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("gauss_seidel_rb_cpu_red", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, M-1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if ((i + j) % 2 == 0) {
                for (int c = 0; c < cn; ++c) {
                    uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                    uint8_t p_top = new_pixelPtr[(i-1)*M*cn + j*cn + c];
                    uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                    uint8_t p_left = new_pixelPtr[i*M*cn + (j-1)*cn + c];
                    uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                    new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
                }
            }
        });
        Kokkos::fence();

        Kokkos::parallel_for("gauss_seidel_rb_cpu_black", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N-1, M-1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if ((i + j) % 2 != 0) {
                for (int c = 0; c < cn; ++c) {
                    uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                    uint8_t p_top = new_pixelPtr[(i-1)*M*cn + j*cn + c];
                    uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                    uint8_t p_left = new_pixelPtr[i*M*cn + (j-1)*cn + c];
                    uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                    new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
                }
            }
        });
        Kokkos::fence();

        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);
}

/**
 * Explications :
    1. Initialisation de Kokkos pour l'exécution GPU.
    2. Allocation de la mémoire sur le GPU.
    3. Utilisation de Kokkos pour la mise à jour parallèle des points rouges et noirs sur le GPU.
    4. Synchronisation : Utilisation de Kokkos::fence pour synchroniser les threads.
    5. Copie des résultats après chaque itération pour l'utilisation dans la suivante.
 */
void gauss_seidel_rb_parallel_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    Kokkos::View<uint8_t***> d_A("A", N, M, cn);
    Kokkos::View<uint8_t***> d_A_new("A_new", N, M, cn);

    Kokkos::parallel_for("copy_to_device", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, cn}),
    KOKKOS_LAMBDA(const int i, const int j, const int c) {
        d_A(i, j, c) = A_copy.data[i * M * cn + j * cn + c];
    });

    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("gauss_seidel_rb_gpu_red", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N - 1, M - 1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if ((i + j) % 2 == 0) {
                for (int c = 0; c < cn; ++c) {
                    uint8_t p_current = d_A(i, j, c);
                    uint8_t p_top = d_A_new(i - 1, j, c);
                    uint8_t p_bottom = d_A(i + 1, j, c);
                    uint8_t p_left = d_A_new(i, j - 1, c);
                    uint8_t p_right = d_A(i, j + 1, c);
                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                    d_A_new(i, j, c) = new_pixel;
                }
            }
        });
        Kokkos::fence();

        Kokkos::parallel_for("gauss_seidel_rb_gpu_black", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {N - 1, M - 1}),
        KOKKOS_LAMBDA(const int i, const int j) {
            if ((i + j) % 2 != 0) {
                for (int c = 0; c < cn; ++c) {
                    uint8_t p_current = d_A(i, j, c);
                    uint8_t p_top = d_A_new(i - 1, j, c);
                    uint8_t p_bottom = d_A(i + 1, j, c);
                    uint8_t p_left = d_A_new(i, j - 1, c);
                    uint8_t p_right = d_A(i, j + 1, c);
                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                    d_A_new(i, j, c) = new_pixel;
                }
            }
        });
        Kokkos::fence();

        Kokkos::deep_copy(d_A, d_A_new);
    }

    Kokkos::parallel_for("copy_to_host", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, M, cn}),
    KOKKOS_LAMBDA(const int i, const int j, const int c) {
        A_copy.data[i * M * cn + j * cn + c] = d_A(i, j, c);
    });

    A_copy(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2)).copyTo(A_new);
}
