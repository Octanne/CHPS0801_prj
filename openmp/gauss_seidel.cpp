#include "gauss_seidel.h"

#include <omp.h>

using namespace std;

void gauss_seidel_sequential(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Make a copy of the input image with a padding of 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width+2,A.size().height+2), A.type());
    // Make black the border
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = (uint8_t*)A_copy.data;
    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;

    // Apply the Jacobi iteration
    for (int iter = 0; iter < iterations; ++iter) {
        // Iterate over the interior pixels
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                for (int c = 0; c < cn; ++c) {
                    // Gauss-Seidel plus compliqué a paralléliser
                    // Car on a besoin des valeurs des pixels voisins déjà calculés
                    // PixelPTR is the instant k and new_pixelPTR is the instant k+1
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
    1. Ajout de bordures : Nous ajoutons une bordure de 1 pixel tout autour de l'image d'entrée pour faciliter le traitement des pixels aux bords de l'image.

    2. Initialisation des matrices : A_copy contient l'image d'entrée avec bordures, A_iter est utilisée pour stocker les nouvelles valeurs de pixels à chaque itération.

    3. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.

    4. Traitement par diagonales :
        Nous traitons les pixels en bandes diagonales.
        Pour chaque diagonale (d), nous itérons sur les pixels (i, j) de cette diagonale.
        Chaque tâche dépend des valeurs des pixels adjacents (i-1, j), (i+1, j), (i, j-1), et (i, j+1).

    5. Dépendances des tâches : Les directives depend(in: ...) et depend(out: ...) d'OpenMP garantissent que les tâches sont exécutées dans le bon ordre en respectant les dépendances des pixels.

    6. Copie des résultats : Après chaque itération, nous copions A_iter dans A_copy pour la prochaine itération.

    7. Sortie des résultats : À la fin des itérations, nous copions le résultat final dans A_new sans les bordures ajoutées.

   Cette méthode garantit que les dépendances entre les pixels sont respectées,
   permettant ainsi une parallélisation efficace tout en maintenant l'exactitude 
   de l'algorithme de Gauss-Seidel.
 * 
*/
void gauss_seidel_parallel_fronts_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'une bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = A_copy.ptr<uint8_t>();
    uint8_t* new_pixelPtr = A_iter.ptr<uint8_t>();

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                // Traitement par diagonales
                for (int d = 1; d < N + M - 3; ++d) {
                    for (int i = std::max(1, d - (M - 2)); i <= std::min(d, N - 2); ++i) {
                        int j = d - i;
                        if (j >= 1 && j < M - 1) {
                            #pragma omp task depend(in: pixelPtr[(i-1)*M*cn + j*cn], pixelPtr[(i+1)*M*cn + j*cn], pixelPtr[i*M*cn + (j-1)*cn], pixelPtr[i*M*cn + (j+1)*cn]) depend(out: new_pixelPtr[i*M*cn + j*cn])
                            {
                                for (int c = 0; c < cn; ++c) {
                                    uint8_t p_current = pixelPtr[i * M * cn + j * cn + c];
                                    uint8_t p_top = new_pixelPtr[(i - 1) * M * cn + j * cn + c];
                                    uint8_t p_bottom = pixelPtr[(i + 1) * M * cn + j * cn + c];
                                    uint8_t p_left = new_pixelPtr[i * M * cn + (j - 1) * cn + c];
                                    uint8_t p_right = pixelPtr[i * M * cn + (j + 1) * cn + c];
                                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                                    new_pixelPtr[i * M * cn + j * cn + c] = new_pixel;
                                }
                            }
                        }
                    }
                    #pragma omp taskwait
                }
            }
        }
        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);
}

/**
 *Explications :

    1. Bordure : On ajoute une bordure d'un pixel autour de l'image d'entrée pour éviter les problèmes de débordement de mémoire lorsque l'on accède aux voisins des pixels aux bords de l'image.

    2. Vagues Diagonales : On traite les éléments de la matrice par vagues diagonales, où d représente la distance de la diagonale par rapport à l'origine (1, 1). Les indices i et j sont calculés en fonction de d pour suivre cette vague.

    3. Parallélisation avec OpenMP : On utilise OpenMP pour paralléliser le traitement des vagues diagonales. Le schedule(dynamic) permet une distribution dynamique des itérations pour un meilleur équilibrage de la charge entre les threads.

    4. Mise à jour des Pixels : Pour chaque pixel (i, j), on calcule la nouvelle valeur en utilisant les voisins directs (haut, bas, gauche, droite) et on met à jour le pixel dans l'image de sortie A_new.

    5. Copie de l'Image : Après chaque itération, on copie l'image mise à jour A_new dans A_copy pour la prochaine itération, en prenant soin de ne copier que la région sans bordure.

  Cette méthode assure que les dépendances sont correctement gérées et 
  que chaque vague est calculée indépendamment, ce qui devrait éviter
  les artefacts tels que les traits horizontaux. 
*/
void gauss_seidel_parallel_fronts_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'une bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = A_copy.ptr<uint8_t>();
    uint8_t* new_pixelPtr = A_iter.ptr<uint8_t>();

    // Allouer de la mémoire sur le GPU
    #pragma omp target enter data map(to: pixelPtr[0:N*M*cn], new_pixelPtr[0:N*M*cn])

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                // Traitement par diagonales
                for (int d = 1; d < N + M - 3; ++d) {
                    for (int i = std::max(1, d - (M - 2)); i <= std::min(d, N - 2); ++i) {
                        int j = d - i;
                        if (j >= 1 && j < M - 1) {
                            #pragma omp task depend(in: pixelPtr[(i-1)*M*cn + j*cn], pixelPtr[(i+1)*M*cn + j*cn], pixelPtr[i*M*cn + (j-1)*cn], pixelPtr[i*M*cn + (j+1)*cn]) depend(out: new_pixelPtr[i*M*cn + j*cn])
                            {
                                #pragma omp target map(to: pixelPtr[0:N*M*cn]) map(from: new_pixelPtr[0:N*M*cn])
                                {
                                    for (int c = 0; c < cn; ++c) {
                                        uint8_t p_current = pixelPtr[i * M * cn + j * cn + c];
                                        uint8_t p_top = new_pixelPtr[(i - 1) * M * cn + j * cn + c];
                                        uint8_t p_bottom = pixelPtr[(i + 1) * M * cn + j * cn + c];
                                        uint8_t p_left = new_pixelPtr[i * M * cn + (j - 1) * cn + c];
                                        uint8_t p_right = pixelPtr[i * M * cn + (j + 1) * cn + c];
                                        uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                                        new_pixelPtr[i * M * cn + j * cn + c] = new_pixel;
                                    }
                                }
                            }
                        }
                    }
                    #pragma omp taskwait
                }
            }
        }
        #pragma omp target update from(new_pixelPtr[0:N*M*cn])
        std::memcpy(pixelPtr, new_pixelPtr, N * M * cn * sizeof(uint8_t));
        #pragma omp target update to(pixelPtr[0:N*M*cn])
        
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copier les résultats du GPU vers l'hôte
    #pragma omp target exit data map(from: new_pixelPtr[0:N*M*cn])
    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);

    // Libération de la mémoire GPU
    #pragma omp target exit data map(delete: pixelPtr[0:N*M*cn], new_pixelPtr[0:N*M*cn])
}

/**
 * Explications :

    1. Bordure : Une bordure d'un pixel est ajoutée autour de l'image d'entrée pour éviter les problèmes de débordement de mémoire.

    2. "Cases rouges" et "cases noires" : Les indices (i, j) sont classifiés comme "rouges" ou "noirs" en utilisant la condition (i + j) % 2. Les pixels rouges sont mis à jour dans la première boucle, et les pixels noirs dans la seconde boucle.

    3. Parallélisation avec OpenMP : Les deux boucles for sont parallélisées indépendamment en utilisant #pragma omp parallel for collapse(2) pour maximiser l'efficacité de la parallélisation.
    
    4. Barrières entre les phases : Une barrière est utilisée pour synchroniser les tâches après la phase rouge afin de garantir que tous les pixels rouges sont mis à jour avant de commencer la phase noire.

    5. Mise à jour des Pixels : Pour chaque pixel, on calcule la nouvelle valeur en utilisant les voisins directs (haut, bas, gauche, droite) et on met à jour le pixel dans l'image de sortie A_new.

    6. Copie de l'Image : Après chaque itération, l'image mise à jour A_new est copiée dans A_copy pour la prochaine itération, en prenant soin de ne copier que la région sans bordure.

  Cette approche maximise la parallélisation tout en gérant 
  correctement les dépendances entre les pixels pour garantir
  une convergence correcte de l'algorithme Gauss-Seidel.
*/
void gauss_seidel_rb_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'une bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    for (int iter = 0; iter < iterations; ++iter) {
        // Mise à jour des "cases rouges"
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < M - 1; ++j) {
                if ((i + j) % 2 == 0) { // "cases rouges"
                    for (int c = 0; c < cn; ++c) {
                        uint8_t p_current = A_copy.at<cv::Vec3b>(i, j)[c];
                        uint8_t p_top = A_copy.at<cv::Vec3b>(i - 1, j)[c];
                        uint8_t p_bottom = A_copy.at<cv::Vec3b>(i + 1, j)[c];
                        uint8_t p_left = A_copy.at<cv::Vec3b>(i, j - 1)[c];
                        uint8_t p_right = A_copy.at<cv::Vec3b>(i, j + 1)[c];
                        uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                        A_new.at<cv::Vec3b>(i - 1, j - 1)[c] = new_pixel;
                    }
                }
            }
        }

        // Synchronize to ensure red phase is done
        #pragma omp barrier

        // Mise à jour des "cases noires"
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < M - 1; ++j) {
                if ((i + j) % 2 != 0) { // "cases noires"
                    for (int c = 0; c < cn; ++c) {
                        uint8_t p_current = A_copy.at<cv::Vec3b>(i, j)[c];
                        uint8_t p_top = A_copy.at<cv::Vec3b>(i - 1, j)[c];
                        uint8_t p_bottom = A_copy.at<cv::Vec3b>(i + 1, j)[c];
                        uint8_t p_left = A_copy.at<cv::Vec3b>(i, j - 1)[c];
                        uint8_t p_right = A_copy.at<cv::Vec3b>(i, j + 1)[c];
                        uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                        A_new.at<cv::Vec3b>(i - 1, j - 1)[c] = new_pixel;
                    }
                }
            }
        }

        A_new.copyTo(A_copy(cv::Rect(1, 1, A_new.cols, A_new.rows)));
        std::cout << "Iteration " << iter << " done\r";
    }

    A_copy(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2)).copyTo(A_new);
}

/**
 * Explications :

    1. Directives OpenMP pour le GPU :
        #pragma omp target est utilisé pour offload les données et les calculs sur le GPU.
        omp_target_alloc alloue de la mémoire sur le GPU.
        omp_target_free libère la mémoire GPU.

    2. Copie de données entre l'hôte et le GPU :
        #pragma omp target enter data map(to: A_copy_data[:N*M*cn], A_new_data[:(N-2)*(M-2)*cn]) copie les données de l'hôte vers le GPU avant le début des itérations.
        #pragma omp target update from et #pragma omp target update to sont utilisés pour synchroniser les données entre l'hôte et le GPU après chaque itération.

    3. Exécution des kernels :
        #pragma omp target teams distribute parallel for collapse(2) est utilisé pour paralléliser les boucles sur le GPU.
*/
void gauss_seidel_rb_parallel_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'une bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    // Allocation de mémoire GPU
    uchar* d_A_copy;
    uchar* d_A_new;
    size_t size = N * M * cn * sizeof(uchar);
    d_A_copy = (uchar*)omp_target_alloc(size, omp_get_default_device());
    d_A_new = (uchar*)omp_target_alloc(size, omp_get_default_device());

    // Copier les données de l'hôte vers le GPU
    uchar* A_copy_data = A_copy.ptr();
    uchar* A_new_data = A_new.ptr();
    #pragma omp target enter data map(to: A_copy_data[:N*M*cn], A_new_data[:(N-2)*(M-2)*cn])
    #pragma omp target update to(A_copy_data[:N*M*cn])

    for (int iter = 0; iter < iterations; ++iter) {
        // Mise à jour des "cases rouges"
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < M - 1; ++j) {
                if ((i + j) % 2 == 0) { // "cases rouges"
                    for (int c = 0; c < cn; ++c) {
                        int index = (i * M + j) * cn + c;
                        int top = ((i - 1) * M + j) * cn + c;
                        int bottom = ((i + 1) * M + j) * cn + c;
                        int left = (i * M + (j - 1)) * cn + c;
                        int right = (i * M + (j + 1)) * cn + c;

                        uchar p_current = A_copy_data[index];
                        uchar p_top = A_copy_data[top];
                        uchar p_bottom = A_copy_data[bottom];
                        uchar p_left = A_copy_data[left];
                        uchar p_right = A_copy_data[right];
                        uchar new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                        A_new_data[((i - 1) * (M - 2) + (j - 1)) * cn + c] = new_pixel;
                    }
                }
            }
        }

        // Synchronize to ensure red phase is done
        #pragma omp barrier

        // Mise à jour des "cases noires"
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < M - 1; ++j) {
                if ((i + j) % 2 != 0) { // "cases noires"
                    for (int c = 0; c < cn; ++c) {
                        int index = (i * M + j) * cn + c;
                        int top = ((i - 1) * M + j) * cn + c;
                        int bottom = ((i + 1) * M + j) * cn + c;
                        int left = (i * M + (j - 1)) * cn + c;
                        int right = (i * M + (j + 1)) * cn + c;

                        uchar p_current = A_copy_data[index];
                        uchar p_top = A_copy_data[top];
                        uchar p_bottom = A_copy_data[bottom];
                        uchar p_left = A_copy_data[left];
                        uchar p_right = A_copy_data[right];
                        uchar new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);
                        A_new_data[((i - 1) * (M - 2) + (j - 1)) * cn + c] = new_pixel;
                    }
                }
            }
        }

        #pragma omp target update from(A_new_data[:(N-2)*(M-2)*cn])
        std::memcpy(A_copy_data + (M + 1) * cn, A_new_data, (N - 2) * (M - 2) * cn * sizeof(uchar));
        #pragma omp target update to(A_copy_data[:N*M*cn])
        
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copier les résultats du GPU vers l'hôte
    #pragma omp target update from(A_new_data[:(N-2)*(M-2)*cn])
    std::memcpy(A_new.ptr(), A_new_data, (N - 2) * (M - 2) * cn * sizeof(uchar));

    // Libération de la mémoire GPU
    omp_target_free(d_A_copy, omp_get_default_device());
    omp_target_free(d_A_new, omp_get_default_device());
}

// Garder car resultat styley
void gauss_seidel_rb_parallel_mais_nan_enfaite_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'une bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = A_copy.data;
    uint8_t* new_pixelPtr = A_iter.data;

    for (int iter = 0; iter < iterations; ++iter) {
        // Mise à jour des "cases rouges"
        #pragma omp target data map(to: pixelPtr[0:N*M*cn]) map(from: new_pixelPtr[0:N*M*cn])
        {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < M - 1; ++j) {
                    if ((i + j) % 2 == 0) { // "cases rouges"
                        for (int c = 0; c < cn; ++c) {
                            uint8_t p_current = pixelPtr[(i * M + j) * cn + c];
                            uint8_t p_top = pixelPtr[((i - 1) * M + j) * cn + c];
                            uint8_t p_bottom = pixelPtr[((i + 1) * M + j) * cn + c];
                            uint8_t p_left = pixelPtr[(i * M + (j - 1)) * cn + c];
                            uint8_t p_right = pixelPtr[(i * M + (j + 1)) * cn + c];
                            uint8_t new_pixel = static_cast<uint8_t>(0.2f * (p_current + p_top + p_bottom + p_left + p_right));
                            new_pixelPtr[(i * M + j) * cn + c] = new_pixel;
                        }
                    }
                }
            }

            // Mise à jour des "cases noires"
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < M - 1; ++j) {
                    if ((i + j) % 2 != 0) { // "cases noires"
                        for (int c = 0; c < cn; ++c) {
                            uint8_t p_current = pixelPtr[(i * M + j) * cn + c];
                            uint8_t p_top = pixelPtr[((i - 1) * M + j) * cn + c];
                            uint8_t p_bottom = pixelPtr[((i + 1) * M + j) * cn + c];
                            uint8_t p_left = pixelPtr[(i * M + (j - 1)) * cn + c];
                            uint8_t p_right = pixelPtr[(i * M + (j + 1)) * cn + c];
                            uint8_t new_pixel = static_cast<uint8_t>(0.2f * (p_current + p_top + p_bottom + p_left + p_right));
                            new_pixelPtr[(i * M + j) * cn + c] = new_pixel;
                        }
                    }
                }
            }
        }

        std::swap(pixelPtr, new_pixelPtr);
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copier le résultat final dans A_new
    cv::Mat result(cv::Size(A.size().width, A.size().height), A.type(), pixelPtr + (M + 1) * cn);
    result.copyTo(A_new);
}