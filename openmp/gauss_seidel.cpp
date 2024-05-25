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
    // Ajout d'un bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = (uint8_t*)A_copy.data;
    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;

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
*/
void gauss_seidel_parallel_fronts_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'un bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = (uint8_t*)A_copy.data;
    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp target data map(to: pixelPtr[0:N*M*cn]) map(from: new_pixelPtr[0:N*M*cn])
        {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < M - 1; ++j) {
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
        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);
}

/**
 * Explications :

    1. Ajout de bordures : Nous ajoutons une bordure de 1 pixel tout autour de l'image d'entrée pour faciliter le traitement des pixels aux bords de l'image.

    2. Initialisation des matrices : A_copy contient l'image d'entrée avec bordures, A_iter est utilisée pour stocker les nouvelles valeurs de pixels à chaque itération.

    3. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.

    4. Phase rouge :
        Nous mettons à jour tous les pixels rouges en parallèle.
        Les pixels rouges sont ceux où (i + j) % 2 == 0.

    5. Barrière de synchronisation : Nous utilisons #pragma omp barrier pour synchroniser toutes les tâches après la phase rouge afin de garantir que tous les pixels rouges sont mis à jour avant de commencer la phase noire.

    6. Phase noire :
        Nous mettons à jour tous les pixels noirs en parallèle.
        Les pixels noirs sont ceux où (i + j) % 2 != 0.

    7. Barrière de synchronisation : Nous utilisons #pragma omp barrier pour synchroniser toutes les tâches après la phase noire afin de garantir que tous les pixels noirs sont mis à jour avant de passer à l'itération suivante.

    8. Mise à jour des pointeurs : Nous échangeons les pointeurs pixelPtr et new_pixelPtr pour préparer la prochaine itération.

    9. Sortie des résultats : À la fin des itérations, nous copions le résultat final dans A_new sans les bordures ajoutées.

   Cette méthode garantit que les dépendances entre les pixels sont respectées,
   permettant ainsi une parallélisation efficace tout en maintenant l'exactitude 
   de l'algorithme de Gauss-Seidel.
*/
void gauss_seidel_rb_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = (uint8_t*)A_copy.data;
    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;

    for (int iter = 0; iter < iterations; ++iter) {
        // Red phase
        #pragma omp parallel for shared(pixelPtr, new_pixelPtr)
        for (int index = 0; index < (N - 2) * (M - 2); ++index) {
            int i = 1 + index / (M - 2);
            int j = 1 + index % (M - 2);

            if ((i + j) % 2 == 0) {  // Red phase condition
                for (int c = 0; c < cn; ++c) {
                    uint8_t p_current = pixelPtr[i * M * cn + j * cn + c];
                    uint8_t p_top = pixelPtr[(i - 1) * M * cn + j * cn + c];
                    uint8_t p_bottom = pixelPtr[(i + 1) * M * cn + j * cn + c];
                    uint8_t p_left = pixelPtr[i * M * cn + (j - 1) * cn + c];
                    uint8_t p_right = pixelPtr[i * M * cn + (j + 1) * cn + c];
                    uint8_t new_pixel = static_cast<uint8_t>(0.2 * (p_current + p_top + p_bottom + p_left + p_right));
                    new_pixelPtr[i * M * cn + j * cn + c] = new_pixel;
                }
            }
        }

        // Synchronize to ensure red phase is done
        #pragma omp barrier

        // Black phase
        #pragma omp parallel for shared(pixelPtr, new_pixelPtr)
        for (int index = 0; index < (N - 2) * (M - 2); ++index) {
            int i = 1 + index / (M - 2);
            int j = 1 + index % (M - 2);

            if ((i + j) % 2 == 1) {  // Black phase condition
                for (int c = 0; c < cn; ++c) {
                    uint8_t p_current = new_pixelPtr[i * M * cn + j * cn + c];
                    uint8_t p_top = new_pixelPtr[(i - 1) * M * cn + j * cn + c];
                    uint8_t p_bottom = new_pixelPtr[(i + 1) * M * cn + j * cn + c];
                    uint8_t p_left = new_pixelPtr[i * M * cn + (j - 1) * cn + c];
                    uint8_t p_right = new_pixelPtr[i * M * cn + (j + 1) * cn + c];
                    uint8_t new_pixel = static_cast<uint8_t>(0.2 * (p_current + p_top + p_bottom + p_left + p_right));
                    new_pixelPtr[i * M * cn + j * cn + c] = new_pixel;
                }
            }
        }

        // Synchronize to ensure black phase is done
        #pragma omp barrier

        std::swap(pixelPtr, new_pixelPtr);
        std::cout << "Iteration " << iter << " done\r";
    }

    //cv::Mat A_result(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2));
    //A_result.copyTo(A_new);
    // Copy the middle of the image to the output image
    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);
}

/**
 * Explications :

    1. Ajout de bordures : Nous ajoutons une bordure de 1 pixel tout autour de l'image d'entrée pour faciliter le traitement des pixels aux bords de l'image.

    2. Initialisation des matrices : A_copy contient l'image d'entrée avec bordures, A_iter est utilisée pour stocker les nouvelles valeurs de pixels à chaque itération.

    3. Boucle d'itération : La boucle principale pour les itérations de Gauss-Seidel.

    4. Phase rouge :
        Nous mettons à jour tous les pixels rouges en parallèle.
        Les pixels rouges sont ceux où (i + j) % 2 == 0.
        Nous utilisons #pragma omp target teams distribute parallel for collapse(2) pour distribuer les calculs sur les équipes et les threads du GPU.

    5. Barrière de synchronisation : Nous utilisons #pragma omp barrier pour synchroniser toutes les tâches après la phase rouge afin de garantir que tous les pixels rouges sont mis à jour avant de commencer la phase noire.

    6. Phase noire :
        Nous mettons à jour tous les pixels noirs en parallèle.
        Les pixels noirs sont ceux où (i + j) % 2 != 0.
        Nous utilisons #pragma omp target teams distribute parallel for collapse(2) pour distribuer les calculs sur les équipes et les threads du GPU.

    7. Mise à jour des pointeurs : Nous échangeons les pointeurs pixelPtr et new_pixelPtr pour préparer la prochaine itération.

    8. Sortie des résultats : À la fin des itérations, nous copions le résultat final dans A_new sans les bordures ajoutées.

   Cette méthode garantit que les dépendances entre les pixels sont respectées,
   permettant ainsi une parallélisation efficace tout en maintenant l'exactitude 
   de l'algorithme de Gauss-Seidel.
*/
void gauss_seidel_rb_parallel_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Ajout d'un bordure de 1 pixel tout autour de l'image d'entrée
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* pixelPtr = (uint8_t*)A_copy.data;
    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp target data map(to: pixelPtr[0:N*M*cn]) map(from: new_pixelPtr[0:N*M*cn])
        {
            // Red phase
            #pragma omp target teams distribute parallel for
            for (int index = 0; index < (N - 2) * (M - 2); ++index) {
                int i = 1 + index / (M - 2);
                int j = 1 + index % (M - 2);

                if ((i + j) % 2 == 0) { // Red phase condition
                    for (int c = 0; c < cn; ++c) {
                        uint8_t p_current = pixelPtr[i * M * cn + j * cn + c];
                        uint8_t p_top = pixelPtr[(i - 1) * M * cn + j * cn + c];
                        uint8_t p_bottom = pixelPtr[(i + 1) * M * cn + j * cn + c];
                        uint8_t p_left = pixelPtr[i * M * cn + (j - 1) * cn + c];
                        uint8_t p_right = pixelPtr[i * M * cn + (j + 1) * cn + c];
                        uint8_t new_pixel = static_cast<uint8_t>(0.2 * (p_current + p_top + p_bottom + p_left + p_right));
                        new_pixelPtr[i * M * cn + j * cn + c] = new_pixel;
                    }
                }
            }

            // Synchronize to ensure red phase is done
            #pragma omp barrier

            // Black phase
            #pragma omp target teams distribute parallel for
            for (int index = 0; index < (N - 2) * (M - 2); ++index) {
                int i = 1 + index / (M - 2);
                int j = 1 + index % (M - 2);

                if ((i + j) % 2 == 1) { // Black phase condition
                    for (int c = 0; c < cn; ++c) {
                        uint8_t p_current = new_pixelPtr[i * M * cn + j * cn + c];
                        uint8_t p_top = new_pixelPtr[(i - 1) * M * cn + j * cn + c];
                        uint8_t p_bottom = new_pixelPtr[(i + 1) * M * cn + j * cn + c];
                        uint8_t p_left = new_pixelPtr[i * M * cn + (j - 1) * cn + c];
                        uint8_t p_right = new_pixelPtr[i * M * cn + (j + 1) * cn + c];
                        uint8_t new_pixel = static_cast<uint8_t>(0.2 * (p_current + p_top + p_bottom + p_left + p_right));
                        pixelPtr[i * M * cn + j * cn + c] = new_pixel;
                    }
                }
            }
        }
        std::swap(pixelPtr, new_pixelPtr);
        std::cout << "Iteration " << iter << " done\r";
    }

    //cv::Mat A_result(cv::Rect(1, 1, A_copy.cols - 2, A_copy.rows - 2));
    //A_result.copyTo(A_new);
    // Copy the middle of the image to the output image
    A_iter(cv::Rect(1, 1, A_iter.cols - 2, A_iter.rows - 2)).copyTo(A_new);
}