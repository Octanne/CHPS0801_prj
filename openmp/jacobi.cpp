#include "jacobi.h"

#include <omp.h>

using namespace std;

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
 * Explications

    1. Ajout de bordures : Nous ajoutons une bordure de 1 pixel tout autour de l'image d'entrée pour simplifier le traitement des pixels aux bords de l'image.

    2. Initialisation des matrices : A_copy contient l'image d'entrée avec bordures, A_iter est utilisée pour stocker les nouvelles valeurs des pixels à chaque itération.

    3. Boucle d'itération : La boucle principale pour les itérations de Jacobi.

    4. Parallélisation avec OpenMP :
        Utilisation de #pragma omp parallel for collapse(2) pour paralléliser les boucles imbriquées sur les pixels. Cette directive permet de paralléliser les calculs sur les pixels intérieurs de l'image.
        Pour chaque pixel intérieur, les nouvelles valeurs sont calculées et stockées dans new_pixelPtr.

    5. Mise à jour des valeurs des pixels : Pour chaque pixel, les nouvelles valeurs sont calculées en utilisant les valeurs des pixels voisins à l'itération précédente.

    6. Mise à jour de l'image : Après chaque itération, A_iter est copié dans A_copy pour la prochaine itération.

    7. Copie des résultats : À la fin des itérations, la partie centrale de A_iter (sans les bordures) est copiée dans A_new.

   Cette méthode utilise OpenMP pour paralléliser efficacement l'algorithme
   de Jacobi sur CPU sans nécessiter de gestion explicite des dépendances, 
   ce qui est approprié pour cet algorithme car nous avons pas besion de la nouvelle valeur des pixels pour calculer les nouvelles valeurs des pixels voisins.
   mais celle d'entrée ainsi pas de dépendances entre les pixels.
*/
void jacobi_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Créer une copie de l'image d'entrée avec une bordure de 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    // Ajouter une bordure noire
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;
    uint8_t* pixelPtr = (uint8_t*)A_copy.data;

    // Appliquer l'itération Jacobi
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                for (int c = 0; c < cn; ++c) {
                    // Pixel actuel
                    uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                    // Pixel supérieur
                    uint8_t p_top = pixelPtr[(i-1)*M*cn + j*cn + c];
                    // Pixel inférieur
                    uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                    // Pixel gauche
                    uint8_t p_left = pixelPtr[i*M*cn + (j-1)*cn + c];
                    // Pixel droit
                    uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                    // Calculer la nouvelle valeur du pixel
                    uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);

                    new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
                }
            }
        }

        // Mettre à jour l'image pour la prochaine itération
        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copier le milieu de l'image dans l'image de sortie
    A_iter(cv::Rect(1, 1, A_iter.cols-2, A_iter.rows-2)).copyTo(A_new);
}

/**
 * Explications

    1. Ajout de bordures : Nous ajoutons une bordure de 1 pixel tout autour de l'image d'entrée pour simplifier le traitement des pixels aux bords de l'image.

    2. Initialisation des matrices : A_copy contient l'image d'entrée avec bordures, A_iter est utilisée pour stocker les nouvelles valeurs des pixels à chaque itération.

    3. Boucle d'itération : La boucle principale pour les itérations de Jacobi.

    4. Parallélisation avec OpenMP :
        Utilisation de #pragma omp target data map pour transférer les données nécessaires vers le GPU.
        Utilisation de #pragma omp target teams distribute parallel for collapse(2) pour paralléliser les boucles imbriquées sur les pixels sur le GPU. Cette directive permet de distribuer les calculs sur les équipes et les threads du GPU.

    5. Mise à jour des valeurs des pixels : Pour chaque pixel, les nouvelles valeurs sont calculées en utilisant les valeurs des pixels voisins à l'itération précédente et stockées dans new_pixelPtr.

    6. Mise à jour de l'image : Après chaque itération, A_iter est copié dans A_copy pour la prochaine itération.

    7. Copie des résultats : À la fin des itérations, la partie centrale de A_iter (sans les bordures) est copiée dans A_new.

   Cette méthode utilise les directives OpenMP target pour offloader les calculs
   sur le GPU et paralléliser efficacement l'algorithme de Jacobi. Aucune gestion 
   explicite des dépendances n'est nécessaire car chaque pixel peut être mis à jour 
   indépendamment dans une itération donnée.
*/
void jacobi_parallel_gpu(cv::Mat& A, cv::Mat& A_new, int iterations) {
    // Créer une copie de l'image d'entrée avec une bordure de 1 pixel
    cv::Mat A_copy(cv::Size(A.size().width + 2, A.size().height + 2), A.type());
    // Ajouter une bordure noire
    cv::copyMakeBorder(A, A_copy, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    int N = A_copy.rows;
    int M = A_copy.cols;
    int cn = A_copy.channels();

    cv::Mat A_iter(A_copy.size(), A_copy.type());

    uint8_t* new_pixelPtr = (uint8_t*)A_iter.data;
    uint8_t* pixelPtr = (uint8_t*)A_copy.data;

    // Appliquer l'itération Jacobi
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp target data map(to: pixelPtr[0:N*M*cn]) map(from: new_pixelPtr[0:N*M*cn])
        {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i < N-1; ++i) {
                for (int j = 1; j < M-1; ++j) {
                    for (int c = 0; c < cn; ++c) {
                        // Pixel actuel
                        uint8_t p_current = pixelPtr[i*M*cn + j*cn + c];
                        // Pixel supérieur
                        uint8_t p_top = pixelPtr[(i-1)*M*cn + j*cn + c];
                        // Pixel inférieur
                        uint8_t p_bottom = pixelPtr[(i+1)*M*cn + j*cn + c];
                        // Pixel gauche
                        uint8_t p_left = pixelPtr[i*M*cn + (j-1)*cn + c];
                        // Pixel droit
                        uint8_t p_right = pixelPtr[i*M*cn + (j+1)*cn + c];
                        // Calculer la nouvelle valeur du pixel
                        uint8_t new_pixel = 0.2 * (p_current + p_top + p_bottom + p_left + p_right);

                        new_pixelPtr[i*M*cn + j*cn + c] = new_pixel;
                    }
                }
            }
        }

        // Mettre à jour l'image pour la prochaine itération
        A_iter.copyTo(A_copy);
        std::cout << "Iteration " << iter << " done\r";
    }

    // Copier le milieu de l'image dans l'image de sortie
    A_iter(cv::Rect(1, 1, A_iter.cols-2, A_iter.rows-2)).copyTo(A_new);
}