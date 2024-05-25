#include "jacobi.h"
#include <iostream>
#include <unistd.h>
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
    for (int it = 0; it < iterations; ++it) {
        for (int i = 1; i < A.rows - 1; ++i) {
            for (int j = 1; j < A.cols - 1; ++j) {
                A_new.at<double>(i, j) = 0.25 * (A.at<double>(i+1, j) + A.at<double>(i-1, j) + A.at<double>(i, j+1) + A.at<double>(i, j-1));
            }
        }
        A = A_new.clone();
    }
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
    Kokkos::initialize();
    {
        for (int it = 0; it < iterations; ++it) {
            Kokkos::parallel_for("jacobi_cpu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {A.rows - 1, A.cols - 1}), KOKKOS_LAMBDA(const int i, const int j) {
                A_new.at<double>(i, j) = 0.25 * (A.at<double>(i+1, j) + A.at<double>(i-1, j) + A.at<double>(i, j+1) + A.at<double>(i, j-1));
            });
            A = A_new.clone();
        }
    }
    Kokkos::finalize();
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
    Kokkos::initialize();
    {
        Kokkos::View<double**> d_A("A", A.rows, A.cols);
        Kokkos::View<double**> d_A_new("A_new", A.rows, A.cols);

        // Copie des données vers le GPU
        Kokkos::parallel_for("copy_to_device", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {A.rows, A.cols}), KOKKOS_LAMBDA(const int i, const int j) {
            d_A(i, j) = A.at<double>(i, j);
        });

        for (int it = 0; it < iterations; ++it) {
            Kokkos::parallel_for("jacobi_gpu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {A.rows - 1, A.cols - 1}), KOKKOS_LAMBDA(const int i, const int j) {
                d_A_new(i, j) = 0.25 * (d_A(i+1, j) + d_A(i-1, j) + d_A(i, j+1) + d_A(i, j-1));
            });
            Kokkos::deep_copy(d_A, d_A_new);
        }

        // Copie des données vers le CPU
        Kokkos::parallel_for("copy_to_host", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {A.rows, A.cols}), KOKKOS_LAMBDA(const int i, const int j) {
            A.at<double>(i, j) = d_A(i, j);
        });
    }
    Kokkos::finalize();
}
