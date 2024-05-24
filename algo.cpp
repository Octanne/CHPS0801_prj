// Jacobi with OPENMP CPU
#include <omp.h>
#include <vector>

void jacobi_iteration(std::vector<std::vector<std::vector<double>>>& A, std::vector<std::vector<std::vector<double>>>& A_new, int iterations) {
    int N = A.size();
    int M = A[0].size();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 1; i < N-1; ++i) {
                    for (int j = 1; j < M-1; ++j) {
                        #pragma omp task firstprivate(i, j)
                        {
                            for (int c = 0; c < 3; ++c) {
                                A_new[i][j][c] = 0.25 * (A[i+1][j][c] + A[i-1][j][c] + A[i][j+1][c] + A[i][j-1][c]);
                            }
                        }
                    }
                }
                #pragma omp taskwait
                #pragma omp parallel for collapse(3)
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < M; ++j) {
                        for (int c = 0; c < 3; ++c) {
                            A[i][j][c] = A_new[i][j][c];
                        }
                    }
                }
            }
        }
    }
}

void initialize(std::vector<std::vector<std::vector<double>>>& A) {
    int N = A.size();
    int M = A[0].size();
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int c = 0; c < 3; ++c) {
                if (i == 0 || i == N-1 || j == 0 || j == M-1) {
                    A[i][j][c] = 1.0; // Boundary conditions
                } else {
                    A[i][j][c] = 0.0; // Initial interior values
                }
            }
        }
    }
}

int main() {
    int N = 1000;
    int M = 1000;
    int iterations = 100;
    std::vector<std::vector<std::vector<double>>> A(N, std::vector<std::vector<double>>(M, std::vector<double>(3, 0)));
    std::vector<std::vector<std::vector<double>>> A_new(N, std::vector<std::vector<double>>(M, std::vector<double>(3, 0)));
    
    initialize(A);
    
    jacobi_iteration(A, A_new, iterations);
    
    return 0;
}

// Jacobi with OPENMP GPU
#include <omp.h>
#include <vector>

void jacobi_iteration(std::vector<std::vector<std::vector<double>>>& A, std::vector<std::vector<std::vector<double>>>& A_new, int iterations) {
    int N = A.size();
    int M = A[0].size();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp target teams distribute parallel for simd collapse(3)
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                for (int c = 0; c < 3; ++c) {
                    A_new[i][j][c] = 0.25 * (A[i+1][j][c] + A[i-1][j][c] + A[i][j+1][c] + A[i][j-1][c]);
                }
            }
        }
        #pragma omp target teams distribute parallel for simd collapse(3)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                for (int c = 0; c < 3; ++c) {
                    A[i][j][c] = A_new[i][j][c];
                }
            }
        }
    }
}

void initialize(std::vector<std::vector<std::vector<double>>>& A) {
    int N = A.size();
    int M = A[0].size();
    #pragma omp target teams distribute parallel for simd collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int c = 0; c < 3; ++c) {
                if (i == 0 || i == N-1 || j == 0 || j == M-1) {
                    A[i][j][c] = 1.0; // Boundary conditions
                } else {
                    A[i][j][c] = 0.0; // Initial interior values
                }
            }
        }
    }
}

int main() {
    int N = 1000;
    int M = 1000;
    int iterations = 100;
    std::vector<std::vector<std::vector<double>>> A(N, std::vector<std::vector<double>>(M, std::vector<double>(3, 0)));
    std::vector<std::vector<std::vector<double>>> A_new(N, std::vector<std::vector<double>>(M, std::vector<double>(3, 0)));
    
    initialize(A);
    
    jacobi_iteration(A, A_new, iterations);
    
    return 0;
}

// Gauss Seidel with OPENMP CPU
#include <omp.h>
#include <vector>

void gauss_seidel_iteration(std::vector<std::vector<std::vector<double>>>& A, int iterations) {
    int N = A.size();
    int M = A[0].size();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 1; i < N-1; ++i) {
                    for (int j = 1; j < M-1; ++j) {
                        #pragma omp task firstprivate(i, j)
                        {
                            for (int c = 0; c < 3; ++c) {
                                A[i][j][c] = 0.25 * (A[i+1][j][c] + A[i-1][j][c] + A[i][j+1][c] + A[i][j-1][c]);
                            }
                        }
                    }
                }
                #pragma omp taskwait
            }
        }
    }
}

void initialize(std::vector<std::vector<std::vector<double>>>& A) {
    int N = A.size();
    int M = A[0].size();
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int c = 0; c < 3; ++c) {
                if (i == 0 || i == N-1 || j == 0 || j == M-1) {
                    A[i][j][c] = 1.0; // Boundary conditions
                } else {
                    A[i][j][c] = 0.0; // Initial interior values
                }
            }
        }
    }
}

int main() {
    int N = 1000;
    int M = 1000;
    int iterations = 100;
    std::vector<std::vector<std::vector<double>>> A(N, std::vector<std::vector<double>>(M, std::vector<double>(3, 0)));
    
    initialize(A);
    
    gauss_seidel_iteration(A, iterations);
    
    return 0;
}

// Gauss Seidel with OPENMP GPU
#include <omp.h>
#include <vector>

void gauss_seidel_iteration(std::vector<std::vector<std::vector<double>>>& A, int iterations) {
    int N = A.size();
    int M = A[0].size();
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp target teams distribute parallel for simd collapse(3)
        for (int i = 1; i < N-1; ++i) {
            for (int j = 1; j < M-1; ++j) {
                for (int c = 0; c < 3; ++c) {
                    A[i][j][c] = 0.25 * (A[i+1][j][c] + A[i-1][j][c] + A[i][j+1][c] + A[i][j-1][c]);
                }
            }
        }
    }
}

void initialize(std::vector<std::vector<std::vector<double>>>& A) {
    int N = A.size();
    int M = A[0].size();
    #pragma omp target teams distribute parallel for simd collapse(3)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int c = 0; c < 3; ++c) {
                if (i == 0 || i == N-1 || j == 0 || j == M-1) {
                    A[i][j][c] = 1.0; // Boundary conditions
                } else {
                    A[i][j][c] = 0.0; // Initial interior values
                }
            }
        }
    }
}

int main() {
    int N = 1000;
    int M = 1000;
    int iterations = 100;
    std::vector<std::vector<std::vector<double>>> A(N, std::vector<std::vector<double>>(M, std::vector<double>(3, 0)));
    
    initialize(A);
    
    gauss_seidel_iteration(A, iterations);
    
    return 0;
}
