#include <chrono>
#include <device_functions.h>
#include <Eigen/Sparse>
#include <random>
#include <vector>

#include "cuda_runtime_api.h"

// Uncomment the following line to verify the correctness of gpu versions.
// #define VERIFY_RESULT

/*!
  \brief  Generate #num_systems linear systems. All systems are identical.
*/
void GenerateLinearSystems(int num_systems, int dimension,
                           std::vector<Eigen::MatrixXf>& A_dense,
                           std::vector<Eigen::VectorXf>& b_dense) {
  A_dense.resize(num_systems);
  b_dense.resize(num_systems);

  for (int i = 0; i < num_systems; ++i) {
    // matrix A
    // make A diagonally dominant
    A_dense[i].resize(dimension, dimension);
    A_dense[i].setZero();
    A_dense[i].diagonal(0).setConstant(10);
    for (int j = 0; j <= 10; ++j) {
      // shift to [0, 1]
      Eigen::VectorXf random_array =
          0.5f * (Eigen::VectorXf::Random(dimension - j) + Eigen::VectorXf::Constant(dimension - j, 1.f));
      A_dense[i].diagonal(j) += random_array;
      A_dense[i].diagonal(-j) += random_array;
    }

    // vector b
    b_dense[i] = Eigen::VectorXf::Random(dimension);
  }
}

/*!
  \brief  Compute offsets for accessing Al, Ad.
  \param  [in] dimension specifies the dimension of each linear system. (400 in the paper.)
  \param  [out] offset: InclusiveSum of [dim_0, dim_1, dim2, ..., dim_{n-1}]. n = num_systems.
                Since in this case all systems have the same dimension, offset = [0, dimension, 2 * dimension, ..., (n - 1) * dimension]
  \param  [out] offset_triplet: InclusiveSum of [3 * max(dim_0, dim_1, dim_2), 3 * max(dim_3, dim_4, dim_5), ..., 3 * max(dim_{n - 3}, dim_{n - 2}, dim_{n - 1})].
                In this case, offset_triplet = [0, 3 * dimension, 6 * dimension, ..., (n / 3) * 3 * dimension].
  \param  [out] dim_triplet: [max(dim_0, dim_1, dim_2), max(dim_3, dim_4, dim_5), ..., max(dim_{n - 3}, dim_{n - 2}, dim_{n - 1}].
*/
void ComputeOffsets(int num_systems, int dimension, int* offset, int* offset_triplet, int* dim_triplet) {
  offset[0] = 0;
  offset_triplet[0] = 0;

  int max_dim_in_triplet = 0;
  int counter_triplet = 0;
  for (int i = 0; i < num_systems; ++i) {
    offset[i + 1] = offset[i] + dimension;
    max_dim_in_triplet = std::max(max_dim_in_triplet, dimension);
    if (counter_triplet == 2) {
      int triplet = i / 3;
      offset_triplet[triplet + 1] =
          offset_triplet[triplet] + max_dim_in_triplet * 3;
      dim_triplet[triplet] = max_dim_in_triplet;

      counter_triplet = 0;
      max_dim_in_triplet = 0;
    } else {
      ++counter_triplet;
    }
  }
}

void TestCpuLdl(int num_systems, int dimension, int sample_size) {
  printf("Test CPU version of LDL...\n");
  // construct dense matrices A and vector b
  std::vector<Eigen::MatrixXf> A_dense;
  std::vector<Eigen::VectorXf> b_dense;
  GenerateLinearSystems(1, dimension, A_dense, b_dense);

  std::vector<Eigen::SparseMatrix<float>> A_sparse;
  for (int i = 0; i < num_systems; ++i) {
    A_sparse.push_back(A_dense[0].sparseView());
    A_sparse[i].makeCompressed();
  }

  // CPU solve begin
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
  for (int k = 0; k < sample_size; k++) {
    for (int i = 0; i < num_systems; i++) {
      // Only analyze once
      if (i == 0) {
        solver.analyzePattern(A_sparse[0]);
      }

      solver.factorize(A_sparse[i]);
      solver.solve(b_dense[0]);
      if (solver.info() != Eigen::Success) {
        printf("CPU solver failed");
      }
    }
  }
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  float average_time = time_span.count() * 1000 / sample_size;
  printf("Time cost of (CPU, num_systems = %d, dimension = %d) = %fms.\n",
         num_systems, dimension, average_time);

  // CPU solve end

  // Since we use Eigen, there's no need to check the results.
}

__global__ void NaiveGPULdl(int num_systems, int n, const float *A,
                            const float *b, float *L, float *D, float *y,
                            float *x) {
  int system_index = blockIdx.x;
  int lane_id = threadIdx.x;

  if (lane_id == 0) {
    // Step 0: LDL decompostion
    for (int j = 0; j < n; j++) {
      float A_jj = A[system_index * n * 11 + j * 11];
      float sum_LD = 0.f;
      for (int k = max(0, j - 10); k < j; k++) {
        float L_jk = L[system_index * n * 10 + k * 10 + (j - 1 - k)];
        float D_k = D[system_index * n + k];
        sum_LD += L_jk * L_jk * D_k;
      }
      float D_j = A_jj - sum_LD;
      D[system_index * n + j] = D_j;

      for (int i = j + 1; i < min(n, j + 11); i++) {
        float A_ij = A[system_index * n * 11 + j * 11 + (i - j)];
        float sum_LLD = 0.f;
        for (int k = max(0, j - 10); k < j; k++) {
          if (i - 1 - k < 10) {
            float L_ik = L[system_index * n * 10 + k * 10 + (i - 1 - k)];
            float L_jk = L[system_index * n * 10 + k * 10 + (j - 1 - k)];
            float D_k = D[system_index * n + k];
            sum_LLD += L_ik * L_jk * D_k;
          }
        }
        float L_ij = (A_ij - sum_LLD) / D_j;
        L[system_index * n * 10 + j * 10 + (i - 1 - j)] = L_ij;
      }
    }

    // Step 1: forward substitution
    for (int j = 0; j < n; j++) {
      float D_j = D[system_index * n + j];
      float b_j = b[system_index * n + j];
      float sum_yLD = 0.f;
      for (int k = max(0, j - 10); k < j; k++) {
        float L_jk = L[system_index * n * 10 + k * 10 + (j - 1 - k)];
        float D_k = D[system_index * n + k];
        float y_k = y[system_index * n + k];
        sum_yLD += y_k * L_jk * D_k;
      }
      float y_j = (b_j - sum_yLD) / D_j;
      y[system_index * n + j] = y_j;
    }

    // Step 2: back substitution
    for (int j = n - 1; j >= 0; j--) {
      float y_j = y[system_index * n + j];
      float sum_Lx = 0.f;
      for (int k = j + 1; k < min(n, j + 11); k++) {
        float L_kj = L[system_index * n * 10 + j * 10 + (k - 1 - j)];
        float x_k = x[system_index * n + k];
        sum_Lx += L_kj * x_k;
      }
      float x_j = y_j - sum_Lx;
      x[system_index * n + j] = x_j;
    }
  }
}

void TestNaiveGpuLdl(int num_systems, int dimension, int sample_size) {
  printf("Test naive GPU version of LDL...\n");
  // construct dense matrices A and vector b
  std::vector<Eigen::MatrixXf> A_dense;
  std::vector<Eigen::VectorXf> b_dense;
  GenerateLinearSystems(1, dimension, A_dense, b_dense);

  float* h_A = (float*)malloc(num_systems * dimension * 11 * sizeof(float));
  float* h_b = (float*)malloc(num_systems * dimension * sizeof(float));
  for (int system_index = 0; system_index < num_systems; system_index++) {
    for (int col = 0; col < dimension; col++) {
      for (int k = 0; k < 11; k++) {
        int row = col + k;
        h_A[system_index * dimension * 11 + col * 11 + k] = A_dense[0](row, col);
      }
      h_b[system_index * dimension + col] = b_dense[0](col);
    }
  }
  float* d_A;
  float* d_b;
  cudaMalloc((void**)&d_A, num_systems * dimension * 11 * sizeof(float));
  cudaMalloc((void**)&d_b, num_systems * dimension * sizeof(float));
  cudaMemcpy(d_A, h_A, num_systems * dimension * 11 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, num_systems * dimension * sizeof(float),
             cudaMemcpyHostToDevice);

  float* h_L = (float*)malloc(num_systems * dimension * 10 * sizeof(float));
  float* h_D = (float*)malloc(num_systems * dimension * sizeof(float));
  float* h_y = (float*)malloc(num_systems * dimension * sizeof(float));
  float* h_x = (float*)malloc(num_systems * dimension * sizeof(float));
  float* d_L;
  float* d_D;
  float* d_y;
  float* d_x;
  cudaMalloc((void**)&d_L, num_systems * dimension * 10 * sizeof(float));
  cudaMalloc((void**)&d_D, num_systems * dimension * sizeof(float));
  cudaMalloc((void**)&d_y, num_systems * dimension * sizeof(float));
  cudaMalloc((void**)&d_x, num_systems * dimension * sizeof(float));
  cudaMemset((void*)d_L, 0, num_systems * dimension * 10 * sizeof(float));
  cudaMemset((void*)d_D, 0, num_systems * dimension * sizeof(float));
  cudaMemset((void*)d_y, 0, num_systems * dimension * sizeof(float));
  cudaMemset((void*)d_x, 0, num_systems * dimension * sizeof(float));

  float total_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < sample_size; i++) {
    NaiveGPULdl<<<num_systems, 32>>>(num_systems, dimension, d_A, d_b, d_L, d_D, d_y, d_x);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);
  float average_time = total_time / sample_size;
  printf("Time cost of (Naive GPU, num_systems = %d, dimension = %d) = %fms.\n",
         num_systems, dimension, average_time);

#ifdef VERIFY_RESULT
  cudaMemcpy(h_x, d_x, num_systems * dimension * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Reconstruct b with x (b = Ax)
  std::vector<Eigen::VectorXf> b_recon(num_systems);
  for (int system_index = 0; system_index < num_systems; system_index++) {
    Eigen::VectorXf local_x(dimension);
    for (int col = 0; col < dimension; col++) {
      local_x[col] = h_x[system_index * dimension + col];
    }
    b_recon[system_index] = A_dense[0] * local_x;
  }

  // Compare the reconstructed b with the original b
  for (int system_index = 0; system_index < num_systems; ++system_index) {
    for (int j = 0; j < b_dense[0].size(); ++j) {
      if (fabs(b_recon[system_index][j] - b_dense[0][j]) >= 1e-4f) {
        printf("Ldlsol reconstruct b fails at matrix %d\n", system_index);
      }
    }
  }
#endif

  free(h_A);
  free(h_b);
  free(h_L);
  free(h_D);
  free(h_y);
  free(h_x);
  cudaFree(d_A);
  cudaFree(d_b);
  cudaFree(d_L);
  cudaFree(d_D);
  cudaFree(d_y);
  cudaFree(d_x);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

// For b and Ad
__forceinline__ __host__ __device__ int GetIndexInTriplet(int row) { return row * 3; }

// For Al
__forceinline__ __host__ __device__ int GetIndexInTriplet(int row, int col) {
  // in each row, data are stored as
  // 9 8 7 6 5 4 3 2 1 0
  // 0 maps to (r, r-1)
  // 9 maps to (r, r-10)
  return col * 30 + row - col - 1;
}

/*!
  \brief  3 systems are processed by 1 warp in parallel. The last two threads in a warp are not used.
*/
__global__ void OptimizedBandLdl3In1(const int *offset_triplet,
                                     const int *dim_triplet, float *Al,
                                     float *Ad, float *b) {
  int local_system_id = threadIdx.x / 10;
  int group10 = local_system_id * 10;
  int local_tid = threadIdx.x - group10;
  int interval = 9 - local_tid;

  int offset = offset_triplet[blockIdx.x] + local_system_id;
  float *Al2 = Al + offset * 10;
  float *Ad2 = Ad + offset;
  float *b2 = b + offset;

  int dimension = dim_triplet[blockIdx.x];

  if (threadIdx.x >= 30) {
    return;
  }

  __shared__ float buf_local_LD[30];
  __shared__ float buf_D[30];
  __shared__ float buf_b[30];
  __shared__ float buf_LD[10][31];

  for (int i = 0; i < 10; ++i) {
    buf_LD[i][threadIdx.x] = 0.f;
  }
  __syncthreads();

  // assume 3 strands have same dof
  int col3 = 0;
  int pointer = 0;
  for (int col = 0; col < dimension; ++col, col3 += 3, ++pointer) {
    if (pointer > 9) {
      pointer = 0;
    }
    // pointer10 is the current index j in [0,9]
    // j in paper
    int pointer10 = pointer + group10;

    int pointer2 = pointer - local_tid - 1;
    // local_tid > pointer - 1
    if (pointer2 < 0) {
      pointer2 += 10;
    }
    // pointer210 points j-10, j-8, ..., j-1
    // k in paper
    int pointer210 = pointer2 + group10;

    // -------------
    // compute D
    // -------------
    int k = col - local_tid - 1;
    float term = 0.f;
    float LD = 0.f;

    // local_tid <= col - 1
    if (k >= 0) {
      LD = buf_LD[pointer2][threadIdx.x];  // L_jk D_k
      term = LD * LD / buf_D[pointer210];  // L_jk^2 D_k
    }
    buf_local_LD[threadIdx.x] = LD;

    // sum L_jk^2 D_k, k = {0,1,2,...,9}
    for (uint32_t iter = 1; iter <= 8; iter <<= 1) {
      float tmp = __shfl_down_sync(0x3FFFFFFF, term, iter);
      if (interval >= iter) {
        term += tmp;
      }
    }

    float term_D = 0.f;
    // local_tid = 0;
    if (interval == 9) {
      // D_j = Ad_j -  sum L_jk^2 D_k
      term = Ad2[col3] - term;
      term_D = term;  // D_j
    }

    // -------------
    // compute LD*y = b
    // -------------
    // row col switch
    term = 0.f;
    int row = col - local_tid - 1;
    // local_tid  < row - 1
    if (row >= 0) {
      // y_k L_jk Dk , k = {0,1,...,9}
      term = LD * buf_b[pointer210];
    }

    // sum term in local_tid = {0,1,2,3,4,5,6,7,8,9}
    // sum y_k L_jk D_k, k = {0,1,...,9}
    for (uint32_t iter = 1; iter <= 8; iter <<= 1) {
      float tmp = __shfl_down_sync(0x3FFFFFFF, term, iter);
      if (interval >= iter) {
        term += tmp;
      }
    }

    // local_tid = 0;
    if (interval == 9) {
      // y_j = (b_j - \sum y_k L_jk D_k)/D_j
      float tmp = (b2[col3] - term) / term_D;
      b2[col3] = tmp;

      // store y_j
      buf_b[pointer10] = tmp;

      // store D_j
      buf_D[pointer10] = term_D;
    }
    __syncthreads();

    // -------------
    // compute L
    // -------------
    row = col + local_tid + 1;
    // buf_LD[pointer][threadIdx.x] = 0.f;
    //  local_tid < dimension - col - 1
    if (row < dimension) {
      float sum = 0.f;
      for (int kk = col - 1; kk >= max(0, row - 10); --kk) {
        sum += Al2[GetIndexInTriplet(row, kk)] * buf_local_LD[group10 + col - 1 - kk];
      }

      int index2 = GetIndexInTriplet(row, col);
      float tmp = Al2[index2] - sum;

      Al2[index2] = tmp / buf_D[pointer10];

      buf_LD[pointer][threadIdx.x] = tmp;
    }
    __syncthreads();
  }

  // Step 5: L'x = y
  int row3 = (dimension - 1) * 3;
  for (int row = dimension - 1; row >= 0; row--, row3 -= 3) {
    float term = 0.f;
    int col = row + local_tid + 1;
    if (col < dimension) {
      term = Al2[GetIndexInTriplet(col, row)] * b2[col * 3];
    }
    for (uint32_t iter = 1; iter <= 8; iter <<= 1) {
      float tmp = __shfl_down_sync(0x3FFFFFFF, term, iter);
      if (interval >= iter) {
        term += tmp;
      }
    }
    if (interval == 9) {
      b2[row3] = b2[row3] - term;
    }
    __syncthreads();
  }
}

/*!
  \brief  Change storage format of A and b. Every 3 systems (i.e., a triplet) are stored together to facilitate coalesced access.
          Data of every triplet are interleaved. See examples below.
*/
// b: [b0_0, b1_0, b2_0,  b0_1, b1_1, b2_1,  b0_2, b1_2, b2_2, ..., b0_399, b1_399, b2_399,                                                    // --> First triplet.   Here bi_j means the j-th element in the i-th system's b.
//     b3_0, b4_0, b5_0,  b3_1, b4_1, b5_1,  b3_2, b4_2, b5_2, ..., b3_399, b4_399, b5_399,                                                    // --> Second triplet
//            ...                                                              ...        ,
//     b{n-3}_0, b{n-2}_0, b{n-1}_0,  b{n-3}_1, b{n-2}_1, b{n-1}_1,  b{n-3}_2, b{n-2}_2, b{n-1}_2, ..., b{n-3}_399, b{n-2}_399, b{n-1}_399,].  // --> n is the number of systems.
//
// Ad: similar to b
//
//                  Column 0 of the system 0                           Column 0 of the system 1                           Column 0 of the system 2                                                   --
//       ______________________|________________________    ______________________|________________________    ______________________|________________________                                        |
//      |                                              |   |                                              |   |                                              |                                        |
// Al: [Al0_{1,0}, Al0_{2,0}, Al0_{3,0}, ..., Al0_{10,0},  Al1_{1,0}, Al1_{2,0}, Al1_{3,0}, ..., Al1_{10,0},  Al2_{1,0}, Al2_{2,0}, Al2_{3,0}, ..., Al2_{10,0},                                       |
//                  Column 1 of the system 0                           Column 1 of the system 1                           Column 1 of the system 2                                                    |
//       ______________________|________________________    ______________________|________________________    ______________________|________________________                                        |
//      |                                              |   |                                              |   |                                              |                                        |
//      Al0_{2,1}, Al0_{3,1}, Al0_{4,1}, ..., Al0_{11,1},  Al1_{2,1}, Al1_{3,1}, Al1_{4,1}, ..., Al1_{11,1},  Al2_{2,1}, Al2_{3,1}, Al2_{4,1}, ..., Al2_{11,1},                                       |
//                                       ...                                                ...                                                ...            ,                                       |-- First triplet.
//                         Column i of the system 0                                       Column i of the system 1                                       Column i of the system 2                     |
//       ____________________________|______________________________    ____________________________|______________________________    ____________________________|______________________________    |
//      |                                                          |   |                                                          |   |                                                          |    |
//      Al0_{i+1, i}, Al0_{i+2, i}, Al0_{i+2, i}, ..., Al0_{i+10, i},  Al1_{i+1, i}, Al1_{i+2, i}, Al1_{i+2, i}, ..., Al1_{i+10, i},  Al2_{i+1, i}, Al2_{i+2, i}, Al2_{i+2, i}, ..., Al2_{i+10, i},   |
//                                       ...                                                ...                                                ...            ,                                       |
//                 Column 398 of the system 0                         Column 398 of the system 1                         Column 398 of the system 2                                                   |
//       ______________________|________________________    ______________________|________________________    ______________________|________________________                                        |
//      |                                              |   |                                              |   |                                              |                                        |
//      Al0_{399, 398}, 0,           ...             , 0,  Al1_{399, 398}, 0,            ...            , 0,  Al2_{399, 398}, 0,           ...             , 0,                                      --

//      ...                                               the remaining triplet are omitted                                                                 ...]
void GenerateAlAdB(int num_systems, std::vector<Eigen::MatrixXf>& A_dense,
                   std::vector<Eigen::VectorXf>& b_dense, const int* offset,
                   const int* offset_triplet, const int* dim_triplet, float* Al,
                   float* Ad, float* b) {
  for (int system_id = 0; system_id < num_systems; system_id++) {
    int triplet = system_id / 3;
    int local_system_id = system_id - triplet * 3;

    int dimension = offset[system_id + 1] - offset[system_id];
    int max_dim_in_triplet = dim_triplet[triplet];

    int offset_d = offset_triplet[triplet] + local_system_id;
    int offset_l = offset_d * 10;
    for (int r = 0; r < dimension; r++) {
      Ad[offset_d + GetIndexInTriplet(r)] = A_dense[0](r, r);
      b[offset_d + GetIndexInTriplet(r)] = b_dense[0](r);
      for (int c = std::max(0, r - 10); c < r; c++) {
        Al[offset_l + GetIndexInTriplet(r, c)] = A_dense[0](r, c);
      }
    }
    // append 1 and 0 for Ad and b. This is only used when each system has a different dimension.
    for (int r = dimension; r < max_dim_in_triplet; r++) {
      Ad[offset_d + GetIndexInTriplet(r)] = 1.f;
      b[offset_d + GetIndexInTriplet(r)] = 0.f;
    }
  }
}

/*!
  \brief  Reconstruct b from x, i.e., reconstructed b = Ax
*/
void ReconstructB(int num_systems, const int* h_offset,
                  const int* h_offset_triplet, const int* h_dim_triplet,
                  const std::vector<Eigen::MatrixXf>& A_dense, const float* h_x,
                  std::vector<Eigen::VectorXf>& b_recon) {
  // reconstruct b = A * x
  b_recon.resize(num_systems);
  for (int i = 0; i < num_systems; ++i) {
    int triplet = i / 3;
    int group = i - triplet * 3;

    int dimension = h_offset[i + 1] - h_offset[i];
    int offset_d = h_offset_triplet[triplet] + group;

    // Result X of system i
    Eigen::VectorXf local_x(dimension);
    for (int r = 0; r < dimension; ++r) {
      local_x[r] = h_x[offset_d + GetIndexInTriplet(r)];
    }

    b_recon[i] = A_dense[0] * local_x;
  }
}

void TestOptimizedGpuLdl3in1(int num_systems, int dimension, int sample_size) {
  printf("Test optimized GPU version of LDL...\n");
  // Since we are solving 3 systems in one warp, num_systems must be a multiple of 3
  if (num_systems % 3 != 0) {
    printf("num_systems must be a multiple of 3!\n");
    return;
  }
  int num_triplets = num_systems / 3;

  // construct dense matrices A and vector b
  std::vector<Eigen::MatrixXf> A_dense;
  std::vector<Eigen::VectorXf> b_dense;
  GenerateLinearSystems(1, dimension, A_dense, b_dense);

  // Offset of each system
  int* h_offset = (int*)malloc((num_systems + 1) * sizeof(int));
  // Offset of each triplet
  int* h_offset_triplet = (int*)malloc((num_triplets + 1) * sizeof(int));
  // Max dimension of triplet (in case where not all systems have the same
  // number of vertices)
  int* h_dim_triplet = (int*)malloc(num_triplets * sizeof(int));
  ComputeOffsets(num_systems, dimension, h_offset, h_offset_triplet, h_dim_triplet);
  int* d_offset_triplet;
  int* d_dim_triplet;
  cudaMalloc((void**)&d_offset_triplet, (num_triplets + 1) * sizeof(int));
  cudaMalloc((void**)&d_dim_triplet, num_triplets * sizeof(int));
  cudaMemcpy(d_offset_triplet, h_offset_triplet,
             (num_triplets + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dim_triplet, h_dim_triplet, num_triplets * sizeof(int),
             cudaMemcpyHostToDevice);

  // convert A and b to data structure compatible with triplets
  // we group every 3 strands into one warp and store them with aosoa
  float* h_Al = (float*)malloc(h_offset_triplet[num_triplets] * 10 * sizeof(float));
  float* h_Ad = (float*)malloc(h_offset_triplet[num_triplets] * sizeof(float));
  float* h_b = (float*)malloc(h_offset_triplet[num_triplets] * sizeof(float));
  float* h_x = (float*)malloc(h_offset_triplet[num_triplets] * sizeof(float));
  GenerateAlAdB(num_systems, A_dense, b_dense, h_offset,
                h_offset_triplet, h_dim_triplet, h_Al, h_Ad, h_b);
  float* d_Al;
  float* d_Ad;
  float* d_b;
  cudaMalloc((void**)&d_Al, h_offset_triplet[num_triplets] * 10 * sizeof(float));
  cudaMalloc((void**)&d_Ad, h_offset_triplet[num_triplets] * sizeof(float));
  cudaMalloc((void**)&d_b, h_offset_triplet[num_triplets] * sizeof(float));
  cudaMemcpy(d_Al, h_Al, h_offset_triplet[num_triplets] * 10 * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ad, h_Ad, h_offset_triplet[num_triplets] * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, h_offset_triplet[num_triplets] * sizeof(float),
             cudaMemcpyHostToDevice);

  float total_time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  for (int i = 0; i < sample_size; i++) {
    OptimizedBandLdl3In1<<<num_triplets, 32, 0>>>(
        d_offset_triplet, d_dim_triplet, d_Al, d_Ad, d_b);
#ifdef VERIFY_RESULT
    // Note: since d_b will be overriden for several times, only the first d_b
    // (storing x) can be used to reconstruct b.
    if (i == 0) {
      cudaMemcpy(h_x, d_b, num_systems * dimension * sizeof(float),
                 cudaMemcpyDeviceToHost);
    }
#endif
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&total_time, start, stop);
  float average_time = total_time / sample_size;
  printf("Time cost of (Optimized GPU, num_systems = %d, dimension = %d) = %fms.\n",
         num_systems, dimension, average_time);

#ifdef VERIFY_RESULT
  std::vector<Eigen::VectorXf> b_recon(num_systems);
  // Reconstruct b with x (b = Ax)
  ReconstructB(num_systems, h_offset, h_offset_triplet, h_dim_triplet, A_dense, h_x, b_recon);

  // Compare the reconstructed b with the original b
  for (int system_index = 0; system_index < num_systems; ++system_index) {
    for (int j = 0; j < b_dense[0].size(); ++j) {
      if (fabs(b_recon[system_index][j] - b_dense[0][j]) >= 1e-4f) {
        printf("Ldlsol reconstruct b fails at matrix %d\n", system_index);
      }
    }
  }
#endif

  free(h_offset);
  free(h_offset_triplet);
  free(h_dim_triplet);
  free(h_Al);
  free(h_Ad);
  free(h_b);
  cudaFree(d_offset_triplet);
  cudaFree(d_dim_triplet);
  cudaFree(d_Al);
  cudaFree(d_Ad);
  cudaFree(d_b);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  // Number of linear systems
  int num_systems = 30000;
  // The dimension of each linear system
  int dimension = 400;
  // How many times do we want to run the kernels to get an average time?
  int sample_size = 20;
  TestOptimizedGpuLdl3in1(num_systems, dimension, sample_size);
  TestNaiveGpuLdl(num_systems, dimension, sample_size);
  TestCpuLdl(num_systems, dimension, sample_size);
  return 0;
}