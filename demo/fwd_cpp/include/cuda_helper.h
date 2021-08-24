/*
 * Copyright (c) [2020] <Tencent>
 *
 * Author: ajian, aoli, percyyuan, jianqiulu
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

inline bool CheckCudaError(cudaError_t cuda_result) {
  if (cuda_result != cudaSuccess) {
    const std::string msg = cudaGetErrorString(cuda_result);
    std::cerr << "CUDA ERROR: " << msg << std::endl;
    return false;
  }

  return true;
}

inline bool CheckCudaKernel(const std::string& name) {
  cudaError_t cuda_result = cudaGetLastError();
  if (cuda_result != cudaSuccess) {
    const std::string msg = cudaGetErrorString(cuda_result);
    std::cerr << name << " failed: " << msg << std::endl;
    return false;
  }
  return true;
}

/**
 * \brief Pinnned 内存分配器
 *
 * \tparam T 数据类型
 */
template <typename T>
class PinnedAllocator {
 public:
  PinnedAllocator() = default;
  ~PinnedAllocator() = default;

  /**
   * \brief 分配内存
   */
  T* allocate(size_t size) {
    T* pointer;
    cudaError cuda_result = cudaMallocHost(reinterpret_cast<void**>(&pointer), sizeof(T) * size);

    if (cuda_result != cudaSuccess) {
      const std::string msg = cudaGetErrorString(cuda_result);
      std::cerr << "PinnedAllocator allocate error: " << msg << std::endl;
      throw std::bad_alloc();
    }

    return pointer;
  }

  /**
   * \brief 释放内存
   */
  void deallocate(T* pointer) {
    if (pointer == nullptr) {
      return;
    }

    cudaError cuda_result = cudaFreeHost(pointer);
    if (cuda_result != cudaSuccess) {
      const std::string msg = cudaGetErrorString(cuda_result);
      std::cerr << "PinnedAllocator deallocate error: " << msg << std::endl;
    }
  }
};

/**
 * \brief 设备内存分配器
 *
 * \tparam T 数据类型
 */
template <typename T>
class DeviceAllocator {
 public:
  DeviceAllocator() = default;
  ~DeviceAllocator() = default;

  /**
   * \brief 分配内存
   */
  T* allocate(size_t size) {
    T* pointer;
    cudaError cuda_result = cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * size);
    if (cuda_result != cudaSuccess) {
      const std::string msg = cudaGetErrorString(cuda_result);
      std::cerr << "DeviceAllocator allocate error: " << msg << std::endl;
      throw std::bad_alloc();
    }

    return pointer;
  }

  /**
   * \brief 释放内存
   */
  void deallocate(T* pointer) {
    if (pointer == nullptr) {
      return;
    }

    cudaError cuda_result = cudaFree(pointer);
    if (cuda_result != cudaSuccess) {
      const std::string msg = cudaGetErrorString(cuda_result);
      std::cerr << "DeviceAllocator deallocate error: " << msg << std::endl;
    }
  }
};

/**
 * \brief Vector 类，支持 Host，Pinned 和 Device 的内存
 *
 * \tparam T 数据类型
 * \tparam TAllocator 内存分配器
 */
template <typename T, typename TAllocator = std::allocator<T>>
class TVector {
 public:
  TVector() {
    size_ = 0;
    data_ = nullptr;
  }
  ~TVector() {
    if (data_ != nullptr) {
      TAllocator().deallocate(data_);
    }
  }

  /**
   * \brief 重新分配大小
   */
  bool Resize(size_t size) {
    size_ = size;

    if (data_ != nullptr) {
      TAllocator().deallocate(data_);
      data_ = nullptr;
    }

    if (size_ == 0) {
      return true;
    }

    try {
      data_ = TAllocator().allocate(size_);
    } catch (std::bad_alloc e) {
      std::cerr << "TVector allocate failed" << std::endl;
      return false;
    }

    return true;
  }

  /**
   * \brief 交换两个 Vector 的内容
   */
  void Swap(TVector& other) {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
  }

  /**
   * \brief 判断是否为空
   */
  bool Empty() const { return size_ == 0; }

  /**
   * \brief 得到大小
   */
  size_t Size() const { return size_; }

  /**
   * \brief 得到数据指针
   */
  T* Data() { return data_; }
  T* Data() const { return data_; }

 private:
  /**
   * \brief 大小
   */
  size_t size_{0};

  /**
   * \brief 数据
   */
  T* data_{nullptr};
};

/**
 * \brief Vector2D类，支持Host，Pinned 和 Device内存
 *
 * 这个类只是一个 Adapter，具体的数据委托 Vector1D 管理
 *
 * \tparam T 数据类型
 * \tparam TVector1D Vector1D 的实现类型
 */
template <typename T, typename TVector1D>
struct TVector2D {
 public:
  TVector2D() = default;
  ~TVector2D() = default;

  /**
   * \brief 重新分配大小
   *
   * \param rows 行数
   * \param cols 列数
   * \return true 成功，false 失败
   */
  bool Resize(size_t rows, size_t cols) {
    rows_ = rows;
    cols_ = cols;

    // 调整为 256 的倍速（效率）
    stride_ = (cols + 0xff) & 0xffffff00;

    return vector_1d_.Resize(rows_ * stride_);
  }

  /**
   * \brief 交换两个 Vector 的内容
   */
  void Swap(TVector2D& other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(stride_, other.stride_);

    vector_1d_.Swap(other.vector_1d_);
  }

  /**
   * \brief 判断是否为空
   */
  bool Empty() const { return rows_ == 0; }

  /**
   * \brief 得到行数
   */
  size_t Rows() const { return rows_; }

  /**
   * \brief 得到列数目
   */
  size_t Cols() const { return cols_; }

  /**
   * \brief 得到跨距
   */
  size_t Stride() const { return stride_; }

  /**
   * \brief 得到每行数据
   */
  T* Data(size_t row = 0) {
    assert(row < rows_);
    return vector_1d_.Data() + stride_ * row;
  }
  T* Data(size_t row = 0) const {
    assert(row < rows_);
    return vector_1d_.Data() + stride_ * row;
  }

 private:
  /**
   * \brief 行数
   */
  size_t rows_{0};
  /**
   * \brief 列数
   */
  size_t cols_{0};
  /**
   * \brief 跨距
   */
  size_t stride_{0};

  TVector1D vector_1d_;
};

template <typename T>
using HostVector = TVector<T, std::allocator<T>>;
template <typename T>
using PinnedVector = TVector<T, PinnedAllocator<T>>;
template <typename T>
using DeviceVector = TVector<T, DeviceAllocator<T>>;

template <typename T>
using HostVector2D = TVector2D<T, HostVector<T>>;
template <typename T>
using PinnedVector2D = TVector2D<T, PinnedVector<T>>;
template <typename T>
using DeviceVector2D = TVector2D<T, DeviceVector<T>>;

template <typename T>
bool MemcpyHostToDevice(T* device, const T* host, int size) {
  return CheckCudaError(cudaMemcpy(device, host, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template <typename T>
bool MemcpyDeviceToHost(T* host, const T* device, int size) {
  return CheckCudaError(cudaMemcpy(host, device, sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
bool MemcpyHostToDeviceAsync(T* device, const T* host, int size, const cudaStream_t& stream) {
  return CheckCudaError(
      cudaMemcpyAsync(device, host, sizeof(T) * size, cudaMemcpyHostToDevice, stream));
}

template <typename T>
bool MemcpyDeviceToHostAsync(T* host, const T* device, int size, const cudaStream_t& stream) {
  return CheckCudaError(
      cudaMemcpyAsync(host, device, sizeof(T) * size, cudaMemcpyDeviceToHost, stream));
}
