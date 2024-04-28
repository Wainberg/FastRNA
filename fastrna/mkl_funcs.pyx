import numpy as np
cimport numpy as np

import cython
from cython.parallel import prange

from fastrna.utils cimport *

# Conversion functions
cdef struct matrix_descr:
    sparse_matrix_type_t type

cdef sparse_matrix_t to_mkl_spmatrix(
        const float[:] data,
        const long[:] indices,
        const long[:] indptr,
        int nrow,
        int ncol,
        bint sptype
        ):

    cdef MKL_INT64 rows = nrow
    cdef MKL_INT64 cols = ncol
    cdef sparse_matrix_t A
    cdef sparse_index_base_t base_index = SPARSE_INDEX_BASE_ZERO

    cdef MKL_INT64* start = &indptr[0]
    cdef MKL_INT64* end = &indptr[1]
    cdef MKL_INT64* index = &indices[0]
    cdef float* values = &data[0]

    if sptype:
        mkl_sparse_s_create = mkl_sparse_s_create_csr_64
    else:
        mkl_sparse_s_create = mkl_sparse_s_create_csc_64

    cdef sparse_status_t create_status = mkl_sparse_s_create(
            &A,
            base_index,
            rows,
            cols,
            start,
            end,
            index,
            values
            )

    return A

cdef np.ndarray[np.float32_t, ndim=2] to_python_dmatrix(
        sparse_matrix_t A,
        bint sptype
        ):

    cdef MKL_INT64 rows
    cdef MKL_INT64 cols
    cdef sparse_index_base_t base_index = SPARSE_INDEX_BASE_ZERO
    cdef MKL_INT64* start
    cdef MKL_INT64* end
    cdef MKL_INT64* index
    cdef float* values
    cdef MKL_INT64 nptr

    if sptype:
        mkl_sparse_s_export = mkl_sparse_s_export_csr_64
        order = 'C'
    else:
        mkl_sparse_s_export = mkl_sparse_s_export_csc_64
        order = 'F'

    export_status = mkl_sparse_s_export(
            A,
            &base_index,
            &rows,
            &cols,
            &start,
            &end,
            &index,
            &values
            )

    if sptype:
        nptr = rows
    else:
        nptr = cols

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros(
            (rows, cols),
            dtype=np.float32,
            order=order
            )
    cdef int nnz = start[nptr]
    if sptype:
        spmatrix_to_dense_csr(
                <float[:nnz]> values,
                <long[:nnz]> index,
                <long[:nptr]> start,
                <long[:nptr]> end,
                rows,
                cols,
                result
                )
    else:
        spmatrix_to_dense_csc(
                <float[:nnz]> values,
                <long[:nnz]> index,
                <long[:nptr]> start,
                <long[:nptr]> end,
                rows,
                cols,
                result
                )

    return result


# Sparse routines
cdef np.ndarray[np.float32_t, ndim=1] mkl_sparse_mv_64(
        const float[:] data,
        const long[:] indices,
        const long[:] indptr,
        int nrow,
        int ncol,
        bint sptype,
        const float[:] vec,
        bint transpose
        ):

    cdef sparse_operation_t operation
    cdef MKL_INT64 shape_out
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
        shape_out = ncol
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE
        shape_out = nrow

    cdef sparse_matrix_t A = to_mkl_spmatrix(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            sptype
            )

    cdef float alpha = 1.
    cdef float beta = 0.
    cdef matrix_descr mat_descript
    mat_descript.type = SPARSE_MATRIX_TYPE_GENERAL

    cdef np.ndarray[np.float32_t, ndim=1] result = np.zeros(shape_out, dtype=np.float32)
    cdef float[:] result_view = result

    status = mkl_sparse_s_mv_64(
            operation,
            alpha,
            A,
            mat_descript,
            &vec[0],
            beta,
            &result_view[0]
            )

    return result

cpdef np.ndarray[np.float32_t, ndim=2] mkl_sparse_gram(
        const float[:] data,
        const long[:] indices,
        const long[:] indptr,
        int nrow,
        int ncol,
        bint transpose
        ):

    cdef sparse_operation_t operation
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE

    cdef sparse_matrix_t A = to_mkl_spmatrix(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            1
            )
    mkl_sparse_order_64(A)
    cdef sparse_matrix_t C

    status = mkl_sparse_syrk_64(
            operation,
            A,
            &C
            )

    cdef np.ndarray[np.float32_t, ndim=2] result = to_python_dmatrix(C, 1)

    mkl_sparse_destroy_64(A)
    mkl_sparse_destroy_64(C)

    return result

cdef np.ndarray[np.float32_t, ndim=2] mkl_sparse_mm(
        const float[:] data,
        const long[:] indices,
        const long[:] indptr,
        int nrow,
        int ncol,
        bint sptype,
        const float[:,:] mat,
        bint transpose
        ):

    cdef sparse_operation_t operation
    cdef MKL_INT64 shape_out
    if transpose:
        operation = SPARSE_OPERATION_TRANSPOSE
        shape_out = ncol
    else:
        operation = SPARSE_OPERATION_NON_TRANSPOSE
        shape_out = nrow

    cdef sparse_matrix_t A = to_mkl_spmatrix(
            data,
            indices,
            indptr,
            nrow,
            ncol,
            sptype
            )

    cdef float alpha = 1.
    cdef float beta = 0.
    cdef matrix_descr mat_descript
    mat_descript.type = SPARSE_MATRIX_TYPE_GENERAL

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros(
            (shape_out, mat.shape[1]),
            dtype=np.float32
            )
    cdef float[:,:] result_view = result

    status = mkl_sparse_s_mm_64(
            operation,
            alpha,
            A,
            mat_descript,
            SPARSE_LAYOUT_ROW_MAJOR,
            &mat[0,0],
            mat.shape[1],
            mat.shape[1],
            beta,
            &result_view[0,0],
            mat.shape[1]
            )

    return result

# Dense routines
cdef np.ndarray[np.float32_t, ndim=2] cblas_ger(
        const float[:] x,
        const float[:] y
        ):

    cdef CBLAS_LAYOUT Layout = CblasRowMajor
    cdef MKL_INT64 m = x.shape[0]
    cdef MKL_INT64 n = y.shape[0]
    cdef float alpha = 1.
    cdef MKL_INT64 incx = 1
    cdef MKL_INT64 incy = 1
    cdef MKL_INT64 lda = n

    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros(
            (m, n),
            dtype=np.float32
            )
    cdef float[:,:] result_view = result
    cblas_sger_64(
            Layout,
            m,
            n,
            alpha,
            &x[0],
            incx,
            &y[0],
            incy,
            &result_view[0,0],
            lda
            )

    return result
