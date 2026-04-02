%module SPTAG

%{
#include "inc/CoreInterface.h"
#include "inc/Core/ResultIterator.h"
%}

%include <std_shared_ptr.i>
%include <stdint.i>
%include <std_map.i>
%include <std_vector.i>

%shared_ptr(AnnIndex)
%shared_ptr(TenantIndexManager)
%shared_ptr(QueryResult)
%shared_ptr(ResultIterator)

// Typemap for GetTenantIds: convert OUTPUT int* to Python list
%typemap(in,numinputs=0) (int* p_tenants, int* p_count) {
    $1 = (int*)malloc(10000 * sizeof(int));  // Allocate buffer for up to 10000 tenants
    $2 = (int*)malloc(sizeof(int));
}

%typemap(argout) (int* p_tenants, int* p_count) {
    PyObject* tenantList = PyList_New(*$2);
    for (int i = 0; i < *$2; i++) {
        PyList_SetItem(tenantList, i, PyLong_FromLong($1[i]));
    }
    $result = tenantList;
    free($1);
    free($2);
}

%include "PythonCommon.i"

%{
#define SWIG_FILE_WITH_INIT
%}

%include "CoreInterface.h"

%pythoncode %{
def _tenant_build_from_numpy(self, vectors, tenant_ids, with_meta_index=True, normalized=False):
    """
    Build per-tenant independent indices from numpy vectors and tenant id array.

    Args:
        vectors: 2D numpy array with shape (N, D), dtype float32 compatible.
        tenant_ids: 1D array-like with length N, each value is an integer tenant id.
        with_meta_index: whether to build metadata index in underlying SPTAG index.
        normalized: whether vectors are already normalized.
    """
    import numpy as _np

    vecs = _np.ascontiguousarray(vectors, dtype=_np.float32)
    tids = _np.asarray(tenant_ids)

    if vecs.ndim != 2:
        raise ValueError("vectors must be a 2D array with shape (N, D)")
    if tids.ndim != 1:
        raise ValueError("tenant_ids must be a 1D array")
    if vecs.shape[0] != tids.shape[0]:
        raise ValueError("vectors row count must equal tenant_ids length")

    # Metadata format expected by C++ BuildFromData: one integer tenant id per line.
    metadata = ("\n".join(str(int(x)) for x in tids.tolist()) + "\n").encode("utf-8")

    return self.BuildFromData(vecs, metadata, int(vecs.shape[0]), bool(with_meta_index), bool(normalized))


def _create_tenant_index_manager(dimension, algo_type="BKT", value_type="Float"):
    """Factory for multi-tenant index manager in Python API."""
    return TenantIndexManager(int(dimension), str(algo_type), str(value_type))


# Python API additions for multi-tenant index construction.
TenantIndexManager.BuildFromNumpy = _tenant_build_from_numpy
CreateTenantIndexManager = _create_tenant_index_manager
%}

%include "../../AnnService/inc/Core/ResultIterator.h"