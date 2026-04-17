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
def _tenant_get_tag_routing_stats(self, tenant_id):
    """Return exact tag routing stats for one tenant as a sorted list of dicts."""
    import struct as _struct

    payload = self.GetTagRoutingStatsBlob(int(tenant_id))
    entry_format = "<Iii"
    entry_size = _struct.calcsize(entry_format)
    if len(payload) % entry_size != 0:
        raise ValueError("Unexpected GetTagRoutingStatsBlob payload length")

    return [
        {
            "tag": int(tag),
            "vector_count": int(vector_count),
            "posting_count": int(posting_count),
        }
        for tag, vector_count, posting_count in _struct.iter_unpack(entry_format, payload)
    ]


def _tenant_get_head_index_cache_state(self):
    """Return estimated HeadIndex usage together with live RSS and RSS high-water settings."""
    return {
        "estimated_usage_bytes": int(self.GetHeadIndexCacheUsage()),
        "safety_factor": float(self.GetHeadIndexCacheSafetyFactor()),
        "current_rss_bytes": int(self.GetCurrentRSSBytes()),
        "rss_high_watermark_bytes": int(self.GetRSSHighWaterMark()),
    }


def _tenant_estimate_pivot_build_plan(
    self,
    tags,
    num_vectors,
    num_tags_per_vec,
    max_nodes=5,
    recall_target=0.99,
    lambda_recall=10.0,
    estimated_recall=1.0,
    level_weights=None,
):
    """Estimate best pivot layer and node count before index build."""
    import json as _json
    import numpy as _np

    tag_arr = _np.ascontiguousarray(tags, dtype=_np.uint32)
    if tag_arr.ndim == 2:
        if int(tag_arr.shape[0]) != int(num_vectors) or int(tag_arr.shape[1]) != int(num_tags_per_vec):
            raise ValueError("tags shape does not match num_vectors/num_tags_per_vec")
    elif tag_arr.ndim == 1:
        expected = int(num_vectors) * int(num_tags_per_vec)
        if int(tag_arr.size) != expected:
            raise ValueError("flat tags length does not match num_vectors * num_tags_per_vec")
    else:
        raise ValueError("tags must be a 1D or 2D uint32 array")

    if level_weights is None:
        weights_csv = b""
    else:
        if len(level_weights) != int(num_tags_per_vec):
            raise ValueError("level_weights length must equal num_tags_per_vec")
        weights_csv = ",".join(str(float(v)) for v in level_weights).encode("utf-8")

    payload = self.EstimatePivotBuildPlan(
        tag_arr,
        int(num_vectors),
        int(num_tags_per_vec),
        int(max_nodes),
        float(recall_target),
        float(lambda_recall),
        float(estimated_recall),
        weights_csv,
    )
    if not payload:
        raise RuntimeError("EstimatePivotBuildPlan returned empty payload")
    return _json.loads(payload.decode("utf-8"))


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
TenantIndexManager.GetTagRoutingStats = _tenant_get_tag_routing_stats
TenantIndexManager.GetHeadIndexCacheState = _tenant_get_head_index_cache_state
TenantIndexManager.EstimatePivotPlan = _tenant_estimate_pivot_build_plan
CreateTenantIndexManager = _create_tenant_index_manager
%}

%include "../../AnnService/inc/Core/ResultIterator.h"