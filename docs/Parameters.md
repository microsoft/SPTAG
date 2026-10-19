# In-memory index parameters

[SDK quick start](QuickStart.md) | [API contracts](SDK.md) | [Documentation index](README.md)

These defaults describe the current BKT/KDT SDK, not SPANN sections or a
benchmark harness. The source of truth is the
[BKT definitions](../AnnService/inc/Core/BKT/ParameterDefinitionList.h) and
[KDT definitions](../AnnService/inc/Core/KDT/ParameterDefinitionList.h).
Use canonical parameter spelling even where an implementation accepts case-insensitive names.

## Set parameters

Python values are strings; set build parameters before `Build`:

```python
index.SetBuildParam("DistCalcMethod", "L2", "Index")
index.SetBuildParam("NumberOfThreads", "4", "Index")
index.SetSearchParam("MaxCheck", "8192", "Index")
```

In C++, call `SetParameter(name, value)` and check its `ErrorCode`.
Python's legacy setters do not return an acceptance status.

## Common BKT/KDT parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `Samples` | int | BKT: 1000; KDT: 100 | Samples used when constructing partition trees |
| `TPTNumber` | int | 32 | Trees used to initialize graph construction |
| `TPTLeafSize` | int | 2000 | Leaf size for graph-initialization trees |
| `NeighborhoodSize` | int | 32 | Graph neighborhood size |
| `GraphNeighborhoodScale` | float | 2.0 | Neighborhood expansion during construction |
| `CEF` | int | 1000 | Candidate count for relative-neighborhood graph construction |
| `MaxCheckForRefineGraph` | int | 8192 | Search budget during graph refinement |
| `NumberOfThreads` | int | 1 | Index worker-thread setting |
| `DistCalcMethod` | string | Cosine | Distance method; the SDK examples explicitly use L2 |
| `MaxCheck` | int | 8192 | Query search budget; tune against recall and latency |

## Algorithm-specific parameters

| Algorithm | Parameter | Type | Default | Meaning |
| --- | --- | --- | --- | --- |
| BKT | `BKTNumber` | int | 1 | Number of balanced k-means trees |
| BKT | `BKTKmeansK` | int | 32 | Branching factor |
| KDT | `KDTNumber` | int | 1 | Number of kd-trees |

## Tuning

Start with the defaults and a ground-truth query set from your application.
Increase `MaxCheck` to explore the recall/latency tradeoff. Construction
parameters such as `TPTNumber`, `NeighborhoodSize`, `CEF`, and refinement
budget also affect index quality, build cost, and memory. Change one group at
a time and record the source revision, data, hardware, and configuration.
See [benchmark methodology](../BENCHMARK_GUIDE.md).

The old inline NNI/CLI tutorial used an obsolete search invocation and is no
longer a runnable SDK guide. It remains in
[Git history](https://github.com/microsoft/SPTAG/blob/49ddc8bd/docs/Parameters.md).
The separate [NNI experiment](../Tools/nni-auto-tune/README.md) is a legacy
integration with its own dependencies, not part of the SDK baseline.
