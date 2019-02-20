#ifdef SWIGPYTHON

%typemap(out) std::shared_ptr<QueryResult>
%{
    {
        $result = PyTuple_New(3);
        int resNum = $1->GetResultNum();
        auto dstVecIDs = PyList_New(resNum);
        auto dstVecDists = PyList_New(resNum);
        auto dstMetadata = PyList_New(resNum);
        int i = 0;
        for (const auto& res : *($1))
        {
            PyList_SetItem(dstVecIDs, i, PyInt_FromLong(res.VID));
            PyList_SetItem(dstVecDists, i, PyFloat_FromDouble(res.Dist));
            i++;
        }
    
        if ($1->WithMeta()) 
        {
            for (i = 0; i < resNum; ++i)
            {
                const auto& metadata = $1->GetMetadata(i);
                PyList_SetItem(dstMetadata, i, PyBytes_FromStringAndSize(reinterpret_cast<const char*>(metadata.Data()),
                                                                         metadata.Length()));
            }
        }

        PyTuple_SetItem($result, 0, dstVecIDs);
        PyTuple_SetItem($result, 1, dstVecDists);
        PyTuple_SetItem($result, 2, dstMetadata);
    }
%}

%typemap(out) std::shared_ptr<RemoteSearchResult>
%{
    {
        $result = PyTuple_New(3);
        auto dstVecIDs = PyList_New(0);
        auto dstVecDists = PyList_New(0);
        auto dstMetadata = PyList_New(0);
        for (const auto& indexRes : $1->m_allIndexResults)
        {
            for (const auto& res : indexRes.m_results)
            {
                PyList_Append(dstVecIDs, PyInt_FromLong(res.VID));
                PyList_Append(dstVecDists, PyFloat_FromDouble(res.Dist));
            }

            if (indexRes.m_results.WithMeta()) 
            {
                for (int i = 0; i < indexRes.m_results.GetResultNum(); ++i)
                {
                    const auto& metadata = indexRes.m_results.GetMetadata(i);
                    PyList_Append(dstMetadata, PyBytes_FromStringAndSize(reinterpret_cast<const char*>(metadata.Data()),
                                                                         metadata.Length()));
                }
            }
        }
        PyTuple_SetItem($result, 0, dstVecIDs);
        PyTuple_SetItem($result, 1, dstVecDists);
        PyTuple_SetItem($result, 2, dstMetadata);
    }
%}

%typemap(in) ByteArray
%{
    $1 = SPTAG::ByteArray((std::uint8_t*)PyBytes_AsString($input), PyBytes_Size($input), false);
%}

#endif
