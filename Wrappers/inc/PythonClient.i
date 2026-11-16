%module SPTAGClient

%{
#include "inc/ClientInterface.h"
#include <exception>
%}

%include <std_shared_ptr.i>
%shared_ptr(AnnClient)
%shared_ptr(RemoteSearchResult)
%shared_ptr(RemoteTenantResult)
%include "PythonCommon.i"
%include <exception.i>

// Avoid overload dispatch bypassing ByteArray's buffer input typemap.
%feature("compactdefaultargs") AnnClient::CreateTenant;

%define TENANT_ALLOW_THREADS(Method)
%exception AnnClient::Method {
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try { $action }
    catch (...) { error = std::current_exception(); }
    Py_END_ALLOW_THREADS
    if (error) {
        try { std::rethrow_exception(error); }
        catch (const std::exception& e) { SWIG_exception(SWIG_RuntimeError, e.what()); }
        catch (...) { SWIG_exception(SWIG_RuntimeError, "Tenant operation failed"); }
    }
}
%enddef

TENANT_ALLOW_THREADS(CreateTenant)
TENANT_ALLOW_THREADS(DeleteTenant)
TENANT_ALLOW_THREADS(InsertTenant)
TENANT_ALLOW_THREADS(DeleteTenantVector)
TENANT_ALLOW_THREADS(SearchTenant)

%typemap(out) std::shared_ptr<RemoteTenantResult>
%{
    {
        const auto& response = *$1;
        PyObject* ids = PyList_New(response.results.size());
        PyObject* distances = PyList_New(response.results.size());
        if (!ids || !distances)
        {
            Py_XDECREF(ids);
            Py_XDECREF(distances);
            SWIG_fail;
        }
        for (size_t i = 0; i < response.results.size(); ++i)
        {
            PyObject* id = PyLong_FromLong(response.results[i].VID);
            PyObject* distance = PyFloat_FromDouble(response.results[i].Dist);
            if (!id || !distance)
            {
                Py_XDECREF(id);
                Py_XDECREF(distance);
                Py_DECREF(ids);
                Py_DECREF(distances);
                SWIG_fail;
            }
            PyList_SET_ITEM(ids, i, id);
            PyList_SET_ITEM(distances, i, distance);
        }
        $result = Py_BuildValue("{s:i,s:s,s:i,s:i,s:O,s:O}",
            "status", static_cast<int>(response.status), "status_name", response.StatusName(),
            "first_id", response.firstID, "count", response.count, "ids", ids, "distances", distances);
        Py_DECREF(ids);
        Py_DECREF(distances);
    }
%}

%{
#define SWIG_FILE_WITH_INIT
%}

%include "ClientInterface.h"