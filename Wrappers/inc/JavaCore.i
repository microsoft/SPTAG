%module JAVASPTAG

%{
#include "inc/CoreInterface.h"
#include "inc/Core/ResultIterator.h"
%}

%include <std_shared_ptr.i>
%include <stdint.i>
%shared_ptr(AnnIndex)
%shared_ptr(QueryResult)
%shared_ptr(ResultIterator)
%include "JavaCommon.i"

%{
#define SWIG_FILE_WITH_INIT
%}

%include "CoreInterface.h"
%include "../../AnnService/inc/Core/SearchResult.h"
%include "../../AnnService/inc/Core/ResultIterator.h"
