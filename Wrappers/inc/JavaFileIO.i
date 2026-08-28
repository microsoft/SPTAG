%module JAVAFileIO

%{
#include "inc/FileIOInterface.h"
%}

%include <std_shared_ptr.i>
%include <stdint.i>
%include "std_string.i"
%include "std_vector.i"
// %shared_ptr(FileIOInterface)
%include "JavaCommon.i"

%include "../../AnnService/inc/Helper/KeyValueIO.h"
%include "../../AnnService/inc/Core/SPANN/ExtraFileController.h"
%include "FileIOInterface.h"