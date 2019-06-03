#ifdef SWIGCSHARP

%{
    struct WrapperArray
	{
	    void * _data;
		size_t _size;
		size_t _itemsize;
	};
%}

%pragma(csharp) imclasscode=%{ 
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)] 
    public struct WrapperArray 
    { 
        public System.IntPtr _data; 
        public ulong _size;
		public ulong _itemsize;
        public WrapperArray(System.IntPtr in_data, ulong in_size, ulong in_itemsize = 0) { _data = in_data; _size = in_size; _itemsize = in_itemsize; } 
    } 
%} 


%typemap(ctype) ByteArray "WrapperArray"
%typemap(imtype) ByteArray "WrapperArray"
%typemap(cstype) ByteArray "byte[]"
%typemap(in) ByteArray {
    $1.SetData((std::uint8_t*)$input._data, $input._size);
}
%typemap(out) ByteArray {
    $result._data = $1.Data();
	$result._size = $1.Length();
}
%typemap(csin,
         pre="unsafe { fixed(byte* ptr$csinput = $csinput) { $modulePINVOKE.WrapperArray temp$csinput = new $modulePINVOKE.WrapperArray( (System.IntPtr)ptr$csinput, (ulong)$csinput.Length );",
		 terminator="} }"
		 ) ByteArray %{ temp$csinput %}

%typemap(csvarin) ByteArray %{ 
    set {
	     unsafe { fixed(byte* ptr$csinput = $csinput) 
		     {
			     $modulePINVOKE.WrapperArray temp$csinput = new $modulePINVOKE.WrapperArray( (System.IntPtr)ptr$csinput, (ulong)$csinput.Length );
				 $imcall;
		     }
		 }
	}
%}

%typemap(csout, excode=SWIGEXCODE) ByteArray %{
    $modulePINVOKE.WrapperArray data = $imcall;$excode
    byte[] ret = new byte[data._size];
	System.Runtime.InteropServices.Marshal.Copy(data._data, ret, 0, (int)data._size);
    return ret; 
%}

%typemap(csvarout) ByteArray %{
    get {
	    $modulePINVOKE.WrapperArray data = $imcall;
        byte[] ret = new byte[data._size];
	    System.Runtime.InteropServices.Marshal.Copy(data._data, ret, 0, (int)data._size);
        return ret; 
	}
%}

%typemap(ctype) std::shared_ptr<QueryResult> "WrapperArray"
%typemap(imtype) std::shared_ptr<QueryResult> "WrapperArray"
%typemap(cstype) std::shared_ptr<QueryResult> "BasicResult[]"
%typemap(out) std::shared_ptr<QueryResult> {
    $result._data = new BasicResult[$1->GetResultNum()];
	$result._size = $1->GetResultNum();
	$result._itemsize = sizeof(BasicResult);
    std::memcpy($result._data, $1->GetResults(), sizeof(BasicResult) * $1->GetResultNum());
}
%typemap(csout, excode=SWIGEXCODE) std::shared_ptr<QueryResult> {
    $modulePINVOKE.WrapperArray data = $imcall;
    BasicResult[] ret = new BasicResult[data._size];
	System.IntPtr ptr = data._data;
    for (ulong i = 0; i < data._size; i++) {
		ret[i] = new BasicResult(ptr, true);
		ptr += (int)data._itemsize;
	}
	$excode
	return ret;
}

%typemap(ctype) std::shared_ptr<RemoteSearchResult> "WrapperArray"
%typemap(imtype) std::shared_ptr<RemoteSearchResult> "WrapperArray"
%typemap(cstype) std::shared_ptr<RemoteSearchResult> "BasicResult[]"
%typemap(out) std::shared_ptr<RemoteSearchResult> {
    int combinelen = 0;
    int nodelen = (int)(($1->m_allIndexResults).size());
    for (int i = 0; i < nodelen; i++) {
        combinelen += $1->m_allIndexResults[i].m_results.GetResultNum();
    }
    $result._data = new BasicResult[combinelen];
	$result._size = combinelen;
	$result._itemsize = sizeof(BasicResult);
	size_t copyed = 0;
	for (int i = 0; i < nodelen; i++) {
	    auto& queryResult = $1->m_allIndexResults[i].m_results;
	    std::memcpy((char*)($result._data) + copyed, queryResult.GetResults(), sizeof(BasicResult) * queryResult.GetResultNum());
	    copyed += sizeof(BasicResult) * queryResult.GetResultNum();
	}
}
%typemap(csout, excode=SWIGEXCODE) std::shared_ptr<RemoteSearchResult> {
    $modulePINVOKE.WrapperArray data = $imcall;
    BasicResult[] ret = new BasicResult[data._size];
	System.IntPtr ptr = data._data;
    for (ulong i = 0; i < data._size; i++) {
		ret[i] = new BasicResult(ptr, true);
		ptr += (int)data._itemsize;
	}
	$excode
	return ret;
}
#endif
