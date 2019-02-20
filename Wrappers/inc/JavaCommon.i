#ifdef SWIGJAVA

%typemap(jni) ByteArray "jbyteArray"
%typemap(jtype) ByteArray "byte[]"
%typemap(jstype) ByteArray "byte[]"
%typemap(in) ByteArray {
    $1.SetData((std::uint8_t*)JCALL2(GetByteArrayElements, jenv, $input, 0));
    $1.SetLength(JCALL1(GetArrayLength, jenv, $input));
}
%typemap(out) ByteArray {
    $result = JCALL1(NewByteArray, jenv, $1.Length());
    JCALL4(SetByteArrayRegion, jenv, $result, 0, $1.Length(), (jbyte *)$1.Data());
}
%typemap(javain) ByteArray "$javainput"
%typemap(javaout) ByteArray { return $jnicall; }

#endif
