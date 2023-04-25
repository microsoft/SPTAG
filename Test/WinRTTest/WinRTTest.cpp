#include <iostream>
#include <winrt/SPTAG.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Security.Cryptography.h>
#include <winrt/Windows.Storage.Streams.h>

extern "C" __declspec(dllexport) winrt::SPTAG::LogLevel SPTAG_GetLoggerLevel() { return winrt::SPTAG::LogLevel::Empty; }

using namespace winrt;
using namespace Windows::Security::Cryptography;
using namespace Windows::Data::Json;

JsonObject CreateJsonObject(bool v) { auto js = JsonObject{}; js.Insert(L"value", JsonValue::CreateBooleanValue(v)); return js; }
JsonObject CreateJsonObject(std::wstring_view v) { auto js = JsonObject{}; js.Insert(L"value", JsonValue::CreateStringValue(v)); return js; }
template<size_t N> JsonObject CreateJsonObject(const wchar_t(&v)[N]) { return CreateJsonObject(winrt::hstring(v)); }
JsonObject CreateJsonObject(double v) { auto js = JsonObject{}; js.Insert(L"value", JsonValue::CreateNumberValue(v)); return js; }
JsonObject CreateJsonObject(nullptr_t) { auto js = JsonObject{}; js.Insert(L"value", JsonValue::CreateNullValue()); return js; }

template<typename T>
winrt::array_view<uint8_t> Serialize(const T& value) {
  JsonObject json = CreateJsonObject(value);
  auto str = json.Stringify();
  auto ibuffer = CryptographicBuffer::ConvertStringToBinary(str, BinaryStringEncoding::Utf16LE);
  auto start = ibuffer.data();
  winrt::com_array<uint8_t> _array(start, start + ibuffer.Length());
  return _array;
}

winrt::hstring Deserialize(winrt::array_view<uint8_t> v) {
  std::wstring_view sv { reinterpret_cast<wchar_t*>(v.begin()), v.size() / sizeof(wchar_t) };
  std::wstring wstr {sv};
  JsonObject js;
  if (JsonObject::TryParse(wstr, js)) {
    auto value = js.GetNamedValue(L"value");
    return value.Stringify();
  } else {
    return winrt::hstring{ wstr };
  }
}

int main()
{
  SPTAG::AnnIndex idx;
  using embedding_t = std::array<float, 1024>;
  auto b = CryptographicBuffer::ConvertStringToBinary(L"first one", BinaryStringEncoding::Utf16LE);
  idx.AddWithMetadata(embedding_t{ 1, 0 }, winrt::com_array<uint8_t>(b.data(), b.data() + b.Length()));
  idx.AddWithMetadata(embedding_t{ 0, 1, 0 }, Serialize(L"second one"));
  idx.AddWithMetadata(embedding_t{ 0, 0.5f, 0.7f, 0 }, Serialize(3.14));
  idx.AddWithMetadata(embedding_t{ 0, 0.7f, 0.5f, 0 }, Serialize(true));
  idx.AddWithMetadata(embedding_t{ 0, 0.7f, 0.8f, 0 }, Serialize(L"fifth"));

  auto res = idx.Search(embedding_t{ 0.f, 0.99f, 0.01f }, 12);
  for (const winrt::SPTAG::SearchResult& r : res) {
    std::wcout << Deserialize(r.Metadata());
    std::wcout << L" -- " << r.Distance() << L"\n";
  }

  auto folder = winrt::Windows::Storage::KnownFolders::DocumentsLibrary();
  auto file = folder.CreateFileAsync(L"vector_index", winrt::Windows::Storage::CreationCollisionOption::ReplaceExisting).get();
  idx.Save(file);

  SPTAG::AnnIndex idx2;
  idx2.Load(file);

}

