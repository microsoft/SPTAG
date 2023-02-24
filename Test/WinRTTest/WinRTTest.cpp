#include <iostream>
#include <winrt/SPTAG.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Data.Json.h>
#include <winrt/Windows.Storage.h>

extern "C" __declspec(dllexport) winrt::SPTAG::LogLevel SPTAG_GetLoggerLevel() { return winrt::SPTAG::LogLevel::Empty; }

using namespace winrt;
int main()
{
  SPTAG::AnnIndex idx;
  using embedding_t = std::array<float, 1024>;
  idx.AddWithMetadata(embedding_t{ 1, 0 }, L"first one");
  idx.AddWithMetadata(embedding_t{ 0, 1, 0 }, L"second one");
  idx.AddWithMetadata(embedding_t{ 0, 0.5f, 0.7f, 0 }, winrt::Windows::Data::Json::JsonValue::CreateBooleanValue(true));
  idx.AddWithMetadata(embedding_t{ 0, 0.7f, 0.5f, 0 }, winrt::Windows::Data::Json::JsonValue::CreateNumberValue(3.14));
  idx.AddWithMetadata(embedding_t{ 0, 0.7f, 0.8f, 0 }, winrt::guid_of<winrt::Windows::Foundation::IInspectable>());

  auto res = idx.Search(embedding_t{ 0.f, 0.99f, 0.01f }, 12);
  for (const winrt::SPTAG::SearchResult& r : res) {
    if (auto m = r.Metadata()) {
      std::wcout << r.Metadata().ToString();
    } else {
      std::wcout << winrt::to_hstring(r.Guid());
    }

    std::wcout << L" -- " << r.Distance() << L"\n";

  }
  auto folder = winrt::Windows::Storage::KnownFolders::DocumentsLibrary();
  auto file = folder.CreateFileAsync(L"vector_index", winrt::Windows::Storage::CreationCollisionOption::ReplaceExisting).get();
  idx.Save(file);

  SPTAG::AnnIndex idx2;
  idx2.Load(file);

  res = idx2.Search(embedding_t{ 0.f, 0.99f, 0.01f }, 12);
  for (const winrt::SPTAG::SearchResult& r : res) {
    if (auto m = r.Metadata()) {
      std::wcout << r.Metadata().ToString();
    } else {
      std::wcout << winrt::to_hstring(r.Guid());
    }

    std::wcout << L" -- " << r.Distance() << L"\n";

  }
}

