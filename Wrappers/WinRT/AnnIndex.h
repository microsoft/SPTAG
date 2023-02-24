#pragma once

#include "AnnIndex.g.h"
#include "SearchResult.g.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SearchQuery.h"
#include <array>
#include <vector>

#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Data.Json.h>

template<typename T>
struct ReadOnlyProperty {
  ReadOnlyProperty() = default;
  ReadOnlyProperty(const T& value) : m_value(value) {}
  T operator()() const noexcept { return m_value; }
  T m_value{};
};

template<typename T>
struct Property : ReadOnlyProperty<T> {
  void operator()(const T& value) { m_value = value; }
};

namespace sptag = ::SPTAG;
namespace winrt::SPTAG::implementation
{
  using namespace winrt::Windows::Data::Json;
  using EmbeddingVector = winrt::array_view<const float>;

  struct SearchResult : SearchResultT<SearchResult> {
    ReadOnlyProperty<JsonValue> Metadata;
    ReadOnlyProperty<float> Distance;
    ReadOnlyProperty<winrt::guid> Guid;

    SearchResult() = default;
    SearchResult(JsonValue v, float d) : Metadata(v), Distance(d) {}
    SearchResult(winrt::guid g, float d) : Metadata(nullptr), Distance(d), Guid(g) {}
  };


  struct AnnIndex : AnnIndexT<AnnIndex>
  {
    sptag::DimensionType m_dimension{ };
    sptag::VectorValueType m_inputValueType{ sptag::VectorValueType::Float };


    AnnIndex() {
      sptag::GetLogger().reset(new sptag::Helper::SimpleLogger(sptag::Helper::LogLevel::LL_Empty));
      m_index = sptag::VectorIndex::CreateInstance(sptag::IndexAlgoType::BKT, sptag::GetEnumValueType<float>());
    }

    void Save(winrt::Windows::Storage::StorageFile file);
    void AddWithMetadata(EmbeddingVector p_data, JsonValue metadata);
    void AddWithMetadata(EmbeddingVector p_data, winrt::hstring metadata);
    void AddWithMetadata(EmbeddingVector p_data, const winrt::guid& guid);
    void Load(winrt::Windows::Storage::StorageFile file);

    SPTAG::SearchResult GetResultFromMetadata(const sptag::BasicResult& r) const;

    winrt::Windows::Foundation::Collections::IVector<SPTAG::SearchResult> Search(EmbeddingVector p_data, uint32_t p_resultNum) const;

    std::shared_ptr<sptag::VectorIndex> m_index;
    template<typename T>
    void _AddWithMetadataImpl(EmbeddingVector p_data, T metadata);
  };
}

namespace winrt::SPTAG::factory_implementation
{
    struct AnnIndex : AnnIndexT<AnnIndex, implementation::AnnIndex>
    {
    };
}
