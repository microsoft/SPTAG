#pragma once

#include "AnnIndex.g.h"
#include "SearchResult.g.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SearchQuery.h"
#include <array>
#include <vector>

#include <winrt/Windows.Storage.h>

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
  using EmbeddingVector = winrt::array_view<const float>;

  struct SearchResult : SearchResultT<SearchResult> {
    winrt::com_array<uint8_t> Metadata() { return winrt::com_array(m_metadata); }
    std::vector<uint8_t> m_metadata;
    ReadOnlyProperty<float> Distance;

    SearchResult() = default;
    SearchResult(winrt::array_view<uint8_t> metadata, float d) : m_metadata(metadata.begin(), metadata.end()), Distance(d) {}
    SearchResult(uint8_t* metadata, size_t length, float d) : m_metadata(metadata, metadata + length), Distance(d) {}
  };


  struct AnnIndex : AnnIndexT<AnnIndex>
  {
    sptag::DimensionType m_dimension{ };
    sptag::VectorValueType m_inputValueType{ sptag::VectorValueType::Float };


    AnnIndex() {
      sptag::GetLogger().reset(new sptag::Helper::SimpleLogger(sptag::Helper::LogLevel::LL_Empty));
      m_index = sptag::VectorIndex::CreateInstance(sptag::IndexAlgoType::BKT, sptag::GetEnumValueType<float>());
    }
    
    void AddWithMetadata(array_view<float const> data, array_view<uint8_t const> metadata);

    void Save(winrt::Windows::Storage::StorageFile file);
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
  struct AnnIndex : AnnIndexT<AnnIndex, implementation::AnnIndex> {};
}
