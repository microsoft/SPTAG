#include "AnnIndex.h"
#include "pch.h"
#if __has_include("AnnIndex.g.cpp")
#include "AnnIndex.g.cpp"
#endif
#include "SearchResult.g.cpp"

namespace winrt::SPTAG::implementation
{
template <typename T, typename = std::enable_if_t<std::is_pod_v<T>>>
sptag::ByteArray GetByteArray(const winrt::array_view<const T> &data)
{
    auto copy = new T[data.size()];
    for (auto i = 0u; i < data.size(); i++)
        copy[i] = data[i];
    const auto byteSize = data.size() * sizeof(data.at(0)) / sizeof(byte);
    auto byteArray = sptag::ByteArray(reinterpret_cast<byte *>(copy), byteSize, true);
    return byteArray;
}

SPTAG::SearchResult AnnIndex::GetResultFromMetadata(const sptag::BasicResult &r) const
{
    return winrt::make<SearchResult>(r.Meta.Data(), r.Meta.Length(), r.Dist);
}

winrt::Windows::Foundation::Collections::IVector<SPTAG::SearchResult> AnnIndex::Search(EmbeddingVector p_data,
                                                                                       uint32_t p_resultNum) const
{
    auto vec = std::vector<SPTAG::SearchResult>{};
    p_resultNum = (std::min)(static_cast<sptag::SizeType>(p_resultNum), m_index->GetNumSamples());
    auto results = std::make_shared<sptag::QueryResult>(p_data.data(), p_resultNum, true);

    if (nullptr != m_index)
    {
        m_index->SearchIndex(*results);
    }
    for (const auto &r : *results)
    {
        auto sr = GetResultFromMetadata(r);
        vec.push_back(sr);
    }
    return winrt::single_threaded_vector<SPTAG::SearchResult>(std::move(vec));
}

void AnnIndex::Load(winrt::Windows::Storage::StorageFile file)
{
    auto path = file.Path();
    if (sptag::ErrorCode::Success != sptag::VectorIndex::LoadIndexFromFile(winrt::to_string(path), m_index))
    {
        throw winrt::hresult_error{};
    }
}

void AnnIndex::Save(winrt::Windows::Storage::StorageFile file)
{
    auto path = file.Path();
    if (sptag::ErrorCode::Success != m_index->SaveIndexToFile(winrt::to_string(path)))
    {
        throw winrt::hresult_error{};
    }
}

template <typename T> void AnnIndex::_AddWithMetadataImpl(EmbeddingVector p_data, T metadata)
{
    if (m_dimension == 0)
    {
        m_dimension = p_data.size();
    }
    else if (m_dimension != static_cast<decltype(m_dimension)>(p_data.size()))
    {
        throw winrt::hresult_invalid_argument{};
    }
    int p_num{1};
    auto byteArray = GetByteArray(p_data);
    std::shared_ptr<sptag::VectorSet> vectors(new sptag::BasicVectorSet(byteArray, m_inputValueType,
                                                                        static_cast<sptag::DimensionType>(m_dimension),
                                                                        static_cast<sptag::SizeType>(p_num)));

    sptag::ByteArray p_meta = GetByteArray(metadata);
    bool p_withMetaIndex{true};
    bool p_normalized{true};

    std::uint64_t *offsets = new std::uint64_t[p_num + 1]{0, metadata.size() * sizeof(metadata[0])};
    sptag::ByteArray offsetsArray(reinterpret_cast<std::uint8_t *>(offsets), p_num + 1, true);
    std::shared_ptr<sptag::MemMetadataSet> meta(new sptag::MemMetadataSet(p_meta, offsetsArray, p_num));

    if (sptag::ErrorCode::Success != m_index->AddIndex(vectors, meta, p_withMetaIndex, p_normalized))
    {
        throw winrt::hresult_error(E_UNEXPECTED);
    }
}

void AnnIndex::AddWithMetadata(array_view<float const> data, array_view<uint8_t const> metadata)
{
    _AddWithMetadataImpl(data, metadata);
}
} // namespace winrt::SPTAG::implementation
