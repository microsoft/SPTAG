#include "pch.h"
#include "AnnIndex.h"
#if __has_include("AnnIndex.g.cpp")
#include "AnnIndex.g.cpp"
#endif
#include "SearchResult.g.cpp"
namespace winrt::SPTAG::implementation
{

  float* CopyArray(const EmbeddingVector& data) {
    auto copy = new float[data.size()];
    for (auto i = 0u; i < data.size(); i++) copy[i] = data[i];
    return copy;
  }

  sptag::ByteArray GetByteArray(const EmbeddingVector& data) {
    auto copy = CopyArray(data);
    auto byteArray = sptag::ByteArray(reinterpret_cast<byte*>(copy), data.size() * sizeof(float) / sizeof(byte), true);
    return byteArray;
  }

  sptag::ByteArray GetByteArray(const winrt::array_view<const uint8_t>& data) {
    auto vec = new uint8_t[data.size()];
    for (auto i = 0u; i < data.size(); i++) vec[i] = data[i];
    auto byteArray = sptag::ByteArray(vec, data.size(), true);
    return byteArray;
  }

  static constexpr auto GuidType = static_cast<JsonValueType>(static_cast<int>(JsonValueType::Object) + 1);

  sptag::ByteArray GetByteArray(const winrt::guid& data) {
    size_t bufferLength = sizeof(uint8_t) + sizeof(data);
    auto result = new uint8_t[bufferLength];
    *result = static_cast<uint8_t>(GuidType);
    memcpy_s(result + sizeof(uint8_t), sizeof(data), &data, sizeof(data));
    return sptag::ByteArray(result, bufferLength, true);
  }

  sptag::ByteArray GetByteArray(const JsonValue& data) {
    size_t bufferLength = sizeof(uint8_t); // 1 byte for value type marker
    uint8_t marker{ static_cast<uint8_t>(data.ValueType()) };
    switch (data.ValueType()) {
    case JsonValueType::Null: {
      auto result = new uint8_t[bufferLength];
      *result = marker;
      return sptag::ByteArray(result, bufferLength, true);
      break; // no additional data needed
    }
    case JsonValueType::Boolean: {
      marker |= data.GetBoolean() ? 0b10000000 : 0;
      auto result = new uint8_t[bufferLength];
      *result = marker;
      return sptag::ByteArray(result, bufferLength, true);
      break; // we encode the bool value in the marker
    }
    case JsonValueType::Number: {
      auto v = data.GetNumber();
      bufferLength += sizeof(double);
      auto result = new uint8_t[bufferLength];
      result[0] = marker;
      memcpy_s(result + 1, sizeof(double), &v, sizeof(double));
      auto byteArray = sptag::ByteArray(result, bufferLength, true);
      return byteArray;
      break;
    }
    case JsonValueType::String: {
      auto str = data.GetString();
      int const size = WINRT_IMPL_WideCharToMultiByte(65001 /*CP_UTF8*/, 0, str.data(), static_cast<int32_t>(str.size()), nullptr, 0, nullptr, nullptr);

      if (size == 0) {
        return{};
      }
      bufferLength += size;
      auto result = new uint8_t[bufferLength + 1];
      result[0] = marker;
      static_assert(sizeof(char) == sizeof(uint8_t));
      WINRT_IMPL_WideCharToMultiByte(65001 /*CP_UTF8*/, 0, str.data(), static_cast<int32_t>(str.size()), reinterpret_cast<char*>(result + 1), size, nullptr, nullptr);
      auto byteArray = sptag::ByteArray(result, bufferLength, true);
      return byteArray;
      break;
    }
    default:
      throw winrt::hresult_invalid_argument{}; break;
    }
  }

  sptag::ByteArray GetByteArray(const winrt::hstring& data) {
    int const size = WINRT_IMPL_WideCharToMultiByte(65001 /*CP_UTF8*/, 0, data.data(), static_cast<int32_t>(data.size()), nullptr, 0, nullptr, nullptr);

    if (size == 0) {
      return{};
    }
    auto result = new uint8_t[size + 1];
    static_assert(sizeof(char) == sizeof(uint8_t));
    WINRT_IMPL_WideCharToMultiByte(65001 /*CP_UTF8*/, 0, data.data(), static_cast<int32_t>(data.size()), reinterpret_cast<char*>(result), size, nullptr, nullptr);
    auto byteArray = sptag::ByteArray(result, size, true);
    return byteArray;
  }

  SPTAG::SearchResult AnnIndex::GetResultFromMetadata(const sptag::BasicResult& r) const {
    auto ba = r.Meta;

    auto kind = static_cast<JsonValueType>(r.Meta.Data()[0] & 0b00001111);
    auto data = ba.Data() + sizeof(uint8_t);
    auto dataSize = ba.Length() - sizeof(uint8_t);
    JsonValue value{ nullptr };
    switch (kind) {
    case JsonValueType::Null:
      value = JsonValue::CreateNullValue(); break;
    case JsonValueType::Boolean:
      value = JsonValue::CreateBooleanValue((ba.Data()[0] & 0b10000000) != 0); break;
    case JsonValueType::Number:
      value = JsonValue::CreateNumberValue(*reinterpret_cast<double*>(data)); break;
    case JsonValueType::String: {
      int const size = WINRT_IMPL_MultiByteToWideChar(65001 /*CP_UTF8*/, 0, reinterpret_cast<char*>(data), static_cast<int32_t>(dataSize), nullptr, 0);

      if (size == 0) {
        throw winrt::hresult_invalid_argument{};
      }

      impl::hstring_builder result(size);
      WINRT_VERIFY_(size, WINRT_IMPL_MultiByteToWideChar(65001 /*CP_UTF8*/, 0, reinterpret_cast<char*>(data), static_cast<int32_t>(dataSize), result.data(), size));

      auto meta = result.to_hstring();
      value = JsonValue::CreateStringValue(meta);
      break;
    }
#pragma warning(push)
#pragma warning(disable:4063)
    case GuidType: {
#pragma warning(pop)
      winrt::guid guid{};
      memcpy_s(&guid, sizeof(guid), data, dataSize);
      return winrt::make<SearchResult>(guid, r.Dist);
    }
    default:
      throw winrt::hresult_invalid_argument{};
    }
    return winrt::make<SearchResult>(value, r.Dist);
  }


  winrt::Windows::Foundation::Collections::IVector<SPTAG::SearchResult> AnnIndex::Search(EmbeddingVector p_data, uint8_t p_resultNum) const {
    auto vec = std::vector<SPTAG::SearchResult>{};
    p_resultNum = static_cast<decltype(p_resultNum)>((std::min)(static_cast<sptag::SizeType>(p_resultNum), m_index->GetNumSamples()));
    auto results = std::make_shared<sptag::QueryResult>(p_data.data(), p_resultNum, true);

    if (nullptr != m_index) {
      m_index->SearchIndex(*results);
    }
    for (const auto& r : *results) {
      auto sr = GetResultFromMetadata(r);
      vec.push_back(sr);
    }
    return winrt::single_threaded_vector<SPTAG::SearchResult>(std::move(vec));
  }

  void AnnIndex::Load(winrt::Windows::Storage::StorageFile file) {
    auto path = file.Path();
    if (sptag::ErrorCode::Success != sptag::VectorIndex::LoadIndexFromFile(winrt::to_string(path), m_index)) {
      throw winrt::hresult_error{};
    }
  }

  void AnnIndex::Save(winrt::Windows::Storage::StorageFile file) {
    auto path = file.Path();
    if (sptag::ErrorCode::Success != m_index->SaveIndexToFile(winrt::to_string(path))) {
      throw winrt::hresult_error{};
    }
  }


  template<typename T>
  void AnnIndex::_AddWithMetadataImpl(EmbeddingVector p_data, T metadata) {
    if (m_dimension == 0) {
      m_dimension = p_data.size();
    } else if (m_dimension != static_cast<decltype(m_dimension)>(p_data.size())) {
      throw winrt::hresult_invalid_argument{};
    }
    int p_num{ 1 };
    auto byteArray = GetByteArray(p_data);
    std::shared_ptr<sptag::VectorSet> vectors(new sptag::BasicVectorSet(byteArray,
      m_inputValueType,
      static_cast<sptag::DimensionType>(m_dimension),
      static_cast<sptag::SizeType>(p_num)));


    sptag::ByteArray p_meta = GetByteArray(metadata);
    bool p_withMetaIndex{ true };
    bool p_normalized{ true };

    std::uint64_t* offsets = new std::uint64_t[p_num + 1]{ 0 };
    if (!sptag::MetadataSet::GetMetadataOffsets(p_meta.Data(), p_meta.Length(), offsets, p_num + 1, '\n')) throw winrt::hresult_invalid_argument{};
    std::shared_ptr<sptag::MetadataSet> meta(new sptag::MemMetadataSet(p_meta, sptag::ByteArray((std::uint8_t*)offsets, (p_num + 1) * sizeof(std::uint64_t), true), (sptag::SizeType)p_num));
    if (sptag::ErrorCode::Success != m_index->AddIndex(vectors, meta, p_withMetaIndex, p_normalized)) {
      throw winrt::hresult_error(E_UNEXPECTED);
    }
  }

  void AnnIndex::AddWithMetadata(EmbeddingVector p_data, JsonValue metadata) {
    _AddWithMetadataImpl(p_data, metadata);
  }

  void AnnIndex::AddWithMetadata(EmbeddingVector p_data, winrt::hstring metadata) {
    AddWithMetadata(p_data, JsonValue::CreateStringValue(metadata));
  }

  void AnnIndex::AddWithMetadata(EmbeddingVector p_data, const winrt::guid& guid) {
    _AddWithMetadataImpl(p_data, guid);
  }



}
