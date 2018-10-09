#include "inc/CoreInterface.h"
#include "inc/Helper/StringConvert.h"


AnnIndex::AnnIndex(SizeType p_dimension)
    : m_algoType(SpaceV::IndexAlgoType::BKT),
      m_inputValueType(SpaceV::VectorValueType::Float),
      m_dimension(p_dimension)
{
    m_inputVectorSize = SpaceV::GetValueTypeSize(m_inputValueType) * m_dimension;
}


AnnIndex::AnnIndex(const char* p_algoType, const char* p_valueType, SizeType p_dimension)
    : m_algoType(SpaceV::IndexAlgoType::Undefined),
      m_inputValueType(SpaceV::VectorValueType::Undefined),
      m_dimension(p_dimension)
{
    SpaceV::Helper::Convert::ConvertStringTo<SpaceV::IndexAlgoType>(p_algoType, m_algoType);
    SpaceV::Helper::Convert::ConvertStringTo<SpaceV::VectorValueType>(p_valueType, m_inputValueType);
    m_inputVectorSize = SpaceV::GetValueTypeSize(m_inputValueType) * m_dimension;
}


AnnIndex::AnnIndex(const std::shared_ptr<SpaceV::VectorIndex>& p_index)
    : m_algoType(p_index->GetIndexAlgoType()),
      m_inputValueType(p_index->AcceptableQueryValueType()),
      m_dimension(p_index->GetFeatureDim()),
      m_index(p_index)
{
    m_inputVectorSize = SpaceV::GetValueTypeSize(m_inputValueType) * m_dimension;
}


AnnIndex::~AnnIndex()
{
}


bool
AnnIndex::Build(ByteArray p_data, SizeType p_num)
{
    if (nullptr == m_index)
    {
        m_index = SpaceV::VectorIndex::CreateInstance(m_algoType, m_inputValueType);
    }
    if (nullptr == m_index || p_num == 0 || m_dimension == 0 || p_data.Length() != p_num * m_inputVectorSize)
    {
        return false;
    }

    std::shared_ptr<SpaceV::VectorSet> vectors(new SpaceV::BasicVectorSet(p_data,
        m_inputValueType,
        static_cast<SpaceV::SizeType>(m_dimension),
        static_cast<SpaceV::SizeType>(p_num)));

    if (SpaceV::ErrorCode::Success != m_index->BuildIndex(vectors, nullptr))
    {
        return false;
    }
    return true;
}


void
AnnIndex::SetBuildParam(const char* p_name, const char* p_value)
{
    if (nullptr == m_index) 
    {
        if (SpaceV::IndexAlgoType::Undefined == m_algoType || 
            SpaceV::VectorValueType::Undefined == m_inputValueType)
        {
            return;    
        }
        m_index = SpaceV::VectorIndex::CreateInstance(m_algoType, m_inputValueType);

    }
    m_index->SetParameter(p_name, p_value);
}


void
AnnIndex::SetSearchParam(const char* p_name, const char* p_value)
{
    if (nullptr != m_index) m_index->SetParameter(p_name, p_value);
}


std::shared_ptr<QueryResult>
AnnIndex::Search(ByteArray p_data, SizeType p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, false);

    if (nullptr != m_index && p_data.Length() == m_inputVectorSize)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

std::shared_ptr<QueryResult>
AnnIndex::SearchWithMetaData(ByteArray p_data, SizeType p_resultNum)
{
    std::shared_ptr<QueryResult> results = std::make_shared<QueryResult>(p_data.Data(), p_resultNum, true);

    if (nullptr != m_index && p_data.Length() == m_inputVectorSize)
    {
        m_index->SearchIndex(*results);
    }
    return std::move(results);
}

bool
AnnIndex::ReadyToServe() const
{
    return m_index != nullptr;
}


bool
AnnIndex::Save(const char* p_savefile) const
{
    return SpaceV::ErrorCode::Success == m_index->SaveIndex(p_savefile);
}


AnnIndex
AnnIndex::Load(const char* p_loaderFile)
{
    std::shared_ptr<SpaceV::VectorIndex> vecIndex;
    auto ret = SpaceV::VectorIndex::LoadIndex(p_loaderFile, vecIndex);
    if (SpaceV::ErrorCode::Success != ret || nullptr == vecIndex)
    {
        return AnnIndex(0);
    }

    return AnnIndex(vecIndex);
}
