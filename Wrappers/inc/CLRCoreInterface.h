// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ManagedObject.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/ResultIterator.h"
#include "inc/Core/MultiIndexScan.h"

using namespace System;

namespace Microsoft
{
    namespace ANN
    {
        namespace SPTAGManaged
        {

            public ref class BasicResult :
                public ManagedObject<SPTAG::BasicResult>
            {
            public:
                BasicResult(SPTAG::BasicResult* p_instance) : ManagedObject(p_instance)
                {
                }

                property int VID
                {
                public:
                    int get()
                    {
                        return m_Instance->VID;
                    }
                private:
                    void set(int p_vid)
                    {
                    }
                }

                property float Dist
                {
                public:
                    float get()
                    {
                        return m_Instance->Dist;
                    }
                private:
                    void set(float p_dist)
                    {
                    }
                }

                property array<Byte>^ Meta
                {
                public:
                    array<Byte>^ get()
                    {
                        array<Byte>^ buf = gcnew array<Byte>(m_Instance->Meta.Length());
                        if (buf->LongLength != 0) Marshal::Copy((IntPtr)m_Instance->Meta.Data(), buf, 0, (int)m_Instance->Meta.Length());
                        return buf;
                    }
                private:
                    void set(array<Byte>^ p_meta)
                    {
                    }
                }

                property bool RelaxedMono
                {
                public:
                    bool get()
                    {
                        return m_Instance->RelaxedMono;
                    }
                private:
                    void set(bool p_relaxedMono)
                    {
                    }
                }
            };

            public ref class RIterator :
                public ManagedObject<std::shared_ptr<ResultIterator>>
            {
            public:
                RIterator(std::shared_ptr<ResultIterator> result_iterator);
                array<BasicResult^>^ Next(int p_batch);
                bool GetRelaxedMono();
                void Close();
            };

            public ref class AnnIndex :
                public ManagedObject<std::shared_ptr<SPTAG::VectorIndex>>
            {
            public:
                AnnIndex(std::shared_ptr<SPTAG::VectorIndex> p_index);

                AnnIndex(String^ p_algoType, String^ p_valueType, int p_dimension);

                void SetBuildParam(String^ p_name, String^ p_value, String^ p_section);

                void SetSearchParam(String^ p_name, String^ p_value, String^ p_section);

                bool LoadQuantizer(String^ p_quantizerFile);

                void SetQuantizerADC(bool p_adc);

                array<Byte>^ QuantizeVector(array<Byte>^ p_data, int p_num);

                array<Byte>^ ReconstructVector(array<Byte>^ p_data, int p_num);

                bool BuildSPANN(bool p_normalized);

                bool BuildSPANNWithMetaData(array<Byte>^ p_meta, int p_num, bool p_withMetaIndex, bool p_normalized);

                bool Build(array<Byte>^ p_data, int p_num);

                bool BuildWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex);

                bool Build(array<Byte>^ p_data, int p_num, bool p_normalized);

                bool BuildWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex, bool p_normalized);

                array<BasicResult^>^ Search(array<Byte>^ p_data, int p_resultNum);

                array<BasicResult^>^ SearchWithMetaData(array<Byte>^ p_data, int p_resultNum);

                RIterator^ GetIterator(array<Byte>^ p_data);

                void UpdateIndex();

                bool Save(String^ p_saveFile);

                array<array<Byte>^>^ Dump();

                bool Add(array<Byte>^ p_data, int p_num);

                bool AddWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex);

                bool Add(array<Byte>^ p_data, int p_num, bool p_normalized);

                bool AddWithMetaData(array<Byte>^ p_data, array<Byte>^ p_meta, int p_num, bool p_withMetaIndex, bool p_normalized);

                bool Delete(array<Byte>^ p_data, int p_num);

                bool DeleteByMetaData(array<Byte>^ p_meta);

                static AnnIndex^ Load(String^ p_loaderFile);

                static AnnIndex^ Load(array<array<Byte>^>^ p_index);

                static AnnIndex^ Merge(String^ p_indexFilePath1, String^ p_indexFilePath2);

            private:

                int m_dimension;

                size_t m_inputVectorSize;
            };

            public ref class MultiIndexScan :
                public ManagedObject<std::shared_ptr<SPTAG::MultiIndexScan>>
            {
            public:
                MultiIndexScan(std::shared_ptr<SPTAG::MultiIndexScan> multi_index_scan);
                MultiIndexScan(array<AnnIndex^>^ indice, array<array<Byte>^>^ p_data, array<float>^ weight, int p_resultNum,
                    bool useTimer, int termCondVal, int searchLimit);
                BasicResult^ Next();
                void Close();
            };

        }
    }
}
