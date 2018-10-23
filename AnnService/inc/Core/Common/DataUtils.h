#ifndef _SPTAG_COMMON_DATAUTILS_H_
#define _SPTAG_COMMON_DATAUTILS_H_

#include <sys/stat.h>
#include <atomic>
#include "CommonUtils.h"
#include "../../Helper/CommonHelper.h"

namespace SPTAG
{
    namespace COMMON
    {
        const int bufsize = 1024 * 1024 * 1024;

        template <typename T>
        void ProcessTSVData(int id, int threadbase, long long blocksize,
            std::string filename, std::string outfile, std::string outmetafile, std::string outmetaindexfile,
            std::atomic_int& numSamples, int& D, DistCalcMethod distCalcMethod) {
            std::ifstream inputStream(filename);
            if (!inputStream.is_open()) {
                std::cerr << "unable to open file " + filename << std::endl;
                throw MyException("unable to open file " + filename);
                exit(1);
            }
            std::ofstream outputStream, metaStream_out, metaStream_index;
            outputStream.open(outfile + std::to_string(id + threadbase), std::ofstream::binary);
            metaStream_out.open(outmetafile + std::to_string(id + threadbase), std::ofstream::binary);
            metaStream_index.open(outmetaindexfile + std::to_string(id + threadbase), std::ofstream::binary);
            if (!outputStream.is_open() || !metaStream_out.is_open() || !metaStream_index.is_open()) {
                std::cerr << "unable to open output file " << outfile << " " << outmetafile << " " << outmetaindexfile << std::endl;
                throw MyException("unable to open output files");
                exit(1);
            }

            std::vector<float> arr;
            std::vector<T> sample;

            int base = 1;
            if (distCalcMethod == DistCalcMethod::Cosine) {
                base = Utils::GetBase<T>();
            }
            long long writepos = 0;
            int sampleSize = 0;
            long long totalread = 0;
            std::streamoff startpos = id * blocksize;

#ifndef _MSC_VER
            int enter_size = 1;
#else
            int enter_size = 1;
#endif
            std::string currentLine;
            size_t index;
            inputStream.seekg(startpos, std::ifstream::beg);
            if (id != 0) {
                std::getline(inputStream, currentLine);
                totalread += currentLine.length() + enter_size;
            }
            std::cout << "Begin thread " << id << " begin at:" << (startpos + totalread) << std::endl;
            while (!inputStream.eof() && totalread <= blocksize) {
                std::getline(inputStream, currentLine);
                if (currentLine.length() <= enter_size || (index = Utils::ProcessLine(currentLine, arr, D, base, distCalcMethod)) < 0) {
                    totalread += currentLine.length() + enter_size;
                    continue;
                }
                sample.resize(D);
                for (int j = 0; j < D; j++) sample[j] = (T)arr[j];

                outputStream.write((char *)(sample.data()), sizeof(T)*D);
                metaStream_index.write((char *)&writepos, sizeof(long long));
                metaStream_out.write(currentLine.c_str(), index);

                writepos += index;
                sampleSize += 1;
                totalread += currentLine.length() + enter_size;
            }
            metaStream_index.write((char *)&writepos, sizeof(long long));
            metaStream_index.write((char *)&sampleSize, sizeof(int));
            inputStream.close();
            outputStream.close();
            metaStream_out.close();
            metaStream_index.close();

            numSamples.fetch_add(sampleSize);

            std::cout << "Finish Thread[" << id << ", " << sampleSize << "] at:" << (startpos + totalread) << std::endl;
        }

        void MergeData(int threadbase, std::string outfile, std::string outmetafile, std::string outmetaindexfile,
            std::atomic_int& numSamples, int D) {
            std::ifstream inputStream;
            std::ofstream outputStream;
            char * buf = new char[bufsize];
            long long * offsets;
            int partSamples;
            int metaSamples = 0;
            long long lastoff = 0;

            outputStream.open(outfile, std::ofstream::binary);
            outputStream.write((char *)&numSamples, sizeof(int));
            outputStream.write((char *)&D, sizeof(int));
            for (int i = 0; i < threadbase; i++) {
                std::string file = outfile + std::to_string(i);
                inputStream.open(file, std::ifstream::binary);
                while (!inputStream.eof()) {
                    inputStream.read(buf, bufsize);
                    outputStream.write(buf, inputStream.gcount());
                }
                inputStream.close();
                remove(file.c_str());
            }
            outputStream.close();

            outputStream.open(outmetafile, std::ofstream::binary);
            for (int i = 0; i < threadbase; i++) {
                std::string file = outmetafile + std::to_string(i);
                inputStream.open(file, std::ifstream::binary);
                while (!inputStream.eof()) {
                    inputStream.read(buf, bufsize);
                    outputStream.write(buf, inputStream.gcount());
                }
                inputStream.close();
                remove(file.c_str());
            }
            outputStream.close();
            delete[] buf;

            outputStream.open(outmetaindexfile, std::ofstream::binary);
            outputStream.write((char *)&numSamples, sizeof(int));
            for (int i = 0; i < threadbase; i++) {
                std::string file = outmetaindexfile + std::to_string(i);
                inputStream.open(file, std::ifstream::binary);

                inputStream.seekg(-((long long)sizeof(int)), inputStream.end);
                inputStream.read((char *)&partSamples, sizeof(int));
                offsets = new long long[partSamples + 1];

                inputStream.seekg(0, inputStream.beg);
                inputStream.read((char *)offsets, sizeof(long long)*(partSamples + 1));
                inputStream.close();
                remove(file.c_str());

                for (int j = 0; j < partSamples + 1; j++)
                    offsets[j] += lastoff;
                outputStream.write((char *)offsets, sizeof(long long)*partSamples);

                lastoff = offsets[partSamples];
                metaSamples += partSamples;
                delete[] offsets;
            }
            outputStream.write((char *)&lastoff, sizeof(long long));
            outputStream.close();

            std::cout << "numSamples:" << numSamples << " metaSamples:" << metaSamples << " D:" << D << std::endl;
        }

        template <typename T>
        void ParseData(std::string filenames, std::string outfile, std::string outmetafile, std::string outmetaindexfile,
            int threadnum, DistCalcMethod distCalcMethod) {
            omp_set_num_threads(threadnum);

            std::atomic_int numSamples = { 0 };
            int D = -1;

            int threadbase = 0;
            std::vector<std::string> inputFileNames = Helper::StrUtils::SplitString(filenames, ",");
            for (std::string inputFileName : inputFileNames)
            {
#ifndef _MSC_VER
                struct stat stat_buf;
                stat(inputFileName.c_str(), &stat_buf);
#else
                struct _stat64 stat_buf;
                int res = _stat64(inputFileName.c_str(), &stat_buf);
#endif
                long long blocksize = (stat_buf.st_size + threadnum - 1) / threadnum;

#pragma omp parallel for
                for (int i = 0; i < threadnum; i++) {
                    ProcessTSVData<T>(i, threadbase, blocksize, inputFileName, outfile, outmetafile, outmetaindexfile, numSamples, D, distCalcMethod);
                }
                threadbase += threadnum;
            }
            MergeData(threadbase, outfile, outmetafile, outmetaindexfile, numSamples, D);
        }
    }
}

#endif // _SPTAG_COMMON_DATAUTILS_H_
