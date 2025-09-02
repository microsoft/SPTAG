#include <bits/stdc++.h>
#include <iostream>

#include "inc/Core/Common.h"
#include "inc/Core/SPANN/ExtraFileController.h"
#include "inc/Core/SPANN/IExtraSearcher.h"
#define CONFLICT_TEST

using namespace SPTAG;

int main(int argc, char *argv[])
{
    int max_blocks = 5;
    int kv_num = 1000;
    int dataset_size = 10000;
    int iter_num = 1000;
    int batch_num = 10;
    int thread_num = 4;
    int timeout_times = 0;
    double read_rate = 0.5;
    std::mutex error_mtx;
    std::vector<std::string> dataset(dataset_size);
    std::vector<int> values(kv_num);
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::microseconds timeout;

    for (int i = 0; i < dataset_size; i++)
    {
        int size = rand() % (max_blocks << 12);
        dataset[i] = "";
        for (int j = 0; j < size; j++)
        {
            dataset[i] += (char)(rand() % 256);
        }
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Data generated\n");
    SPTAG::SPANN::Options opt;
    opt.m_indexDirectory = "/tmp";
    opt.m_ssdMappingFile = "pbfile";
    opt.m_postingPageLimit = max_blocks * 2;
    opt.m_spdkBatchSize = 64;
    SPANN::FileIO fileIO(opt);

    bool single_thread_test = false;
    bool multi_thread_test = false;
    bool multi_get_test = false;
    bool mixed_read_write_test = false;
    bool timeout_test = true;
    bool conflict_test = false;

    SPTAG::SPANN::ExtraWorkSpace workspace;
    workspace.Initialize(4096, 2, 64, 4 * PageSize, true, false);

    // 单线程存取
    if (!single_thread_test)
    {
        goto MultiThreadTest;
    }
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iter_num; iter++)
    {
        // 随机生成一些数据
        for (int i = 0; i < kv_num; i++)
        {
            values[i] = rand() % dataset_size;
        }
        // 写入数据
        for (int key = 0; key < kv_num; key++)
        {
            fileIO.Put(key, dataset[values[key]], MaxTimeout, &(workspace.m_diskRequests));
        }
        // 读取数据
        for (int key = 0; key < kv_num; key++)
        {
            std::string readValue;
            fileIO.Get(key, &readValue, MaxTimeout, &(workspace.m_diskRequests));
            if (dataset[values[key]] != readValue)
            {
                std::cout << "Error: key " << key << " value not match" << std::endl;
                std::cout << "True value: ";
                for (auto c : dataset[values[key]])
                {
                    std::cout << (int)c << " ";
                }
                std::cout << std::endl;
                std::cout << "Read value: ";
                for (auto c : readValue)
                {
                    std::cout << (int)c << " ";
                }
                std::cout << std::endl;
                return 0;
            }
        }
        // if (error) {
        //     std::cout << "Error in single thread iter " << iter << std::endl;
        //     return 0;
        // }
    }
    end = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Single thread test passed\n");
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Single thread time: %d ms\n",
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Single thread IOPS: %fk\n", (double)iter_num * kv_num * 2 /
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    fileIO.GetStat();

    // 多线程存取
MultiThreadTest:
    if (!multi_thread_test)
    {
        goto MultiGetTest;
    }
    iter_num = 10;
    kv_num = 100000;
    thread_num = 4;
    values.resize(kv_num);
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iter_num; iter++)
    {
        std::vector<std::thread> threads;
        std::vector<int> thread_keys[thread_num];
        std::vector<bool> errors(thread_num);
        // 随机生成一些数据
        for (int i = 0; i < kv_num; i++)
        {
            values[i] = rand() % dataset_size;
        }
        for (int i = 0; i < kv_num; i++)
        {
            int thread_id = rand() % thread_num;
            thread_keys[thread_id].push_back(i);
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads.emplace_back([&fileIO, &values, &thread_keys, &dataset, i]() {
                SPTAG::SPANN::ExtraWorkSpace workspace;
                workspace.Initialize(4096, 2, 64, 4 * PageSize, true, false);

                for (auto key : thread_keys[i])
                {
                    fileIO.Put(key, dataset[values[key]], MaxTimeout, &(workspace.m_diskRequests));
                }
            });
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        threads.clear();
        for (int i = 0; i < thread_num; i++)
        {
            thread_keys[i].clear();
        }
        for (int i = 0; i < kv_num; i++)
        {
            int thread_id = rand() % thread_num;
            thread_keys[thread_id].push_back(i);
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads.emplace_back([&fileIO, &values, &thread_keys, &dataset, &errors, &error_mtx, i]() {
                SPTAG::SPANN::ExtraWorkSpace workspace;
                workspace.Initialize(4096, 2, 64, 4 * PageSize, true, false);

                for (auto key : thread_keys[i])
                {
                    std::string readValue;
                    fileIO.Get(key, &readValue, MaxTimeout, &(workspace.m_diskRequests));
                    if (dataset[values[key]] != readValue)
                    {
                        // errors[i].push_back({key, readValue});
                        errors[i] = true;
                        std::lock_guard<std::mutex> lock(error_mtx);
                        std::cout << "Error: key " << key << " value not match" << std::endl;
                        std::cout << "True value: ";
                        for (auto c : dataset[values[key]])
                        {
                            std::cout << (int)c << " ";
                        }
                        std::cout << std::endl;
                        std::cout << "Read value: ";
                        for (auto c : readValue)
                        {
                            std::cout << (int)c << " ";
                        }
                        std::cout << std::endl;
                        return;
                    }
                }
            });
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        threads.clear();
        for (int i = 0; i < thread_num; i++)
        {
            if (errors[i])
            {
                std::cout << "Error in multi thread iter " << iter << std::endl;
                return 0;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Multi thread test passed\n");
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Multi thread time: %d ms\n",
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Multi thread IOPS: %fk\n", (double)iter_num * kv_num * 2 /
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    fileIO.GetStat();

    // MultiGet测试
MultiGetTest:
    if (!multi_get_test)
    {
        goto MixReadWriteTest;
    }
    iter_num = 100;
    kv_num = 100000;
    thread_num = 4;
    values.resize(kv_num);
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iter_num; iter++)
    {
        std::vector<std::thread> threads;
        std::vector<int> thread_keys[thread_num];
        std::vector<bool> errors(thread_num);
        // 随机生成一些数据
        for (int i = 0; i < kv_num; i++)
        {
            values[i] = rand() % dataset_size;
        }
        for (int i = 0; i < kv_num; i++)
        {
            int thread_id = rand() % thread_num;
            thread_keys[thread_id].push_back(i);
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads.emplace_back([&fileIO, &values, &thread_keys, &errors, &error_mtx, &dataset, i]() {
                SPTAG::SPANN::ExtraWorkSpace workspace;
                workspace.Initialize(4096, 2, 64, 4 * PageSize, true, false);

                for (auto key : thread_keys[i])
                {
                    fileIO.Put(key, dataset[values[key]], MaxTimeout, &(workspace.m_diskRequests));
                }
                std::vector<std::string> readValues;
                std::vector<int> keys;
                int num = thread_keys[i].size();
                for (int k = 0; k < num; k += 256)
                {
                    readValues.clear();
                    keys.clear();
                    for (int j = k; j < std::min(k + 256, num); j++)
                    {
                        keys.push_back(thread_keys[i][j]);
                    }
                    fileIO.MultiGet(keys, &readValues, MaxTimeout, &(workspace.m_diskRequests));
                    for (int j = 0; j < keys.size(); j++)
                    {
                        if (dataset[values[keys[j]]] != readValues[j])
                        {
                            errors[i] = true;
                            std::lock_guard<std::mutex> lock(error_mtx);
                            std::cout << "Error: key " << keys[j] << " value not match" << std::endl;
                            std::cout << "True value: ";
                            for (auto c : dataset[values[keys[j]]])
                            {
                                std::cout << (int)c << " ";
                            }
                            std::cout << std::endl;
                            std::cout << "Read value: ";
                            for (auto c : readValues[j])
                            {
                                std::cout << (int)c << " ";
                            }
                            std::cout << std::endl;
                            return;
                        }
                    }
                }
            });
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        threads.clear();
        for (int i = 0; i < thread_num; i++)
        {
            if (errors[i])
            {
                std::cout << "Error in MultiGet test iter " << iter << std::endl;
                return 0;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "MultiGet test passed\n");
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "MultiGet time: %d ms\n",
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "MultiGet IOPS: %fk\n", (double)iter_num * kv_num * 2 /
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    fileIO.GetStat();

    // 多线程混合读写存取
MixReadWriteTest:
    if (!mixed_read_write_test)
    {
        goto TimeoutTest;
    }
    batch_num = 10;
    iter_num = 100000;
    read_rate = 0.5;
    values.resize(kv_num);
    start = std::chrono::high_resolution_clock::now();
    for (int batch = 0; batch < batch_num; batch++)
    {
        std::vector<std::thread> threads;
        std::vector<int> thread_keys[thread_num];
        std::vector<bool> errors(thread_num, false);
        for (int i = 0; i < kv_num; i++)
        {
            int thread_id = rand() % thread_num;
            thread_keys[thread_id].push_back(i);
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads.emplace_back([&fileIO, &values, &thread_keys, &dataset, &errors, &error_mtx, i, max_blocks,
                                  read_rate, iter_num, dataset_size]() {
                SPTAG::SPANN::ExtraWorkSpace workspace;
                workspace.Initialize(4096, 2, 64, 4 * PageSize, true, false);

                int num = thread_keys[i].size();
                int read_count = 0, write_count = 0;
                bool is_read = false;
                bool is_merge = false;
                for (auto key : thread_keys[i])
                {
                    values[key] = rand() % dataset_size;
                    fileIO.Put(key, dataset[values[key]], MaxTimeout, &(workspace.m_diskRequests));
                }
                for (int j = 0; j < iter_num; j++)
                {
                    int key = thread_keys[i][rand() % num];
                    if (rand() < RAND_MAX * read_rate)
                    {
                        is_read = true;
                    }
                    else
                    {
                        is_read = false;
                        if (rand() < RAND_MAX * 0.5)
                        {
                            is_merge = true;
                        }
                        else
                        {
                            is_merge = false;
                        }
                    }
                    if (is_read)
                    {
                        std::string readValue;
                        fileIO.Get(key, &readValue, MaxTimeout, &(workspace.m_diskRequests));
                        if (dataset[values[key]] != readValue)
                        {
                            // errors[i].push_back({key, readValue});
                            errors[i] = true;
                            std::lock_guard<std::mutex> lock(error_mtx);
                            std::cout << "Error: key " << key << " value not match" << std::endl;
                            std::cout << "True value: ";
                            for (auto c : dataset[values[key]])
                            {
                                std::cout << (int)c << " ";
                            }
                            std::cout << std::endl;
                            std::cout << "Read value: ";
                            for (auto c : readValue)
                            {
                                std::cout << (int)c << " ";
                            }
                            std::cout << std::endl;
                            return;
                        }
                        read_count++;
                    }
                    else if (is_merge)
                    {
                        std::string mergeValue = "";
                        int size = rand() % (max_blocks << 12);
                        for (int k = 0; k < size; k++)
                        {
                            mergeValue += (char)(rand() % 256);
                        }
                        fileIO.Merge(key, mergeValue, MaxTimeout, &(workspace.m_diskRequests));
                        write_count++;
                        read_count++;
                        std::string readValue;
                        fileIO.Get(key, &readValue, MaxTimeout, &(workspace.m_diskRequests));
                        if (readValue != dataset[values[key]] + mergeValue)
                        {
                            errors[i] = true;
                            std::lock_guard<std::mutex> lock(error_mtx);
                            std::cout << "Error: key " << key << " value not match" << std::endl;
                            std::cout << "True value: ";
                            for (auto c : mergeValue)
                            {
                                std::cout << (int)c << " ";
                            }
                            std::cout << std::endl;
                            std::cout << "Read value: ";
                            for (auto c : readValue)
                            {
                                std::cout << (int)c << " ";
                            }
                            std::cout << std::endl;
                            return;
                        }
                        fileIO.Put(key, dataset[values[key]], MaxTimeout, &(workspace.m_diskRequests));
                        write_count++;
                        read_count++;
                    }
                    else
                    {
                        values[key] = rand() % dataset_size;
                        fileIO.Put(key, dataset[values[key]], MaxTimeout, &(workspace.m_diskRequests));
                        write_count++;
                    }
                }
            });
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        threads.clear();
        for (int i = 0; i < thread_num; i++)
        {
            if (errors[i])
            {
                std::cout << "Error in batch " << batch << std::endl;
                return 0;
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mix read write test passed\n");
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mix read write time: %d ms\n",
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Mix read write IOPS: %fk\n", (double)batch_num * iter_num /
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    fileIO.GetStat();

    // 超时测试
TimeoutTest:
    if (!timeout_test)
    {
        goto ConflictTest;
    }
    batch_num = 10;
    timeout = std::chrono::microseconds(0);
    iter_num = 50;
    kv_num = 10;
    values.resize(kv_num);
    for (int iter = 0; iter < iter_num; iter++)
    {
        for (int i = 0; i < kv_num; i++)
        {
            values[i] = rand() % dataset_size;
            fileIO.Put(i, dataset[values[i]], MaxTimeout, &(workspace.m_diskRequests));
        }
        for (int key = 0; key < kv_num; key++)
        {
            std::string readValue;
            auto result = fileIO.Get(key, &readValue, timeout, &(workspace.m_diskRequests));
            if (result == ErrorCode::Fail)
            {
                timeout_times++;
            }
        }
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Timeout times: %d\n", timeout_times);
    iter_num = 10;
    kv_num = 1000;
    values.resize(kv_num);
    for (int iter = 0; iter < iter_num; iter++)
    {
        for (int i = 0; i < kv_num; i++)
        {
            values[i] = rand() % dataset_size;
            fileIO.Put(i, dataset[values[i]], MaxTimeout, &(workspace.m_diskRequests));
        }
        for (int key = 0; key < kv_num; key++)
        {
            std::string readValue;
            auto result = fileIO.Get(key, &readValue, MaxTimeout, &(workspace.m_diskRequests));
            if (result == ErrorCode::Fail)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error while get key %d\n", key);
                return 0;
            }
            if (dataset[values[key]] != readValue)
            {
                SPTAGLIB_LOG(Helper::LogLevel::LL_Error, "Error: key %d value not match\n", key);
                return 0;
            }
        }
    }
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Timeout test passed\n");
    fileIO.GetStat();

#ifdef CONFLICT_TEST
    // 冲突测试
ConflictTest:
    if (!conflict_test)
    {
        goto End;
    }
    batch_num = 10;
    iter_num = 1000;
    kv_num = 2048;
    thread_num = 4;
    values.resize(kv_num);
    // start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < batch_num; iter++)
    {
        std::vector<std::thread> threads;
        std::vector<bool> errors(thread_num);
        // 随机生成一些数据
        for (int i = 0; i < kv_num; i++)
        {
            values[i] = rand() % dataset_size;
        }
        std::string init_str(PageSize, -1);
        for (int i = 0; i < kv_num; i++)
        {
            fileIO.Put(i, init_str, MaxTimeout, &(workspace.m_diskRequests));
        }
        for (int i = 0; i < thread_num; i++)
        {
            threads.emplace_back([&fileIO, &errors, &error_mtx, &dataset, iter_num, kv_num, i]() {
                SPTAG::SPANN::ExtraWorkSpace workspace;
                workspace.Initialize(4096, 2, 64, 4 * PageSize, true, false);

                std::vector<std::string> readValues;
                std::vector<int> keys;
                std::vector<int> values(kv_num);
                for (int i = 0; i < kv_num; i++)
                {
                    values[i] = rand() % dataset.size();
                }
                for (int k = 0; k < iter_num; k++)
                {
                    int key = rand() % kv_num;
                    bool is_read = false;
                    bool batch_read = false;
                    if (rand() < RAND_MAX * 0.66)
                    {
                        is_read = true;
                        if (rand() < RAND_MAX * 0.5)
                        {
                            batch_read = true;
                        }
                    }
                    if (is_read && !batch_read)
                    {
                        std::string readValue;
                        fileIO.Get(key, &readValue, MaxTimeout, &(workspace.m_diskRequests));
                        if ((int)readValue[0] != i)
                        {
                            ;
                        }
                        else
                        {
                            readValue[0] = dataset[values[key]][0];
                            if (readValue != dataset[values[key]])
                            {
                                errors[i] = true;
                                std::lock_guard<std::mutex> lock(error_mtx);
                                std::cout << "Error: key " << key << " value not match" << std::endl;
                                std::cout << "True value: ";
                                std::cout << i << " ";
                                for (int j = 1; j < PageSize; j++)
                                {
                                    std::cout << (int)dataset[values[key]][j] << " ";
                                }
                                std::cout << std::endl;
                                std::cout << "Read value: ";
                                readValue[0] = (char)i;
                                for (auto c : readValue)
                                {
                                    std::cout << (int)c << " ";
                                }
                                std::cout << std::endl;
                                return;
                            }
                        }
                    }
                    else if (is_read && batch_read)
                    {
                        std::vector<std::string> readValues;
                        std::vector<int> keys;
                        int num = 256;
                        std::set<int> key_set;
                        while (key_set.size() < num)
                        {
                            key_set.insert(rand() % kv_num);
                        }
                        keys.assign(key_set.begin(), key_set.end());
                        fileIO.MultiGet(keys, &readValues, MaxTimeout, &(workspace.m_diskRequests));
                        for (int j = 0; j < keys.size(); j++)
                        {
                            if ((int)readValues[j][0] != i)
                            {
                                ;
                            }
                            else
                            {
                                readValues[j][0] = dataset[values[keys[j]]][0];
                                if (readValues[j] != dataset[values[keys[j]]])
                                {
                                    errors[i] = true;
                                    std::lock_guard<std::mutex> lock(error_mtx);
                                    std::cout << "Error: key " << keys[j] << " value not match" << std::endl;
                                    std::cout << "True value: ";
                                    std::cout << i << " ";
                                    for (int k = 1; k < PageSize; k++)
                                    {
                                        std::cout << (int)dataset[values[keys[j]]][k] << " ";
                                    }
                                    std::cout << std::endl;
                                    std::cout << "Read value: ";
                                    readValues[j][0] = (char)i;
                                    for (auto c : readValues[j])
                                    {
                                        std::cout << (int)c << " ";
                                    }
                                    std::cout << std::endl;
                                    return;
                                }
                            }
                        }
                    }
                    else
                    {
                        values[key] = rand() % dataset.size();
                        std::string tmp_str(dataset[values[key]]);
                        tmp_str[0] = (char)i;
                        fileIO.Put(key, tmp_str, MaxTimeout, &(workspace.m_diskRequests));
                    }
                }
            });
        }
        for (auto &thread : threads)
        {
            thread.join();
        }
        threads.clear();
        for (int i = 0; i < thread_num; i++)
        {
            if (errors[i])
            {
                std::cout << "Error in Conflict test iter " << iter << std::endl;
                return 0;
            }
        }
    }
    // end = std::chrono::high_resolution_clock::now();
    SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Conflict test passed\n");
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Conflict test time: %d ms\n",
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // SPTAGLIB_LOG(Helper::LogLevel::LL_Info, "Conflict test IOPS: %fk\n", (double)iter_num * kv_num * 2 /
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    fileIO.GetStat();
#endif
End:
    std::cout << "Test passed" << std::endl;
    return 0;
}
