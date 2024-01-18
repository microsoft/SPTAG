#include "inc/Core/Common/WAL.h"

void SimpleFileWAL::Init(const std::string path)
{
	dir = path;
	metaIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string metaPath = std::string(dir) + std::string("/") + std::string(metaFile);
	// check whether any file exist
	if (!metaIO->Initialize(metaPath.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedOpenFile;

	metaindexIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string metaindexPath = std::string(dir) + std::string("/") + std::string(metaindexFile);
	// check whether any file exist
	if (!metaindexIO->Initialize(metaindexPath.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedOpenFile;

	indexIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string indexPath = std::string(dir) + std::string("/") + std::string(indexFile);
	// check whether any file exist
	if (!indexIO->Initialize(indexPath.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedOpenFile;
}

ErrorCode SimpleFileWAL::Start()
{
	metaIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string metaPath = std::string(dir) + std::string("/") + std::string(metaFile);
	// check whether any file exists
	if (!metaIO->Initialize(metaPath.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedOpenFile;

	metaindexIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string metaindexPath = std::string(dir) + std::string("/") + std::string(metaindexFile);
	// check whether any file exists
	if (!metaindexIO->Initialize(metaindexPath.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedOpenFile;

	indexIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string indexPath = std::string(dir) + std::string("/") + std::string(indexFile);
	// check whether any file exists
	if (!indexIO->Initialize(indexPath.c_str(), std::ios::binary | std::ios::out)) return ErrorCode::FailedOpenFile;
}

bool SimpleFileWAL::Append(VectorSet* indexdataSet, MetadataSet* metadataSet)
{
	((IndexDataSet*)indexdataSet)->SaveIndexDataSet(indexOut);
	metadataSet->SaveMetadata(metaIO, metaindexIO);
	return true;
}

bool SimpleFileWAL::Replay(VectorIndex* index) 
{
	
	//open for read
	//check whether index file exists
	metaIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string metaPath = std::string(dir) + std::string("/") + std::string(metaFile);
	if (!metaIO->Initialize(metaPath.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
	//check whether index file exists
	metaindexIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string metaindexPath = std::string(dir) + std::string("/") + std::string(metaindexFile);
	if (!metaindexIO->Initialize(metaindexPath.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;
	//check whether index file exists
	indexIO = std::shared_ptr<Helper::DiskIO>(new Helper::SimpleFileIO());
	std::string indexPath = std::string(dir) + std::string("/") + std::string(indexFile);
	if (!indexIO->Initialize(indexPath.c_str(), std::ios::binary | std::ios::in)) return ErrorCode::FailedOpenFile;

	//replay



	//rename
	metaIO = nullptr;
	metaindexIO = nullptr;
	indexIO = nullptr;
}