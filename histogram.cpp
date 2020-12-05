#define _CRT_SECURE_NO_WARNINGS

#include "serial.hpp"
#include "cuda.hpp"
#include "interface.hpp"

#include "cli/args.hpp"
#include "system/stopwatch.hpp"
#include "system/file.hpp"
#include "system/mmap_file.hpp"

#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <cstdint>


template<typename T = std::uint8_t, typename RES = std::uint32_t>
std::unique_ptr<IHistogramAlgorithm<T, RES>> getAlgorithm(const std::string& algoName, bpp::ProgramArguments& args, bool quiet = false)
{
	using map_t = std::map<std::string, std::unique_ptr<IHistogramAlgorithm<T, RES>>>;


	map_t algorithms;
	algorithms["serial"] = std::make_unique<SerialHistogramAlgorithm<T, RES>>();

	// PLACE ADDITIONAL ALGORITHMS HERE ...

	algorithms["naive"] = std::make_unique<CudaNaiveAlgorithm<T,RES>>();
	algorithms["atomic"] = std::make_unique<CudaAtomicAlgorithm<T,RES>>();
	algorithms["privatized"] = std::make_unique<CudaPrivatizedAlgorithm<T,RES>>();
	algorithms["aggregated"] = std::make_unique<CudaAggregatedAlgorithm<T,RES>>();
	algorithms["atomic_shm"] = std::make_unique<CudaAtomicShmAlgorithm<T,RES>>();
	algorithms["overlap"] = std::make_unique<CudaOverlapAlgorithm<T,RES>>();
	algorithms["final"] = std::make_unique<CudaFinalAlgorithm<T,RES>>();

	auto it = algorithms.find(algoName);
	if (it == algorithms.end()) {
		throw (bpp::RuntimeError() << "Unkown algorithm '" << algoName << "'.");
	}

	if (!quiet) {
		std::cout << "Selected algorithm: " << algoName << std::endl;
	}
	return std::move(it->second);
}


template<typename T = std::uint8_t, typename RES = std::uint32_t>
bool verify(const std::vector<RES>& res, const std::vector<RES>& correctRes, T fromValue)
{
	if (res.size() != correctRes.size()) {
		std::cerr << std::endl << "Error: Result size mismatch (" << res.size() << " values found, but " << correctRes.size() << " values expected)!" << std::endl;
		return false;
	}

	std::size_t errorCount = 0;
	for (std::size_t i = 0; i < res.size(); ++i) {
		if (res[i] != correctRes[i]) {
			if (errorCount == 0) std::cout << std::endl;
			if (++errorCount <= 10) {
				std::cerr << "Error in bucket [" << i << "]: " << res[i] << " != " << correctRes[i] << " (expected)" << std::endl;
			}
		}
	}

	if (errorCount > 0) {
		std::cerr << "Total errors found: " << errorCount << std::endl;
	}

	return errorCount == 0;
}


template<typename T = std::uint8_t, typename RES = std::uint32_t>
void saveResults(const std::string& fileName, const std::vector<RES>& result, T fromValue)
{
	bpp::File file(fileName);
	file.open();
	bpp::TextWriter writer(file, "\n", "\t");

	for (std::size_t i = 0; i < result.size(); ++i) {
		writer.writeToken(i+fromValue);
		writer.writeToken(result[i]);
		writer.writeLine();
	}

	file.close();
}


template<typename T = std::uint8_t, typename RES = std::uint32_t>
void run(bpp::ProgramArguments& args)
{
	auto algoName = args.getArgString("algorithm").getValue();
	auto algorithm = getAlgorithm<T, RES>(algoName, args);

	std::cout << "MMaping file '" << args[0] << "' ..." << std::endl;
	bpp::MMapFile file;
	file.open(args[0]);

	auto repeatInput = args.getArgInt("repeatInput").getValue();
	const T* data = (const T*)file.getData();
	std::size_t length = file.length();
	std::vector<T> repeatedData;
	if (repeatInput > 1) {
		std::cout << "Repeating input file " << repeatInput << "x ..." << std::endl;
		repeatedData.reserve(length * repeatInput);
		while (repeatInput > 0) {
			for (std::size_t i = 0; i < length; ++i) repeatedData.push_back(data[i]);
			--repeatInput;
		}

		length = repeatedData.size();
		data = &repeatedData[0];
	}
	else
		file.populate();

	bpp::Stopwatch stopwatch;
	double totalTime = 0.0;

	T fromValue = (T)args.getArgInt("fromValue").getValue();
	T toValue = (T)args.getArgInt("toValue").getValue();
	if (fromValue > toValue) std::swap(fromValue, toValue);
	std::cout << "Initialize (range: " << (int)fromValue << ".." << (int)toValue << ", data length " << length << ") ..." << std::endl;
	algorithm->initialize(data, length, fromValue, toValue, args);

	std::cout << "Preparations ... "; std::cout.flush();
	stopwatch.start();
	algorithm->prepare();
	stopwatch.stop();
	totalTime += stopwatch.getMiliseconds();
	std::cout << stopwatch.getMiliseconds() << " ms" << std::endl;

	std::cout << "Execution ... "; std::cout.flush();
	stopwatch.start();
	algorithm->run();
	stopwatch.stop();
	totalTime += stopwatch.getMiliseconds();
	std::cout << stopwatch.getMiliseconds() << " ms" << std::endl;

	std::cout << "Finalization ... "; std::cout.flush();
	stopwatch.start();
	algorithm->finalize();
	stopwatch.stop();
	totalTime += stopwatch.getMiliseconds();
	std::cout << stopwatch.getMiliseconds() << " ms" << std::endl;

	auto result = algorithm->getResult();
	if (args.getArgBool("verify").getValue() && algoName != "serial") {
		std::cout << "Verifying results ... "; std::cout.flush();
		auto baseAlgorithm = getAlgorithm<T, RES>("serial", args, true);
		baseAlgorithm->initialize(data, length, fromValue, toValue, args);
		baseAlgorithm->prepare();
		baseAlgorithm->run();
		baseAlgorithm->finalize();

		if (verify(algorithm->getResult(), baseAlgorithm->getResult(), (T)args.getArgInt("fromValue").getValue()))
			std::cout << "OK" << std::endl;
		else
			std::cout << "FAILED" << std::endl;

		baseAlgorithm->cleanup();
	}

	if (args.getArg("save").isPresent()) {
		auto saveToFile = args.getArgString("save").getValue();
		std::cout << "Saving results to " << saveToFile << " ..." << std::endl;
		saveResults(saveToFile, algorithm->getResult(), (T)args.getArgInt("fromValue").getValue());
	}

	algorithm->cleanup();
	std::cout << "And we're done here." << std::endl;
}


int main(int argc, char* argv[])
{
	/*
	 * Arguments
	 */
	bpp::ProgramArguments args(1, 1);
	args.setNamelessCaption(0, "Input file");

	try {
		args.registerArg<bpp::ProgramArguments::ArgString>("algorithm", "Which algorithm is to be tested.", false, "serial");
		args.registerArg<bpp::ProgramArguments::ArgString>("save", "Path to a file to which the histogram is saved", false);
		args.registerArg<bpp::ProgramArguments::ArgBool>("verify", "Results will be automatically verified using serial algorithm as baseline.");

		args.registerArg<bpp::ProgramArguments::ArgInt>("fromValue", "Ordinal value of the first character in histogram.", false, 0, 0, 255);
		args.registerArg<bpp::ProgramArguments::ArgInt>("toValue", "Ordinal value of the last character in histogram.", false, 127, 0, 255);
		args.registerArg<bpp::ProgramArguments::ArgInt>("repeatInput", "Enlarge data input by loading input file multiple times.", false, 1, 1);

		args.registerArg<bpp::ProgramArguments::ArgInt>("blockSize", "CUDA block size (threads in a block).", false, 256, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("itemsPerThread", "How many items are processed by one thread.", false, 64, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("privCopies", "Number of privatized copies.", false, 8, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("chunkSize", "Number of input data items transfered in a single operation.", false, 4194304, 1, std::numeric_limits<bpp::ProgramArguments::ArgInt::value_t>::max());
		args.registerArg<bpp::ProgramArguments::ArgInt>("numStreams", "Number of CUDA streams used to dispatch chunk processing.", false, 32, 1, std::numeric_limits<bpp::ProgramArguments::ArgInt::value_t>::max());
		args.registerArg<bpp::ProgramArguments::ArgBool>("pinned", "If pinned memory should be used.");

		// Process the arguments ...
		args.process(argc, argv);
	}
	catch (bpp::ArgumentException& e) {
		std::cout << "Invalid arguments: " << e.what() << std::endl << std::endl;
		args.printUsage(std::cout);
		return 1;
	}

	try {
		run<std::uint8_t, std::uint32_t>(args);
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		return 2;
	}

	return 0;
}
