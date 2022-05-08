#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL\cl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <chrono>
#include <thread>

void upperTriangularMatrix(std::vector<std::vector<float>>& matrix, std::vector<float>& constantTerms) {
	unsigned int n = matrix.size();
	float ratio;

	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			ratio = matrix[j][i] / matrix[i][i];
			for (int k = 0; k < n; k++) {
				matrix[j][k] -= matrix[i][k] * ratio;
			}
			constantTerms[j] -= constantTerms[i] * ratio;
		}
	}
}

std::vector<float> gaussEliminationSolve(std::vector<std::vector<float>>& matrix, std::vector<float>& constantTerms) {
	upperTriangularMatrix(matrix, constantTerms);

	float difference;
	unsigned int n = matrix.size();
	std::vector<float> solution(n);

	for (int i = n - 1; i >= 0; i--) {
		difference = 0;
		if (i == n - 1) {
			solution[i] = constantTerms[i] / matrix[i][i];
		}
		else {
			for (int j = i + 1; j < n; j++) {
				difference += matrix[i][j] * solution[j];
			}
			solution[i] = (constantTerms[i] - difference) / matrix[i][i];
		}
	}

	return solution;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char* srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}


int main() {

	srand(time(NULL));

	const int N = 100;
	int min = 1;
	int max = 20;

	//dane dla algorytmu c++
	//macierz wspolczynnikow:
	std::vector<std::vector<float>> cpp_matrix(N, std::vector<float>(N));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			//cpp_matrix[i].push_back(rand() % max + min);
			cpp_matrix[i][j] = rand() % max + min;
		}
	}

	//wektor wyrazow wolnych:
	std::vector<float> cpp_constant_terms(N);
	for (int i = 0; i < N; i++) {
		//cpp_constant_terms.push_back(rand() % max + min);
		cpp_constant_terms[i] = rand() % max + min;
	}

	//dane dla algorytmu OpenCL
	//konieczne jest przekszta³cenie macierzy do postaci tablicy jednowymiarowej
	//tablica reprezentujaca macierz:
	float *cl_matrix = new float[N*N];

	//przekszta³cenie macierzy (tablicy 2d) do tablicy jednowymiarowej:
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cl_matrix[i * N + j] = cpp_matrix[i][j];
		}
	}

	//rozmiar tablicy w bajtach:
	int cl_matrix_size = N * N * sizeof(float);

	//tablicowy wektor wyrazow wolnych i jego rozmiar:
	float cl_constant_terms[N];
	for (int i = 0; i < N; i++) {
		cl_constant_terms[i] = cpp_constant_terms[i];
	}
	int cl_constant_terms_size = N * sizeof(float);

	//oraz tablica przechowujaca rozwiazanie i jej rozmiar:
	float cl_solution[N];
	int cl_solution_size = N * sizeof(float);


	std::cout << "main matrix before transformation:\n";
	for (int i = 0; i < N * N; i++) {
		std::cout << cl_matrix[i] << " ";
		if ((i + 1) % N == 0)
			std::cout << "\n";
	}

	std::cout << "\nconstant terms before transformation:\n";
	for (int i = 0; i < N; i++) {
		std::cout << cl_constant_terms[i] << " ";
	}

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	//wykonanie algorytmu C++:
	auto t1 = high_resolution_clock::now();
	std::vector<float> cpp_solution = gaussEliminationSolve(cpp_matrix, cpp_constant_terms);
	auto t2 = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(t2 - t1);

	/* Getting number of milliseconds as a double. */
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout <<"\n\n\n" << ms_int.count() << "ms\n";
	std::cout << ms_double.count() << "ms\n\n\n";

	//error value
	cl_int err;

	//czas
	cl_event event;

	//get the first platform id
	cl_platform_id myp;
	err = clGetPlatformIDs(1, &myp, NULL);

	//get the first fpga device in the platform
	cl_device_id device;
	err = clGetDeviceIDs(myp, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//create an opencl context for the fpga device
	cl_context context;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

	//a program is a collection of kernels
	cl_program program = CreateProgram(context,device,"GaussianElimination.cl");

	//create an opencl command queue
	cl_command_queue queue;
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

	//create kernels from the program
	cl_kernel kernel = clCreateKernel(program, "GaussianElimination", &err);

	//allocate memory on device
	//for main matrix:
	cl_mem a;
	a = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_matrix_size, NULL, &err);

	//for constant terms:
	cl_mem b;
	b = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_constant_terms_size, NULL, &err);

	//for solution:
	cl_mem c;
	c = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_solution_size, NULL, &err);

	//transfer memory
	err = clEnqueueWriteBuffer(queue, a, CL_TRUE, 0, cl_matrix_size, cl_matrix, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, b, CL_TRUE, 0, cl_constant_terms_size, cl_constant_terms, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(queue, c, CL_TRUE, 0, cl_solution_size, cl_solution, 0, NULL, NULL);

	//set up the kernel argument list
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c);
	err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&N);

	//launch the kernel
	err = clEnqueueTask(queue, kernel, 0, NULL, &event);

	//transfer result buffer back
	err = clEnqueueReadBuffer(queue, c, CL_TRUE, 0, cl_solution_size, cl_solution, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, a, CL_TRUE, 0, cl_matrix_size, cl_matrix, 0, NULL, NULL);
	err = clEnqueueReadBuffer(queue, b, CL_TRUE, 0, cl_constant_terms_size, cl_constant_terms, 0, NULL, NULL);

	clWaitForEvents(1, &event);

	cl::finish();

	cl_ulong time_start;
	cl_ulong time_end;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

	double nanoSeconds = time_end - time_start;
	printf("\n\n\n\nOpenCl Execution time is: %0.20f milliseconds \n\n\n\n\n", nanoSeconds / 1000000.0);
	 
	//wyniki dla OpenCL:
	/*std::cout << "\n\nmain matrix after transformation for OpenCL:\n";
	for (int i = 0; i < N * N; i++) {
		std::cout << std::fixed << cl_matrix[i] << " ";
		if ((i+1) % N == 0)
			std::cout << "\n";
	}

	std::cout << "\nconstant terms after transformation for OpenCL:\n";
	for (int i = 0; i < N; i++) {
		std::cout << cl_constant_terms[i] << " ";
	}*/
	
	std::cout << "\n\nsolution for OpenCL:\n";
	for (int i = 0; i < N; i++) {
		std::cout << "x" << i+1 << " = " << cl_solution[i] << "\n";
	}
	std::cout << "\n\n";


	//wyniki dla C++:
	/*std::cout << "\n\nmain matrix after transformation for C++:\n";
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << cpp_matrix[i][j] << " ";
		}
		std::cout << "\n";
	}*/

	/*std::cout << "\nconstant terms after transformation for C++:\n";
	for (int i = 0; i < N; i++) {
		std::cout << cpp_constant_terms[i] << " ";
	}*/

	std::cout << "\n\nsolution for C++:\n";
	for (int i = 0; i < N; i++) {
		std::cout << "x" << i + 1 << " = " << cpp_solution[i] << "\n";
	}
	std::cout << "\n\n";

	return 0;
}