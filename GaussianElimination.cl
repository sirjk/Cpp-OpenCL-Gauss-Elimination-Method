__kernel void GaussianElimination(__global float* matrix, __global float* constantTerms, __global float* solution, int size){
	
	float ratio;
	float difference;
	
	//utworzenie macierzy trojkatnej gornej
	
	for (int i = 0; i < size - 1; i++) {
		for (int j = i + 1; j < size; j++) {
			ratio = matrix[j*size + i] / matrix[i*size + i];
			for (int k = 0; k < size; k++) {
				matrix[j*size + k] -= matrix[i*size + k] * ratio;
			}
			constantTerms[j] -= constantTerms[i] * ratio;
		}
	}

	//znalezienie rozwiazania
	for (int i = size - 1; i >= 0; i--) {
		difference = 0;
		if (i == size - 1) {
			solution[i] = constantTerms[i] / matrix[i*size + i];
		}
		else {
			for (int j = i + 1; j < size; j++) {
				difference += matrix[i*size + j] * solution[j];
			}
			solution[i] = (constantTerms[i] - difference) / matrix[i*size + i];
		}
	}
	
}
