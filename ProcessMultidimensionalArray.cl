__kernel void ProcessMultidimensionalArray(__global int* data){

	
	size_t id = (get_global_id(1) * get_global_size(0)) + get_global_id(1);
	
	data[id] = data[id] * 2;

}