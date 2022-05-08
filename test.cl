__kernel void test(__global int* data, int N){

	int id = get_global_id(0);
	int our_value = array[id];
	int x = id%width
	
	int i;
	for(i=0;i<N;i++){	
		data[i] += 1;
	}
	
}