__kernel void yoloswag420blazeit360noscope(__global float *input, int size, __global float *rand, float prob, float scale)
{
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}