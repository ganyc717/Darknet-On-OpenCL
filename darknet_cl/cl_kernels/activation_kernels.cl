typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;


float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}

float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate_kernel(float x){return x;}
float logistic_activate_kernel(float x){return 1.f/(1.f + exp(-x));}
float loggy_activate_kernel(float x){return 2.f/(1.f + exp(-x)) - 1;}
float relu_activate_kernel(float x){return x*(x>0);}
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
float tanh_activate_kernel(float x){return (2.f/(1 + exp(-2*x)) - 1);}
float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}

float stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2);
    else return (x - n) + floor(x/2);
}
 

float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
float linear_gradient_kernel(float x){return 1;}
float logistic_gradient_kernel(float x){return (1-x)*x;}
float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}

float relu_gradient_kernel(float x){return (x>0);}
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
float ramp_gradient_kernel(float x){return (x>0)+.1f;}
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
float tanh_gradient_kernel(float x){return 1-x*x;}
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
float stair_gradient_kernel(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}

float activate_kernel(float x, int a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

float gradient_kernel(float x, int a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

__kernel void binary_gradient_array_kernel(__global float *x, __global float *dy, int n, int s, int a, __global float *dx)
{
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) {
        float de = dy[id];
        dx[b*s + i] = x2*de;
        dx[b*s + s/2 + i] = x1*de; 
    }
}

__kernel void binary_activate_array_kernel(__global float *x, int n, int s, int a, __global float *y)
{
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1*x2;
}


__kernel void activate_array_kernel(__global float *x, int n, int a)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n) x[i] = activate_kernel(x[i], a);
}

__kernel void gradient_array_kernel(__global float *x, int n, int a, __global float *delta)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}