float get_pixel_kernel(__global float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

float3 rgb_to_hsv_kernel(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y; 
    float b = rgb.z;

    float h, s, v;
    float max_ = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min_ = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max_ - min_;
    v = max_;
    if(max_ == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max_;
        if(r == max_){
            h = (g - b) / delta;
        } else if (g == max_) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return (float3)(h, s, v);
}

float3 hsv_to_rgb_kernel(float3 hsv)
{
    float h = hsv.x;
    float s = hsv.y; 
    float v = hsv.z;

    float r, g, b;
    float f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int) floor(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return (float3)(r, g, b);
}

float bilinear_interpolate_kernel(__global float *image, int w, int h, float x, float y, int c)
{
    int ix = (int) floor(x);
    int iy = (int) floor(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__kernel void levels_image_kernel(__global float *image, __global float *rand, int batch, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift)
{
    int size = batch * w * h;

    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    float rshift = rand[0];
    float gshift = rand[1];
    float bshift = rand[2];
    float r0 = rand[8*id + 0];
    float r1 = rand[8*id + 1];
    float r2 = rand[8*id + 2];
    float r3 = rand[8*id + 3];

    saturation = r0*(saturation - 1) + 1;
    saturation = (r1 > .5f) ? 1.f/saturation : saturation;
    exposure = r2*(exposure - 1) + 1;
    exposure = (r3 > .5f) ? 1.f/exposure : exposure;

    size_t offset = id * h * w * 3;
    //image += offset;
    float r = image[x + w*(y + h*0) + offset];
    float g = image[x + w*(y + h*1) + offset];
    float b = image[x + w*(y + h*2) + offset];
    float3 rgb = (float3)(r,g,b);
    if(train){
        float3 hsv = rgb_to_hsv_kernel(rgb);
        hsv.y *= saturation;
        hsv.z *= exposure;
        rgb = hsv_to_rgb_kernel(hsv);
    } else {
        shift = 0;
    }
    image[x + w*(y + h*0) + offset] = rgb.x*scale + translate + (rshift - .5f)*shift;
    image[x + w*(y + h*1) + offset] = rgb.y*scale + translate + (gshift - .5f)*shift;
    image[x + w*(y + h*2) + offset] = rgb.z*scale + translate + (bshift - .5f)*shift;
}

__kernel void forward_crop_layer_kernel(__global float *input, __global float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle, __global float *output)
{
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(id >= size) return;

    float cx = w/2.f;
    float cy = h/2.f;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    float r4 = rand[8*b + 4];
    float r5 = rand[8*b + 5];
    float r6 = rand[8*b + 6];
    float r7 = rand[8*b + 7];

    float dw = (w - crop_width)*r4;
    float dh = (h - crop_height)*r5;
    flip = (flip && (r6 > .5f));
    angle = 2*angle*r7 - angle;
    if(!train){
        dw = (w - crop_width)/2.f;
        dh = (h - crop_height)/2.f;
        flip = 0;
        angle = 0;
    }

    input += w*h*c*b;

    float x = (flip) ? w - dw - j - 1 : j + dw;    
    float y = i + dh;

    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;

    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);
}