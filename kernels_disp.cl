__kernel void display(__global float *x, __global float *y, __global float *z,
    __global int *image, int maxx, int maxy, float al, float be, float mu, int co)
{
    int gid = get_global_id(0);

    float xx = x[gid] * mu;
    float yy = y[gid] * mu;
    float zz = z[gid] * mu;

    float xx1 = cos(al) * xx - sin(al) * yy;
    float yy1 = sin(al) * xx + cos(al) * yy;
    float zz1 = zz;

    float xx2 = xx1;
    float yy2 = cos(be) * yy1 - sin(be) * zz1;
    float zz2 = sin(be) * yy1 + cos(be) * zz1;

    if(co==0)
    {
        int xxx = round(xx2 - 0.5f * yy2 + maxx / 2.0f);
        int yyy = round(-zz2 - 0.5f * yy2 + maxy / 2.0f);


        if(xxx >= 0 && xxx < maxx && yyy >=0 && yyy < maxy)
        {
            atomic_inc(&(image[(yyy * maxx + xxx) * 4+1]));
            atomic_inc(&(image[(yyy * maxx + xxx) * 4+2]));
        }
    }
    else
    {
        int xxx = round(xx2 + maxx / 2.0f);
        int yyy = round(-zz2 + maxy / 2.0f);

        if(xxx >= 0 && xxx < maxx && yyy >=0 && yyy < maxy)
        {
            atomic_inc(&(image[(yyy * maxx + xxx) * 4+1]));
            atomic_inc(&(image[(yyy * maxx + xxx) * 4+2]));
        }

        xxx = round(xx2 - 0.2f * yy2 + maxx / 2.0f);
        yyy = round(-zz2 + maxy / 2.0f);

        if(xxx >= 0 && xxx < maxx && yyy >=0 && yyy < maxy)
        {
            atomic_inc(&(image[(yyy * maxx + xxx) * 4+0]));
        }
    }
}