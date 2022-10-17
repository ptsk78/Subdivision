void doRungeKuttaStep(float ss, float *x, float *y, float *z, void (*derivative(float, float, float, float*, float*, float*)))
{
    float dx1, dy1, dz1;
    derivative(*x, *y, *z, &dx1, &dy1, &dz1);

    float x2, y2, z2;
    x2 = *x + 0.5f * ss * dx1;
    y2 = *y + 0.5f * ss * dy1;
    z2 = *z + 0.5f * ss * dz1;

    float dx2, dy2, dz2;
    derivative(x2, y2, z2, &dx2, &dy2, &dz2);

    float x3, y3, z3;
    x3 = *x + 0.5f * ss * dx2;
    y3 = *y + 0.5f * ss * dy2;
    z3 = *z + 0.5f * ss * dz2;

    float dx3, dy3, dz3;
    derivative(x3, y3, z3, &dx3, &dy3, &dz3);

    float x4, y4, z4;
    x4 = *x + ss * dx3;
    y4 = *y + ss * dy3;
    z4 = *z + ss * dz3;

    float dx4, dy4, dz4;
    derivative(x4, y4, z4, &dx4, &dy4, &dz4);

    *x = *x + ss * (dx1 + 2.0f * dx2 + 2.0f * dx3 + dx4) / 6.0f;
    *y = *y + ss * (dy1 + 2.0f * dy2 + 2.0f * dy3 + dy4) / 6.0f;
    *z = *z + ss * (dz1 + 2.0f * dz2 + 2.0f * dz3 + dz4) / 6.0f;
}

inline bool isOutside(float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z,
    float nx, float ny, float nz)
{
    if(nx < min_x || nx > max_x || ny < min_y || ny > max_y || nz < min_z || nz > max_z)
    {
        return true;
    }
    return false;
}

void chaotic(float *x, float *y, float *z)
{
    float nx,ny, nz;
    
    nx = *y;
    ny = *z;
    nz = 1.4f + 0.1f * (*x) + 0.3f * (*y) - (*z) * (*z);
    
    *x = nx;
    *y = ny;
    *z = nz;
}

void derivativeRossler(float x, float y, float z, float *dx, float *dy, float *dz)
{
    *dx = (-y-z);
    *dy = (x+0.2f*y);
    *dz = (0.2f+z*(x-5.7f));
}

void derivativeLorenz(float x, float y, float z, float *dx, float *dy, float *dz)
{
    *dx = (10.0f*(y-x));
    *dy = (x*(28.0f-z)-y);
    *dz = (x*y-8.0f/3.0f*z);
}

void derivativeAizawa(float x, float y, float z, float *dx, float *dy, float *dz)
{
    *dx = ((z-0.7f)*x-3.5f*y);
    *dy = (3.5f*x+(z-0.7f)*y);
    *dz = (0.6f+0.95f*z-z*z*z/3.0f-(x*x+y*y)*(1.0f+0.25f*z)+0.1f*z*x*x*x);
}

void derivativeHalvorsen(float x, float y, float z, float *dx, float *dy, float *dz)
{
    *dx = (-1.89f*x-4.0f*y-4.0*z-y*y);
    *dy = (-1.89f*y-4.0f*z-4.0*x-z*z);
    *dz = (-1.89f*z-4.0f*x-4.0*y-x*x);
}

void dostepContinuous(float *x, float *y, float *z,
                      float min_x, float max_x,
                      float min_y, float max_y,
                      float min_z, float max_z,
                      int numSteps, float ss,
                      void (*derivative(float, float, float, float*, float*, float*)))
{
    float nx, ny, nz;

    nx = *x;
    ny = *y;
    nz = *z;
    for(int i=0;i<numSteps;i++)
    {
        doRungeKuttaStep(ss, &nx, &ny, &nz, derivative);
        if(isOutside(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz))
        {
            break;
        }
    }

    *x = nx;
    *y = ny;
    *z = nz;
}

enum whichSystem { sys_chaotic = 0, sys_rossler = 1, sys_lorenz = 2, sys_aizawa = 3, sys_halvorsen = 4};

__kernel void dostep(__global long *active, __global long *result,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z,
    int system, int numSteps, float ss,
    int r, int tp)
{
    long gid = get_global_id(0);

    long tmp = gid;
    long x1 = tmp % tp;
    tmp /= tp;
    long y1 = tmp % tp;
    tmp /= tp;
    long z1 = tmp % tp;
    tmp /= tp;
    long whichbox = active[tmp];
    long x2 = whichbox % (1<<r);
    whichbox /= (1<<r);
    long y2 = whichbox % (1<<r);
    whichbox /= (1<<r);
    long z2 = whichbox;

    float x = min_x + (max_x-min_x) * ((float)x2) / ((float)(1<<r)) + (max_x-min_x) / ((float)(1<<r)) * (1.0f + 2.0f * ((float)x1)) / (2.0f * ((float)tp));
    float y = min_y + (max_y-min_y) * ((float)y2) / ((float)(1<<r)) + (max_y-min_y) / ((float)(1<<r)) * (1.0f + 2.0f * ((float)y1)) / (2.0f * ((float)tp));
    float z = min_z + (max_z-min_z) * ((float)z2) / ((float)(1<<r)) + (max_z-min_z) / ((float)(1<<r)) * (1.0f + 2.0f * ((float)z1)) / (2.0f * ((float)tp));

    switch(system)
    {
        case sys_chaotic:
            chaotic(&x, &y, &z);
            break;
        case sys_rossler:
            dostepContinuous(&x, &y, &z, min_x, max_x, min_y, max_y, min_z, max_z, numSteps, ss, &derivativeRossler);
            break;
        case sys_lorenz:
            dostepContinuous(&x, &y, &z, min_x, max_x, min_y, max_y, min_z, max_z, numSteps, ss, &derivativeLorenz);
            break;
        case sys_aizawa:
            dostepContinuous(&x, &y, &z, min_x, max_x, min_y, max_y, min_z, max_z, numSteps, ss, &derivativeAizawa);
            break;
        case sys_halvorsen:
            dostepContinuous(&x, &y, &z, min_x, max_x, min_y, max_y, min_z, max_z, numSteps, ss, &derivativeHalvorsen);
            break;
        default:
            break;
    }

    if(isOutside(min_x, max_x, min_y, max_y, min_z, max_z, x, y, z))
    {
        result[gid] = -1;
    }
    else
    {
        x1 = (int)((float)(x-min_x) / (max_x-min_x) * ((float)(1<<(r+1))));
        y1 = (int)((float)(y-min_y) / (max_x-min_y) * ((float)(1<<(r+1))));
        z1 = (int)((float)(z-min_z) / (max_x-min_z) * ((float)(1<<(r+1))));
        if((x1>=0)&&(x1<(1<<(r+1)))&&(x1>=0)&&(x1<(1<<(r+1)))&&(x1>=0)&&(x1<(1<<(r+1))))
        {
            result[gid] = x1 + y1 * (1<<(r+1)) + z1 * (1<<(r+1)) * (1<<(r+1));
        }
        else
        {
            result[gid] = -1;
        }
    }
}
