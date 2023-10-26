// RUN: %{ispc} --target=host --nowrap --nostdlib -O2 %s -o %t.o

#if   TARGET_WIDTH == 2
#define INITIALIZER 1,0,
#elif TARGET_WIDTH == 4
#define INITIALIZER 1,0,1,0,
#elif TARGET_WIDTH == 8
#define INITIALIZER 1,0,1,0,1,0,1,0,
#elif TARGET_WIDTH == 16
#define INITIALIZER 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
#elif TARGET_WIDTH == 32
#define INITIALIZER 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
#elif TARGET_WIDTH == 64
#define INITIALIZER 1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
#endif

void foo( uniform float dst[], const uniform uint32 baseIndex, const varying float src )
{
    static const varying bool mask = { INITIALIZER };  // crashes
    if ( mask != 0 )
    {
        dst[baseIndex+programIndex] = src;
    }
}
