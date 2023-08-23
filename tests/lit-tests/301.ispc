// RUN: %{ispc} --target=host --nostdlib --nowrap -o %t.o %s

static unmasked void lHomogeneousRast();

typedef unmasked void (*ARasterizeFunc)();

struct ARasterizerMethods {
    ARasterizeFunc RasterizeGrid;
};

static uniform ARasterizerMethods lHomogeneousMethods = {
    lHomogeneousRast
};
