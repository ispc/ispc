// RUN: %{ispc} %s --target=gen9-x16 --arch=xe64 --emit-llvm-text --nowrap -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// REQUIRES: XE_ENABLED
struct vec3f {
    float x;
    float y;
    float z;
};
struct Volume {
    float num;
    vec3f dimensions;
};
float square1(void *uniform  _self, const vec3f &localCoordinates) {
    Volume* uniform self = (Volume* uniform)_self;
    const uniform vec3f test = self->dimensions;
    if (localCoordinates.x < 0.f ||
      localCoordinates.x > test.x - 1.f ||
      localCoordinates.y < 0.f ||
      localCoordinates.y > test.y - 1.f ||
      localCoordinates.z < 0.f ||
      localCoordinates.z > test.z - 1.f) {
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK-NOT: logical_op_done
    return 0;
  }
  return self->num*localCoordinates.x;
}

float square2(void *uniform  _self, const vec3f &localCoordinates, uniform float x, uniform float y, uniform float z) {
  Volume* uniform self = (Volume* uniform)_self;
  if (localCoordinates.x < 0.f ||
    localCoordinates.x > x - 1.f ||
    localCoordinates.y < 0.f ||
    localCoordinates.y > y - 1.f ||
    localCoordinates.z < 0.f ||
    localCoordinates.z > z - 1.f) {
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK: or <16 x i1>
  // CHECK-NOT: logical_op_done
    return 0;
  }
  return self->num*localCoordinates.x;
}

float square3(void *uniform  _self, const vec3f &localCoordinates) {
    Volume* uniform self = (Volume* uniform)_self;
    if (localCoordinates.x < 0.f ||
      localCoordinates.x > self->dimensions.x - 1.f ||
      localCoordinates.y < 0.f ||
      localCoordinates.y > self->dimensions.y - 1.f ||
      localCoordinates.z < 0.f ||
      localCoordinates.z > self->dimensions.z - 1.f) {
// CHECK: logical_op_done
// CHECK: logical_op_done
// CHECK: logical_op_done
    return 0;
  }

  return self->num*localCoordinates.x;
}

task void test1(void* uniform _self, int* uniform out){
  vec3f coord = {out[0], out[1], out[2]};
  out[programIndex]=square1(_self, coord);
}

task void test2(void* uniform _self, int* uniform out){
  vec3f coord = {out[0], out[1], out[2]};
  out[programIndex]=square2(_self, coord, out[0], out[1], out[2]);
}

task void test3(void* uniform _self, int* uniform out){
  vec3f coord = {out[0], out[1], out[2]};
  out[programIndex]=square3(_self, coord);
}
