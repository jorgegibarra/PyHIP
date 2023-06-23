import ctypes
from pyhip import hip, hiprtc
source = """
extern "C" __global__ void set(int *a) {
  *a = 10;
}
"""
prog = hiprtc.hiprtcCreateProgram(source, 'set', [], [])
device_properties = hip.hipGetDeviceProperties(0)
print(f"Compiling kernel for {device_properties.gcnArchName}")
if hip.hipGetPlatformName() == 'amd': # offload arch is amd only
  hiprtc.hiprtcCompileProgram(prog, [f'--offload-arch={device_properties.gcnArchName}'])
else:
  hiprtc.hiprtcCompileProgram(prog, [])
code = hiprtc.hiprtcGetCode(prog)
module = hip.hipModuleLoadData(code)
kernel = hip.hipModuleGetFunction(module, 'set')
ptr = hip.hipMalloc(4)

class PackageStruct(ctypes.Structure):
  _fields_ = [("a", ctypes.c_void_p)]

struct = PackageStruct(ptr)
hip.hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, 0, struct)
res = ctypes.c_int(0)
hip.hipMemcpy_dtoh(ctypes.byref(res), ptr, 4)
print(res.value)