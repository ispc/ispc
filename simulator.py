import threading, os, time
from subprocess import Popen, PIPE


class Simulator(object):
    def __init__(self, platform, simu_path, port, verbose=False, flags=""):
# __INTEL_EMBARGO_BEGIN__
        self._fulsim_proc = None
        self._tbxport = port
        self._platform = platform
        self._verbose = verbose;
        self._fulsim_path = simu_path

        if platform == "skl":
            fulsim_tbx = f"-socket tcp:{port}"
            fulsim_flags = f"-device skl.2.a0 {flags}"
        elif platform == "pvc":
            fulsim_tbx = f"-socket tcp:{port}"
            fulsim_flags = f"-device pvc.8x8x8.B0 {flags}" \
                           f"-swsbcheck on"
        elif platform == "dg2":
            fulsim_tbx = f"-socket tcp:{port}"
            # DG2 128EU - dg2.2x4x16
            # DG2 512EU - dg2.8x4x16
            fulsim_flags = f"-device dg2.2x4x16.b0 {flags}" \
                           f"-swsbcheck on"
        elif platform == "mtl":
            fulsim_tbx = f"-socket tcp:{port}"
            fulsim_flags = f"-device mtl.2x4x16.a0 {flags}" \
                           f"-swsbcheck on"

        elif simu_path != "":
            raise SystemError(f"Unsupported Fulsim platform \"{platform}\"")

        # prepare Fulsim command line
        self._cmd = f"{simu_path}/AubLoad {fulsim_tbx} {fulsim_flags} "\
                    "-msglevel terse"

        self._environ = dict()
        self._environ['TbxPort'] = f"{self._tbxport}"
        self._environ['SetCommandStreamReceiver'] = "2"
        # --- this is temp solution for problem with runtime pre-kernel code ---
        self._environ['OverrideRevision'] = "3"
        self._environ['RebuildPrecompiledKernels'] = "1"
        # ----------------------------------------------------------------------
        self._environ['ProductFamilyOverride'] = f"{self._platform}"
# __INTEL_EMBARGO_END__
        pass

    def _start(self):
# __INTEL_EMBARGO_BEGIN__
        # start Fulsim process
        self._fulsim_proc = Popen(self._cmd.split(),
                                 universal_newlines=True,
                                 stdout=PIPE, stderr=PIPE, bufsize=1)
        fulsimReady = False
        timeoutIterations = 60
        lines = []
        # wait for Fulsim to be ready..
        for i in range(timeoutIterations):
            for line in self._fulsim_proc.stdout:
                lines.append(line)
                if self._verbose:
                    print(f"F: {line}", end='')
                if line.startswith('DONE'):
                    fulsimReady = True
                    break
            if fulsimReady:
                break
            time.sleep(1)

        if not fulsimReady:
            print("ERROR: Timeout occured... ")
            print("\n".join(map(str,lines)))
            self._kill()
            return

        # separate thread generating fulsim ouput as program runs
        def fulsim_output_f(fulsim_proc):
            for line in fulsim_proc.stdout:
                if self._verbose:
                    print(f"F: {line}", end='')

        self._fulsim_thread = threading.Thread(target=fulsim_output_f,
                                              args=(self._fulsim_proc,))
        self._fulsim_thread.daemon = True
        self._fulsim_thread.start()
# __INTEL_EMBARGO_END__
        pass

    def _kill(self):
# __INTEL_EMBARGO_BEGIN__
        try:
            if os.name == 'nt':
                Popen(['taskkill', '/F', '/T', '/PID',
                       str(self._fulsim_proc.pid)], shell=True)
            else:
                self._fulsim_proc.terminate()
                os.kill(self._fulsim_proc.pid, 0)
                os.kill(self._fulsim_proc.pid, 9)
        except Exception as e:
            raise Exception(f"Cannot kill pid {self._fulsim_proc.pid}: "\
                            f"Error {str(e)}")

        self._fulsim_proc = None
# __INTEL_EMBARGO_END__
        pass

    def __enter__(self):
# __INTEL_EMBARGO_BEGIN__
        if self._fulsim_path == "":
            return
        self._start()

        for k,v in self._environ.items():
            os.environ[k] = v
# __INTEL_EMBARGO_END__
        pass

    def __exit__(self, exception_type, exception_value, traceback):
# __INTEL_EMBARGO_BEGIN__
        if self._fulsim_path == "":
            return
        self._kill()
# __INTEL_EMBARGO_END__
        pass



