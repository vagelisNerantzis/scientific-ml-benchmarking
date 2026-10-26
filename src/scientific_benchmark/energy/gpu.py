"""Optional NVML board-power integration, not total system or process energy."""
from threading import Event, Thread
from time import perf_counter
import warnings

class GPUEnergyMonitor:
    def __init__(self, *, device_index=0, interval=0.1, enabled=False):
        if interval<=0:
            raise ValueError('Sampling interval must be positive')
        self.enabled,self.device_index,self.interval = enabled,device_index,interval
        self.available=False; self.energy_wh=None; self.samples=[]; self.error=None

    def __enter__(self):
        self.samples=[];self.stop=Event();self.initialized=False;self.available=False
        self.energy_wh=None;self.error=None
        if self.enabled:
            try:
                import pynvml
                self.nvml=pynvml; pynvml.nvmlInit();self.initialized=True
                self.handle=pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._sample();self.available=True
                self.thread=Thread(target=self._run,daemon=True);self.thread.start()
            except Exception as exc:
                self.error=type(exc).__name__
                warnings.warn('NVML GPU energy unavailable: '+self.error,RuntimeWarning)
                if self.initialized:
                    self.nvml.nvmlShutdown();self.initialized=False
        return self

    def _sample(self):
        self.samples.append((perf_counter(),self.nvml.nvmlDeviceGetPowerUsage(self.handle)/1000))

    def _run(self):
        while not self.stop.wait(self.interval):
            try:
                self._sample()
            except Exception as exc:
                self.error=type(exc).__name__;return

    def __exit__(self,*exc):
        if not self.available:
            return
        self.stop.set();self.thread.join()
        try:
            self._sample()
            if self.error is None:
                self.energy_wh=sum((b[0]-a[0])*(a[1]+b[1])/2
                                   for a,b in zip(self.samples,self.samples[1:]))/3600
        except Exception as error:
            self.error=type(error).__name__
        finally:
            self.nvml.nvmlShutdown()
        if self.error:
            self.available=False
            warnings.warn('Incomplete GPU telemetry; energy omitted',RuntimeWarning)
