from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count, Pool, TimeoutError
from time import sleep, strftime, gmtime, time
from signal import signal, SIG_IGN, SIGINT
from functools import partial
import re
import os


class TestError(Exception):
    """
    Base Class for Test Exceptions.
    """
    pass


class TestTypeError(TestError):
    """
    Class for Exception by not set type og test.
    """
    pass


class AbstractTest(object):
    """
    Abstract Class for tests.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def run_test(self, time):
        pass


class TestCPU(AbstractTest):
    """
    Test for load CPU. It can load by CPU count.
    By defaults load = 100% for all CPU core.
    """

    def __init__(self, cpu_core=None, cpu_util=None):
        self.cpu_core = cpu_core or cpu_count()
        self.cpu_util = cpu_util or 100

    @staticmethod
    def init_worker():
        signal(SIGINT, SIG_IGN)

    @staticmethod
    def _f(x, timeout):
        time_end = time() + timeout
        while time_end >= time():
            try:
                x * x
            except KeyboardInterrupt:
                break

    def run_test(self, timeout):
        pool = Pool(self.cpu_core, TestCPU.init_worker)
        try:
            result = pool.map(partial(TestCPU._f, timeout=timeout), range(self.cpu_core))
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            print('Test finish by ^C')
        else:
            print('Test finish by timeout.')
            pool.close()
            pool.join()


class TestMem(AbstractTest):
    """
    Test for load memory. It can load by size or percentage of free memory.
    """

    def __init__(self, size=None, chunk=None):
        self.chunk = chunk or 1024 * 1024 * 100
        self.size = size or '100%'
        self.measurement = {'GB': 1_073_741_824, 'MB': 1_048_576, 'KB': 1_024}
        self.meminfo = TestMem._detect_memory()
        self.bytes = self._size_to_bytes()
        self.list_obj = []

    @staticmethod
    def _detect_memory():
        mem = None
        try:
            with open('/proc/meminfo') as f:
                mem = {l.split(':')[0]: (re.sub('\D'', ''', l),re.sub('.*\d ', '', a).upper()) for l in f.readlines()}
        except FileNotFoundError:
            print('File /proc/meminfo not found.\nYou can\'t use percentage.')
        return mem

    def _size_to_bytes(self):
        measurement = re.sub('\d', '', self.size).upper()
        amount = int(re.sub('\D', '', self.size))
        if measurement == '%':
            if self.meminfo:
                free_in_bytes = self.meminfo['MemFree'][0] * self.measurement[self.meminfo['MemFree'][1]]
                return int(amount * free_in_bytes / 100)
            return None
        MMB = {'G': 'GB', 'M': 'MB', 'K': 'KB'}
        return amount * self.measurement[MMB.get(measurement, measurement)]

    def _get_chunk(self):
        while self.bytes // self.chunk > 0:
            current = self.bytes
            self.bytes = self.bytes - self.chunk
            yield current - self.bytes
        else:
            yield self.bytes

    def run_test(self, time):
        if self.bytes:
            for chunk in self._get_chunk():
                try:
                    self.list_obj.append(' ' * chunk)
                except MemoryError:
                    print('Not enough memory.')
                    break
                except KeyboardInterrupt:
                    break
                else:
                    print('Load {}/{}.'.format(chunk, self.bytes))
                    sleep(2)
            print(strftime("Test load. Waiting for %Hh:%Mm:%Ss", gmtime(time)))
            try:
                sleep(time)
            except KeyboardInterrupt:
                self.list_obj.clear()
                return 'Test finish.'
            else:
                self.list_obj.clear()
                return 'Test finish.'


class WriteTestHDD(AbstractTest):
    """
    Test for load disc. It use dd with flag direct to write without FS caches.
    It write file of 1 GB by defaultsx by loop for set time.
    """
    def __init__(self, mount, file_size=1024):
        self.mount = mount
        self.file_size = file_size
        self.free_space = self._get_free_space()

    def _get_free_space(self):
        stats = os.statvfs(self.mount)
        return stats.f_frsize * stats.f_bavail / 1024 / 1024

    def run_test(self, time):
        pass


class StressTest:

    def __init__(self, test_time, test_type=None):
        self.test_time = test_time
        self.test_type = test_type
        if self.test_type is None:
            raise TestTypeError('You have to set test type.')

    def run_test(self):
        return self.test_type.run_test(self.test_time)

def _main():
    test_type = StressTest(test_time=5)


# if __name__ == '__main__':
#     _main()