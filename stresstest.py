from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count, Pool, TimeoutError
from time import sleep, strftime, gmtime, clock, time
import signal
from functools import partial
import re
import os
import argparse
import subprocess


class TestError(Exception):
    """
    Base Class for Test Exceptions.
    """
    pass


class TestTypeError(TestError):
    """
    Class for Exception by not set type of test.
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


def unwrap_func(*args, **kwargs):
    TestCPU._f(*args, **kwargs)



class TestCPU(AbstractTest):
    """
    Test for load CPU. It can load by CPU count.
    By defaults load = 100% for all CPU core.
    """

    def __init__(self, cpu_core=None, cpu_util=None):
        self.cpu_core = cpu_core or cpu_count()
        self.cpu_util = cpu_util or 100

    def __str__(self):
        return 'TestCPU'

    def __repr__(self):
        return 'TestCPU'

    @staticmethod
    def _init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    @staticmethod
    def _f(x, load):
        time_of_run = 0.1
        percent_cpu = load
        cpu_time_utilisation = float(percent_cpu) / 100
        on_time = time_of_run * cpu_time_utilisation
        off_time = time_of_run * (1 - cpu_time_utilisation)
        while True:
            start_time = clock()
            while clock() - start_time < on_time:
                x * x
            sleep(off_time)

    def run_test(self, timeout):
        pool = Pool(self.cpu_core, initializer=TestCPU._init_worker)
        try:
            # result = pool.map_async(unwrap_func, range(self.cpu_core))
            result = pool.map_async(partial(unwrap_func, load=self.cpu_util), range(self.cpu_core))
            result.get(timeout)
        except KeyboardInterrupt:
            pool.terminate()
            print('Test finish by ^C')
        except TimeoutError:
            pool.terminate()
            print('Test finish by timeout.')
        finally:
            pool.join()


class TestMem(AbstractTest):
    """
    Test for load memory. It can load by size or percentage of free memory.
    """

    def __init__(self, size=None, chunk=None):
        self.chunk = chunk or 1024 * 1024 * 100
        self.size = size or '100%'
        self.measurement = {'GB': 1073741824, 'MB': 1048576, 'KB': 1024}
        self.meminfo = TestMem._detect_memory()
        self.bytes = self._size_to_bytes()
        self.list_obj = []

    @staticmethod
    def _detect_memory():
        mem = None
        try:
            with open('/proc/meminfo') as f:
                mem = {l.split(':')[0]: (int(re.sub('\D', '', l)), re.sub('.*\d ', '', l[:-1]).upper()) for l in f.readlines()}
        except IOError:
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

    def run_test(self, timeout):
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
            print(strftime("Test load. Waiting for %Hh:%Mm:%Ss", gmtime(timeout)))
            try:
                sleep(timeout)
            except KeyboardInterrupt:
                del self.list_obj[:]
                return 'Test finish.'
            else:
                del self.list_obj[:]
                return 'Test finish.'


class WriteTestHDD(AbstractTest):
    """
    Test for load disc. It use dd with flag direct to write without FS caches.
    It write file of 1 GB by defaultsx by loop for set time.
    """
    def __init__(self, mount=None, bs=None, count=None):
        self.measurement = {'GB': 1073741824, 'MB': 1048576, 'KB': 1024}
        self.mount = mount or '.'
        self.bs = self._size_to_bytes(bs) or self._size_to_bytes('1G')
        self.count = count or 1
        self.free_space = self._get_free_space()

    def _get_free_space(self):
        stats = os.statvfs(self.mount)
        return stats.f_frsize * stats.f_bavail

    def _size_to_bytes(self, size):
        if size is None:
            return None
        measurement = re.sub('\d', '', size).upper()
        amount = int(re.sub('\D', '', size))
        MMB = {'G': 'GB', 'M': 'MB', 'K': 'KB'}
        return amount * self.measurement[MMB.get(measurement, measurement)]

    def run_test(self, timeout):
        if self.free_space < self.bs * self.count:
            raise TestError('Not enough free space on {}. Required {} bytes'.format(self.mount, self.bs * self.count))
        if os.path.exists('{}/test.img'.format(self.mount)):
            raise TestError('File {} exist.'.format(self.mount))
        cmd_list = ['dd', 'if=/dev/zero', 'of={}/test.img'.format(self.mount),
                    'bs={}'.format(self.bs), 'count={}'.format(self.count), 'oflag=direct', 'conv=fdatasync']
        time_finish = time() + timeout
        while time_finish > time():
            try:
                a = subprocess.Popen(cmd_list, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = a.communicate()
                print(out)
            except KeyboardInterrupt:
                print('Test finish by ^C')
            finally:
                os.remove('{}/test.img'.format(self.mount))
        return 'Test finish.'


class StressTest(object):

    def __init__(self, test_time, test_type=None):
        self.test_time = test_time * 60
        self.test_type = test_type
        if self.test_type is None:
            raise TestTypeError('You have to set test type.')

    def run_test(self):
        return self.test_type.run_test(self.test_time)


def _main():
    parser = argparse.ArgumentParser(description='Script for simulate system load.')
    parser.add_argument('--type', choices=('memory', 'cpu', 'hdd'), required=True, help='Type of generate load.')
    parser.add_argument('--time', type=int, required=True, help='How long in minute load.')
    parser.add_argument('--cpu-count', type=int, help='How much core to use. By default is all.')
    parser.add_argument('--cpu-util', type=int, help='How much load to generate. By default is 100%')
    parser.add_argument('--mem-size', type=str, help='Size of memory to load.')
    parser.add_argument('--mem-chunk', type=int, help='Size of chunk in bytes. By default chunk is 100MiB.')
    parser.add_argument('--hdd-mount', type=str, help='Mount to write. By default is near the script.')
    parser.add_argument('--hdd-bs', type=str, help='Write up to BYTES bytes at a time. By default is 1GiB.')
    parser.add_argument('--hdd-count', type=int, help='Size of chunk in bytes. By default chunk is 100MiB.')
    args = parser.parse_args()
    test = None
    if args.type == 'memory':
        test = StressTest(test_time=args.time, test_type=TestMem(size=args.mem_size))
    if args.type == 'cpu':
        test = StressTest(test_time=args.time, test_type=TestCPU(cpu_core=args.cpu_count, cpu_util=args.cpu_util))
    if args.type == 'hdd':
        test = StressTest(test_time=args.time, test_type=WriteTestHDD(mount=args.hdd_mount, bs=args.hdd_bs,
                                                                      count=args.hdd_count))
    if test:
        test.run_test()


if __name__ == '__main__':
    _main()
