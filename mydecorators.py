import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{0} ({1}, {2}) {3:.2f} sec'.format(method.__name__, args, kw, te-ts))
        return result

    return timed
