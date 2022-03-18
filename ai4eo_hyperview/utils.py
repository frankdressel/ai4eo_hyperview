import inspect
import os

class MTimeMixin:
    """
        Mixin that flags a task as incomplete if any requirement
        is incomplete or has been updated more recently than this task
        This is based on http://stackoverflow.com/a/29304506, but extends
        it to support multiple input / output dependencies.
    """

    def complete(self):
        def to_list(obj):
            if type(obj) in (type(()), type([])):
                return obj
            else:
                if type(obj) == dict:
                    return list(obj.values())
                else:
                    return [obj]

        def mtime(path):
            return os.path.getmtime(path)

        if not all(os.path.exists(out.path) for out in to_list(self.output())):
            return False

        self_mtime = min(mtime(out.path) for out in to_list(self.output()))
        if mtime(inspect.getsourcefile(self.__class__)) > self_mtime:
            return False

        # the below assumes a list of requirements, each with a list of outputs. YMMV
        for el in to_list(self.requires()):
            if not el.complete():
                return False
            for output in to_list(el.output()):
                if mtime(output.path) > self_mtime:
                    return False

        return True

def mse(data, ref):
    return ((data - ref)**2).mean()
