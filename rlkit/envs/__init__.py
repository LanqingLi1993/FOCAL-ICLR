import os
import importlib


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn


# automatically import any envs in the envs/ directory

def _tabulate(module_name):
    print("import module: {}".format(module_name))

for file in os.listdir(os.path.dirname(__file__)):
    if 'MUJOCO_PY_MJPRO_PATH' in os.environ.keys():
        mjpro_path = os.environ.get('MUJOCO_PY_MJPRO_PATH')
        mjpro = os.path.basename(mjpro_path)
    else:
        mjpro = None
    if mjpro is not None:
        if file.endswith('.py') and not file.startswith('_') and not file.startswith('non_mujoco'):
            if mjpro != 'mjpro131':
                if not file.startswith('walker_rand_params'):
                    module = file[:file.find('.py')]
                    importlib.import_module('rlkit.envs.' + module)
                    _tabulate(module)
            else:
                if file.startswith('walker_rand_params'):
                    module = file[:file.find('.py')]
                    importlib.import_module('rlkit.envs.' + module)
                    _tabulate(module)
    else:
        if file.endswith('.py') and file.startswith('non_mujoco'):
            module = file[:file.find('.py')]
            importlib.import_module('rlkit.envs.' + module)
            _tabulate(module)

