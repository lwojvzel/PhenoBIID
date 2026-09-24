"""Run an unchanged preprocessing CLI in a fresh workspace with read guards."""
import json
import os
from pathlib import Path
import runpy
import sys


def allowed(path, writing, workspace, project, raw_files, environment):
    path = Path(path).resolve()
    if writing:
        return (path.is_relative_to(workspace) or path.is_relative_to(Path('/tmp'))
                or path == Path('/dev/null'))
    if not path.is_relative_to(project):
        return True
    return path in raw_files or path.is_relative_to(environment)


def main():
    workspace = Path(__file__).resolve().parent
    registration = json.loads((workspace / 'registration.json').read_text())
    project = Path(registration['original_project'])
    raw_files = {Path(p).resolve() for p in registration['raw_files']}
    environment = Path(sys.prefix).resolve()
    def guard(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if not allowed(os.fsdecode(args[0]), writing, workspace, project, raw_files, environment):
            raise PermissionError('Original processed/reference access or external write denied')
    sys.addaudithook(guard)
    for target, mode in ((project / 'Data/processed/crop_yield_growing_season/lat.npy', 'rb'),
                         (next(iter(raw_files)), 'ab')):
        try:
            open(target, mode)
        except PermissionError:
            pass
        else:
            raise AssertionError('File guard probe failed')
    script = workspace / 'core' / sys.argv[1]
    if script.parent != workspace / 'core' or script.name not in registration['scripts']:
        raise ValueError('Unregistered preprocessing entrypoint')
    sys.dont_write_bytecode = True
    sys.argv = [str(script), *sys.argv[2:]]
    os.chdir(workspace)
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
