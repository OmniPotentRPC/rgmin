project(
    'rgmin',
    'c',
    'cpp',
    version: '0.3.0',
    meson_version: '>=1.3.0',
    default_options: ['buildtype=debugoptimized', 'warning_level=3'],
    license: 'MIT',
)

fs = import('fs')
_msproot = meson.project_source_root()
_incdir = include_directories('include', 'tests/support')
if not fs.is_file(_msproot / 'include' / 'xts.h')
    error('missing shipped header include/xts.h')
endif

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cargo_prog = find_program('cargo', required: true)
_py = find_program('python3', 'python')

# Native Cargo dependencies share the consumer's compiler and libc ABI.
_cargo_env = {
    'CC': ' '.join(cc.cmd_array()),
    'CXX': ' '.join(cpp.cmd_array()),
    'RGMIN_NATIVE_TOOLCHAIN': cc.get_id() + ':' + cc.version() + '/' + cpp.get_id() + ':' + cpp.version(),
}

# Native Rust and C objects use one linker and libc ABI.
rustc_prog = find_program('rustc', required: true)
_rust_host = ''
foreach line : run_command(rustc_prog, '-vV', check: true).stdout().split('\n')
    if line.startswith('host: ')
        _rust_host = line.substring(6).strip()
    endif
endforeach
if _rust_host == ''
    error('rustc did not report its host target')
endif
_rust_linker_key = 'CARGO_TARGET_' + _rust_host.to_upper().replace('-', '_') + '_LINKER'
_cargo_env += {_rust_linker_key: cc.cmd_array()[-1]}

_rust_sys_deps = [dependency('threads')]
if host_machine.system() == 'linux'
    # Name-only: find_library('dl') against a pixi gcc is /usr/lib/libdl.so
    # and Meson records that host dir in the consumer's DT_RPATH.
    if cc.has_link_argument('-ldl')
        _rust_sys_deps += declare_dependency(link_args: ['-ldl'])
    else
        _rust_sys_deps += cc.find_library('dl', required: true)
    endif
    if cc.has_link_argument('-lm')
        _rust_sys_deps += declare_dependency(link_args: ['-lm'])
    else
        _rust_sys_deps += cc.find_library('m', required: true)
    endif
endif

_so = host_machine.system() == 'darwin' ? '.dylib' : '.so'
_cargo_shared = 'librgmin' + _so
_cargo_static = 'librgmin.a'
_profile = get_option('buildtype') == 'debug' ? 'debug' : 'release'

# Cargo rustc compiles src/**/*.rs and reruns on build.rs,
# cbindgen.toml, and include/. depend_files lists that set plus Cargo.lock.
_rgmin_rs_raw = run_command(
    _py,
    '-c',
    'import pathlib, sys; r = pathlib.Path(sys.argv[1]); print("\\n".join(sorted(str(p.relative_to(r)) for p in r.glob("src/**/*.rs"))))',
    _msproot,
    check: true,
).stdout().strip()
if _rgmin_rs_raw != ''
    _rgmin_rs = _rgmin_rs_raw.split('\n')
else
    _rgmin_rs = []
endif

_rgmin_meta = []
foreach _f : ['build.rs', 'cbindgen.toml']
    if not fs.is_file(_msproot / _f)
        error('missing ' + _f)
    endif
    _rgmin_meta += _f
endforeach

_rgmin_inc_raw = run_command(
    _py,
    '-c',
    'import pathlib, sys; r = pathlib.Path(sys.argv[1]); print("\\n".join(sorted(str(p.relative_to(r)) for p in list(r.glob("include/**/*.h")) + list(r.glob("include/**/*.hpp")))))',
    _msproot,
    check: true,
).stdout().strip()
if _rgmin_inc_raw != ''
    _rgmin_inc = _rgmin_inc_raw.split('\n')
else
    error('missing rgmin include headers')
endif

rgmin_cargo = custom_target(
    'rgmin-cargo',
    input: files('Cargo.toml', 'src/lib.rs'),
    depend_files: files('Cargo.lock') + files(_rgmin_meta) + files(_rgmin_rs) + files(_rgmin_inc),
    output: [_cargo_shared, _cargo_static],
    command: [
        _py,
        files('scripts/meson_cargo_build.py'),
        cargo_prog.full_path(),
        _msproot,
        meson.current_build_dir() / 'cargo-target',
        _profile,
        _cargo_shared,
        _cargo_static,
        '@OUTPUT0@',
        '@OUTPUT1@',
        get_option('rgmin_features'),
    ],
    env: _cargo_env,
    console: true,
)

rgmin_dep = declare_dependency(
    include_directories: _incdir,
    link_with: rgmin_cargo[1],
    dependencies: _rust_sys_deps,
    compile_args: ['-DUSE_RGMIN'],
)
meson.override_dependency('rgmin', rgmin_dep)
