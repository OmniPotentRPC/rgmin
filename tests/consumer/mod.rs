//! Build librgmin through cargo and run C and C++ programs against it.
//!
//! `cargo test` compiles the cdylib and staticlib only as inputs to the
//! integration tests and leaves them in its build directory, whose layout
//! changes between cargo releases. `cargo build --lib` for the same target,
//! profile and features finds that build fresh, copies the libraries to
//! where `cargo build` puts them, and names them in its JSON messages.
//!
//! A dev-dependency that a build script also pulls in, such as `cc` or
//! `serde_json`, would make `cargo test` build that crate for the build
//! script differently from `cargo build`, and the two would stop sharing
//! the library. So this module reads the JSON and picks the compiler itself.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

const TARGET: &str = env!("RGMIN_BUILD_TARGET");
const HOST: &str = env!("RGMIN_BUILD_HOST");

enum Library {
    /// The shared library, and the import library MSVC links it through.
    Shared {
        path: PathBuf,
        import: Option<PathBuf>,
    },
    Static(PathBuf),
}

/// A compiler and the words before the first argument, as `CC` gives them.
struct Compiler {
    words: Vec<String>,
    cpp: bool,
}

impl Compiler {
    fn msvc(&self) -> bool {
        Path::new(&self.words[0]).file_stem().is_some_and(|stem| {
            stem.eq_ignore_ascii_case("cl") || stem.eq_ignore_ascii_case("clang-cl")
        })
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.words[0]);
        command
            .args(&self.words[1..])
            .args(match (self.msvc(), self.cpp) {
                (true, true) => &["-nologo", "-std:c++17", "-EHsc"][..],
                (true, false) => &["-nologo", "-std:c11"],
                (false, true) => &["-std=c++17"],
                (false, false) => &["-std=c11"],
            });
        command
    }

    fn builds_hello(&self, dir: &Path) -> bool {
        let (file, text) = if self.cpp {
            ("hello.cpp", "#include <cstdio>\nint main() { return 0; }\n")
        } else {
            (
                "hello.c",
                "#include <stdio.h>\nint main(void) { return 0; }\n",
            )
        };
        let source = dir.join(file);
        fs::write(&source, text).expect("write the compiler probe");
        self.command()
            .current_dir(dir)
            .arg(&source)
            .arg("-o")
            .arg(dir.join(format!("hello{}", env::consts::EXE_SUFFIX)))
            .output()
            .is_ok_and(|probe| probe.status.success())
    }
}

/// Compile `source` against `include/` and librgmin, run it, and return
/// its output. A `.cpp` source is C++17, any other C11.
pub fn compile_and_run(source: &str, name: &str) -> Output {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cpp = source.ends_with(".cpp");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::create_dir_all(&dir).expect("create the consumer directory");
    let exe = dir.join(format!("{name}{}", env::consts::EXE_SUFFIX));
    let compiler = compiler(cpp, &dir);
    let library = library();
    let mut command = compiler.command();
    command
        .current_dir(&dir)
        .arg(root.join(source))
        .arg("-I")
        .arg(root.join("include"))
        .env_remove("CPATH")
        .env_remove(if cpp {
            "CPLUS_INCLUDE_PATH"
        } else {
            "C_INCLUDE_PATH"
        });
    if compiler.msvc() {
        let Library::Shared {
            import: Some(import),
            ..
        } = library
        else {
            panic!("cargo named no rgmin.dll.lib for MSVC to link");
        };
        let mut fe = OsString::from("-Fe");
        fe.push(&exe);
        command.arg(fe).arg("-link").arg(import);
    } else {
        command.arg("-o").arg(&exe);
        match library {
            Library::Shared { path, .. } => {
                let lib = path.parent().expect("shared library directory");
                command.arg("-L").arg(lib).arg("-lrgmin");
                if !cfg!(windows) {
                    let mut rpath = OsString::from("-Wl,-rpath,");
                    rpath.push(lib);
                    command.arg(rpath);
                }
            }
            Library::Static(archive) => {
                command.arg(archive).args(["-lpthread", "-ldl", "-lm"]);
            }
        }
    }
    let program = &compiler.words[0];
    let variable = if cpp { "CXX" } else { "CC" };
    let compiled = command.output().unwrap_or_else(|err| {
        panic!("spawn {program}: {err}; set {variable}_{TARGET} or {variable} to a compiler")
    });
    assert!(
        compiled.status.success(),
        "{program} failed on {source}:\n{}{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    let mut run = runner(&exe);
    if cfg!(windows)
        && let Library::Shared { path, .. } = library
    {
        let lib = path.parent().expect("shared library directory");
        let search = env::var_os("PATH").unwrap_or_default();
        let paths = std::iter::once(lib.to_path_buf()).chain(env::split_paths(&search));
        run.env("PATH", env::join_paths(paths).expect("join PATH"));
    }
    let ran = run.output().expect("run the consumer");
    assert!(
        ran.status.success(),
        "{source} exit {}:\n{}\n{}",
        ran.status,
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );
    ran
}

fn library() -> &'static Library {
    static LIBRARY: OnceLock<Library> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let cargo = env::var_os("CARGO").unwrap_or_else(|| env!("CARGO").into());
        let mut command = Command::new(cargo);
        command
            .current_dir(root)
            .args(["build", "--lib", "--locked"])
            .arg("--message-format=json-render-diagnostics")
            .args(["--no-default-features", "--features"])
            .arg(env!("RGMIN_BUILD_FEATURES"))
            .arg(if cfg!(debug_assertions) {
                "--profile=test"
            } else {
                "--release"
            });
        // `cargo test --target`, even with the host triple, puts
        // CARGO_TARGET_TMPDIR under a directory named for the triple.
        let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
        if tmp
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|dir| dir == TARGET)
        {
            command.args(["--target", TARGET]);
        }
        let built = command.output().expect("spawn cargo");
        assert!(
            built.status.success(),
            "cargo build --lib failed:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let files = artifact_paths(&String::from_utf8_lossy(&built.stdout));
        let named = |name: &str| {
            files
                .iter()
                .find(|file| file.file_name().is_some_and(|file| file == name))
                .cloned()
        };
        let shared = format!(
            "{}rgmin{}",
            env::consts::DLL_PREFIX,
            env::consts::DLL_SUFFIX
        );
        let archive = if cfg!(target_env = "msvc") {
            "rgmin.lib"
        } else {
            "librgmin.a"
        };
        match (named(&shared), named(archive)) {
            (Some(path), _) => Library::Shared {
                path,
                import: named("rgmin.dll.lib"),
            },
            (None, Some(archive)) => Library::Static(archive),
            (None, None) => panic!("cargo build --lib named no C library: {files:?}"),
        }
    })
}

/// The paths in the `filenames` arrays of cargo's JSON messages.
fn artifact_paths(messages: &str) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for line in messages.lines() {
        let Some((_, list)) = line.split_once(r#""filenames":["#) else {
            continue;
        };
        let mut chars = list.chars();
        while chars.next() == Some('"') {
            paths.push(json_string(&mut chars).into());
            if chars.next() != Some(',') {
                break;
            }
        }
    }
    paths
}

/// Read a JSON string up to its closing quote. Cargo writes its messages
/// with serde_json, which escapes only quotes, backslashes and control
/// characters.
fn json_string(chars: &mut std::str::Chars<'_>) -> String {
    let mut text = String::new();
    while let Some(c) = chars.next() {
        let c = match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('b') => '\u{8}',
                Some('f') => '\u{c}',
                Some('n') => '\n',
                Some('r') => '\r',
                Some('t') => '\t',
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    u32::from_str_radix(&hex, 16)
                        .ok()
                        .and_then(char::from_u32)
                        .unwrap_or(char::REPLACEMENT_CHARACTER)
                }
                Some(c) => c,
                None => break,
            },
            c => c,
        };
        text.push(c);
    }
    text
}

/// The C or C++ compiler for the target. The variables the `cc` crate
/// reads choose it: `CC_<target>`, `CC_<target_with_underscores>`,
/// `TARGET_CC` or `HOST_CC`, then `CC`, with `CXX` for C++. Without them
/// it is `cl` for MSVC, the triple without its vendor and `-gcc` or `-g++`
/// for another target, such as `aarch64-linux-gnu-gcc`, and else the first
/// of `cc`, `gcc` and `clang`, or `c++`, `g++` and `clang++`, that builds
/// a program.
fn compiler(cpp: bool, dir: &Path) -> Compiler {
    let tool = if cpp { "CXX" } else { "CC" };
    let side = if TARGET == HOST { "HOST" } else { "TARGET" };
    let set = [
        format!("{tool}_{TARGET}"),
        format!("{tool}_{}", TARGET.replace('-', "_")),
        format!("{side}_{tool}"),
        tool.to_string(),
    ]
    .into_iter()
    .find_map(|name| env::var(name).ok().filter(|value| !value.trim().is_empty()));
    let new = |words: Vec<String>| Compiler { words, cpp };
    if let Some(value) = set {
        return new(value.split_whitespace().map(String::from).collect());
    }
    if TARGET.ends_with("-msvc") {
        return new(vec!["cl".into()]);
    }
    let gnu = if cpp { "g++" } else { "gcc" };
    if TARGET != HOST {
        let mut parts: Vec<&str> = TARGET.split('-').collect();
        if parts.len() > 2 {
            parts.remove(1);
        }
        return new(vec![format!("{}-{gnu}", parts.join("-"))]);
    }
    let names = if cpp {
        ["c++", "g++", "clang++"]
    } else {
        ["cc", "gcc", "clang"]
    };
    names
        .into_iter()
        .map(|name| new(vec![name.into()]))
        .find(|candidate| candidate.builds_hello(dir))
        .unwrap_or_else(|| new(vec![names[0].into()]))
}

/// `exe` under the cargo runner for the target, when one is set.
fn runner(exe: &Path) -> Command {
    let key = format!(
        "CARGO_TARGET_{}_RUNNER",
        TARGET.to_uppercase().replace(['-', '.'], "_")
    );
    let runner = env::var(key).unwrap_or_default();
    let mut words = runner.split_whitespace();
    let Some(program) = words.next() else {
        return Command::new(exe);
    };
    let mut command = Command::new(program);
    command.args(words).arg(exe);
    command
}
