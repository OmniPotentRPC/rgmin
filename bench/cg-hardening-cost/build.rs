fn main() {
    // Directory that holds librgpot.so and liblinkcell.so. Set by the
    // builder. No path is compiled into this file.
    let dir = std::env::var("CG_LIB_DIR").expect("set CG_LIB_DIR to the shared-library directory");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=rgpot");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    println!("cargo:rerun-if-env-changed=CG_LIB_DIR");
}
