use vergen::EmitBuilder;

fn main() {
    let _ = EmitBuilder::builder()
        .all_build()
        .all_rustc()
        .emit();
}
