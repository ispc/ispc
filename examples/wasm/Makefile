all: wasm_hello_world

tmp_folder:
	mkdir tmp_folder

wasm_hello_world: hello_world.o hello_world.o ../../builtins/builtins.c tmp_folder
	../../build/bin/ispc hello_world.ispc -O2 --target=wasm-i32x4 --nostdlib --emit-llvm-text -o tmp_folder/hello_world.ispc.ll
	emcc -DWASM hello_world.c -O2 -s EXPORTED_FUNCTIONS='["_main"]' -c -o tmp_folder/hello_world.o
	emcc -DWASM_IMPLEMENTATION ../../builtins/builtins.c -O2 -s EXPORTED_FUNCTIONS='["___wasm_do_print"]' -c -o tmp_folder/builtins.o
	cd tmp_folder && llc -filetype=obj hello_world.ispc.ll -o hello_world.ispc.o
	cd tmp_folder && emcc hello_world.o hello_world.ispc.o builtins.o -o index.html
	cd tmp_folder && python3 -m http.server
clean:
	rm -rf tmp_folder

# Don't forget to enable wasm-simd extension
#   Chrome: goto chrome://flags/#enable-webassembly