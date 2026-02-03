To run OpenJPEG, 

```
<!-- Get into openjpeg folder -->
cd /Users/yaellyshkow/Desktop/iSyntaxtoj2k/openjpeg

<!-- Remake and set flags -->
cmake --build build -- -j 
ISY_COEFF_DIR="/Users/yaellyshkow/Desktop/hugh_coeffs" \
ISY_BASE_LEVEL=7 \
OPJ_EXTERNAL_DWT=1 \

<!-- Compress with opj_compress -->
./build/bin/opj_compress \
  -i ~/Desktop/isyntaxtoj2k/openjpeg/tile_output.ppm \
  -o out.j2k \
  -n 2 -b 64,64 -mct 0ISY_DUMP_PREIDWT_DEC=1 ./build/bin/opj_decompress -i out.j2k -o /tmp/out.pgx

<!-- Run python script to view results -->
python3 src/lib/openjp2/ycocg_recon.py /tmp/out_0.pgx /tmp/out_1.pgx /tmp/out_2.pgx /tmp/recon.ppm
open /tmp/recon.ppm
```

and libisyntax,

```
cd /Users/yaellyshkow/Desktop/iSyntaxtoj2k/libisyntax

cmake --build build -- -j                           
ISY_DUMP_PREIDWT_BIN=1 ISY_DWT_LEVELS=1 \
ISY_WANT_SCALE=3 ISY_WANT_TX=10 ISY_WANT_TY=10 \
./isyntax_example testslide.isyntax 3 10 10 tile_output.png

ISY_DUMP_PREIDWT_BIN=1 ISY_DWT_LEVELS=1 \
ISY_WANT_SCALE=2 ISY_WANT_TX=20 ISY_WANT_TY=20 \
./isyntax_example testslide.isyntax 2 20 20 tile_output.png
```