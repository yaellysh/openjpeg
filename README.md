To run OpenJPEG, 

```cd /Users/yaellyshkow/Desktop/iSyntaxtoj2k/openjpeg

cmake --build build -- -j                          
OPJ_EXTERNAL_DWT=1 ISY_DUMP_PREIDWT_DEC=1 ISY_SCALE=3 ISY_TX=10 ISY_TY=10   ISY_DUMP_PREIDWT_ENC=1 ./build/bin/opj_compress \
  -i ~/Desktop/isyntaxtoj2k/openjpeg/tile_output.ppm \
  -o out.j2k \
  -n 2 -b 64,64 -t 256,256 -mct 0

ISY_DUMP_PREIDWT_DEC=1 ./build/bin/opj_decompress -i out.j2k -o /tmp/out.pgx

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