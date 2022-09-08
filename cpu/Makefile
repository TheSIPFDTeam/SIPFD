####  Makefile for compilation on Unix-like operative systems  ####
#SHELL := /bin/bash # Use bash syntax
SHELL := /bin/sh # Use bash syntax
# COMPILER
CC=gcc
OS := $(shell uname)
LDFLAGS=
ifeq ($(OS),Darwin)
	# Flag options for MacOS system
	ifeq ($(COMPILER),clang)
		CC=/usr/local/opt/llvm/bin/clang
		OSFLAG=-Xpreprocessor -I/usr/local/opt/llvm/include
		LDFLAGS+=-L/usr/local/opt/llvm/lib
	else
		CC=gcc-11
		OSFLAG=-Xpreprocessor -I/usr/local/opt/libomp/include #-Xclang
		LDFLAGS+=-L/usr/local/opt/libomp/lib
	endif
	#MACOSX_FLAGS=-lomp
else
	# Flag option for Linux system (Windows should be also included)
	OSFLAG=-no-pie
endif

LDFLAGS+=-L/usr/local/opt/openssl/lib
CPPFLAGS=-I/usr/local/opt/openssl/include

CURVEMODEL=_mont_
ifeq ("$(MODEL)", "shortw")
	CURVEMODEL=_shortw_
else ifeq ("$(MODEL)","mont")
	CURVEMODEL=_mont_
endif

PROCEDURE=_vowgcs_
ifeq ("$(ALGORITHM)","mitm")
	PROCEDURE=_mitm_
else ifeq ("$(ALGORITHM)","vowgcs")
	PROCEDURE=_vowgcs_
else ifeq ("$(ALGORITHM)","instance_generator")
	PROCEDURE=_instance_generator_
else ifeq ("$(ALGORITHM)","none")
	PROCEDURE=_noprocedure_
endif

ARCHITECTURE=_x64_
ifeq "$(ARCH)" "x64"
	ARCHITECTURE=_x64_
else ifeq "$(ARCH)" "x86"
	ARCHITECTURE=_x86_
else ifeq "$(ARCH)" "ARM"
	ARCHITECTURE=_x86_
else ifeq "$(ARCH)" "ARM64"
	ARCHITECTURE=_x64_
endif

ARITHMETIC=_assembly_
ifeq "$(ARITH)" "asm"
	ARITHMETIC=_assembly_
	ifneq "$(ARCHITECTURE)" "_x64_"
		$(error ASM arithmetic only available for x64)
	endif
else ifeq "$(ARITH)" "fiat"
	ARITHMETIC=_fiat_
endif

AR=ar rcs
RANLIB=ranlib

# INCLUDE FOLDER
INC_DIR+= -I./src
FOPENMP=$(OSFLAG) -fopenmp
CFLAGS_ALWAYS=-lm -lssl -lcrypto -fopenmp

ifeq "$(ARITHMETIC)" "_assembly_"
	CFLAGS=-D $(CURVEMODEL) -D $(PROCEDURE) -D $(ARCHITECTURE) -D $(ARITHMETIC) -Wall -no-pie -O3 -march=native -flto -mtune=native -funroll-loops -fomit-frame-pointer
	OBJECTS_69=objs69/p69-$(PROCEDURE).o objs69/fp69_asm_.o objs/stats.o objs/rng.o
	OBJECTS_72=objs72/p72-$(PROCEDURE).o objs72/fp72_asm_.o objs/stats.o objs/rng.o
	OBJECTS_75=objs75/p75-$(PROCEDURE).o objs75/fp75_asm_.o objs/stats.o objs/rng.o
	OBJECTS_76=objs76/p76-$(PROCEDURE).o objs76/fp76_asm_.o objs/stats.o objs/rng.o
	OBJECTS_82=objs82/p82-$(PROCEDURE).o objs82/fp82_asm_.o objs/stats.o objs/rng.o
	OBJECTS_87=objs87/p87-$(PROCEDURE).o objs87/fp87_asm_.o objs/stats.o objs/rng.o
	OBJECTS_89=objs89/p89-$(PROCEDURE).o objs89/fp89_asm_.o objs/stats.o objs/rng.o
	OBJECTS_95=objs95/p95-$(PROCEDURE).o objs95/fp95_asm_.o objs/stats.o objs/rng.o
	OBJECTS_99=objs99/p99-$(PROCEDURE).o objs99/fp99_asm_.o objs/stats.o objs/rng.o
	OBJECTS_100=objs100/p100-$(PROCEDURE).o objs100/fp100_asm_.o objs/stats.o objs/rng.o
	OBJECTS_107=objs107/p107-$(PROCEDURE).o objs107/fp107_asm_.o objs/stats.o objs/rng.o
	OBJECTS_111=objs111/p111-$(PROCEDURE).o objs111/fp111_asm_.o objs/stats.o objs/rng.o
	OBJECTS_117=objs117/p117-$(PROCEDURE).o objs117/fp117_asm_.o objs/stats.o objs/rng.o
	OBJECTS_118=objs118/p118-$(PROCEDURE).o objs118/fp118_asm_.o objs/stats.o objs/rng.o
	OBJECTS_124=objs124/p124-$(PROCEDURE).o objs124/fp124_asm_.o objs/stats.o objs/rng.o
	OBJECTS_126=objs126/p126-$(PROCEDURE).o objs126/fp126_asm_.o objs/stats.o objs/rng.o
	OBJECTS_129=objs129/p129-$(PROCEDURE).o objs129/fp129_asm_.o objs/stats.o objs/rng.o
	OBJECTS_131=objs131/p131-$(PROCEDURE).o objs131/fp131_asm_.o objs/stats.o objs/rng.o
	OBJECTS_140=objs140/p140-$(PROCEDURE).o objs140/fp140_asm_.o objs/stats.o objs/rng.o
	OBJECTS_152=objs152/p152-$(PROCEDURE).o objs152/fp152_asm_.o objs/stats.o objs/rng.o
	OBJECTS_164=objs164/p164-$(PROCEDURE).o objs164/fp164_asm_.o objs/stats.o objs/rng.o
	OBJECTS_176=objs176/p176-$(PROCEDURE).o objs176/fp176_asm_.o objs/stats.o objs/rng.o
	OBJECTS_184=objs184/p184-$(PROCEDURE).o objs184/fp184_asm_.o objs/stats.o objs/rng.o
	OBJECTS_194=objs194/p194-$(PROCEDURE).o objs194/fp194_asm_.o objs/stats.o objs/rng.o
	OBJECTS_196=objs196/p196-$(PROCEDURE).o objs196/fp196_asm_.o objs/stats.o objs/rng.o
	OBJECTS_204=objs204/p204-$(PROCEDURE).o objs204/fp204_asm_.o objs/stats.o objs/rng.o
	OBJECTS_216=objs216/p216-$(PROCEDURE).o objs216/fp216_asm_.o objs/stats.o objs/rng.o
	OBJECTS_224=objs224/p224-$(PROCEDURE).o objs224/fp224_asm_.o objs/stats.o objs/rng.o
	OBJECTS_236=objs236/p236-$(PROCEDURE).o objs236/fp236_asm_.o objs/stats.o objs/rng.o
	OBJECTS_244=objs244/p244-$(PROCEDURE).o objs244/fp244_asm_.o objs/stats.o objs/rng.o
	OBJECTS_252=objs252/p252-$(PROCEDURE).o objs252/fp252_asm_.o objs/stats.o objs/rng.o
else
	CFLAGS=-D $(CURVEMODEL) -D $(PROCEDURE) -D $(ARCHITECTURE) -D $(ARITHMETIC) -Wall -O3 -march=native -flto -mtune=native -funroll-loops -fomit-frame-pointer
	OBJECTS_69=objs69/p69-$(PROCEDURE).o objs69/fp69$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_70=objs70/p70-$(PROCEDURE).o objs70/fp70$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_72=objs72/p72-$(PROCEDURE).o objs72/fp72$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_75=objs75/p75-$(PROCEDURE).o objs75/fp75$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_76=objs76/p76-$(PROCEDURE).o objs76/fp76$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_82=objs82/p82-$(PROCEDURE).o objs82/fp82$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_87=objs87/p87-$(PROCEDURE).o objs87/fp87$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_89=objs89/p89-$(PROCEDURE).o objs89/fp89$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_95=objs95/p95-$(PROCEDURE).o objs95/fp95$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_99=objs99/p99-$(PROCEDURE).o objs99/fp99$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_100=objs100/p100-$(PROCEDURE).o objs100/fp100$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_107=objs107/p107-$(PROCEDURE).o objs107/fp107$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_111=objs111/p111-$(PROCEDURE).o objs111/fp111$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_117=objs117/p117-$(PROCEDURE).o objs117/fp117$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_118=objs118/p118-$(PROCEDURE).o objs118/fp118$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_124=objs124/p124-$(PROCEDURE).o objs124/fp124$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_126=objs126/p126-$(PROCEDURE).o objs126/fp126$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_129=objs129/p129-$(PROCEDURE).o objs129/fp129$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_131=objs131/p131-$(PROCEDURE).o objs131/fp131$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_140=objs140/p140-$(PROCEDURE).o objs140/fp140$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_152=objs152/p152-$(PROCEDURE).o objs152/fp152$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_164=objs164/p164-$(PROCEDURE).o objs164/fp164$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_176=objs176/p176-$(PROCEDURE).o objs176/fp176$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_184=objs184/p184-$(PROCEDURE).o objs184/fp184$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_194=objs194/p194-$(PROCEDURE).o objs194/fp194$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_196=objs196/p196-$(PROCEDURE).o objs196/fp196$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_204=objs204/p204-$(PROCEDURE).o objs204/fp204$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_216=objs216/p216-$(PROCEDURE).o objs216/fp216$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_224=objs224/p224-$(PROCEDURE).o objs224/fp224$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_236=objs236/p236-$(PROCEDURE).o objs236/fp236$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_244=objs244/p244-$(PROCEDURE).o objs244/fp244$(ARCHITECTURE).o objs/stats.o objs/rng.o
	OBJECTS_252=objs252/p252-$(PROCEDURE).o objs252/fp252$(ARCHITECTURE).o objs/stats.o objs/rng.o
endif

TESTS=tests69 tests72 tests76 tests82 tests87 tests89 tests95 tests99 tests100 tests107 \
tests111 tests117 tests118 tests124 tests126 tests129 tests131 tests140 tests152 tests164 tests176 tests184 \
tests194 tests196 tests204 tests216 tests224 tests236 tests244 tests252

all: clean $(TESTS)

objs/stats.o: src/stats.c
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) -c $(CFLAGS) $< -o $@

objs/rng.o: src/rng.c
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) -c $(CFLAGS) $< -o $@

objs69/fp69$(ARCHITECTURE).o: src/primes/p69/fp69$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs69/fp69_asm_.o: src/primes/p69/fp69_asm_.s src/primes/p69/p69asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs69/p69-$(PROCEDURE).o: src/primes/p69/p69.c
	@echo "\033[1;33m++++++ lib69\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib69: $(OBJECTS_69)
	rm -rf lib69 SIPFD69
	mkdir lib69 SIPFD69
	$(AR) lib69/libsidh.a $^
	$(RANLIB) lib69/libsidh.a

objs70/fp70$(ARCHITECTURE).o: src/primes/p70/fp70$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs70/fp70_asm_.o: src/primes/p70/fp70_asm_.s src/primes/p70/p70asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs70/p70-$(PROCEDURE).o: src/primes/p70/p70.c
	@echo "\033[1;33m++++++ lib70\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib70: $(OBJECTS_70)
	rm -rf lib70 SIPFD70
	mkdir lib70 SIPFD70
	$(AR) lib70/libsidh.a $^
	$(RANLIB) lib70/libsidh.a

objs72/fp72$(ARCHITECTURE).o: src/primes/p72/fp72$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs72/fp72_asm_.o: src/primes/p72/fp72_asm_.s src/primes/p72/p72asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs72/p72-$(PROCEDURE).o: src/primes/p72/p72.c
	@echo "\033[1;33m++++++ lib72\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib72: $(OBJECTS_72)
	rm -rf lib72 SIPFD72
	mkdir lib72 SIPFD72
	$(AR) lib72/libsidh.a $^
	$(RANLIB) lib72/libsidh.a

objs75/fp75$(ARCHITECTURE).o: src/primes/p75/fp75$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs75/fp75_asm_.o: src/primes/p75/fp75_asm_.s src/primes/p75/p75asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs75/p75-$(PROCEDURE).o: src/primes/p75/p75.c
	@echo "\033[1;33m++++++ lib75\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib75: $(OBJECTS_75)
	rm -rf lib75 SIPFD75
	mkdir lib75 SIPFD75
	$(AR) lib75/libsidh.a $^
	$(RANLIB) lib75/libsidh.a

objs76/fp76$(ARCHITECTURE).o: src/primes/p76/fp76$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs76/fp76_asm_.o: src/primes/p76/fp76_asm_.s src/primes/p76/p76asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs76/p76-$(PROCEDURE).o: src/primes/p76/p76.c
	@echo "\033[1;33m++++++ lib76\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib76: $(OBJECTS_76)
	rm -rf lib76 SIPFD76
	mkdir lib76 SIPFD76
	$(AR) lib76/libsidh.a $^
	$(RANLIB) lib76/libsidh.a

objs82/fp82$(ARCHITECTURE).o: src/primes/p82/fp82$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs82/fp82_asm_.o: src/primes/p82/fp82_asm_.s src/primes/p82/p82asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs82/p82-$(PROCEDURE).o: src/primes/p82/p82.c
	@echo "\033[1;33m++++++ lib82\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib82: $(OBJECTS_82)
	rm -rf lib82 SIPFD82
	mkdir lib82 SIPFD82
	$(AR) lib82/libsidh.a $^
	$(RANLIB) lib82/libsidh.a

objs87/fp87$(ARCHITECTURE).o: src/primes/p87/fp87$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs87/fp87_asm_.o: src/primes/p87/fp87_asm_.s src/primes/p87/p87asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs87/p87-$(PROCEDURE).o: src/primes/p87/p87.c
	@echo "\033[1;33m++++++ lib87\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib87: $(OBJECTS_87)
	rm -rf lib87 SIPFD87
	mkdir lib87 SIPFD87
	$(AR) lib87/libsidh.a $^
	$(RANLIB) lib87/libsidh.a

objs89/fp89$(ARCHITECTURE).o: src/primes/p89/fp89$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs89/fp89_asm_.o: src/primes/p89/fp89_asm_.s src/primes/p89/p89asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs89/p89-$(PROCEDURE).o: src/primes/p89/p89.c
	@echo "\033[1;33m++++++ lib89\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib89: $(OBJECTS_89)
	rm -rf lib89 SIPFD89
	mkdir lib89 SIPFD89
	$(AR) lib89/libsidh.a $^
	$(RANLIB) lib89/libsidh.a

objs95/fp95$(ARCHITECTURE).o: src/primes/p95/fp95$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs95/fp95_asm_.o: src/primes/p95/fp95_asm_.s src/primes/p95/p95asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs95/p95-$(PROCEDURE).o: src/primes/p95/p95.c
	@echo "\033[1;33m++++++ lib95\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib95: $(OBJECTS_95)
	rm -rf lib95 SIPFD95
	mkdir lib95 SIPFD95
	$(AR) lib95/libsidh.a $^
	$(RANLIB) lib95/libsidh.a

objs99/fp99$(ARCHITECTURE).o: src/primes/p99/fp99$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs99/fp99_asm_.o: src/primes/p99/fp99_asm_.s src/primes/p99/p99asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs99/p99-$(PROCEDURE).o: src/primes/p99/p99.c
	@echo "\033[1;33m++++++ lib99\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib99: $(OBJECTS_99)
	rm -rf lib99 SIPFD99
	mkdir lib99 SIPFD99
	$(AR) lib99/libsidh.a $^
	$(RANLIB) lib99/libsidh.a

objs100/fp100$(ARCHITECTURE).o: src/primes/p100/fp100$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs100/fp100_asm_.o: src/primes/p100/fp100_asm_.s src/primes/p100/p100asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs100/p100-$(PROCEDURE).o: src/primes/p100/p100.c
	@echo "\033[1;33m++++++ lib100\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib100: $(OBJECTS_100)
	rm -rf lib100 SIPFD100
	mkdir lib100 SIPFD100
	$(AR) lib100/libsidh.a $^
	$(RANLIB) lib100/libsidh.a

objs107/fp107$(ARCHITECTURE).o: src/primes/p107/fp107$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs107/fp107_asm_.o: src/primes/p107/fp107_asm_.s src/primes/p107/p107asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs107/p107-$(PROCEDURE).o: src/primes/p107/p107.c
	@echo "\033[1;33m++++++ lib107\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib107: $(OBJECTS_107)
	rm -rf lib107 SIPFD107
	mkdir lib107 SIPFD107
	$(AR) lib107/libsidh.a $^
	$(RANLIB) lib107/libsidh.a

objs111/fp111$(ARCHITECTURE).o: src/primes/p111/fp111$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs111/fp111_asm_.o: src/primes/p111/fp111_asm_.s src/primes/p111/p111asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs111/p111-$(PROCEDURE).o: src/primes/p111/p111.c
	@echo "\033[1;33m++++++ lib111\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib111: $(OBJECTS_111)
	rm -rf lib111 SIPFD111
	mkdir lib111 SIPFD111
	$(AR) lib111/libsidh.a $^
	$(RANLIB) lib111/libsidh.a

objs117/fp117$(ARCHITECTURE).o: src/primes/p117/fp117$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs117/fp117_asm_.o: src/primes/p117/fp117_asm_.s src/primes/p117/p117asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs117/p117-$(PROCEDURE).o: src/primes/p117/p117.c
	@echo "\033[1;33m++++++ lib117\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib117: $(OBJECTS_117)
	rm -rf lib117 SIPFD117
	mkdir lib117 SIPFD117
	$(AR) lib117/libsidh.a $^
	$(RANLIB) lib117/libsidh.a

objs118/fp118$(ARCHITECTURE).o: src/primes/p118/fp118$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs118/fp118_asm_.o: src/primes/p118/fp118_asm_.s src/primes/p118/p118asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs118/p118-$(PROCEDURE).o: src/primes/p118/p118.c
	@echo "\033[1;33m++++++ lib118\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib118: $(OBJECTS_118)
	rm -rf lib118 SIPFD118
	mkdir lib118 SIPFD118
	$(AR) lib118/libsidh.a $^
	$(RANLIB) lib118/libsidh.a

objs124/fp124$(ARCHITECTURE).o: src/primes/p124/fp124$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs124/fp124_asm_.o: src/primes/p124/fp124_asm_.s src/primes/p124/p124asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs124/p124-$(PROCEDURE).o: src/primes/p124/p124.c
	@echo "\033[1;33m++++++ lib124\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib124: $(OBJECTS_124)
	rm -rf lib124 SIPFD124
	mkdir lib124 SIPFD124
	$(AR) lib124/libsidh.a $^
	$(RANLIB) lib124/libsidh.a

objs126/fp126$(ARCHITECTURE).o: src/primes/p126/fp126$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs126/fp126_asm_.o: src/primes/p126/fp126_asm_.s src/primes/p126/p126asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs126/p126-$(PROCEDURE).o: src/primes/p126/p126.c
	@echo "\033[1;33m++++++ lib126\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib126: $(OBJECTS_126)
	rm -rf lib126 SIPFD126
	mkdir lib126 SIPFD126
	$(AR) lib126/libsidh.a $^
	$(RANLIB) lib126/libsidh.a

objs129/fp129$(ARCHITECTURE).o: src/primes/p129/fp129$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs129/fp129_asm_.o: src/primes/p129/fp129_asm_.s src/primes/p129/p129asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs129/p129-$(PROCEDURE).o: src/primes/p129/p129.c
	@echo "\033[1;33m++++++ lib129\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib129: $(OBJECTS_129)
	rm -rf lib129 SIPFD129
	mkdir lib129 SIPFD129
	$(AR) lib129/libsidh.a $^
	$(RANLIB) lib129/libsidh.a

objs131/fp131$(ARCHITECTURE).o: src/primes/p131/fp131$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs131/fp131_asm_.o: src/primes/p131/fp131_asm_.s src/primes/p131/p131asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs131/p131-$(PROCEDURE).o: src/primes/p131/p131.c
	@echo "\033[1;33m++++++ lib131\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib131: $(OBJECTS_131)
	rm -rf lib131 SIPFD131
	mkdir lib131 SIPFD131
	$(AR) lib131/libsidh.a $^
	$(RANLIB) lib131/libsidh.a

objs140/fp140$(ARCHITECTURE).o: src/primes/p140/fp140$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs140/fp140_asm_.o: src/primes/p140/fp140_asm_.s src/primes/p140/p140asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs140/p140-$(PROCEDURE).o: src/primes/p140/p140.c
	@echo "\033[1;33m++++++ lib140\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib140: $(OBJECTS_140)
	rm -rf lib140 SIPFD140
	mkdir lib140 SIPFD140
	$(AR) lib140/libsidh.a $^
	$(RANLIB) lib140/libsidh.a

objs152/fp152$(ARCHITECTURE).o: src/primes/p152/fp152$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs152/fp152_asm_.o: src/primes/p152/fp152_asm_.s src/primes/p152/p152asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs152/p152-$(PROCEDURE).o: src/primes/p152/p152.c
	@echo "\033[1;33m++++++ lib152\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib152: $(OBJECTS_152)
	rm -rf lib152 SIPFD152
	mkdir lib152 SIPFD152
	$(AR) lib152/libsidh.a $^
	$(RANLIB) lib152/libsidh.a

objs164/fp164$(ARCHITECTURE).o: src/primes/p164/fp164$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs164/fp164_asm_.o: src/primes/p164/fp164_asm_.s src/primes/p164/p164asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs164/p164-$(PROCEDURE).o: src/primes/p164/p164.c
	@echo "\033[1;33m++++++ lib164\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib164: $(OBJECTS_164)
	rm -rf lib164 SIPFD164
	mkdir lib164 SIPFD164
	$(AR) lib164/libsidh.a $^
	$(RANLIB) lib164/libsidh.a

objs176/fp176$(ARCHITECTURE).o: src/primes/p176/fp176$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs176/fp176_asm_.o: src/primes/p176/fp176_asm_.s src/primes/p176/p176asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs176/p176-$(PROCEDURE).o: src/primes/p176/p176.c
	@echo "\033[1;33m++++++ lib176\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib176: $(OBJECTS_176)
	rm -rf lib176 SIPFD176
	mkdir lib176 SIPFD176
	$(AR) lib176/libsidh.a $^
	$(RANLIB) lib176/libsidh.a

objs184/fp184$(ARCHITECTURE).o: src/primes/p184/fp184$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs184/fp184_asm_.o: src/primes/p184/fp184_asm_.s src/primes/p184/p184asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs184/p184-$(PROCEDURE).o: src/primes/p184/p184.c
	@echo "\033[1;33m++++++ lib184\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib184: $(OBJECTS_184)
	rm -rf lib184 SIPFD184
	mkdir lib184 SIPFD184
	$(AR) lib184/libsidh.a $^
	$(RANLIB) lib184/libsidh.a

objs194/fp194$(ARCHITECTURE).o: src/primes/p194/fp194$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs194/fp194_asm_.o: src/primes/p194/fp194_asm_.s src/primes/p194/p194asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs194/p194-$(PROCEDURE).o: src/primes/p194/p194.c
	@echo "\033[1;33m++++++ lib194\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib194: $(OBJECTS_194)
	rm -rf lib194 SIPFD194
	mkdir lib194 SIPFD194
	$(AR) lib194/libsidh.a $^
	$(RANLIB) lib194/libsidh.a

objs196/fp196$(ARCHITECTURE).o: src/primes/p196/fp196$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs196/fp196_asm_.o: src/primes/p196/fp196_asm_.s src/primes/p196/p196asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs196/p196-$(PROCEDURE).o: src/primes/p196/p196.c
	@echo "\033[1;33m++++++ lib196\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib196: $(OBJECTS_196)
	rm -rf lib196 SIPFD196
	mkdir lib196 SIPFD196
	$(AR) lib196/libsidh.a $^
	$(RANLIB) lib196/libsidh.a

objs204/fp204$(ARCHITECTURE).o: src/primes/p204/fp204$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs204/fp204_asm_.o: src/primes/p204/fp204_asm_.s src/primes/p204/p204asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs204/p204-$(PROCEDURE).o: src/primes/p204/p204.c
	@echo "\033[1;33m++++++ lib204\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib204: $(OBJECTS_204)
	rm -rf lib204 SIPFD204
	mkdir lib204 SIPFD204
	$(AR) lib204/libsidh.a $^
	$(RANLIB) lib204/libsidh.a

objs216/fp216$(ARCHITECTURE).o: src/primes/p216/fp216$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs216/fp216_asm_.o: src/primes/p216/fp216_asm_.s src/primes/p216/p216asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs216/p216-$(PROCEDURE).o: src/primes/p216/p216.c
	@echo "\033[1;33m++++++ lib216\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib216: $(OBJECTS_216)
	rm -rf lib216 SIPFD216
	mkdir lib216 SIPFD216
	$(AR) lib216/libsidh.a $^
	$(RANLIB) lib216/libsidh.a

objs224/fp224$(ARCHITECTURE).o: src/primes/p224/fp224$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs224/fp224_asm_.o: src/primes/p224/fp224_asm_.s src/primes/p224/p224asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs224/p224-$(PROCEDURE).o: src/primes/p224/p224.c
	@echo "\033[1;33m++++++ lib224\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib224: $(OBJECTS_224)
	rm -rf lib224 SIPFD224
	mkdir lib224 SIPFD224
	$(AR) lib224/libsidh.a $^
	$(RANLIB) lib224/libsidh.a

objs236/fp236$(ARCHITECTURE).o: src/primes/p236/fp236$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs236/fp236_asm_.o: src/primes/p236/fp236_asm_.s src/primes/p236/p236asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs236/p236-$(PROCEDURE).o: src/primes/p236/p236.c
	@echo "\033[1;33m++++++ lib236\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib236: $(OBJECTS_236)
	rm -rf lib236 SIPFD236
	mkdir lib236 SIPFD236
	$(AR) lib236/libsidh.a $^
	$(RANLIB) lib236/libsidh.a

objs244/fp244$(ARCHITECTURE).o: src/primes/p244/fp244$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs244/fp244_asm_.o: src/primes/p244/fp244_asm_.s src/primes/p244/p244asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs244/p244-$(PROCEDURE).o: src/primes/p244/p244.c
	@echo "\033[1;33m++++++ lib244\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib244: $(OBJECTS_244)
	rm -rf lib244 SIPFD244
	mkdir lib244 SIPFD244
	$(AR) lib244/libsidh.a $^
	$(RANLIB) lib244/libsidh.a

objs252/fp252$(ARCHITECTURE).o: src/primes/p252/fp252$(ARCHITECTURE).c
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

objs252/fp252_asm_.o: src/primes/p252/fp252_asm_.s src/primes/p252/p252asm_api.h
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $(ASMFLAGS) $< -o $@

objs252/p252-$(PROCEDURE).o: src/primes/p252/p252.c
	@echo "\033[1;33m++++++ lib252\033[0m"
	@mkdir -p $(@D)
	$(CC) $(INC_DIR) $(CPPFLAGS) -c $(CFLAGS) $< -o $@

lib252: $(OBJECTS_252)
	rm -rf lib252 SIPFD252
	mkdir lib252 SIPFD252
	$(AR) lib252/libsidh.a $^
	$(RANLIB) lib252/libsidh.a

tests%: lib%
	$(eval BN=$(basename $@))
	$(eval P=$(BN:tests%=%))
	@echo "\033[0;33m++++++ SIPFD$(P)\033[0m"
	$(CC) $(CPPFLAGS) $(CFLAGS) -D _TEST_ $(LDFLAGS) -L./lib$(P) tests/arith_tests/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/arith_tests$(ARCHITECTURE)
	@if test "$(PROCEDURE)" = "_mitm_" ; then \
  	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-basic/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-basic$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-basic/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-basic$(ARCHITECTURE) ;\
	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-dfs/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-dfs$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-dfs/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-dfs$(ARCHITECTURE) ;\
	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-memory-limited/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-memory-limited$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-memory-limited/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-memory-limited$(ARCHITECTURE) ;\
	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-dfs-memory-limited/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-dfs-memory-limited$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/mitm-dfs-memory-limited/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/mitm-dfs-memory-limited$(ARCHITECTURE) ;\
	elif test "$(PROCEDURE)" = "_vowgcs_" ; then \
	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/vowgcs/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/vow-gcs$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/vowgcs/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/vow-gcs$(ARCHITECTURE) ;\
	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/vowrigged/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/vow-rigged$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/vowrigged/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/vow-rigged$(ARCHITECTURE) ;\
	elif test "$(PROCEDURE)" = "_instance_generator_" ; then \
	echo "\033[0;32m$(CC) $(CPPFLAGS) $(CFLAGS) $(FOPENMP) $(LDFLAGS) -L./lib$(P) tests/vow_instance_generator/p$(P).c -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/instance_generator$(ARCHITECTURE)\033[0m" ;\
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -L./lib$(P) tests/vow_instance_generator/p$(P).c $(CFLAGS_ALWAYS) -lsidh $(MACOSX_FLAGS) -o SIPFD$(P)/instance_generator$(ARCHITECTURE);\
	fi;
	@echo ""

help:
	@echo "\n++++++ macOS instructions"
	@echo "usage: make ARCH=[x86/x64] COMPILER=[gcc/clang] MODEL=[mont/shortw] ALGORITHM=[mitm/vowgcs]-B"
	@echo "usage (by default, ARCH=x64 COMPILER=gcc, MODEL=mont, ALGORITHM=vowgcs, ARITH=asm): make -B"
	@echo "usage: make clean\n"
	@echo "\n++++++ Linux instructions"
	@echo "usage: make ARCH=[x86/x64] CC=[gcc/clang] MODEL=[mont/shortw] ALGORITHM=[mitm/vowgcs]-B"
	@echo "usage (by default, ARCH=x64 CC=gcc, MODEL=shortw, ALGORITHM=mitm): make -B"
	@echo "usage: make clean\n"

clean:
	@echo "\033[1;33m++++++ Clean compilation\033[0m"
	rm -rf fiat-crypto-p*.log
	rm -rf objs
	rm -rf objs* SIPFD* lib*
	@echo ""
