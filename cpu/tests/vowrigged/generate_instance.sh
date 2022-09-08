cd ../..
make ALGORITHM=instance_generator MODEL=mont ARITH=fiat tests$1
SIPFD$1/instance_generator_x64_ > tests/vowrigged/fixed_instance_p$1.h