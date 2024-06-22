CC      = gcc

CFLAGS	=	-Wall -g
LDLibs	=	-lm
BUILD	=	xor

math: math_structs.o
	$(CC) -o math_structs math_structs.o

all:	$(BUILD)

clean:
	/bin/rm -f $(BUILD) *.o core

xor:	xor.o ann.o layer.o math_structs.o math_funcs.o helper.o 
	$(CC) $(CFLAGS) -o xor xor.o ann.o layer.o math_structs.o math_funcs.o helper.o $(LDLibs)

mnist:	mnist.o ann.o layer.o adam.o math_structs.o math_funcs.o math_r4t.o helper.o 
	$(CC) $(CFLAGS) -o mnist mnist.o ann.o layer.o adam.o math_structs.o math_funcs.o math_r4t.o helper.o $(LDLibs)