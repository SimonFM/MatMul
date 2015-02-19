
CC       = gcc

# Allows passing extra compilation flags.
# Intended for passing `FLAG="-DDEBUG"`
FLAG     =

OPT      = -O0

CFLAGS   = -std=gnu11 -DNCORES=$(shell getconf _NPROCESSORS_ONLN)
CFLAGS  += $(OPT)
CFLAGS  += $(FLAG)

LDFLAGS  = -lpthread -fopenmp

TARGET   = matMul

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
