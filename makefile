
CC       = gcc

# Allows passing extra compilation flags.
# Intended for passing `FLAG="-DDEBUG"`
FLAG     =

CFLAGS   = -std=gnu11 -DNCORES=$(shell getconf _NPROCESSORS_ONLN)
CFLAGS  += $(FLAG)

LDFLAGS  = -lpthread

TARGET   = matMul

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
