COMPOPT=-fast 
ACCOPT=-acc -gpu=ccall -Minfo=accel
CC=nvc
TGT=spmv

.PHONY: all
all: $(TGT).bin $(TGT).dbg $(TGT).lin

$(TGT).bin: $(TGT).c Makefile
	$(CC) $(COMPOPT) $(ACCOPT) $< -o $@
$(TGT).dbg: $(TGT).c Makefile
	$(CC) $(COMPOPT) -g -gpu=debug $(ACCOPT) $(TGT).c -o $@
$(TGT).lin: $(TGT).c Makefile
	$(CC) $(COMPOPT) -g -gpu=lineinfo $(ACCOPT) $(TGT).c -o $@
clean:
	rm -f $(TGT).bin $(TGT).dbg $(TGT).lin
