COMPOPT=-fast 
ACCOPT=-acc -gpu=ccall -Minfo=accel
CC=nvfortran
TGT=vadd

.PHONY: all
all: $(TGT).bin $(TGT).dbg $(TGT).lin

$(TGT).bin: $(TGT).F03 Makefile
	$(CC) $(COMPOPT) $(ACCOPT) $< -o $@
$(TGT).dbg: $(TGT).F03 Makefile
	$(CC) $(COMPOPT) -gpu=debug $(ACCOPT) $(TGT).F03 -o $@
$(TGT).lin: $(TGT).F03 Makefile
	$(CC) $(COMPOPT) -gpu=lineinfo $(ACCOPT) $(TGT).F03 -o $@
clean:
	rm -f $(TGT).bin $(TGT).dbg $(TGT).lin
