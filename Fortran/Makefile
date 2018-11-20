OBJECTS = adog_mod.o input_func.o test1.o 
MODULES = adog_mod.mod input_func.mod

FFLAGS = -g

.PHONY: test1 clean 

test1: test1.exe
	./test1.exe

test1.exe: $(MODULES) $(OBJECTS)
	gfortran $(FFLAGS) $(OBJECTS) -o test1.exe

%.o : %.f
	gfortran $(FFLAGS) -c  $< 

%.mod: %.f
	gfortran $(FFLAGS) -c $<

clean:
	rm -f *.o *.exe *.mod

