#PREFIX = Specify the install path

objects=  Trans_filter.o NDIRECT2_exe.o  NDIRECT2_1x1.o \
		NDIRECT2_1x1s2.o NDIRECT2_3x3.o NDIRECT2_7x7.o 
		 

all:$(objects)
	g++ $(objects) -shared -fPIC -fopenmp -mavx512f -O2 -o libnd2.so
	ar rcs libnd2.a $(objects)

Trans_filter.o:Trans_filter.c 
	g++ -c -fPIC Trans_filter.c -mavx512f -O2 -o Trans_filter.o
NDIRECT2_exe.o:NDIRECT2_exe.c 
	g++ -c -fPIC -mavx512f NDIRECT2_exe.c -O2 -o NDIRECT2_exe.o
NDIRECT_direct_1x1s1.o:NDIRECT2_1x1s1.c
	g++ -c -fPIC -mavx512f -fopenmp NDIRECT2_1x1s1.c -O2 -o NDIRECT2_1x1s1.o
NDIRECT_direct_1x1s2.o:NDIRECT2_1x1s2.c 
	g++ -c -fPIC -mavx512f -fopenmp NDIRECT2_1x1s2.c -O2 -o NDIRECT2_1x1s2.o
NDIRECT2_3x3.o:NDIRECT2_3x3.c 
	g++ -c -fPIC -mavx512f -fopenmp NDIRECT2_3x3.c -O2 -o NDIRECT2_3x3.o
NDIRECT2_7x7.o:NDIRECT2_7x7.c
	g++ -c -fPIC -mavx512f -fopenmp NDIRECT2_7x7.c -O2 -o NDIRECT2_7x7.o


install:
		mkdir -p $(PREFIX)/ND2
		mkdir -p $(PREFIX)/ND2/lib
		cp -f ./libnd2.so $(PREFIX)/ND2/lib/
		cp -f ./libnd2.a $(PREFIX)/ND2/lib/
		mkdir -p $(PREFIX)/ND2/include
		cp -f ./*.h $(PREFIX)/ND2/include/

.PHONY:clean
clean:
	rm -rf $(PREFIX)/ND2 *.o *.a *.so
