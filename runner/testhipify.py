import os
import fileinput
import os.path
from sys import platform

from runner import patch_gen, patch_gen2, patch_gen3
'''
import patch_gen
import patch_gen2
import patch_gen3
from patch_gen import *
from patch_gen2 import *
from patch_gen3 import *
'''
#import argparse
cuda_path = '/usr/local/cuda-12.0/targets/x86_64-linux/include'
user_platform=''
try:
	with open('user_platform.txt','r') as f:
			user_platform=f.read()
	f.close()				
except FileNotFoundError:
	pass	
def getListOfFiles(dirName):
    listOfFile=os.listdir(dirName)
    allFiles=list()
    for entry in listOfFile:
        fullPath=os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles=allFiles+getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def sorting(filename):
  infile = open(filename)
  words = []
  for line in infile:
    temp = line.split()
    for i in temp:
      words.append(i)
  infile.close()
  words.sort()
  outfile = open("final_ignored_samples1.txt", "w")
  for i in words:
    outfile.writelines(i)
    outfile.writelines("\n")
  outfile.close()
  os.rename("final_ignored_samples1.txt","final_ignored_samples.txt")
"""
def prepend_line(file_name, line):
	#line='#include "HIPCHECK.h"'
    dummy_file = file_name + '.bak'
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        write_obj.write(line + '\n')
        for line in read_obj:
            write_obj.write(line)
    os.remove(file_name)
    os.rename(dummy_file, file_name)
	"""

def prepend_line(file_name, line):
	result=check_for_word(file_name,line)
	if result==0:
		p=os.path.dirname(file_name)
		file=open(file_name,'r')
		lines = file.readlines()
		for elem in lines:
			if elem == '#include <stdio.h>\n':
				index=lines.index(elem)
				lines.insert(index+8,line)
			else:
				continue	
		with open(p+'/'+'a.cu.hip','w') as fp:
			for item in lines:
				fp.write(item)
		file.close()
		os.remove(file_name)
		os.rename(p+'/a.cu.hip', file_name)
	

def check_for_word(file_name,word):
	file = open(file_name, 'r')
	linelist = file.readlines()
	file.close()
	for line in linelist:
		if str(word) in line:
			return 1
	return 0
		 


def setup():
	global cuda_path
	global user_platform
	#cuda_path = '/usr/local/cuda-12.0/targets/x86_64-linux/include'
	print("Enter Nvidia or AMD as per your system specifications.")
	user_platform=input()
	with open('user_platform.txt','w') as f:
		f.write(str(user_platform))
	f.close()	
	print ('Confirm the following CUDA Installation path for compilation:')
	print('CUDA Path:'+cuda_path)
	print('If Path is incorrect,please provide current path by typing CUDA or press any key to continue')
	user_input=input()
	if user_input.lower() == 'cuda':
		print('Enter path of your CUDA installation')
		cuda_path=input()  
	os.system('gcc --version')
	print('Enter gcc to install gcc compiler, or any other button to continue.')
	user_input=input()
	if user_input.lower() == 'gcc':
		os.system('sudo apt install gcc')
	print("Enter 'requirements' to install python packages dependencies")
	user_input=input()
	if user_input.lower() == 'requirements':
		os.system('pip install -r requirements.txt')
	print("Enter 'samples' to install latest version of CUDA Samples")
	user_input=input()
	if user_input.lower() == 'samples':
		os.system('cp -r src-original/patches src/')
		os.system('cp -r src-original/samples src/')
		os.system('rm -rf src/samples')
		os.chdir('src/')
		os.system('git clone https://github.com/NVIDIA/cuda-samples.git')
		os.system('mv cuda-samples samples')
		os.chdir('../')
		os.system('cp -r src-original/samples/Common/ src/samples/')
		os.chdir('src/samples')
		os.system('rm .gitignore')
		os.system('rm README.md')
		os.system('rm CHANGELOG.md')
		os.system('rm -rf .git')	
		os.system('rm LICENSE')
	print("Enter 'generate' to hipify additional files.")
	user_input=input()
	if user_input.lower() == 'generate':
		patch_gen.generate_all('src/samples/Samples')
		patch_gen2.generate_all('src/samples/Samples')
		patch_gen3.generate_all('src/samples/Samples')	
	print("Enter 'omp' to install OpenMP in your system, or any other button to continue.")
	user_input=input()
	if user_input.lower() == 'omp':
		os.system('sudo apt install libomp-dev')
		os.system('echo |cpp -fopenmp -dM |grep -i open')
		print('Enter number of threads ')
		x=int(input())
		os.system('export OMP_NUM_THREADS='+str(x))
		print("Always add -fopenmp flag on compilation.")
	print("Enter 'mpi' to install OpenMPI, or any other button to continue.It's better to install latest version from this link manually:https://sites.google.com/site/rangsiman1993/comp-env/program-install/install-openmpi")
	user_input=input()
	if user_input.lower()=='mpi':
		print('cd ~')
		os.chdir(os.path.expanduser("~"))
		print('wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.3.tar.gz')
		os.system('wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.3.tar.gz')
		print('tar -xzvf openmpi-3.1.3.tar.gz')
		os.system('tar -xzvf openmpi-3.1.3.tar.gz')
		os.system('mv -r ')
		print('cd openmpi-3.1.3')
		os.system('cd openmpi-3.1.3')
		os.chdir('openmpi-3.1.3')
		print('pwd')
		os.system('pwd')
		print('./configure --prefix=/usr/local/')
		os.system('./configure --prefix=/usr/local/')
		print('./configure --prefix=/usr/local/openmpi-3.1.3/')
		os.system('./configure --prefix=/usr/local/openmpi-3.1.3/')
		print('sudo make all install')
		os.system('sudo make all install')
		print('After make install is completed, mpirun or orterun executable should be at /usr/local/bin/.')
		print('echo "export PATH=$PATH:/usr/local/bin" >> $HOME/.bashrc')
		os.system('echo "export PATH=$PATH:/usr/local/bin" >> $HOME/.bashrc')
		print('echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" > $HOME/.bashrc')
		os.system('echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib" > $HOME/.bashrc')
		print('export PATH=$PATH:/usr/local/bin')
		os.system('export PATH=$PATH:/usr/local/bin')
		print('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib')
		os.system('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib')
		print('source $HOME/.bashrc')
		os.system('source $HOME/.bashrc')
		print('mpirun --version')
		os.system('mpirun --version')
	'''	
	listOfFiles=getListOfFiles('src/samples/Samples')
	print("Do you also want to generate files of extension cu.cpp for compilation on Nvidia devices?")
	user_input=input()
	if user_input.lower() == 'yes' or user_input.lower() == 'y':
		for elem in listOfFiles:
			if elem.endswith('.cu'):
				with open('final_ignored_samples.txt','r') as f:
					if elem in f.read():
						print("Ignoring this sample "+elem)
					else:
						elem2=elem+'.cpp'
						if os.path.exists(elem2)==False:
							print('Writing to '+elem2)
							with open(elem+'.hip','r') as f1, open(elem2,'a') as f2:
								for line in f1:
									f2.write(line)
	'''								
						
					
						
def parenthesis_check(file_name):
	string=''
	p=os.path.dirname(file_name)
	#f = open("a.cu.hip", "w")
	file=open(file_name,'r')
	file2=open(file_name,'r')
	while 1:
		char=file.read(1)
		if not char:
			break
		if char=='{' or char=='}':
			string+=char
	#print(string)
	result=check(string)
	#print(result)
	if result==1:
		#for line in file2:
			#print(line)
		lines = file2.readlines()
		lines = lines[:-1]	
		for elem in reversed(lines):
			if '}\n' not in elem:
				lines = lines[:-1]	
			else:
				break	
				
		#print(lines)	
		with open(p+'/a.cu.hip','w') as fp:
			for item in lines:
				fp.write(item)
		file.close()
		file2.close()
		os.remove(file_name)
		os.rename(p+'/a.cu.hip', file_name)

			
def parenthesis_check_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu.hip'):  ##or elem.endswith('.cpp') 
			parenthesis_check(elem)			


open_list = ["{"]
close_list = ["}"]
def check(myStr):
    stack = []
    for i in myStr:
        if i in open_list:
            stack.append(i)
        elif i in close_list:
            pos = close_list.index(i)
            if ((len(stack) > 0) and
                (open_list[pos] == stack[len(stack)-1])):
                stack.pop()
            else:
                return 1
    if len(stack) == 0:
        return 0
    else:
        return 1
  
def ftale(x):
	generate(x)
	apply_patches_individually(x)
	compilation_1(x)
	compilation_2(x)
	runsample(x)
	
def generate_all(y):
	global user_platform
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
				#generate(elem)	
				with open('final_ignored_samples.txt','r') as f:
					if elem in f.read():
						print("Ignoring this sample "+elem)
					else:
						generate(elem)						
	apply_patches()
	#find . -type f -name '*.cu.hip' -print -delete	
	#print("Do you also want to generate files of extension cu.cpp for compilation on Nvidia devices?")
	#user_input=input()
	#if user_input.lower() == 'yes' or user_input.lower() == 'y':
	if user_platform.lower()=='nvidia' :
		for elem in listOfFiles:
			if elem.endswith('.cu'):
				with open('final_ignored_samples.txt','r') as f:
					if elem in f.read():
						print("Ignoring this sample "+elem)
					else:
						elem2=elem+'.cpp'
						if os.path.exists(elem2)==False:
							print('Writing to '+elem2)
							with open(elem+'.hip','r') as f1, open(elem2,'a') as f2:
								for line in f1:
									f2.write(line)			
		

def compilation_1_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					compilation_1(elem)	

def compilation_2_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					compilation_2(elem)		

def runsample_all(y):
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					runsample(elem)	
	os.chdir('src/')
	print("In your src folder:")				
	print("Number of converted samples:")
	os.system('find . -name "*.cu.hip" | wc -l')
	print("Number of executables .out / .o:")
	os.system('find . -name "*.out" | wc -l')
	os.system('find . -name "*.o" | wc -l')
	print("Number of Ignored Samples:")
	os.system('cat ../final_ignored_samples.txt | wc -l')	
	os.chdir('../src-original')
	print("In src-original folder:")
	print("Number of converted samples:")
	os.system('find . -name "*.cu.hip" | wc -l')
	print("Number of executables .out / .o:")
	os.system('find . -name "*.out" | wc -l')
	os.system('find . -name "*.o" | wc -l')
	print("Number of Ignored Samples:")
	os.system('cat final_ignored_samples.txt | wc -l')																		

def generate(x):
	x=x.replace('"', '')
	p=os.path.dirname(x)
	q=os.path.basename(x)
	p=p.replace("\\","/")
	os.system("cd "+p)
	"""
	with open(p+"/"+q, 'r') as fp:
		lines = fp.readlines()
		for row in lines:
			word = '#include <GL/glu.h>'
			if row.find(word) == 0:
				flag=1
			else:
				flag=0	

	if flag==1:
		print("GL Headers found")	
		"""
	#$ sed 's/checkCudaErrors/HIPCHECK/g' asyncAPI.cu.hip
	#command="hipify-clang -Isrc/samples/Common "+x+" > "+x+".hip"
	command="hipify-perl "+x+" > "+x+".hip"
	print(command)
	os.system(command)
	prepend_line(x+".hip",'#include "HIPCHECK.h"\n')
	prepend_line(x+".hip",'#include "rocprofiler.h"\n')
	textToSearch="checkCudaErrors"
	textToReplace="HIPCHECK"
	fileToSearch=p+"/"+q+".hip"
	
	textToSearch1="#include <helper_cuda.h>"
	textToReplace1='#include "helper_cuda_hipified.h"'
	textToSearch2="#include <helper_functions.h>"
	textToReplace2='#include "helper_functions.h"'
	
	tempFile=open(fileToSearch,'r+')
	for line in fileinput.input(fileToSearch):
		tempFile.write(line.replace(textToSearch,textToReplace))
	tempFile.close()	
	
	tempFile=open(fileToSearch,'r+')
	for line in fileinput.input(fileToSearch):
		tempFile.write(line.replace(textToSearch1,textToReplace1))
	tempFile.close()
	tempFile=open(fileToSearch,'r+')
	for line in fileinput.input(fileToSearch):
		tempFile.write(line.replace(textToSearch2,textToReplace2))	
	tempFile.close()
	
	parenthesis_check(x+".hip")

def apply_patches():
	"""
	#command='git apply --reject --whitespace=fix src/patches/*.patch'
	command='git am src/patches/*.patch'
	print(command)
	os.system(command)
	os.system('rm -rf *.rej')
	"""
	y="src/patches"
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.patch'):  
			command='git apply --reject --whitespace=fix '+elem
			print(command)
			os.system(command)
			os.system('find . -name "*.rej" -type f -delete')

def apply_patches_individually(x):
	patch_path='src/patches'
	search_path=x+'.hip'
	patch_files=[]
	dir=os.listdir(patch_path)
	for fname in dir:
		if os.path.isfile(patch_path+os.sep+fname):
			f=open(patch_path+os.sep+fname,'r')
			if search_path in f.read():
				#print('found path in patch file '+fname)
				patch_files.append(fname)
			'''	
			else:
				print('Not found')
			'''	
			f.close()
	for patch in patch_files:
		command='git apply --reject --whitespace=fix '+patch_path+'/'+patch
		print(command)
		os.system(command)
		os.system('find . -name "*.rej" -type f -delete')			
	
def compilation_1(x):
	global cuda_path
	global user_platform
	cpp=[]
	print(user_platform)
	x=x.replace('"', '')
	p=os.path.dirname(x)
	p=p.replace("\\","/")
	if x=='src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu' and user_platform.lower() == 'amd':
		command='hipcc -I src/samples/Common src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu.hip src/samples/Samples/0_Introduction/simpleMPI/simpleMPI_hipified.cpp -lmpi -o src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.out'
		print(command)
		os.system(command)
	elif x=='src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu' or x=='/src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu' and user_platform.lower() == 'amd':
		command='hipcc -I src/samples/Common -fgpu-rdc src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu.hip src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu.hip -o src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.out'
		print(command)
		os.system(command)	
	elif x=='src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu' and user_platform.lower() == 'amd':
		command='hipcc -I src/samples/Common -fopenmp src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu.hip -o src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.out'
		print(command)
		os.system(command)
	if x=='src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu' and user_platform.lower() == 'nvidia':
		command='hipcc -I src/samples/Common src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu.cpp src/samples/Samples/0_Introduction/simpleMPI/simpleMPI_hipified.cpp -lmpi -o src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.out'
		print(command)
		os.system(command)
	elif x=='src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu' or x=='/src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu' and user_platform.lower() == 'nvidia':
		command='hipcc -I src/samples/Common -fgpu-rdc src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu.cpp src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu.cpp -o src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.out'
		print(command)
		os.system(command)	
	elif x=='src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu' and user_platform.lower() == 'nvidia':
		command='hipcc -I src/samples/Common -fopenmp src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu.cpp -o src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.out'
		print(command)
		os.system(command)		
	elif user_platform.lower()=='nvidia':
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.cpp"):
				cpp.append(file)	
	elif user_platform.lower()=='amd':	
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.hip"):
				cpp.append(file)	
		
			
		

	cpp = [p+'/'+y for y in cpp]
	command='hipcc -I src/samples/Common -I '+cuda_path+' '+' '.join(cpp)+' -o '+p+'/'+os.path.basename(os.path.dirname(x))+'.out'
	print(command)
	os.system(command)	

	

def compilation_2(x):
	global cuda_path
	global user_platform
	cpp=[]
	x=x.replace('"', '')
	p=os.path.dirname(x)
	p=p.replace("\\","/")
	if x=='src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu' and user_platform.lower() == 'amd':
		command='hipcc -use-staticlib -I src/samples/Common src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu.hip src/samples/Samples/0_Introduction/simpleMPI/simpleMPI_hipified.cpp -lmpi -o src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.out'
		print(command)
		os.system(command)
	elif x=='src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu' or x=='/src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu' and user_platform.lower() == 'amd':
		command='hipcc -use-staticlib -I src/samples/Common -fgpu-rdc src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu.hip src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu.hip -o src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.out'
		print(command)
		os.system(command)	
	elif x=='src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu' and user_platform.lower() == 'amd':
		command='hipcc -use-staticlib -I src/samples/Common -fopenmp src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu.hip -o src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.out'
		print(command)
		os.system(command)
	if x=='src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu' and user_platform.lower() == 'nvidia':
		command='hipcc -use-staticlib -I src/samples/Common src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.cu.cpp src/samples/Samples/0_Introduction/simpleMPI/simpleMPI_hipified.cpp -lmpi -o src/samples/Samples/0_Introduction/simpleMPI/simpleMPI.out'
		print(command)
		os.system(command)
	elif x=='src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu' or x=='/src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu' and user_platform.lower() == 'nvidia':
		command='hipcc -use-staticlib -I src/samples/Common -fgpu-rdc src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleDeviceLibrary.cu.cpp src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.cu.cpp -o src/samples/Samples/0_Introduction/simpleSeparateCompilation/simpleSeparateCompilation.out'
		print(command)
		os.system(command)	
	elif x=='src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu' and user_platform.lower() == 'nvidia':
		command='hipcc -use-staticlib -I src/samples/Common -fopenmp src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.cu.cpp -o src/samples/Samples/0_Introduction/cudaOpenMP/cudaOpenMP.out'
		print(command)
		os.system(command)		
	elif user_platform.lower()=='nvidia':
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.cpp"):
				cpp.append(file)	
	elif user_platform.lower()=='amd':	
		for file in os.listdir(p):
			if file.endswith("_hipified.cpp") or file.endswith(".cu.hip"):
				cpp.append(file)		
			
		

	cpp = [p+'/'+y for y in cpp]
	command='hipcc -use-staticlib -I src/samples/Common -I '+cuda_path+' '+' '.join(cpp)+' -o '+p+'/'+os.path.basename(os.path.dirname(x))+'.out'
	print(command)
	os.system(command)

def runsample(x):	
	print('Processing Sample:'+x)
	command='./'+os.path.dirname(x)+'/'+os.path.basename(os.path.dirname(x))+'.out'
	print(command)
	os.system(command)
	
	

	

		#os.system("cd "+p)
		#os.system("echo cd "+p)
		#os.system(r'echo sed -i.bak "s/checkCudaErrors/HIPCHECK/g" '+q)
		#os.system('sed -i.bak ""{}"" '.format(s)+q)

        #/data/driver/testhipify/src/samples/Samples/
		
		#command="/opt/rocm/bin/hipcc -I /home/taccuser/testhipify/src/samples/Common -I /home/taccuser/testhipify/src/samples/Common/GL -I /home/taccuser/testhipify/src/samples/Common/UtilNPP -I /home/taccuser/testhipify/src/samples/Common/data -I /home/taccuser/testhipify/src/samples/Common/lib/x64 "+p+"/"+os.path.basename(x)+".hip"
		#/opt/rocm-5.4.0-10890/bin
	#command='/opt/rocm-5.4.0-10890/bin/hipcc -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common/UtilNPP -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common/data -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common/lib/x64 '+p+'/'+q+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out'
	#print(command)
	#os.system(command)

    #hipcc  square.cpp -o square.out(done)
	#command='hipcc '+p+'/'+os.path.basename(x)+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out'
	#print(command)
	#os.system(command)
	#/home/user/hip/bin/hipcc -use-staticlib  square.cpp -o square.out.static
	#command='/opt/rocm-5.4.0-10890/bin/hipcc -use-staticlib '+p+'/'+q+'.hip -o '+p+'/'+os.path.splitext(x)[0]+'.out.static'
	#print(command)
	#os.system(command)
	#./square.out
	#command='./'+os.path.splitext(x)[0]+'.out'
	#print(command)
	#os.system(command)
	#command2='hipcc -I /long_pathname_so_that_rpms_can_package_the_debug_info/data/driver/testhipify/src/samples/Common asyncAPI.cu.hip -o asynAPI.out'
    

def fall(y):
	"""
	y=y.replace('"', '')
	listOfFiles=getListOfFiles(y)
	for elem in listOfFiles:
		if elem.endswith('.cu'):  ##or elem.endswith('.cpp') 
			with open('final_ignored_samples.txt','r') as f:
				if elem in f.read():
					print("Ignoring this sample "+elem)
				else:
					ftale(elem)
	"""
	generate_all(y)
	compilation_1_all(y)
	compilation_2_all(y)
	runsample_all(y)


def rem(z):
	print("This script automates sample exclusion.Please backup any paths provided by you to avoid loss or overwriting.")
	input("Press Enter to continue...")
	a=open("samples_to_be_ignored.txt","w")
	a.truncate(0)	
	b=open("final_ignored_samples.txt","w")
	b.close()
	z=z.replace('"','')
	#ignore_list = ['<GL/', '<screen', '<drm.h>','FDTD3dGPU.h','#include <d312.h>\n',' #include <GLES3/gl31.h>\n','#include <windows.h>\n','#include <omp.h>\n','#include "nvmedia_image_nvscibuf.h"\n','#include "graphics_interface.c"\n','#include <DirectXMath.h>\n']
	'''
	ignore_list = ['<GL/','<screen/screen.h>', '<drm.h>','"FDTD3dGPU.h"','<d3d12.h>',
	' <GLES3/gl31.h>','<EGL/egl.h>','<GLFW/glfw3.h>','"cudla.h"','#include <d312.h>\n',
	' #include <GLES3/gl31.h>\n','#include <windows.h>\n','#include "nvmedia_image_nvscibuf.h"\n',
	'#include "graphics_interface.c"\n','#include <DirectXMath.h>\n','#include "cuda_gl_interop.h"',
	'#include "cudaEGL.h"','#include "cudaEGL.h"\n','"cuda_gl_interop.h"','#include "cuda_runtime.h"\n',
	'#include "cudla.h"\n','#include "nvscisync.h"\n','#include "FDTD3d.h"\n','#include "windows.h"\n',
	'#include "builtin_types.h"\n','#include "hipfft.h"\n','#include "screen/screen.h"\n','#include "Windows.h"\n',
	'#include "cuda_d3d9_interop.h"\n','#include "drm.h"\n','#include "cuda_runtime_api.h"\n','#include "GLFW/glfw3.h"\n',
	'#include "cuda/barrier"\n','#include "cuda_runtime.h"\n','#include "cooperative_groups/reduce.h"\n',
	'#include "cuda_bf16.h"\n','#include "mma.h"\n','#include "cuda/pipeline"\n','"builtin_types.h"']
	'''
	ignore_list = ['<GL/','<screen/screen.h>', '<drm.h>','"FDTD3dGPU.h"','<d3d12.h>',
	' <GLES3/gl31.h>','<EGL/egl.h>','<GLFW/glfw3.h>','"cudla.h"','#include <d312.h>',
	' #include <GLES3/gl31.h>','#include <windows.h>','#include "nvmedia_image_nvscibuf.h"',
	'#include "graphics_interface.c"','#include <DirectXMath.h>','#include "cuda_gl_interop.h"',
	'#include "cudaEGL.h"','#include "cudaEGL.h"','"cuda_gl_interop.h"','#include "cuda_runtime.h"',
	'#include "cudla.h"','#include "nvscisync.h"','#include "FDTD3d.h"','#include "windows.h"',
	'#include "builtin_types.h"','#include "hipfft.h"','#include "screen/screen.h"','#include "Windows.h"',
	'#include "cuda_d3d9_interop.h"','#include "drm.h"','#include "cuda_runtime_api.h"','#include "GLFW/glfw3.h"',
	'#include "cuda/barrier"','#include "cuda_runtime.h"','#include "cooperative_groups/reduce.h"',
	'#include "cuda_bf16.h"','#include "mma.h"','#include "cuda/pipeline"','"builtin_types.h"']
	listofFiles=getListOfFiles(z)
	for elem in listofFiles:
		if elem.endswith('.cu'):
			with open(elem) as f:
				for line in f:
					if any(word in line for word in ignore_list):
						#a.write(elem+"\n"+line)
						a.write(elem+"\n")
		elif elem.endswith('_hipified.cpp'):
			with open(elem) as f:
				for line in f:
					if any(word in line for word in ignore_list):
						for file in os.listdir(os.path.dirname(elem)):
							if file.endswith('.cu'):
								a.write(os.path.dirname(elem)+'/'+file+"\n")

						
	a.close()
	"""
	lines_seen = set()
	outfile = open('final_ignored_samples.txt', "w")
	infile = open('samples_to_be_ignored.txt', "r")
	for line in infile:
		if line not in lines_seen: # not a duplicate
			outfile.write(line)
			lines_seen.add(line)
	outfile.close()
	"""
	with open('samples_to_be_ignored.txt') as fp:
		data1 = fp.read()	
	with open('custom_samples_path.txt') as fp:
		data2 = fp.read()
	data1 += data2
	with open ('final_ignored_samples.txt', 'a') as fp:
		fp.write(data1)
	uniqlines = set(open('final_ignored_samples.txt').readlines())	
	bar = open('final_ignored_samples.txt', 'w')
	bar.writelines(uniqlines)
	bar.close()
	fin = open('final_ignored_samples.txt', "rt")
	data = fin.read()
	data = data.replace('\\', '/')
	fin.close()
	fin = open("final_ignored_samples.txt", "wt")
	fin.write(data)
	fin.close()
	lines_seen = set() # holds lines already seen
	outfile = open('final_ignored_samples1.txt', "w")
	for line in open('final_ignored_samples.txt', "r"):
		#outfile.writelines(sorted(lines_seen))
		if line not in lines_seen: # not a duplicate
			outfile.write(line)
			lines_seen.add(line)
	outfile.close()
	os.remove('final_ignored_samples.txt')
	if platform=="linux" or platform == "linux2":
		os.system('sort final_ignored_samples1.txt > final_ignored_samples.txt')
		#os.rename("final_ignored_samples1.txt","final_ignored_samples.txt")
		os.remove('final_ignored_samples1.txt')
	else:
		sorting('final_ignored_samples1.txt')
	
	#os.rename("final_ignored_samples1.txt","final_ignored_samples.txt")
	os.remove('samples_to_be_ignored.txt')
'''
def main():
	parser=argparse.ArgumentParser(description ='HIPIFY Cuda Samples.Please avoid and ignore samples with graphical operations')
	parser.add_argument("-a", "--all", help='To run hipify-perl for all sample:python testhipify.py --all "[PATH TO SAMPLE FOLDER]"')
	parser.add_argument("-b", "--generate", help='Generate .hip files')
	parser.add_argument("-c", "--compile1", help='Compile .hip files')
	parser.add_argument("-d", "--compile2", help='Compile .hip files with static libraries')
	parser.add_argument("-e", "--execute", help='Execute .out files')
	parser.add_argument("-f", "--generate_all", help='Generate all .hip files')
	parser.add_argument("-g", "--compile1_all", help='Compile all .hip files')
	parser.add_argument("-i", "--compile2_all", help='Compile all .hip files with static libraries')
	parser.add_argument("-j", "--execute_all", help='Execute all .out files')
	parser.add_argument("-k", "--parenthesis_check", help='Remove last parts from cu.hip files which are out of bounds.')
	parser.add_argument("-l", "--parenthesis_check_all", help='Remove all last parts from cu.hip files which are out of bounds.')
	parser.add_argument("-p", "--patch", help='Apply all patches in src/patches',action='store_true')
	parser.add_argument("-t", "--tale", help='To run hipify-perl for single sample:python testhipify.py -t "[PATH TO SAMPLE]"')
	parser.add_argument("-x", "--remove", help='Remove any sample relating to graphical operations e.g.DirectX,Vulcan,OpenGL,OpenCL and so on.')
	parser.add_argument("-s", "--setup", help='Configure dependencies.',action='store_true')
	args=parser.parse_args()
	if args.tale:
			x=args.tale
			##print(x)
			ftale(x)
	if args.all:
			y=args.all
			##print(y)
			fall(y)
	if args.remove:
			z=args.remove
			rem(z)
	if args.generate:
			a=args.generate
			generate(a)
	if args.patch:
			apply_patches()	
	if args.compile1:
			b=args.compile1
			compilation_1(b)
	if args.compile2:
			c=args.compile2
			compilation_2(c)
	if args.execute:
			d=args.execute
			runsample(d)	
	if args.generate_all:
			a=args.generate_all
			generate_all(a)
	if args.compile1_all:
			b=args.compile1_all
			compilation_1_all(b)
	if args.compile2_all:
			c=args.compile2_all
			compilation_2_all(c)
	if args.execute_all:
			d=args.execute_all
			runsample_all(d)
	if args.parenthesis_check:
			e=args.parenthesis_check
			parenthesis_check(e)
	if args.parenthesis_check_all:
			f=args.parenthesis_check_all
			parenthesis_check_all(f)
	if args.setup:
			setup()

if __name__ == "__main__":
    main()  		
'''
