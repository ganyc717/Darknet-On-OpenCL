'''
This python script is to generate buildin kernel source file,
if you change or add the kernel source *.cl, please run this script and
generate new cl_kernel_source.h
'''
import os
import shutil
HEADER_FILE_NAME = "cl_kernel_source.h"

def extension(file_path):
    return os.path.splitext(file_path)[1]

def main():
    files = os.listdir("./")
    files = [file for file in files if extension(file) == ".cl"]

    with open(HEADER_FILE_NAME, "w") as des:
        des.writelines("#ifndef CL_KERNEL_SOURCE\n")
        des.writelines("#define CL_KERNEL_SOURCE\n")
        des.writelines("#include<map>\n")
        for file in files:
            source_name = "std::string " + file[:-3] + " = \"\\n\\\n"
            des.writelines(source_name)
            with open(file, "r") as src:
                while True:
                    line = src.readline()
                    if line == "":
                        break
                    else:
                        line = line.replace("\n","") + "\\n\\\n"
                        des.writelines(line)
            end = "\"; \n"
            des.writelines(end)

        des.writelines("std::map<std::string,std::string> source_map = {\n")
        for file in files:
            des.writelines("std::make_pair(\"" + file + "\"," + file[:-3] + "),\n")
        des.writelines("};\n")
        des.writelines("#endif")
    shutil.copyfile(HEADER_FILE_NAME,"../include/"+HEADER_FILE_NAME)

if __name__ == "__main__":
    main()