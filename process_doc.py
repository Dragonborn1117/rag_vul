import os
import subprocess

def traversal_files(path):
    doc_dirs = []
    
    print("traversal files...")
    
    for item in os.scandir(path):
        if item.is_dir():
          file = item.path + '/README.md'
          doc_dirs.append(file)
    
    index = 0
    for file in doc_dirs:
        src_file = file
        dst_file = f'test/doc/{index}.md'
        index += 1
        subprocess.run(["cp", src_file, dst_file])

    
file_path = 'darpa/samples-master/cqe-challenges'
traversal_files(file_path)
